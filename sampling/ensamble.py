import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd


from .sampler import find_crossing
from .sampler import my_while



class vectorize_target:

    def __init__(self, target):

        self.nlogp = jax.vmap(target.nlogp)
        self.grad_nlogp = jax.vmap(target.grad_nlogp)
        self.transform = jax.vmap(target.transform)
        self.prior_draw = jax.vmap(target.prior_draw)
        self.d = target.d
        if hasattr(target, 'variance'):
            self.variance = target.variance



class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L=None, eps=None, varE_wanted = 0.01):
        """Args:
                Target: the target distribution class
        """

        self.Target = vectorize_target(Target)

        self.grad_evals_per_step = 1.0
        self.varE_wanted = varE_wanted

        if (not (L is None)) and (not (eps is None)):
            self.set_hyperparameters(L, eps)


    def set_hyperparameters(self, L, eps):
        self.L = L
        self.eps = eps
        self.nu = jnp.sqrt((jnp.exp(2 * self.eps / L) - 1.0) / self.Target.d)


    def energy(self, x, r):
        return (self.Target.d-1) * r + self.Target.nlogp(x)


    def random_unit_vector(self, key, num_chains):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape = (num_chains, self.Target.d), dtype = 'float64')
        normed_u = u / jnp.sqrt(jnp.sum(jnp.square(u), axis = 1))[:, None]
        return normed_u, key


    def partially_refresh_momentum(self, u, key):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(key)
        noise = self.nu * jax.random.normal(subkey, shape= u.shape, dtype=u.dtype)

        return (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise), axis = 1))[:, None], key


    def update_momentum(self, eps, g, u):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g), axis = 1)).T
        e = - g / g_norm[:, None]
        ue = jnp.sum(u*e, axis = 1)
        sh = jnp.sinh(eps * g_norm / (self.Target.d - 1))
        ch = jnp.cosh(eps * g_norm / (self.Target.d - 1))
        th = jnp.tanh(eps * g_norm / (self.Target.d-1))
        delta_r = jnp.log(ch) + jnp.log1p(ue * th)

        return (u + e * (sh + ue * (ch - 1))[:, None]) / (ch + ue * sh)[:, None], delta_r


    def hamiltonian_dynamics(self, x, u, g, key):
        """leapfrog"""

        z = x / self.sigma # go to the latent space

        # half step in momentum
        uu, delta_r1 = self.update_momentum(self.eps * 0.5, g * self.sigma, u)

        # full step in x
        zz = z + self.eps * uu
        xx = self.sigma * zz # go back to the configuration space
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(self.eps * 0.5, gg * self.sigma, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)
        return xx, uu, l, gg, kinetic_change, key


    def dynamics(self, x, u, g, key):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, l, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, key)

        # bounce
        uu, key = self.partially_refresh_momentum(uu, key)

        return xx, uu, l, gg, kinetic_change, key


    def full_b(self, X):

        X_sq = jnp.average(jnp.square(X), axis= 0)

        def step(F2, index):
            x_sq = X_sq[index, :]
            F2_new = (F2 * index + x_sq) / (index + 1)  # Update <f(x)> with a Kalman filter
            b = jnp.sqrt(jnp.average(jnp.square((F2_new - self.Target.variance) / self.Target.variance)))

            return F2_new, b

        return jax.lax.scan(step, jnp.zeros(self.Target.d), xs=jnp.arange(len(X_sq)))[1]


    def virial_loss(self, x, g):
        """loss^2 = (1/d) sum_i (virial_i - 1)^2"""

        virials = jnp.average(x*g, axis=0) #should be all close to 1 if we have reached the typical set
        return jnp.sqrt(jnp.average(jnp.square(virials - 1.0)))


    def initialize(self, random_key, x_initial, num_chains):

        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains + 1)
                x = self.Target.prior_draw(keys_all[1:])
                key = keys_all[0]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')

        else:  # initial x is given
            x = jnp.copy(x_initial)


        l, g = self.Target.grad_nlogp(x)

        ### initial velocity ###
        virials = jnp.average(x * g, axis=0)
        loss = jnp.sqrt(jnp.average(jnp.square(virials - 1.0)))
        sgn = -2.0 * (virials < 1.0) + 1.0
        u = - g / jnp.sqrt(jnp.sum(jnp.square(g), axis = 1))[:, None] # initialize momentum in the direction of the gradient of log p
        u = u * sgn[None, :]

        # u, key = self.random_unit_vector(key, num_chains) #random velocity orientations

        return loss, x, u, l, g, key


    def burn_in(self, loss, x, u, l, g, key):

        max_burn_in = 200


        def burn_in_step(state):

            index, loss, x, u, l, g, key, _ = state
            self.sigma = jnp.std(x, axis=0)  # diagonal conditioner

            xx, uu, ll, gg, kinetic_change, key = self.dynamics(x, u, g, key)  # update particles by one step

            loss_new = self.virial_loss(xx, gg)

            #if the loss has not decreased, let's reduce the stepsize
            no_nans = jnp.all(jnp.isfinite(xx))
            tru = (loss_new < loss)*no_nans #loss went down and there were no nans
            false = (1 - tru)
            new_eps = self.eps * (false * 0.5 + tru * 1.0)
            self.set_hyperparameters(self.L, new_eps)

            #lets update the state if the loss went down
            loss = loss_new * tru + loss * false
            x = xx * tru + x * false
            u = uu * tru + u * false
            l = ll * tru + l * false
            g = gg * tru + g * false

            energy_change = kinetic_change + ll - l

            return index + 1, loss, x, u, l, g, key, energy_change


        condition = lambda state: (state[1] > 0.2)*(state[0] < max_burn_in)  # false if the burn-in should be ended

        burn_in_steps, loss, x, u, l, g, key, energy_change = my_while(condition, burn_in_step, (0, loss, x, u, l, g, key, jnp.zeros(l.shape)))

        #after you are done with developing, replace, my_while with jax.lax.while_loop

        if burn_in_steps == max_burn_in:
            print('Burn-in exceeded the predescribed number of iterations, loss = '.format(loss))

        return burn_in_steps, x, u, l, g, key, energy_change



    def sample(self, num_steps, num_chains, x_initial='prior', random_key= None, output = 'normal', thinning= 1):


        state = self.initialize(random_key, x_initial, num_chains) #initialize

        burn_in_steps, x, u, l, g, key, energy_change = self.burn_in(*state) #burn-in

        print(burn_in_steps)
        exit()

        ### prepare for sampling ###
        self.sigma = jnp.std(x, axis=0)
        varE = jnp.std(energy_change)**2 / self.Target.d
        self.set_hyperparameters(self.L, self.eps * jnp.power(self.varE_wanted / varE, 1.0/6.0)) # assume var[E] ~ eps^6 for the estimator used her




        ### sampling ###

        X = jnp.empty((num_chains, num_steps, self.Target.d)) # we will store the samples here
        X = X.at[:, 0, :].set(x) #initial condition

        for i_step in range(1, num_steps):

            x, u, ll, g, kinetic_change, key = self.dynamics(x, u, g, key)  # update particles by one step
            energy_change = kinetic_change + ll - l
            l = ll

            if i_step<10: #at the begining we adjust the stepsize according to the energy fluctuations
                varE = jnp.std(energy_change) ** 2 / self.Target.d
                self.set_hyperparameters(self.L, self.eps * jnp.power(self.varE_wanted / varE, 0.25))  # assume var[E] ~ eps^4

            X = X.at[:, i_step, :].set(x) #store the sample


        ### return results ###

        if output == 'ess': #we return the number of sampling steps (needed for b2 < 0.1) and the number of burn-in steps
            b2 = self.full_b(X)
            print(self.eps)
            plt.plot(b2)
            plt.show()
            no_nans = 1-jnp.any(jnp.isnan(b2))
            cutoff_reached = b2[-1] < 0.1
            return (find_crossing(b2, 0.1), burn_in_steps) if (no_nans and cutoff_reached) else (np.inf, burn_in_steps)


        else:

            if output == 'normal': #return the samples X
                return X[::thinning]

            else:
                raise ValueError('output = ' + output + 'is not a valid argument for the Sampler.sample')

