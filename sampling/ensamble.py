import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd


class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L=None, eps=None):
        """Args:
                Target: the target distribution class
        """

        self.Target = Target

        self.grad_evals_per_step = 1.0

        if (not (L is None)) and (not (eps is None)):
            self.set_hyperparameters(L, eps)


    def set_hyperparameters(self, L, eps):
        self.L = L
        self.eps = eps
        self.nu = jnp.sqrt((jnp.exp(2 * self.eps / L) - 1.0) / self.Target.d)


    def random_unit_vector(self, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape=(self.Target.d,), dtype='float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def partially_refresh_momentum(self, u, key):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(key)
        noise = self.nu * jax.random.normal(subkey, shape=(self.Target.d,), dtype='float64')

        return (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise))), key


    def update_momentum(self, eps, g, u):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        sh = jnp.sinh(eps * g_norm / (self.Target.d - 1))
        ch = jnp.cosh(eps * g_norm / (self.Target.d - 1))

        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh)


    def hamiltonian_dynamics(self, x, g, u, key):
        """leapfrog"""

        z = x / self.Sigma_sq

        # half step in momentum
        uu = self.update_momentum(self.eps * 0.5, g * self.Sigma_sq, u)

        # full step in x
        xx = z + self.eps * uu
        gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu = self.update_momentum(self.eps * 0.5, gg * self.Sigma_sq, uu)

        return xx, gg, uu, key


    def dynamics(self, x, u, g, key):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, gg, uu, key = self.hamiltonian_dynamics(x, g, u, key)

        # bounce
        uu, key = self.partially_refresh_momentum(uu, key)

        return xx, uu, gg, key


    def full_b(self, X_sq):

        def step(F2, index):
            x_sq = X_sq[index, :]
            F2_new = (F2 * index + x_sq) / (index + 1)  # Update <f(x)> with a Kalman filter
            b = jnp.sqrt(jnp.average(jnp.square((F2_new - self.Target.variance) / self.Target.variance)))

            return F2_new, b

        return jax.lax.scan(step, jnp.zeros(self.Target.d), xs=jnp.arange(len(X_sq)))[1]


    def sample(self, num_chains, num_steps, x_initial='prior', random_key=None):
        """Run multiple chains. The initial conditions for each chain are drawn with self.Target.prior_draw"""

        ### initialization ###

        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains * 2)
                x = jnp.array([self.Target.prior_draw(keys_all[num_chains + i]) for i in range(num_chains)])
                keys = keys_all[:num_chains]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError(
                    'x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')

        else:  # initial x is given
            x = jnp.copy(x_initial)
            keys = jax.random.split(key, num_chains)

        g = self.Target.grad_nlogp(x)
        u = - g / jnp.sqrt(jnp.sum(jnp.square(g)))  # initialize momentum in the direction of the gradient of log p

        dynamics_parallel = jax.vmap(self.dynamics)

        ### burn-in ###

        max_burn_in = 1000
        burn_in_steps = max_burn_in
        # self.entropy = jnp.average(self.Target.nlogp(x))

        for i_step in range(1, max_burn_in):

            self.Sigma_sq = jnp.std(x, axis=0)

            x, u, g, keys = dynamics_parallel(x, u, g, keys)  # update particles by one step

            virial = np.average(np.average(g * x, axis=1), axis=0)

            if virial < 1.1:  # end of burn-in
                burn_in_steps = i_step
                break

            # entropy = jnp.average(self.Target.nlogp(x))  # if self.entropy < entropy:  #     self.set_hyperparameters(self.L, self.eps * 0.5)  #     print('reducing step size: {}'.format(self.eps))  # self.entropy = entropy

            # renormalized

        if burn_in_steps == max_burn_in:
            print('Burn-in exceeded the predescribed number of iterations.')

        ### will be added: digonal preconditioning and perhaps L-BFGS? ###

        ### sampling ###
        self.set_hyperparameters(self.L, self.eps * 0.75)
        print('reducing step size: {}'.format(self.eps))

        X = jnp.empty((num_chains, num_steps, self.Target.d))
        X = X.at[:, 0, :].set(x)

        for i_step in range(1, steps):

            x, u, g, keys = dynamics_parallel(x, u, g, keys)  # update particles by one step

            X = X.at[:, i_step, :].set(x)

        return X, burn_in_steps