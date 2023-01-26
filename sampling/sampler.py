import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from .jump_identification import remove_jumps
from .correlation_length import ess_corr

jax.config.update('jax_enable_x64', True)


lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L = None, eps = None, integrator = 'MN', generalized= True):
        """Args:
                Target: the target distribution class
                L: momentum decoherence scale (can be tuned automatically)
                eps: integration step-size (can be tuned automatically)
                integrator: 'LF' (leapfrog) or 'MN' (minimal norm). Typically MN performs better.
                generalized: True (Langevin-like momentum decoherence) or False (bounces).
        """

        self.Target = Target

        ### integrator ###
        if integrator == "LF": #leapfrog (first updates the velocity)
            self.hamiltonian_dynamics = self.leapfrog
            self.grad_evals_per_step = 1.0
        elif integrator== 'MN': #minimal norm integrator (velocity)
            self.hamiltonian_dynamics = self.minimal_norm
            self.grad_evals_per_step = 2.0
        elif integrator == 'RM':
            self.hamiltonian_dynamics = self.randomized_midpoint
            self.grad_evals_per_step = 1.0
        else:
            print('integrator = ' + integrator + 'is not a valid option.')


        ### decoherence mechanism ###
        self.dynamics = self.dynamics_generalized if generalized else self.dynamics_bounces


        if (not (L is None)) and (not (eps is None)):
            self.set_hyperparameters(L, eps)



    def set_hyperparameters(self, L, eps):
        self.L = L
        self.eps= eps
        self.nu = jnp.sqrt((jnp.exp(2 * self.eps / L) - 1.0) / self.Target.d)


    def random_unit_vector(self, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def partially_refresh_momentum(self, u, key):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(key)
        z = self.nu * jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')

        return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key


    def update_momentum(self, eps, g, u):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        sh = jnp.sinh(eps * g_norm / (self.Target.d-1))
        ch = jnp.cosh(eps * g_norm / (self.Target.d-1))
        th = jnp.tanh(eps * g_norm / (self.Target.d-1))
        delta_r = jnp.log(ch) + jnp.log1p(ue * th)

        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), delta_r


    def leapfrog(self, x, u, g, key):
        """leapfrog"""

        #half step in momentum
        uu, r1 = self.update_momentum(self.eps * 0.5, g, u)

        #full step in x
        xx = x + self.eps * uu
        ll, gg = self.Target.grad_nlogp(xx)

        #half step in momentum
        uu, r2 = self.update_momentum(self.eps * 0.5, gg, uu)

        return xx, uu, ll, gg, (r1 + r2)*(self.Target.d-1), key



    def minimal_norm(self, x, u, g, key):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V

        uu, r1 = self.update_momentum(self.eps * lambda_c, g, u)

        xx = x + self.eps * 0.5 * uu
        ll, gg = self.Target.grad_nlogp(xx)

        uu, r2 = self.update_momentum(self.eps * (1 - 2 * lambda_c), gg, uu)

        xx = xx + self.eps * 0.5 * uu
        ll, gg = self.Target.grad_nlogp(xx)

        uu, r3 = self.update_momentum(self.eps * lambda_c, gg, uu)

        return xx, uu, ll, gg, (r1 + r2 + r3) * (self.Target.d-1), key


    #
    # def randomized_midpoint(self, x, u, g, r, key):
    #
    #     key1, key2 = jax.random.split(key)
    #
    #     xx = x + jax.random.uniform(key2) * self.eps * u
    #
    #     gg = self.Target.grad_nlogp(xx)
    #
    #     uu, r1 = self.update_momentum(self.eps, gg, u)
    #
    #     xx = self.update_position_RM(xx, )
    #
    #
    #     return xx, uu, gg, r1 * (self.Target.d-1), key1



    def dynamics_bounces(self, state):
        """One step of the dynamics (with bounces)"""
        x, u, l, g, E, key, time = state

        # Hamiltonian step
        xx, uu, ll, gg, dK, key = self.hamiltonian_dynamics(x, u, g, key)

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps
        do_bounce = time > self.L
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, ll, gg, E+dK + ll -l, key, time


    def dynamics_generalized(self, state):
        """One step of the generalized dynamics."""

        x, u, l, g, E, key, time = state

        # Hamiltonian step
        xx, uu, ll, gg, dK, key = self.hamiltonian_dynamics(x, u, g, key)

        # bounce
        uu, key = self.partially_refresh_momentum(uu, key)

        return xx, uu, ll, gg, E+dK+ll-l, key, 0.0



    def get_initial_conditions(self, x_initial, random_key):

        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        ### initial conditions ###
        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                key, prior_key = jax.random.split(key)
                x = self.Target.prior_draw(prior_key)
            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError(
                    'x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else: #initial x is given
            x = x_initial

        l, g = self.Target.grad_nlogp(x)

        u, key = self.random_unit_vector(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, l, g, key



    def sample(self, num_steps, x_initial = 'prior', random_key= None, ess=False, monitor_energy= False, final_state = False, burn_in = 0):
        """Args:
               num_steps: number of integration steps to take.
               x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
               eps: jax radnom seed, e.g. jax.random.PRNGKey(42).
               ess: if True, it only ouputs the Effective Sample Size. In this case self.Target.variance = <x_i^2>_true should be defined.
               monitor_energy: also tracks the energy error (energy should be conserved but there are numerical errors). Outputs samples, weights and energy (at each step).
               final_state: only returns the final x of the chain (not self.Target.transform(x)!)

            Returns:
                samples (shape = (num_steps, self.Target.d))
                weights (shape = (num_steps))

                Except if ess == True, monitor_energy == True or final_state == True.
        """

        def step(state, useless):
            """Tracks transform(x) as a function of number of iterations"""

            x, u, l, g, E, key, time = self.dynamics(state)

            return (x, u, l, g, E, key, time), (self.Target.transform(x), E)



        def b_step(state_track, useless):
            """Only tracks b as a function of number of iterations."""

            x, u, l, g, E, key, time = self.dynamics(state_track[0])
            W, F2 = state_track[1]

            F2 = (W * F2 + jnp.square(self.Target.transform(x)))/ (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))
            #bias = jnp.average((F2 - self.Target.variance) / self.Target.variance)

            return ((x, u, l, g, E, key, time), (W, F2)), bias


        x, u, l, g, key = self.get_initial_conditions(x_initial, random_key)


        ### do sampling ###

        if ess:  # only track the bias

            _, b = jax.lax.scan(b_step, init=((x, u, l, g, 0.0, key, 0.0), (1.0, jnp.square(x))), xs=None, length=num_steps)


            no_nans = 1-jnp.any(jnp.isnan(b))
            cutoff_reached = b[-1] < 0.1

            # plt.plot(bias, '.')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

            return ess_cutoff_crossing(b) * no_nans * cutoff_reached / self.grad_evals_per_step #return 0 if there are nans, or if the bias cutoff was not reached


        else: # track the full transform(x)

            state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, key, 0.0), xs=None, length=num_steps)
            if final_state: #only return the final x
                return state[0]
            elif monitor_energy: #return the samples X and the energy E
                return track[0][burn_in:], track[1][burn_in:]
            else: #return the samples X
                return track[0][burn_in:]


    def parallel_sample(self, num_chains, num_steps, x_initial = 'prior', random_key= None, ess= False, monitor_energy= False, final_state = False, num_cores= 1):
        """Run multiple chains. The initial conditions for each chain are drawn with self.Target.prior_draw"""

        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key


        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains * 2)
                x0 = jnp.array([self.Target.prior_draw(keys_all[num_chains+i]) for i in range(num_chains)])
                keys = keys_all[:num_chains]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else: #initial x is given
            x0 = jnp.copy(x_initial)
            keys = jax.random.split(key, num_chains)


        f = lambda i: self.sample(num_steps, x_initial= x0[i], random_key=keys[i], ess=ess, monitor_energy=monitor_energy, final_state= final_state)

        if num_cores != 1: #run the chains on parallel cores
            parallel_function = jax.pmap(jax.vmap(f))
            results = parallel_function(jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores))
            ### reshape results ###
            if type(results) is tuple: #each chain returned a tuple
                results_reshaped =[]
                for i in range(len(results)):
                    res = jnp.array(results[i])
                    results_reshaped.append(res.reshape([num_chains, ] + [res.shape[j] for j in range(2, len(res.shape))]))
                return results_reshaped

            else:
                return results.reshape([num_chains, ] + [results.shape[j] for j in range(2, len(results.shape))])


        else: #run chains serially on a single core

            return jax.vmap(f)(jnp.arange(num_chains))



    def tune_hyperparameters(self, x_initial = 'prior', random_key= None, dialog = False):

        varE_wanted = 0.0005             # targeted energy variance per dimension
        burn_in, samples = 2000, 1000


        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key


        self.set_hyperparameters(np.sqrt(self.Target.d), 0.6)

        key, subkey = jax.random.split(key)
        x0 = self.sample(burn_in, x_initial, random_key= subkey, final_state= True)
        props = (key, np.inf, 0.0, False)
        if dialog:
            print('Hyperparameter tuning (first stage)')

        def tuning_step(props):

            key, eps_inappropriate, eps_appropriate, success = props

            # get a small number of samples
            key_new, subkey = jax.random.split(key)
            X, E = self.sample(samples, x0, subkey, monitor_energy= True)

            # remove large jumps in the energy
            E -= jnp.average(E)
            E = remove_jumps(E)

            ### compute quantities of interest ###

            # typical size of the posterior
            x1 = jnp.average(X, axis= 0) #first moments
            x2 = jnp.average(jnp.square(X), axis=0) #second moments
            sigma = jnp.sqrt(jnp.average(x2 - jnp.square(x1))) #average variance over the dimensions

            # energy fluctuations
            varE = jnp.std(E)**2 / self.Target.d #variance per dimension
            no_divergences = np.isfinite(varE)

            ### update the hyperparameters ###

            if no_divergences:
                L_new = sigma * jnp.sqrt(self.Target.d)
                eps_new = self.eps * jnp.power(varE_wanted / varE, 0.25) #assume var[E] ~ eps^4
                success = jnp.abs(1.0 - varE / varE_wanted) < 0.2 #we are done

            else:
                L_new = self.L

                if self.eps < eps_inappropriate:
                    eps_inappropriate = self.eps

                eps_new = jnp.inf #will be lowered later


            #update the known region of appropriate eps

            if not no_divergences: # inappropriate epsilon
                if self.eps < eps_inappropriate: #it is the smallest found so far
                    eps_inappropriate = self.eps

            else: # appropriate epsilon
                if self.eps > eps_appropriate: #it is the largest found so far
                    eps_appropriate = self.eps

            # if suggested new eps is inappropriate we switch to bisection
            if eps_new > eps_inappropriate:
                eps_new = 0.5 * (eps_inappropriate + eps_appropriate)
            self.set_hyperparameters(L_new, eps_new)

            if dialog:
                word = 'bisection' if (not no_divergences) else 'update'
                print('varE / varE wanted: {} ---'.format(np.round(varE / varE_wanted, 4)) + word + '---> eps: {}, sigma = L / sqrt(d): {}'.format(np.round(eps_new, 3), np.round(L_new / np.sqrt(self.Target.d), 3)))

            return key_new, eps_inappropriate, eps_appropriate, success


        ### first stage: L = sigma sqrt(d)  ###
        for i in range(10): # = maxiter
            props = tuning_step(props)
            if props[-1]: # success == True
                break

        ### second stage: L = epsilon(best) / ESS(correlations)  ###
        if dialog:
            print('Hyperparameter tuning (second stage)')

        n = np.logspace(2, np.log10(2500), 6).astype(int) # = [100, 190, 362, 689, 1313, 2499]
        n = np.insert(n, [0, ], [1, ])
        X = np.empty((n[-1] + 1, self.Target.d))
        X[0] = x0
        for i in range(1, len(n)):
            key, subkey = jax.random.split(key)
            X[n[i-1]:n[i]] = self.sample(n[i] - n[i-1], x_initial= X[n[i-1]-1], random_key= subkey, monitor_energy=True)[0]
            ESS = ess_corr(X[:n[i]])
            if dialog:
                print('n = {0}, ESS = {1}'.format(n[i], ESS))
            if n[i] > 10.0 / ESS:
                break

        L = 0.4 * self.eps / ESS # = 0.4 * correlation length
        self.set_hyperparameters(L, self.eps)

        if dialog:
            print('L / sqrt(d) = {}, ESS(correlations) = {}'.format(L / np.sqrt(self.Target.d), ESS))
            print('-------------')



def ess_cutoff_crossing(bias):

    def find_crossing(carry, b):
        above_threshold = b > 0.1
        never_been_below = carry[1] * above_threshold
        return (carry[0] + never_been_below, never_been_below), above_threshold

    crossing_index = jax.lax.scan(find_crossing, init= (0, 1), xs = bias, length=len(bias))[0][0]

    return 200.0 / np.sum(crossing_index)


def point_reduction(num_points, reduction_factor):
    """reduces the number of points for plotting purposes"""

    indexes = np.concatenate((np.arange(1, 1 + num_points // reduction_factor, dtype=int),
                              np.arange(1 + num_points // reduction_factor, num_points, reduction_factor, dtype=int)))
    return indexes


