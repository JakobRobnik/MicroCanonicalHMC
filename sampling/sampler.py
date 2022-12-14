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

    def __init__(self, Target, L = None, eps = None, integrator = 'MN', generalized= True, K = 1):
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
        elif integrator == 'LFp': #position leapfrog (first updates the position)
            self.hamiltonian_dynamics = self.position_leapfrog
            self.grad_evals_per_step = 1.0
        elif integrator== 'MN': #minimal norm integrator (velocity)
            self.hamiltonian_dynamics = self.minimal_norm
            self.grad_evals_per_step = 2.0
        elif integrator == 'MNp': #minimal norm (position)
            self.hamiltonian_dynamics = self.position_minimal_norm
            self.grad_evals_per_step = 2.0
        elif integrator == 'RM':
            self.hamiltonian_dynamics = self.randomized_midpoint
            self.grad_evals_per_step = 1.0
            print('Assumed RM takes 1 grad eval / step')
        else:
            print('integrator = ' + integrator + 'is not a valid option.')


        ### decoherence mechanism ###
        self.dynamics = self.dynamics_generalized if generalized else self.dynamics_bounces
        if K != 1:
            self.dynamics = self.dynamicsK
        self.K= K

        if (not (L is None)) and (not (eps is None)):
            self.set_hyperparameters(L, eps)



    def set_hyperparameters(self, L, eps):
        self.L = L
        self.eps= eps
        self.nu = jnp.sqrt((jnp.exp(2 * self.eps * self.K / L) - 1.0) / self.Target.d)


    def energy(self, x, r):
        return self.Target.d * r + self.Target.nlogp(x)


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


    def update_momentum(self, eps, g, u, r):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        sh = jnp.sinh(eps * g_norm / (self.Target.d-1))
        ch = jnp.cosh(eps * g_norm / (self.Target.d-1))
        th = jnp.tanh(eps * g_norm / (self.Target.d-1))
        delta_r = jnp.log(ch) + jnp.log1p(ue * th)

        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), r + delta_r


    def leapfrog(self, x, g, u, r, key):
        """leapfrog"""

        #half step in momentum
        uu, rr = self.update_momentum(self.eps * 0.5, g, u, r)

        #full step in x
        xx = x + self.eps * uu
        gg = self.Target.grad_nlogp(xx)

        #half step in momentum
        uu, rr = self.update_momentum(self.eps * 0.5, gg, uu, rr)

        return xx, gg, uu, rr, key


    def position_leapfrog(self, x, g, u, r, key):
        """position leapfrog"""

        #half step in x
        xx = x + 0.5 * self.eps * u
        gg = self.Target.grad_nlogp(xx)

        #full step in momentum
        uu, rr = self.update_momentum(self.eps, gg, u, r)

        #half step in x
        xx = xx + 0.5 * self.eps * uu

        return xx, gg, uu, rr, key


    def minimal_norm(self, x, g, u, r, key):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V

        uu, rr = self.update_momentum(self.eps * lambda_c, g, u, r)

        xx = x + self.eps * 0.5 * uu
        gg = self.Target.grad_nlogp(xx)

        uu, rr = self.update_momentum(self.eps * (1 - 2 * lambda_c), gg, uu, rr)

        xx = xx + self.eps * 0.5 * uu
        gg = self.Target.grad_nlogp(xx)

        uu, rr = self.update_momentum(self.eps * lambda_c, gg, uu, rr)

        return xx, gg, uu, rr, key


    def position_minimal_norm(self, x, g, u, r, key):

        # T V T V T

        xx = x + lambda_c * self.eps * u
        gg = self.Target.grad_nlogp(xx)

        uu, rr = self.update_momentum(0.5*self.eps, gg, u, r)

        xx = xx + (1 - 2*lambda_c) * self.eps* uu
        gg = self.Target.grad_nlogp(xx)

        uu, rr = self.update_momentum(0.5 * self.eps, gg, uu, rr)

        xx = xx + lambda_c * self.eps* uu

        return xx, gg, uu, rr, key


    def randomized_midpoint(self, x, g, u, r, key):

        key1, key2 = jax.random.split(key)

        xx = x + jax.random.uniform(key2) * self.eps * u

        gg = self.Target.grad_nlogp(xx)

        uu, rr = self.update_momentum(self.eps, gg, u, r)

        xx = self.update_position_RM(xx, )


        return xx, gg, uu, rr, key1




    def dynamics_bounces(self, state):
        """One step of the dynamics (with bounces)"""
        x, u, g, r, key, time = state

        # Hamiltonian step
        xx, gg, uu, rr, key = self.hamiltonian_dynamics(x, g, u, r, key)

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps
        do_bounce = time > self.L
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, gg, rr, key, time


    def dynamicsK(self, state):
        """One step of the dynamics (with K > 1 langevin)"""
        x, u, g, r, key, time = state

        # Hamiltonian step
        xx, gg, uu, rr, key = self.hamiltonian_dynamics(x, g, u, r, key)

        # bounce
        u_bounce, key = self.partially_refresh_momentum(uu, key)

        time += self.eps
        do_bounce = time > self.K
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, gg, rr, key, time


    def dynamics_generalized(self, state):
        """One step of the generalized dynamics."""

        x, u, g, r, key, time = state

        # Hamiltonian step
        xx, gg, uu, rr, key = self.hamiltonian_dynamics(x, g, u, r, key)

        # bounce
        uu, key = self.partially_refresh_momentum(uu, key)

        return xx, uu, gg, rr, key, 0.0



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

        g = self.Target.grad_nlogp(x)
        r = 0.5 * self.Target.d - self.Target.nlogp(x) / (self.Target.d-1) # initialize r such that all the chains have the same energy = d / 2

        u, key = self.random_unit_vector(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, g, r, key



    def sample(self, num_steps, x_initial = 'prior', random_key= None, ess=False, monitor_energy= False, final_state = False):
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

            x, u, g, r, key, time = self.dynamics(state)

            return (x, u, g, r, key, time), self.Target.transform(x)


        def step_with_energy(state, useless):
            """Tracks transform(x) and the energy as a function of number of iterations"""

            x, u, g, r, key, time = self.dynamics(state)

            return (x, u, g, r, key, time), (x, self.energy(x, r))


        def b_step(state_track, useless):
            """Only tracks b as a function of number of iterations."""

            x, u, g, r, key, time = self.dynamics(state_track[0])
            W, F2 = state_track[1]

            F2 = (W * F2 + jnp.square(self.Target.transform(x)))/ (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))
            #bias = jnp.average((F2 - self.Target.variance) / self.Target.variance)

            return ((x, u, g, r, key, time), (W, F2)), bias


        x, u, g, r, key = self.get_initial_conditions(x_initial, random_key)


        ### do sampling ###

        if ess:  # only track the bias

            _, b = jax.lax.scan(b_step, init=((x, u, g, r, key, 0.0), (1.0, jnp.square(x))), xs=None, length=num_steps)


            no_nans = 1-jnp.any(jnp.isnan(b))
            cutoff_reached = b[-1] < 0.1

            # plt.plot(bias, '.')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

            return ess_cutoff_crossing(b) * no_nans * cutoff_reached / self.grad_evals_per_step #return 0 if there are nans, or if the bias cutoff was not reached


        else: # track the full transform(x)

            if monitor_energy:
                return jax.lax.scan(step_with_energy, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)[1]

            elif final_state:
                return jax.lax.scan(step, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)[0][0]

            else:
                return jax.lax.scan(step, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)[1]



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



    def tune_hyperparameters(self, x_initial = 'prior', random_key= None):

        varE_wanted = 0.0005             # targeted energy variance per dimension
        burn_in, samples = 2000, 1000

        dialog = False

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



    def full_b(self, x_arr):

        def step(moments, index):
            F2, W = moments
            x = x_arr[index, :]
            F2 = (F2 * index + jnp.square(self.Target.transform(x)))/ (index + 1)  # Update <f(x)> with a Kalman filter
            b = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))

            return (F2, ), b

        def step_parallel(moments, index):
            F2, W = moments
            x = x_arr[:, index, :]
            F2 = (F2 * index + jnp.square(self.Target.transform(x))) / index # Update <f(x)> with a Kalman filter
            W += 1
            b = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance), axis = 1))

            return (F2, W), b


        if len(x_arr.shape) == 2: #single chain
            return jax.lax.scan(step, (jnp.zeros(self.Target.d), ), xs= jnp.arange(len(x_arr)))[1]

        else:
            num_chains = x_arr.shape[0]
            return jax.lax.scan(step_parallel, (jnp.zeros((num_chains, self.Target.d)), ), xs=jnp.arange(x_arr.shape[1]))[1]



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


