import jax
import jax.numpy as jnp
import numpy as np

from .correlation_length import ess_corr


jax.config.update('jax_enable_x64', True)

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator



class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L = None, eps = None,
                 integrator = 'MN', generalized= True):
        """Args:
                Target: the target distribution class
                L: momentum decoherence scale
                eps: integration step-size
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
        # elif integrator == 'RM':
        #     self.hamiltonian_dynamics = self.randomized_midpoint
        #     self.grad_evals_per_step = 1.0
        else:
            print('integrator = ' + integrator + 'is not a valid option.')


        ### decoherence mechanism ###
        self.dynamics = self.dynamics_generalized if generalized else self.dynamics_bounces

        self.sigma = jnp.ones(self.Target.d) #diagonal preconditioning

        ### autotuning parameters ###
        self.varEwanted = 5e-4#1e-3 #targeted energy variance Var[E]/d
        neff = 50 #effective number of steps used to determine the stepsize in the adaptive step
        self.gamma = (neff - 1.0) / (neff + 1.0) #forgeting factor in the adaptive step
        self.sigma_xi= 1.5 # determines how much do we trust the stepsize predictions from the too large and too small stepsizes
        self.frac_tune1 = 0.1 # num_samples/num2 steps will be used to autotune L
        self.frac_tune2 = 0.1

        if L != None:
            self.L = L
        else: #default value (works if the target is well preconditioned). If you are not happy with the default value and have not run the grid search we suggest runing sample with the option tune= 'expensive'.
            self.L = jnp.sqrt(Target.d)

        if eps != None:
            self.eps = eps
        else: #defualt value (assumes preconditioned target and even then it might not work). Unless you have done a grid search to determine this value we suggest runing sample with the option tune= 'cheap' or tune= 'expensive'.
            self.eps = jnp.sqrt(Target.d) * 0.4



    def random_unit_vector(self, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def partially_refresh_momentum(self, u, nu, key):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(key)
        z = nu * jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')

        return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key

    # naive update
    # def update_momentum(self, eps, g, u):
    #     """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
    #     g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
    #     e = - g / g_norm
    #     ue = jnp.dot(u, e)
    #     sh = jnp.sinh(eps * g_norm / (self.Target.d-1))
    #     ch = jnp.cosh(eps * g_norm / (self.Target.d-1))
    #     th = jnp.tanh(eps * g_norm / (self.Target.d-1))
    #     delta_r = jnp.log(ch) + jnp.log1p(ue * th)
    #
    #     return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), delta_r


    def update_momentum(self, eps, g, u):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
        similar to the implementation: https://github.com/gregversteeg/esh_dynamics
        There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        delta = eps * g_norm / (self.Target.d-1)
        zeta = jnp.exp(-delta)
        uu = e *(1-zeta)*(1+zeta + ue * (1-zeta)) + 2*zeta* u
        delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1-ue)*zeta**2)
        return uu/jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r


    def leapfrog(self, x, u, g, key, eps):
        """leapfrog"""

        z = x / self.sigma # go to the latent space

        # half step in momentum
        uu, delta_r1 = self.update_momentum(eps * 0.5, g * self.sigma, u)

        # full step in x
        zz = z + eps * uu
        xx = self.sigma * zz # go back to the configuration space
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(eps * 0.5, gg * self.sigma, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, key


    def minimal_norm(self, x, u, g, key, eps):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V
        z = x / self.sigma # go to the latent space

        #V (momentum update)
        uu, r1 = self.update_momentum(eps * lambda_c, g * self.sigma, u)

        #T (postion update)
        zz = z + 0.5 * eps * uu
        xx = self.sigma * zz # go back to the configuration space
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r2 = self.update_momentum(eps * (1 - 2 * lambda_c), gg * self.sigma, uu)

        #T (postion update)
        zz = zz + 0.5 * eps * uu
        xx = self.sigma * zz  # go back to the configuration space
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r3 = self.update_momentum(eps * lambda_c, gg, uu)

        #kinetic energy change
        kinetic_change = (r1 + r2 + r3) * (self.Target.d-1)

        return xx, uu, ll, gg, kinetic_change, key


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



    def dynamics_bounces(self, x, u, g, key, time, L, eps):
        """One step of the dynamics (with bounces)"""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, key, eps)

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += eps
        do_bounce = time > L
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, ll, gg, kinetic_change, key, time


    def dynamics_generalized(self, x, u, g, key, time, L, eps):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, key, eps)

        # Langevin-like noise
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = self.partially_refresh_momentum(uu, nu, key)

        return xx, uu, ll, gg, kinetic_change, key, time + eps


    def nan_reject(self, x, u, l, g, t, xx, uu, ll, gg, tt, eps, eps_max, kk):
        """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""
        tru = jnp.all(jnp.isfinite(xx))
        false = (1 - tru)
        return tru,\
               jnp.nan_to_num(xx) * tru + x * false, \
               jnp.nan_to_num(uu) * tru + u * false, \
               jnp.nan_to_num(ll) * tru + l * false, \
               jnp.nan_to_num(gg) * tru + g * false, \
               jnp.nan_to_num(tt) * tru + t * false, \
               eps_max * tru + 0.8 * eps * false, \
               jnp.nan_to_num(kk) * tru


    def dynamics_adaptive(self, state, L):
        """One step of the dynamics with the adaptive stepsize"""

        x, u, l, g, E, Feps, Weps, eps_max, key, t = state

        eps = jnp.power(Feps/Weps, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (eps > eps_max) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences

        # dynamics
        xx, uu, ll, gg, kinetic_change, key, tt = self.dynamics(x, u, g, key, t, L, eps)

        # step updating
        success, xx, uu, ll, gg, time, eps_max, kinetic_change = self.nan_reject(x, u, l, g, t, xx, uu, ll, gg, tt, eps, eps_max, kinetic_change)

        DE = kinetic_change + ll - l  # energy difference
        EE = E + DE  # energy
        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = ((DE ** 2) / (self.Target.d * self.varEwanted)) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * self.sigma_xi)))  # the weight which reduces the impact of stepsizes which are much larger on much smaller than the desired one.
        Feps = self.gamma * Feps + w * (xi/jnp.power(eps, 6.0))  # Kalman update the linear combinations
        Weps = self.gamma * Weps + w

        return xx, uu, ll, gg, EE, Feps, Weps, eps_max, key, time, eps * success



    ### sampling routine ###

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
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else: #initial x is given
            x = x_initial
        l, g = self.Target.grad_nlogp(x)

        u, key = self.random_unit_vector(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, l, g, key

    #
    # def burn_in(self, x0, u0, l0, g0, key0):
    #     """assuming the prior is wider than the posterior"""
    #
    #     #adam = Adam(g0)
    #
    #     maxsteps = 250
    #     maxsteps_per_level = 50
    #
    #     Ls = []
    #     #self.sigma = np.load('simga.npy')
    #     #self.sigma = jnp.sqrt(self.Target.variance)
    #
    #
    #     def nan_reject(x, u, l, g, xx, uu, ll, gg):
    #         """if there are nans, let's reduce the stepsize, and not update the state"""
    #         no_nans = jnp.all(jnp.isfinite(xx))
    #         tru = no_nans
    #         false = (1 - tru)
    #         new_eps = self.eps * (false * 0.5 + tru * 1.0)
    #         X = jnp.nan_to_num(xx) * tru + x * false
    #         U = jnp.nan_to_num(uu) * tru + u * false
    #         L = jnp.nan_to_num(ll) * tru + l * false
    #         G = jnp.nan_to_num(gg) * tru + g * false
    #         return new_eps,X, U, L, G
    #
    #     def burn_in_step(state):
    #
    #         index, stationary, x, u, l, g, key, time = state
    #         #self.sigma = adam.sigma_estimate()  # diagonal conditioner
    #
    #         xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time)
    #         #energy_change = kinetic_change + ll - l
    #         #energy_condition = energy_change**2 / self.Target.d < 10000
    #         new_eps, xx, uu, ll, gg = nan_reject(x, u, l, g, xx, uu, ll, gg)
    #         self.set_hyperparameters(self.L, new_eps)
    #
    #         #adam.step(gg)
    #         Ls.append(ll)
    #
    #         if len(Ls) > 10:
    #             stationary = np.std(Ls[-10:]) / np.sqrt(self.Target.d * 0.5) < 1.2
    #         else:
    #             stationary = False
    #
    #         return index + 1, stationary, xx, uu, ll, gg, key, time
    #
    #
    #     condition = lambda state: (state[0] < maxsteps_per_level) and not state[1] # false if the burn-in should be ended
    #
    #
    #     x, u, l, g, key = x0, u0, l0, g0, key0
    #     total_steps = 0
    #     new_level = True
    #     l_plateau = np.inf
    #     while new_level and total_steps < maxsteps:
    #         steps, stationary, x, u, l, g, key, time = my_while(condition, burn_in_step, (0, False, x, u, l, g, key, 0.0))
    #         total_steps += steps
    #         l_plateau_new = np.average(Ls[-10:])
    #         diff = np.abs(l_plateau_new - l_plateau) / np.sqrt(self.Target.d * 0.5)
    #         new_level = diff > 1.0
    #         l_plateau = l_plateau_new
    #         self.eps = self.eps * 0.5
    #
    #     # plt.plot(Ls)
    #     # plt.yscale('log')
    #     # plt.show()
    #     # after you are done with developing, replace, my_while with jax.lax.while_loop
    #     #self.sigma = adam.sigma_estimate()  # diagonal conditioner
    #     #np.save('simga.npy', self.sigma)
    #     # plt.plot(self.sigma/np.sqrt(self.Target.variance), 'o')
    #     # plt.yscale('log')
    #     # plt.show()
    #
    #     return total_steps, x, u, l, g, key
    #

    def sample(self, num_steps, num_chains = 1, x_initial = 'prior', random_key= None, output = 'normal', tune = 'cheap', adaptive = False):
        """Args:
               num_steps: number of integration steps to take.

               num_chains: number of independent chains, defaults to 1. If different than 1, jax will parallelize the computation with the number of available devices (CPU, GPU, TPU),
               as returned by jax.local_device_count().

               x_initial: initial condition for x, shape: (d, ). Defaults to 'prior' in which case the initial condition is drawn from the prior distribution (self.Target.prior_draw).

               random_key: jax random seed, defaults to jax.random.PRNGKey(0)

               output: determines the output of the function:

                        'normal': samples, burn in steps.
                            samples were transformed by the Target.transform to save memory and have shape: (num_samples, len(Target.transform(x)))

                        'expectation': exepcted value of transform(x)
                            most memory efficient.

                        'details': samples, energy, L, eps

                        'ess': Effective Sample Size per gradient evaluation, float.
                            In this case, self.Target.variance = <x_i^2>_true should be defined.

        Warning: for most purposes the burn-in samples should be removed. Example usage:

        all_samples, burnin = Sampler.sample(10000)
        samples = all_samples[burnin:, :]

        """

        if num_chains == 1:
            return self.single_chain_sample(num_steps, x_initial, random_key, output, tune, adaptive) #the function which actually does the sampling

        else:
            num_cores = jax.local_device_count()
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


            f = lambda i: self.single_chain_sample(num_steps, x0[i], keys[i], output, tune, adaptive)

            if num_cores != 1: #run the chains on parallel cores
                parallel_function = jax.pmap(jax.vmap(f))
                results = parallel_function(jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores))
                if output == 'ess':
                    bsq = jnp.average(results.reshape(results.shape[0] * results.shape[1], results.shape[2]), axis = 0)
                    # import matplotlib.pyplot as plt
                    # plt.plot(jnp.sqrt(bsq))
                    # plt.yscale('log')
                    # plt.show()
                    cutoff_reached = bsq[-1] < 0.01
                    return (200.0 / (find_crossing(bsq, 0.01) *self.grad_evals_per_step) ) * cutoff_reached

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



    def single_chain_sample(self, num_steps, x_initial, random_key, output, tune, adaptive):
        """sampling routine. It is called by self.sample"""

        ### initial conditions ###
        x, u, l, g, key = self.get_initial_conditions(x_initial, random_key)
        L, eps = self.L, self.eps #the initial values, given at the class initialization (or set to the default values)

        ### auto-tune the hyperparameters L and eps ###
        if tune == 'none': #no tuning
            None
        else:
            L, eps, x, u, l, g, key = self.tune1(x, u, l, g, key, L, eps, (int)(num_steps * self.frac_tune1), (int)(num_steps * self.frac_tune2)) #the cheap tuning (100 steps)

            if tune == 'cheap': # this is it
                None
            elif tune == 'full': #if we want to further improve L tuning we go to the second stage (which can cost up to 2500 steps)
                L = self.tune2(x, u, l, g, key, L, eps)
            else:
                raise ValueError('tune = ' + output + ' is not a valid argument for the Sampler.sample')


        #print(L, eps)

        ### sampling ###

        if adaptive: #adaptive stepsize

            if output == 'normal' or output == 'detailed':
                X, W, _, E = self.sample_adaptive_normal(num_steps, x, u, l, g, key, L, eps)

                if output == 'detailed':
                    return X, W, E, L
                else:
                    return X, W

            elif output == 'ess':  # return the samples X
                return self.sample_adaptive_ess(num_steps, x, u, l, g, key, L, eps)

            else:
                raise ValueError('output = ' + output + ' is not a valid argument for the Sampler.sample')


        else: #fixed stepsize

            if output == 'normal' or output == 'detailed':
                X, _, E = self.sample_normal(num_steps, x, u, l, g, key, L, eps)

                if output == 'detailed':
                    return X, E, L, eps
                else:
                    return X

            elif output == 'ess':
                return self.sample_ess(num_steps, x, u, l, g, key, L, eps)

            else:
                raise ValueError('output = ' + output + 'is not a valid argument for the Sampler.sample')


    ### for loops which do the sampling steps: ###

    def sample_normal(self, num_steps, x, u, l, g, key, L, eps):
        """Stores transform(x) for each step."""
        
        def step(state, useless):

            x, u, l, g, E, key, time = state
            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key, time), (self.Target.transform(xx), ll, EE)

        state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, key, 0.0), xs=None, length=num_steps)

        return track
        # index_burnin = burn_in_ending(L)//thinning



    def sample_expectation(self, num_steps, x, u, l, g, key, L, eps):
        """Stores no history but keeps the expected value of transform(x)."""
        
        def step(state, useless):
            
            x, u, g, key, time = state[0]
            x, u, _, g, _, key, time = self.dynamics(x, u, g, key, time, L, eps,)
            W, F = state[1]
        
            F = (W * F + self.Target.transform(x)) / (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            return ((x, u, g, key, time), (W, F)), None


        return jax.lax.scan(step, init=(x, u, g, key, 0.0), xs=None, length=num_steps)[0][1][1]



    def sample_ess(self, num_steps, x, u, l, g, key, L, eps):
        """Stores the bias of the second moments for each step."""
        
        def step(state_track, useless):
            
            x, u, l, g, E, key, time = state_track[0]
            x, u, ll, g, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps)
            W, F2 = state_track[1]
        
            F2 = (W * F2 + jnp.square(self.Target.transform(x))) / (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            bias = jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance))
            # bias = jnp.average((F2 - self.Target.variance) / self.Target.variance)
        
            return ((x, u, ll, g, E + kinetic_change + ll - l, key, time), (W, F2)), bias

        
        _, b = jax.lax.scan(step, init=((x, u, l, g, 0.0, key, 0.0), (1, jnp.square(self.Target.transform(x)))), xs=None, length=num_steps)

        nans = jnp.any(jnp.isnan(b))

        return b #+ nans * 1e5 #return a large bias if there were nans



    def sample_adaptive_normal(self, num_steps, x, u, l, g, key, L, eps):
        """Stores transform(x) for each iteration. It uses the adaptive stepsize."""

        def step(state, useless):
            
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state, L)

            return (x, u, l, g, E, Feps, Weps, eps_max, key, time), (self.Target.transform(x), l, E, eps)

        state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, key, 0.0), xs=None, length=num_steps)
        X, nlogp, E, eps = track
        W = jnp.concatenate((0.5 * (eps[1:] + eps[:-1]), 0.5 * eps[-1:]))  # weights (because Hamiltonian time does not flow uniformly if the step size changes)
        
        return X, W, nlogp, E


    def sample_adaptive_ess(self, num_steps, x, u, l, g, key, L, eps):
        """Stores the bias of the second moments for each step."""

        def step(state, useless):
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state[0], L)

            W, F2 = state[1]
            w = eps
            F2 = (W * F2 + w * jnp.square(self.Target.transform(x))) / (W + w)  # Update <f(x)> with a Kalman filter
            W += w
            bias = jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance))

            return ((x, u, l, g, E, Feps, Weps, eps_max, key, time), (W, F2)), bias



        _, b = jax.lax.scan(step, init= ((x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, key, 0.0),
                                                 (eps, jnp.square(self.Target.transform(x)))),
                                    xs=None, length=num_steps)

        return b  # + nans * 1e5 #return a large bias if there were nans


    ### tuning phase: ###

    def tune1(self, x, u, l, g, key, L, eps, num_steps1, num_steps2):
        """cheap hyperparameter tuning"""

        gamma_save = self.gamma
        neff = 150.0
        self.gamma = (neff - 1)/(neff + 1.0)

        def step(state, outer_weight):
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state[0], L)
            W, F1, F2 = state[1]
            w = outer_weight * eps
            zero_prevention = 1-outer_weight
            F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
            F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
            W += w

            return ((x, u, l, g, E, Feps, Weps, eps_max, key, time), (W, F1, F2)), (eps, E)

        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2))) # we use the last num_steps2 to compute the typical width of the posterior
        state, track = jax.lax.scan(step, init=((x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, key, 0.0), (0.0, jnp.zeros(len(x)), jnp.zeros(len(x)))), xs= outer_weights, length= num_steps1 + num_steps2)
        eps, E = track
        # import matplotlib.pyplot as plt
        # plt.subplot(2, 1, 1)
        # plt.plot(np.square(E[1:]-E[:-1])/self.Target.d, '.-')
        # plt.yscale('log')
        # plt.subplot(2, 1, 2)
        # plt.plot(eps, '.-')
        # plt.show()
        xx, uu, ll, gg, keyy = state[0][0], state[0][1], state[0][2], state[0][3], state[0][-2]
        F1, F2 = state[1][1], state[1][2]
        sigma2 = jnp.average(F2 - jnp.square(F1))
        self.gamma = gamma_save
        return jnp.sqrt(sigma2 * self.Target.d), eps[-1], xx, uu, ll, gg, keyy



    def tune2(self, x, u, l, g, key, L, eps):
        """more expensive hyperparameter tuning to improve L (5 effective samples or max 2500 samples)"""

        dialog = False

        n = np.logspace(2, np.log10(2500), 6).astype(int) # = [100, 190, 362, 689, 1313, 2499]
        n = np.insert(n, [0, ], [1, ])
        Xall = np.empty((n[-1] + 1, self.Target.d))
        Xall[0] = x
        for i in range(1, len(n)):
            key, subkey, keyu = jax.random.split(key)
            xx = Xall[n[i-1]-1]
            uu = self.random_unit_vector(keyu)
            ll, gg = self.Target.grad_nlogp(xx)
            Xall[n[i-1]:n[i]] = self.sample_full(n[i] - n[i-1], xx, uu, ll, gg, subkey)

            ESS = ess_corr(Xall[:n[i]])
            if dialog:
                print('n = {0}, ESS = {1}'.format(n[i], ESS))
            if n[i] > 10.0 / ESS:
                break

        L = 0.4 * eps / ESS # = 0.4 * correlation length

        if dialog:
            print('L / sqrt(d) = {}, ESS(correlations) = {}'.format(L / np.sqrt(self.Target.d), ESS))
            print('-------------')

        return L

    def sample_full(self, num_steps, x, u, l, g, key, L, eps):
        """Stores full x for each step. Used in tune2."""

        def step(state, useless):
            x, u, l, g, E, key, time = state
            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key, time), (xx,)

        state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, key, 0.0), xs=None, length=num_steps)

        return track  # index_burnin = burn_in_ending(L)//thinning




def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    def step(carry, element):
        """carry = (, 1 if (array[i] > cutoff for all i < current index) else 0"""
        above_threshold = element > cutoff
        never_been_below = carry[1] * above_threshold  #1 if (array[i] > cutoff for all i < current index) else 0
        return (carry[0] + never_been_below, never_been_below), above_threshold

    state, track = jax.lax.scan(step, init=(0, 1), xs=array, length=len(array))

    return state[0]
    #return jnp.sum(track) #total number of indices for which array[m] < cutoff



def ess_cutoff_crossing(bias):

    return 200.0 / find_crossing(bias, 0.1)



def point_reduction(num_points, reduction_factor):
    """reduces the number of points for plotting purposes"""

    indexes = np.concatenate((np.arange(1, 1 + num_points // reduction_factor, dtype=int),
                              np.arange(1 + num_points // reduction_factor, num_points, reduction_factor, dtype=int)))
    return indexes



def burn_in_ending(loss):
    loss_avg = jnp.median(loss[len(loss)//2:])
    return 2 * find_crossing(loss - loss_avg, 0.0) #we add a safety factor of 2

    ### plot the removal ###
    # t= np.arange(len(loss))
    # plt.plot(t[:i*2], loss[:i*2], color= 'tab:red')
    # plt.plot(t[i*2:], loss[i*2:], color= 'tab:blue')
    # plt.yscale('log')
    # plt.show()



def my_while(cond_fun, body_fun, initial_state):
    """see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html"""

    state = initial_state

    while cond_fun(state):
        state = body_fun(state)

    return state


