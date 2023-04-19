import jax
import jax.numpy as jnp
import numpy as np

from .correlation_length import ess_corr


jax.config.update('jax_enable_x64', True)

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator



class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L = None, eps = None,
                 integrator = 'MN', varEwanted = 5e-4,
                 diagonal_preconditioning= True, sg = False,
                 frac_tune1 = 0.1, frac_tune2 = 0.1, frac_tune3 = 0.1):
        """Args:
                Target: the target distribution class

                L: momentum decoherence scale (it is then automaticaly tuned before the sampling starts unless you turn-off the tuning by setting frac_tune2 and 3 to zero (see below))

                eps: initial integration step-size (it is then automaticaly tuned before the sampling starts unless you turn-off the tuning by setting all frac_tune1 and 2 to zero (see below))

                integrator: 'LF' (leapfrog) or 'MN' (minimal norm). Typically MN performs better.

                varEwanted: if your posteriors are biased try smaller values (or larger values: perhaps the convergence is too slow). This is perhaps the parameter whose default value is the least well determined.

                diagonal_preconditioning: if you already have your own preconditioning or if you suspect diagonal preconditioning is not useful, turn this off as it can also make matters worse
                                          (but it can also be very useful if you did not precondition the parameters (make their posterior variances close to 1))

                frac_tune1: (num_samples * frac_tune1) steps will be used as a burn-in and to autotune the stepsize

                frac_tune2: (num_samples * frac_tune2) steps will be used to autotune L (should be around 10 effective samples long for the optimal performance)

                frac_tune3: (num_samples * frac_tune3) steps will be used to improve the L tuning (should be around 10 effective samples long for the optimal performance). This stage is not neccessary if the posterior is close to a Gaussian and does not change much in general.
                            It can be memory intensive in high dimensions so try turning it off if you have problems with the memory.
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
            raise ValueError('integrator = ' + integrator + 'is not a valid option.')


        ### option of stochastic gradient ###
        self.sg = sg
        self.dynamics = self.dynamics_generalized_sg if sg else self.dynamics_generalized

        ### preconditioning ###
        self.diagonal_preconditioning = diagonal_preconditioning

        ### autotuning parameters ###

        # length of autotuning
        self.frac_tune1 = frac_tune1 # num_samples * frac_tune1 steps will be used to autotune eps
        self.frac_tune2 = frac_tune2 # num_samples * frac_tune2 steps will be used to approximately autotune L
        self.frac_tune3 = frac_tune3 # num_samples * frac_tune3 steps will be used to improve L tuning.

        self.varEwanted = varEwanted # 1e-3 #targeted energy variance Var[E]/d
        neff = 50 # effective number of steps used to determine the stepsize in the adaptive step
        self.gamma = (neff - 1.0) / (neff + 1.0) # forgeting factor in the adaptive step
        self.sigma_xi= 1.5 # determines how much do we trust the stepsize predictions from the too large and too small stepsizes

        self.Lfactor = 0.4 #in the third stage we set L = Lfactor * (configuration space distance bewteen independent samples)


        ### default eps and L ###
        if L != None:
            self.L = L
        else: #default value (works if the target is well preconditioned). If you are not happy with the default value and have not run the grid search we suggest runing sample with the option tune= 'expensive'.
            self.L = jnp.sqrt(Target.d)
        if eps != None:
            self.eps = eps
        else: #defualt value (assumes preconditioned target and even then it might not work). Unless you have done a grid search to determine this value we suggest runing sample with the option tune= 'cheap' or tune= 'expensive'.
            self.eps = jnp.sqrt(Target.d) * 0.4



    def random_unit_vector(self, random_key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def partially_refresh_momentum(self, u, nu, random_key):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(random_key)
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


    def leapfrog(self, x, u, g, random_key, eps, sigma):
        """leapfrog"""

        z = x / sigma # go to the latent space

        # half step in momentum
        uu, delta_r1 = self.update_momentum(eps * 0.5, g * sigma, u)

        # full step in x
        zz = z + eps * uu
        xx = sigma * zz # go back to the configuration space
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(eps * 0.5, gg * sigma, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, random_key

    # def leapfrog(self, x, u, g, random_key, eps, sigma):
    #     """leapfrog"""
    #
    #     z = x / sigma  # go to the latent space
    #
    #     # half step in x
    #     zz = z + 0.5 * eps * u
    #     l, gg = self.Target.grad_nlogp(sigma * zz)
    #
    #     # full step in momentum
    #     uu, delta_r = self.update_momentum(eps, gg * sigma, u)
    #
    #     # half step in x
    #     zz += 0.5 * eps * uu
    #     xx = sigma * zz  # go back to the configuration space
    #
    #     l = self.Target.nlogp(xx)
    #     kinetic_change = delta_r * (self.Target.d - 1)
    #
    #     return xx, uu, l, gg, kinetic_change, random_key

    def leapfrog_sg(self, x, u, g, random_key, eps, sigma, data):
        """leapfrog"""

        z = x / sigma # go to the latent space

        # half step in momentum
        uu, delta_r1 = self.update_momentum(eps * 0.5, g * sigma, u)

        # full step in x
        zz = z + eps * uu
        xx = sigma * zz # go back to the configuration space
        l, gg = self.Target.grad_nlogp(xx, data)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(eps * 0.5, gg * sigma, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, random_key

    #
    # def leapfrog_sg(self, x, u, g, random_key, eps, sigma, data):
    #     """leapfrog"""
    #
    #     z = x / sigma # go to the latent space
    #
    #     # half step in x
    #     zz = z + 0.5 * eps * u
    #     xx = sigma * zz # go back to the configuration space
    #     l, gg = self.Target.grad_nlogp(xx, data)
    #
    #     # half step in momentum
    #     uu, delta_r = self.update_momentum(eps, gg * sigma, u)
    #
    #     # full step in x
    #     zz += 0.5 * eps * uu
    #     xx = sigma * zz # go back to the configuration space
    #
    #     kinetic_change = delta_r * (self.Target.d-1)
    #
    #     return xx, uu, l, gg, kinetic_change, random_key



    def minimal_norm(self, x, u, g, random_key, eps, sigma):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V
        z = x / sigma # go to the latent space

        #V (momentum update)
        uu, r1 = self.update_momentum(eps * lambda_c, g * sigma, u)

        #T (postion update)
        zz = z + 0.5 * eps * uu
        xx = sigma * zz # go back to the configuration space
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r2 = self.update_momentum(eps * (1 - 2 * lambda_c), gg * sigma, uu)

        #T (postion update)
        zz = zz + 0.5 * eps * uu
        xx = sigma * zz  # go back to the configuration space
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r3 = self.update_momentum(eps * lambda_c, gg, uu)

        #kinetic energy change
        kinetic_change = (r1 + r2 + r3) * (self.Target.d-1)

        return xx, uu, ll, gg, kinetic_change, random_key


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



    def dynamics_bounces(self, x, u, g, random_key, time, L, eps, sigma):
        """One step of the dynamics (with bounces)"""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, random_key, eps, sigma)

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += eps
        do_bounce = time > L
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, ll, gg, kinetic_change, key, time


    def dynamics_generalized(self, x, u, g, random_key, time, L, eps, sigma):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, random_key, eps, sigma)

        # Langevin-like noise
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = self.partially_refresh_momentum(uu, nu, key)

        return xx, uu, ll, gg, kinetic_change, key, time + eps



    def dynamics_generalized_sg(self, x, u, g, random_key, time, L, eps, sigma):
        """One sweep over the entire dataset. Perfomrs self.Target.num_batches steps with the stochastic gradient."""

        #reshufle data and arange in batches

        key_reshuffle, key = jax.random.split(random_key)
        data_shape = self.Target.data.shape
        data = jax.random.permutation(key_reshuffle, self.Target.data).reshape(self.Target.num_batches, data_shape[0]//self.Target.num_batches, data_shape[1])

        def substep(state, data_batch):
            x, u, l, g, key, K, t = state
            # Hamiltonian step
            xx, uu, ll, gg, dK, key = self.leapfrog_sg(x, u, g, key, eps, sigma, data_batch)

            # Langevin-like noise
            nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
            uu, key = self.partially_refresh_momentum(uu, nu, key)

            return (xx, uu, ll, gg, key, K + dK, t + eps), None

        xx, uu, ll, gg, key, kinetic_change, time = jax.lax.scan(substep, init= (x, u, 0.0, g, key, 0.0, time), xs= data, length= self.Target.num_batches)[0]

        return xx, uu, ll, gg, kinetic_change, key, time



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


    def dynamics_adaptive(self, state, L, sigma):
        """One step of the dynamics with the adaptive stepsize"""

        x, u, l, g, E, Feps, Weps, eps_max, key, t = state

        eps = jnp.power(Feps/Weps, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (eps > eps_max) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences

        # dynamics
        xx, uu, ll, gg, kinetic_change, key, tt = self.dynamics(x, u, g, key, t, L, eps, sigma)

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

        if self.sg:
            key_reshuffle, key = jax.random.split(key)
            data_batch = jax.random.permutation(key_reshuffle, self.Target.data)[0: len(self.Target.data) // self.Target.num_batches]
            l, g = self.Target.grad_nlogp(x, data_batch)

        else:
            l, g = self.Target.grad_nlogp(x)

        u, key = self.random_unit_vector(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, l, g, key



    def sample(self, num_steps, num_chains = 1, x_initial = 'prior', random_key= None, output = 'normal', adaptive = False):
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
                            most memory efficient. If you are after memory it might be usefull to turn off the third tuning stage

                        'detailed': samples, energy for each step, L and eps used for sampling

                        'ess': Effective Sample Size per gradient evaluation, float.
                            In this case, self.Target.variance = <x_i^2>_true should be defined.

               adaptive: use the adaptive step size for sampling. This is experimental and not well developed yet.
        """

        if num_chains == 1:
            results = self.single_chain_sample(num_steps, x_initial, random_key, output, adaptive) #the function which actually does the sampling
            if output == 'ess':
                cutoff_reached = results[-1] < 0.01
                return (100.0 / (find_crossing(results, 0.01) * self.grad_evals_per_step)) * cutoff_reached
            else:
                return results
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


            f = lambda i: self.single_chain_sample(num_steps, x0[i], keys[i], output, adaptive)

            if num_cores != 1: #run the chains on parallel cores
                parallel_function = jax.pmap(jax.vmap(f))
                results = parallel_function(jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores))
                if output == 'ess' or output == 'ess funnel':
                    bsq = jnp.average(results.reshape(results.shape[0] * results.shape[1], results.shape[2]), axis = 0)

                    import matplotlib.pyplot as plt
                    plt.plot(jnp.sqrt(bsq))
                    plt.plot([0, len(bsq)], np.ones(2) * 0.1, '--', color = 'black', alpha= 0.5)
                    plt.yscale('log')
                    plt.show()

                    cutoff_reached = bsq[-1] < 0.01
                    return (100.0 / (find_crossing(bsq, 0.01) *self.grad_evals_per_step) ) * cutoff_reached

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



    def single_chain_sample(self, num_steps, x_initial, random_key, output, adaptive):
        """sampling routine. It is called by self.sample"""

        ### initial conditions ###
        x, u, l, g, key = self.get_initial_conditions(x_initial, random_key)
        L, eps = self.L, self.eps #the initial values, given at the class initialization (or set to the default values)

        sigma = jnp.ones(self.Target.d)  # no diagonal preconditioning

        ### auto-tune the hyperparameters L and eps ###
        if self.frac_tune1 + self.frac_tune2 + self.frac_tune3 != 0.0:
            L, eps, sigma, x, u, l, g, key = self.tune12(x, u, l, g, key, L, eps, sigma, (int)(num_steps * self.frac_tune1), (int)(num_steps * self.frac_tune2)) #the cheap tuning (100 steps)
            if self.frac_tune3 != 0: #if we want to further improve L tuning we go to the second stage (which is a bit slower)
                L, x, u, l, g, key = self.tune3(x, u, l, g, key, L, eps, sigma, (int)(num_steps * self.frac_tune3))

        ### sampling ###

        if adaptive: #adaptive stepsize

            if output == 'normal' or output == 'detailed':
                X, W, _, E = self.sample_adaptive_normal(num_steps, x, u, l, g, key, L, eps, sigma)

                if output == 'detailed':
                    return X, W, E, L
                else:
                    return X, W

            elif output == 'ess':  # return the samples X
                return self.sample_adaptive_ess(num_steps, x, u, l, g, key, L, eps, sigma)
            elif output == 'expectation':
                raise ValueError('output = ' + output + ' is not yet implemented for the adaptive step-size. Let me know if you need it.')
            else:
                raise ValueError('output = ' + output + ' is not a valid argument for the Sampler.sample')


        else: #fixed stepsize

            if output == 'normal' or output == 'detailed':
                X, _, E = self.sample_normal(num_steps, x, u, l, g, key, L, eps, sigma)
                if output == 'detailed':
                    return X, E, L, eps
                else:
                    return X
            elif output == 'expectation':
                return self.sample_expectation(num_steps, x, u, l, g, key, L, eps, sigma)

            elif output == 'ess':
                return self.sample_ess(num_steps, x, u, l, g, key, L, eps, sigma)

            elif output == 'ess funnel':
                return self.sample_ess_funnel(num_steps, x, u, l, g, key, L, eps, sigma)

            else:
                raise ValueError('output = ' + output + 'is not a valid argument for the Sampler.sample')


    ### for loops which do the sampling steps: ###

    def sample_normal(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores transform(x) for each step."""
        
        def step(state, useless):

            x, u, l, g, E, key, time = state
            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key, time), (self.Target.transform(xx), ll, EE)

        state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, random_key, 0.0), xs=None, length=num_steps)

        return track
        # index_burnin = burn_in_ending(L)//thinning



    def sample_expectation(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores no history but keeps the expected value of transform(x)."""
        
        def step(state, useless):
            
            x, u, g, key, time = state[0]
            x, u, _, g, _, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
            W, F = state[1]
        
            F = (W * F + self.Target.transform(x)) / (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            return ((x, u, g, key, time), (W, F)), None


        return jax.lax.scan(step, init=(x, u, g, random_key, 0.0), xs=None, length=num_steps)[0][1][1]



    def sample_ess(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores the bias of the second moments for each step."""
        
        def step(state_track, useless):
            
            x, u, l, g, E, key, time = state_track[0]
            x, u, ll, g, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
            W, F2 = state_track[1]
        
            F2 = (W * F2 + jnp.square(self.Target.transform(x))) / (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            bias_d = jnp.square(F2 - self.Target.second_moments) / self.Target.variance_second_moments
            bias = jnp.average(bias_d)
            #bias = jnp.max(bias_d)

            return ((x, u, ll, g, E + kinetic_change + ll - l, key, time), (W, F2)), bias

        
        _, b = jax.lax.scan(step, init=((x, u, l, g, 0.0, random_key, 0.0), (1, jnp.square(self.Target.transform(x)))), xs=None, length=num_steps)

        #nans = jnp.any(jnp.isnan(b))

        return b #+ nans * 1e5 #return a large bias if there were nans


    def sample_ess_funnel(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores the bias of the second moments for each step."""

        def step(state_track, useless):
            x, u, l, g, E, key, time = state_track[0]
            eps1 = eps * jnp.exp(0.5 * x[-1])
            eps_max = eps *0.5#* jnp.exp(0.5)
            too_large = eps1 > eps_max
            eps_real = eps1 * (1-too_large) + eps_max * too_large
            x, u, ll, g, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps_real, sigma)
            W, F2 = state_track[1]
            F2 = (W * F2 + eps_real * jnp.square(self.Target.transform(x))) / (W + eps_real)  # Update <f(x)> with a Kalman filter
            W += eps_real
            bias = jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance))
            # bias = jnp.average((F2 - self.Target.variance) / self.Target.variance)

            return ((x, u, ll, g, E + kinetic_change + ll - l, key, time), (W, F2)), bias

        _, b = jax.lax.scan(step, init=((x, u, l, g, 0.0, random_key, 0.0), (eps * jnp.exp(0.5 * x[-1]), jnp.square(self.Target.transform(x)))),
                            xs=None, length=num_steps)

        return b  # + nans * 1e5 #return a large bias if there were nans


    def sample_adaptive_normal(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores transform(x) for each iteration. It uses the adaptive stepsize."""

        def step(state, useless):
            
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state, L, sigma)

            return (x, u, l, g, E, Feps, Weps, eps_max, key, time), (self.Target.transform(x), l, E, eps)

        state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, random_key, 0.0), xs=None, length=num_steps)
        X, nlogp, E, eps = track
        W = jnp.concatenate((0.5 * (eps[1:] + eps[:-1]), 0.5 * eps[-1:]))  # weights (because Hamiltonian time does not flow uniformly if the step size changes)
        
        return X, W, nlogp, E


    def sample_adaptive_ess(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores the bias of the second moments for each step."""

        def step(state, useless):
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state[0], L, sigma)

            W, F2 = state[1]
            w = eps
            F2 = (W * F2 + w * jnp.square(self.Target.transform(x))) / (W + w)  # Update <f(x)> with a Kalman filter
            W += w
            bias = jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance))

            return ((x, u, l, g, E, Feps, Weps, eps_max, key, time), (W, F2)), bias



        _, b = jax.lax.scan(step, init= ((x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, random_key, 0.0),
                                                 (eps, jnp.square(self.Target.transform(x)))),
                                    xs=None, length=num_steps)

        return b  # + nans * 1e5 #return a large bias if there were nans


    ### tuning phase: ###

    def tune12(self, x, u, l, g, random_key, L_given, eps, sigma_given, num_steps1, num_steps2):
        """cheap hyperparameter tuning"""

        # during the tuning we will be using a different gamma
        gamma_save = self.gamma # save the old value
        neff = 150.0
        self.gamma = (neff - 1)/(neff + 1.0)
        sigma = sigma_given

        def step(state, outer_weight):
            """one adaptive step of the dynamics"""
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state[0], L, sigma)
            W, F1, F2 = state[1]
            w = outer_weight * eps
            zero_prevention = 1-outer_weight
            F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
            F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
            W += w

            return ((x, u, l, g, E, Feps, Weps, eps_max, key, time), (W, F1, F2)), eps

        L = L_given

        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        #initial state
        state = ((x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, random_key, 0.0), (0.0, jnp.zeros(len(x)), jnp.zeros(len(x))))

        # run the steps
        state, eps = jax.lax.scan(step, init=state, xs= outer_weights, length= num_steps1 + num_steps2)

        # determine L
        F1, F2 = state[1][1], state[1][2]
        variances = F2 - jnp.square(F1)
        sigma2 = jnp.average(variances)
        # print(F1)
        # print(F2 / self.Target.second_moments)

        #variances = self.Target.second_moments
        #resc = jnp.diag(1.0/jnp.sqrt(variances))
        #Sigma = resc @ self.Target.Cov @ resc
        #print(jnp.linalg.cond(Sigma) / jnp.linalg.cond(self.Target.Cov))

        # optionally we do the diagonal preconditioning (and readjust the stepsize)
        if self.diagonal_preconditioning:

            # diagonal preconditioning
            sigma = jnp.sqrt(variances)
            L = jnp.sqrt(self.Target.d)

            # state = ((state[0][0], state[0][1], state[0][2], state[0][3], 0.0, jnp.power(eps[-1], -6.0) * 1e-5, 1e-5, jnp.inf, state[0][-2], 0.0),
            #         (0.0, jnp.zeros(len(x)), jnp.zeros(len(x))))

            # print(L, eps[-1])
            # print(sigma**2 / self.Target.variance)

            #readjust the stepsize
            steps = num_steps2 // 3 #we do some small number of steps
            state, eps = jax.lax.scan(step, init= state, xs= jnp.ones(steps), length= steps)

        else:
            L = jnp.sqrt(sigma2 * self.Target.d)
        #print(L, eps[-1])
        xx, uu, ll, gg, key = state[0][0], state[0][1], state[0][2], state[0][3], state[0][-2] # the final state
        self.gamma = gamma_save #set gamma to the previous value
        #print(L, eps[-1])
        return L, eps[-1], sigma, xx, uu, ll, gg, key #return the tuned hyperparameters and the final state



    def tune3(self, x, u, l, g, random_key, L, eps, sigma, num_steps):
        """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

        X, xx, uu, ll, gg, key = self.sample_full(num_steps, x, u, l, g, random_key, L, eps, sigma)
        ESS = ess_corr(X)
        Lnew = self.Lfactor * eps / ESS # = 0.4 * correlation length

        return Lnew, xx, uu, ll, gg, key


    def sample_full(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores full x for each step. Used in tune2."""

        def step(state, useless):
            x, u, l, g, E, key,   time = state
            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key, time), xx

        state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, random_key, 0.0), xs=None, length=num_steps)
        xx, uu, ll, gg, key = state[0], state[1], state[2], state[3], state[5]
        return track, xx, uu, ll, gg, key




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


