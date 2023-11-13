from enum import Enum
import jax
import jax.numpy as jnp

from . import dynamics
from .correlation_length import ess_corr
from .sampler import OutputType, find_crossing


OutputType = Enum('Output', ['normal', 'ess'])


class Sampler:
    """Metropolis-Hastings algorithm with HMC or MCHMC for generating the proposal."""

    def __init__(self, Target, N, eps, N2 = jnp.inf, integrator = dynamics.minimal_norm, hmc = False, adjust = True):
        
        self.Target = Target
        self.hyp = (N, N2, eps, jnp.ones(Target.d))
        
        update_momentum, full_refresh, partial_refresh, N2nu = dynamics.setup(self.Target.d, True, hmc)

        ### integrator ###
        hamiltonian_step, num_grads = integrator(T= dynamics.update_position(self.Target.grad_nlogp), 
                                                 V= update_momentum,
                                                 d= self.Target.d)
        
        self.grad_evals_per_step = num_grads * N
        self.ma_step = dynamics.ma_step(hamiltonian_step, full_refresh, partial_refresh, N2nu, adjust)
        
        self.step_ess = self.step_ess_adjusted if adjust else self.step_ess_unadjusted
        
        

    def get_initial_conditions(self, x_initial, random_key):

        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        ### initial conditions ###
        if x_initial is None:  # draw the initial x from the prior
            key, prior_key = jax.random.split(key)
            x_initial = self.Target.prior_draw(prior_key)
        
        l, g = self.Target.grad_nlogp(x_initial)

        return x_initial, l, g, key



    def sample(self, num_steps, num_chains = 1, x_initial = None, random_key= None, output = OutputType.normal):
        """Args:
               num_steps: number of integration steps to take.

               num_chains: number of independent chains, defaults to 1. If different than 1, jax will parallelize the computation with the number of available devices (CPU, GPU, TPU),
               as returned by jax.local_device_count().

               x_initial: initial condition for x, shape: (d, ). Defaults to None in which case the initial condition is drawn from the prior distribution (self.Target.prior_draw).

               random_key: jax random seed, defaults to jax.random.PRNGKey(0)

               output: determines the output of the function:

                        'normal': samples, acceptance.
                            samples were transformed by the Target.transform to save memory and have shape: (num_samples, len(Target.transform(x)))

                        'ess': Effective Sample Size per gradient evaluation, float.
                            In this case, self.Target.variance = <x_i^2>_true should be defined.

                thinning: only one every 'thinning' steps is stored. Defaults to 1.
                        This is not the recommended solution to save memory. It is better to use the transform functionality.
                        If this is not sufficient consider saving only the expected values, by setting output= 'expectation'.
        """
        
        # check that ground truth is known for this target
        if output == OutputType.ess:
            for ground_truth in ['second_moments', 'variance_second_moments']:
                if not hasattr(self.Target, ground_truth):
                    raise AttributeError("Target." + ground_truth + " should be defined if you want to use output = ess.")
        
        if num_chains == 1:
            results, acc = self.single_chain_sample(num_steps, x_initial, random_key, output) #the function which actually does the sampling
            if output == OutputType.ess:
                return self.ESS(results), acc

            else:
                return results, acc
        else:
            num_cores = jax.local_device_count()
            if random_key is None:
                key = jax.random.PRNGKey(0)
            else:
                key = random_key

            if x_initial is None:  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains * 2)
                x0 = jnp.array([self.Target.prior_draw(keys_all[num_chains+i]) for i in range(num_chains)])
                keys = keys_all[:num_chains]

            else: #initial x is given
                x0 = jnp.copy(x_initial)
                keys = jax.random.split(key, num_chains)


            f = lambda i: self.single_chain_sample(num_steps, x0[i], keys[i], output)

            if num_cores != 1: #run the chains on parallel cores
                parallel_function = jax.pmap(jax.vmap(f))
                results, acc = parallel_function(jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores))
                if output == OutputType.ess:
                    return self.ESS(results.reshape(num_chains, num_steps)), acc

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

                results, acc = jax.vmap(f)(jnp.arange(num_chains))

                if output == OutputType.ess:
                    return self.ESS(results), acc

                else: 
                    return results, acc



    def single_chain_sample(self, num_steps, x_initial, random_key, output):
        """sampling routine. It is called by self.sample"""
        
        ### initial conditions ###
        x, l, g, key = self.get_initial_conditions(x_initial, random_key)

        
        if output == OutputType.normal:
            return self.sample_normal(num_steps, x, l, g, key)
            
        elif output == OutputType.ess:
            return self.sample_ess(num_steps, x, l, g, key)
        
        

    def sample_normal(self, num_steps, x, l, g, random_key):
        """Stores transform(x) for each step."""
        
        def step(_state, useless):
            state, acc = self.ma_step(*_state, *self.hyp)
            return state, (self.Target.transform(state[0]), acc)


        if thinning == 1:
            return jax.lax.scan(step, init=(x, l, g, random_key), xs=None, length=num_steps)[1]

        else:
            return self.sample_thinning(num_steps, x, l, g, random_key, hyp, thinning)


    def sample_ess(self, num_steps, x, l, g, random_key):
        """Stores the bias of the second moments for each step."""
     
        b, acc = jax.lax.scan(self.step_ess, init=((x, l, g, random_key), (0., jnp.zeros(len(self.Target.transform(x))))), xs=None, length=num_steps)[1]
  
        return b, jnp.sum(acc)/(num_steps)



    def step_ess_adjusted(self, state_track, useless):
        
        state, acc = self.ma_step(*state_track[0], *self.hyp)
        
        W, F2 = state_track[1]
        F2 = (W * F2 + jnp.square(self.Target.transform(state[0]))) / (W + 1)  # Update <f(x)> with a Kalman filter
        W += 1
        
        bias_d = jnp.square(F2 - self.Target.second_moments) / self.Target.variance_second_moments
        bias_avg = jnp.average(bias_d)
        #bias_max = jnp.max(bias_d)
        return (state, (W, F2)), (bias_avg, acc)

    
    def step_ess_unadjusted(self, state_track, useless):
        
        state, track = self.ma_step(*state_track[0], *self.hyp)
        
        W, F2 = state_track[1]
        F = jnp.average(jnp.square(jax.vmap(self.Target.transform)(track)), axis = 0)
        F2 = (W * F2 + F) / (W + 1)  # Update <f(x)> with a Kalman filter
        W += 1
        
        bias_d = jnp.square(F2 - self.Target.second_moments) / self.Target.variance_second_moments
        bias_avg = jnp.average(bias_d)
        #bias_max = jnp.max(bias_d)
        return (state, (W, F2)), (bias_avg, 1.)



    def ESS(self, results):
        #bsq = jnp.average(results.reshape(results.shape[0] * results.shape[1], results.shape[2]), axis = 0)
        if len(results.shape) > 1:
            bsq = jnp.median(results, axis = 0)
        else:
            bsq = results
            
        cutoff_reached = bsq[-1] < 0.01
        return (100. / (find_crossing(bsq, 0.01) * self.grad_evals_per_step)) * cutoff_reached

