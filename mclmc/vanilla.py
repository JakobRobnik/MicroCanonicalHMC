from enum import Enum
import jax
import jax.numpy as jnp

from . import dynamics
from .correlation_length import ess_corr
from .sampler import OutputType, find_crossing
from .stepsize_adaptation import dual_averaging, predictor

OutputType = Enum('Output', ['normal', 'ess'])


class Sampler:
    """
    If ajdust = True, we do a MH accept/reject after num_steps steps of the dynamics and collect a sample.
    There are three strategies of refreshment:
        full: do full refreshment every num_steps steps of Hamiltonian dynamics
        full Langevin: do full refreshment every num_steps steps of Langevin dynamics
        Langevin: do Langevin dynamics
    
    If adjust = False, all steps are collected as samples.
    There are two strategies of refreshment:
        full: do full refreshment every num_steps steps of Hamiltonian dynamics
        Langevin: do Langevin dynamics
        
    """
    def __init__(self, Target, 
                 steps_per_sample, eps, num_decoherence = jnp.inf,
                 integrator = dynamics.minimal_norm, hmc = False, adjust = True, full_refreshment = True):
        
        self.Target = Target
        
        if not full_refreshment and jnp.isinf(num_decoherence):
            raise ValueError("For the Langevin trajectories num_decoherence should be specifed and cannot be infinite.")
        
        self.hyp = (steps_per_sample, num_decoherence, eps, jnp.ones(Target.d))
        
        update_momentum, full_refresh, partial_refresh, get_nu = dynamics.setup(self.Target.d, True, hmc)

        ### integrator ###
        hamiltonian_step, grads_per_step = integrator(T= dynamics.update_position(self.Target.grad_nlogp), 
                                                 V= update_momentum,
                                                 d= self.Target.d)
        
        self.grads_per_sample = grads_per_step * steps_per_sample
        id = lambda u, k: (u, k)
        full = lambda u, k: full_refresh(k)
        self.ma_step = dynamics.ma_step(hamiltonian_step, full if full_refreshment else id, partial_refresh, get_nu, adjust)

        self.full_refresh = full_refresh
        
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
        u, key = self.full_refresh(key)
        return x_initial, u, l, g, key



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
        dyn = self.get_initial_conditions(x_initial, random_key)

        
        if output == OutputType.normal:
            return self.sample_normal(num_steps, dyn)
            
        elif output == OutputType.ess:
            return self.sample_ess(num_steps, dyn)

        
        
    def sample_normal(self, num_steps, dyn):
        """Stores transform(x) for each step."""
        
        def step(_state, useless):
            state, acc = self.ma_step(*_state, *self.hyp)
            return state, (self.Target.transform(state[0]), acc)

        return jax.lax.scan(step, init= dyn, xs=None, length=num_steps)[1]



    def sample_ess(self, num_steps, dyn):
        """Stores the bias of the second moments for each step."""
     
        b, acc = jax.lax.scan(self.step_ess, init=(dyn, (0., jnp.zeros(len(self.Target.transform(dyn[0]))))), xs=None, length=num_steps)[1]
  
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



    def adaptation_dual_averaging(self, num_steps, x_initial= None, random_key= None):
        
        ### initial conditions ###
        dyn = self.get_initial_conditions(x_initial, random_key)
        eps = self.hyp[2]
        adap = jnp.zeros(4)
        
        acc_prob_wanted = 0.7
        
        def step(_state, useless):
            dyn, eps, adap = _state
            dyn, acc = self.ma_step(*dyn, self.hyp[0], self.hyp[1], eps, self.hyp[3])
            adap = dual_averaging(acc, adap, acc_prob_wanted)
            return (dyn, jnp.exp(adap[0]), adap), (self.Target.transform(dyn[0]), acc, adap)
        
        
        _state, track = jax.lax.scan(step, init= (dyn, eps, adap), xs=None, length=num_steps)
        
        X, acc, adap = track
        
        import matplotlib.pyplot as plt
        plt.figure(figsize= (10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(jnp.exp(adap[:, 0]), '.', markersize = 5, color = 'chocolate', label = 'current stepsize')
        plt.plot(jnp.exp(adap[:, 1]), '-', lw = 3, color = 'xkcd:ruby', label = 'predicted stepsize')
        plt.ylabel('stepsize')
        plt.xlabel('# samples')
        plt.yscale('log')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(acc, '.', color = 'tab:blue', label = 'actual')
        plt.plot(acc_prob_wanted - adap[:, 2], color = 'black', label = 'estimated average')
        plt.plot(jnp.arange(len(acc)), jnp.ones(len(acc)) * acc_prob_wanted, lw = 3, color = 'black', alpha = 0.5, label = 'desired average')
        plt.ylabel('acceptance probability')
        plt.xlabel('# samples')
        plt.legend()
        
        plt.savefig('dual_averaging_burnin.png')
        plt.show()
    
    
    def adaptation_predictor(self, num_steps, x_initial= None, random_key= None):
        
        ### initial conditions ###
        dyn = self.get_initial_conditions(x_initial, random_key)
        eps = self.hyp[2]
        counter = 0
        
        acc_prob_wanted = 0.7
        
        data = jnp.zeros((2, num_steps))
        
        
        def step(_state, useless):
            dyn, eps, data, counter = _state
            dyn, acc = self.ma_step(*dyn, self.hyp[0], self.hyp[1], eps, self.hyp[3])
            data = data.at[:, counter].set(jnp.array([eps, acc]))
            counter += 1            
            eps = predictor(data, counter, acc_prob_wanted)
            return (dyn, eps, data, counter), eps#(self.Target.transform(dyn[0]), acc, eps)
        
        
        state= (dyn, eps, data, counter)
        eps = []
        for i in range(num_steps):
            state, _eps = step(state, None)
            eps.append(_eps)
            
        #_state, track = jax.lax.scan(step, init= (dyn, eps, data, counter), xs=None, length=num_steps)

        import matplotlib.pyplot as plt
        plt.figure(figsize= (10, 4))
        plt.plot(eps, '.', markersize = 5, color = 'chocolate', label = 'stepsize')
        plt.ylabel('stepsize')
        plt.xlabel('# samples')
        plt.yscale('log')
        plt.legend()
        plt.savefig('predictor.png')
        plt.show()
        

    def ESS(self, results):
        #bsq = jnp.average(results.reshape(results.shape[0] * results.shape[1], results.shape[2]), axis = 0)
        if len(results.shape) > 1:
            bsq = jnp.median(results, axis = 0)
        else:
            bsq = results
            
        cutoff_reached = bsq[-1] < 0.01
        return (100. / (find_crossing(bsq, 0.01) * self.grads_per_sample)) * cutoff_reached

