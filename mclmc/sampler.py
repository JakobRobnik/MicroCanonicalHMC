## style note: general preference here for functional style (e.g. global function definitions, purity, code sharing)

from enum import Enum
import jax
import jax.numpy as jnp

from . import dynamics
from . import tune



class Target():
  """#Class for target distribution
  
  E.g. 
  
  ```python
  Target(d=2, nlogp = lambda x: 0.5*jnp.sum(jnp.square(x)))
```

  defines a Gaussian.
  
  """

  def __init__(self, d, nlogp):
    self.d = d
    """dimensionality of the target distribution"""
    self.nlogp = nlogp
    """ negative log probability of target distribution (i.e. energy function)"""
    self.grad_nlogp = jax.value_and_grad(self.nlogp)
    """ function which computes nlogp and its gradient"""

  def transform(self, x):
    """ a transformation of the samples from the target distribution"""
    return x

  def prior_draw(self, key):
    """**Args**: jax random key
       
       **Returns**: one random sample from the prior
    """

    raise Exception("Not implemented")


OutputType = Enum('Output', ['normal', 'detailed', 'ess'])
""" @private """




class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, 
                 Target : Target,
                 L = None, eps = None,
                 integrator = dynamics.minimal_norm, varEwanted = 5e-4,
                 diagonal_preconditioning= False,
                 frac_tune1 = 0.1, frac_tune2 = 0.1, frac_tune3 = 0.1,
                 boundary = None
                 ):
        """Args:
                Target: the target distribution class

                **L**: momentum decoherence scale (it is then automaticaly tuned before the sampling starts unless you turn-off the tuning by setting frac_tune2 and 3 to zero (see below))

                **eps**: initial integration step-size (it is then automaticaly tuned before the sampling starts unless you turn-off the tuning by setting all frac_tune1 and 2 to zero (see below))

                **integrator**: dynamics.leapfrog or dynamics.minimal_norm. Typically minimal_norm performs better.

                **varEwanted**: if your posteriors are biased try smaller values (or larger values: perhaps the convergence is too slow). This is perhaps the parameter whose default value is the least well determined.

                **diagonal_preconditioning**: if you already have your own preconditioning or if you suspect diagonal preconditioning is not useful, turn this off as it can also make matters worse
                                          (but it can also be very useful if you did not precondition the parameters (make their posterior variances close to 1))

                **frac_tune1**: (num_samples * frac_tune1) steps will be used as a burn-in and to autotune the stepsize

                **frac_tune2**: (num_samples * frac_tune2) steps will be used to autotune L (should be around 10 effective samples long for the optimal performance)

                **frac_tune3**: (num_samples * frac_tune3) steps will be used to improve the L tuning (should be around 10 effective samples long for the optimal performance). This stage is not neccessary if the posterior is close to a Gaussian and does not change much in general.
                            It can be memory intensive in high dimensions so try turning it off if you have problems with the memory.
                
                **boundary**: use if some parameters need to be constrained (a Boundary class object)
        """

        self.Target = Target

        ### kernel ###
        self.integrator = integrator
        self.boundary = boundary

        hamiltonian_step, self.grad_evals_per_step = self.integrator(T= dynamics.update_position(self.Target.grad_nlogp, boundary), 
                                                                     V= dynamics.update_momentum(self.Target.d))
        self.step = dynamics.mclmc(hamiltonian_step, *dynamics.partial_refresh(self.Target.d))
        self.full_refresh = dynamics.full_refresh(self.Target.d)

        
        ### hyperparameters ###
        self.hyp = tune.Hyperparameters(L if L!= None else jnp.sqrt(self.Target.d), 
                                        eps if eps != None else jnp.sqrt(self.Target.d) * 0.25, 
                                        jnp.ones(self.Target.d))


        ### adaptation ###
        tune12 = tune.tune12(self.step, self.Target.d, diagonal_preconditioning, jnp.array([frac_tune1, frac_tune2]), varEwanted, 1.5, 150)
        tune3 = tune.tune3(self.step, frac_tune3, 0.4)

        if frac_tune3 != 0.:
            tune3 = tune.tune3(self.step, frac= frac_tune3, Lfactor= 0.4)
            self.schedule = [tune12, tune3]
        else:
            self.schedule = [tune12, ]
        
    

    ### sampling routine ###

    def initialize(self, x_initial, random_key):

        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        ### initial conditions ###
        if x_initial is None:  # draw the initial x from the prior
            key, prior_key = jax.random.split(key)
            x = self.Target.prior_draw(prior_key)
        else:
            x = x_initial
            
        l, g = self.Target.grad_nlogp(x)

        u, key = self.full_refresh(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p
        
        return dynamics.State(x, u, l, g, key)
        


    def sample(self, num_steps, num_chains = 1, x_initial = None, random_key= None, output = OutputType.normal, thinning= 1):
        """Args:
               num_steps: number of integration steps to take.

               num_chains: number of independent chains, defaults to 1. If different than 1, jax will parallelize the computation with the number of available devices (CPU, GPU, TPU),
               as returned by jax.local_device_count().

               x_initial: initial condition for x, shape: (d, ). Defaults to None in which case the initial condition is drawn from the prior distribution (self.Target.prior_draw).

               random_key: jax random seed, defaults to jax.random.PRNGKey(0)

               output: determines the output of the function:

                        'normal': samples
                            samples were transformed by the Target.transform to save memory and have shape: (num_samples, len(Target.transform(x)))

                        'detailed': samples, energy error at each step and -log p(x) at each step

                        'ess': Effective Sample Size per gradient evaluation, float.
                            In this case, ground truth E[x_i^2] and Var[x_i^2] should be known and defined as self.Target.second_moments and self.Target.variance_second_moments 

                        Note: in all cases the hyperparameters that were used for sampling can be accessed through Sampler.hyp
                        
                thinning: only one every 'thinning' steps is stored. Defaults to 1 (the output then contains (num_steps / thinning) samples)
                        This is not the recommended solution to save memory. It is better to use the transform functionality, when possible.
        """
        
        if output == OutputType.ess:
            for ground_truth in ['second_moments', 'variance_second_moments']:
                if not hasattr(self.Target, ground_truth):
                    raise AttributeError("Target." + ground_truth + " should be defined if you want to use output = ess.")
        
        if num_chains == 1:
            results = self.single_chain_sample(num_steps, x_initial, random_key, output, thinning) #the function which actually does the sampling
            if output == OutputType.ess:
                return self.bias_plot(results)

            else:
                return results
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


            f = lambda i: self.single_chain_sample(num_steps, x0[i], keys[i], output, thinning)

            if num_cores != 1: #run the chains on parallel cores
                parallel_function = jax.pmap(jax.vmap(f))
                results = parallel_function(jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores))
                if output == OutputType.ess:
                    return self.bias_plot(results.reshape(num_chains, num_steps))

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

                results = jax.vmap(f)(jnp.arange(num_chains))

                if output == OutputType.ess:
                    return self.bias_plot(results)

                else: 
                    return results



    def single_chain_sample(self, num_steps, x_initial, random_key, output, thinning):
        """sampling routine. It is called by self.sample"""
         ### initial conditions ###
        dyn = self.initialize(x_initial, random_key)
        
        hyp = self.hyp
        
        ### tuning ###
        dyn, hyp = tune.run(dyn, hyp, self.schedule, num_steps)
        self.hyp = hyp

        
        ### sampling ###

        if output == OutputType.normal or output == OutputType.detailed:
            X, l, E = self.sample_normal(num_steps, dyn, hyp, thinning)
            if output == OutputType.detailed:
                return X, E, l
            else:
                return X

        elif output == OutputType.ess:
            return self.sample_ess(num_steps, dyn, hyp)
        
        
        
    def build_kernel(self, thinning : int):
        """kernel for sampling_normal"""
        
        def kernel_with_thinning(dyn, hyp):
            
            def substep(state, _):
                _dyn, energy_change = self.step(state[0], hyp)
                return (_dyn, energy_change), None
            
            return jax.lax.scan(substep, init= (dyn, 0.), xs=None, length= thinning)[0] #do 'thinning' steps without saving

        if thinning == 1:
            return self.step
        else:
            return kernel_with_thinning 
        
           
    def sample_normal(self, num_steps : int, _dyn : dynamics.State, hyp : tune.Hyperparameters, thinning : int):
        """Stores transform(x) for each step."""
        
        kernel = self.build_kernel(thinning)
        
        def step(state, _):

            dyn, energy_change = kernel(state, hyp)
 
            return dyn, (self.Target.transform(dyn.x), dyn.l, energy_change)
        
            
        return jax.lax.scan(step, init= _dyn, xs=None, length= num_steps // thinning)[1]



    def sample_ess(self, num_steps : int, _dyn : dynamics.State, hyp : tune.Hyperparameters):
        """Stores the bias of the second moments for each step."""
        
        def step(state_track, useless):
            dyn, kalman_state = state_track
            dyn, _ = self.step(dyn, hyp)
            kalman_state = kalman_step(kalman_state)
            return (dyn, kalman_state), bias(kalman_state[1])

        def kalman_step(state, x):
            W, F2 = state        
            F2 = (W * F2 + jnp.square(self.Target.transform(x))) / (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            return W, F2
        
        def bias(x2):
            bias_d = jnp.square(x2 - self.Target.second_moments) / self.Target.variance_second_moments
            bavg2 = jnp.average(bias_d)
            #bmax2 = jnp.max(bias_d)
            return bavg2
            
            
        _, b = jax.lax.scan(step, init=(_dyn, (1, jnp.square(self.Target.transform(_dyn.x)))), xs=None, length=num_steps)

        return b



    def bias_plot(self, results):
        if len(results.shape)>1:
            bsq = jnp.median(results, axis = 0)
        else:
            bsq = results    


        cutoff_reached = bsq[-1] < 0.01
        return (100. / (find_crossing(bsq, 0.01) * self.grad_evals_per_step)) * cutoff_reached


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

