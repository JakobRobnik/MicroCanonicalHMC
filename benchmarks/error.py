import jax
import jax.numpy as jnp



def err(f_true, var_f, contract = jnp.max):
    """Computes the error b^2 = (f - f_true)^2 / var_f
        Args:
            f: E_sampler[f(x)], can be a vector
            f_true: E_true[f(x)]
            var_f: Var_true[f(x)]
            contract: how to combine a vector f in a single number, can be for example jnp.average or jnp.max
            
        Returns:
            contract(b^2)
    """    
    
    def _err(f):
        bsq = jnp.square(f - f_true) / var_f
        return contract(bsq)
    
    return jax.vmap(_err)



def grads_to_low_error(err_t, low_error= 0.01, grad_evals_per_step= 1):
    """Uses the error of the expectation values to compute the effective sample size neff
        b^2 = 1/neff"""
    
    cutoff_reached = err_t[-1] < low_error
    return find_crossing(err_t, low_error) * grad_evals_per_step, cutoff_reached
    
    
    
def ess(err_t, neff= 100, grad_evals_per_step = 1):
    
    low_error = 1./neff
    cutoff_reached = err_t[-1] < low_error
    crossing = find_crossing(err_t, low_error)
    
    return (neff / (crossing * grad_evals_per_step)) * cutoff_reached



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



def cumulative_avg(samples):
    return jnp.cumsum(samples, axis = 0) / jnp.arange(1, samples.shape[0] + 1)[:, None]



if __name__ == '__main__':
    
    # example usage
    d = 100
    n = 1000
    
    # in reality we would generate the samples with some sampler
    samples = jnp.square(jax.random.normal(jax.random.PRNGKey(42), shape = (n, d)))  
    f = cumulative_avg(samples)
      
    # ground truth 
    favg, fvar = jnp.ones(d), jnp.ones(d) * 2
    
    # error after using some number of samples
    err_t = err(favg, fvar, jnp.average)(f)
    
    # effective sample size
    ess_per_sample = ess(err_t)
    
    print("Effective sample size / sample: {0:.3}".format(ess_per_sample))
