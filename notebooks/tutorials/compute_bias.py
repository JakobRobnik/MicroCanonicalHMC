import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt



def get_bias(x, xsq0, varxsq0):
    """Args:
            x: samples. shape = (num_chains, num_steps, d)
            xsq0[i] = E[x_i^2] (ground truth value)
            varxsq0[i] = Var[x_i^2] (ground truth value)
       Returns:
            bias^2, as defined in the MEADS paper
    """
    
    second_moments = jnp.cumsum(jnp.square(x), axis = 1) / jnp.arange(1, x.shape[1] + 1)[None, :, None]
    z = jnp.square(second_moments - xsq0[None, None, :]) / varxsq0[None, None, :] # this quantity converges to 1/neff by CLT
    return jnp.average(jnp.average(z, 2), 0) # average over dimensions and chains
    
    


def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    def step(carry, element):
        """carry = (, 1 if (array[i] > cutoff for all i < current index) else 0"""
        above_threshold = element > cutoff
        never_been_below = carry[1] * above_threshold  #1 if (array[i] > cutoff for all i < current index) else 0
        return (carry[0] + never_been_below, never_been_below), above_threshold

    state, track = jax.lax.scan(step, init=(0, 1), xs=array, length=len(array))

    return state[0].astype(int)
    #return jnp.sum(track) #total number of indices for which array[m] < cutoff



def example():
    d = 100

    # ground truth for the standard Gaussian
    xsq0 = jnp.ones(d)
    varxsq0 = 2 * xsq0

    #exact samples
    num_chains, num_samples = 5, 200
    X = jax.random.normal(key = jax.random.PRNGKey(42), shape = (num_chains, num_samples, d))

    bsq= get_bias(X, xsq0, varxsq0)

    print(find_crossing(bsq, 0.01))
    #print('bsq = 0.01 at ' + str(find_crossing(bsq, 0.01) + ' steps.'))

    plt.plot(bsq, 'o-', label = 'exact samples')
    plt.plot(1./np.arange(len(bsq)), color = 'black', label= '1/n')
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel(r'$b^2$')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


example()