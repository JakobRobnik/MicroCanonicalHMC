import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd
import os
 
import tensorflow as tf
import tensorflow_probability as tfp 


num_cores = 128 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

from sampling.ensemble import Sampler as EnsembleSampler
from sampling.sampler import Sampler as SequentialSampler


from benchmarks.benchmarks_mchmc import *
from benchmarks.german_credit import Target as GermanCredit
from benchmarks.brownian import Target as Brownian
from benchmarks.IRT import Target as IRT


num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)


targets = [[Banana(prior = 'prior'), 100, 30000],
        [IllConditionedGaussianGamma(prior = 'prior'), 500, 100000],
        [GermanCredit(), 400, 100000],
        [Brownian(), 500, 100000],
        [IRT(), 700, 100000],
        [StochasticVolatility(), 1500, 50000]]


def ess_cross_chain(x, chains = 128, nt = 100):
    """Number of gradient evaluations (all chains) / effective sample size.
    Computed from cross chain correlations (using tensorflow probability's effective_sample_size: https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/effective_sample_size)
        Args:
            x: shape = (chains, time, dimensions)
            chains: how many chains to group together
            nt: only the last nt time points are taken
    """
    chains = 128
    repeat = x.shape[0]//chains
    X = jnp.swapaxes(jnp.reshape(x[:, -nt:, :], (repeat, chains, nt, x.shape[2])), 0, 3) # shape = (dimensions, chains, time, repeat)
    y = jnp.swapaxes(jnp.concatenate((X, jnp.square(X))), 0, 2)  #shape = (time, chains, 2 * dimensions, repeat)
    y = tf.convert_to_tensor(np.array(y), dtype = 'float64') 
    ess = tfp.mcmc.effective_sample_size(y, cross_chain_dims= 1, filter_beyond_positive_pairs=True).numpy() #shape = (2 * dimensions, repeat)
    
    return 2 * chains * nt / jnp.median(jnp.min(ess, axis = 0))
    
    
def ensemble_run():
    chains = 4096

    for i in [0, ]:
        target, num_steps, _ = targets[i]
        print(target.name)
        sampler = EnsembleSampler(target, chains)
        x, grads = sampler.sample(num_steps)
        np.save('x.npy', x)
        #ess_cross_chain(x)
        #print(grads, ess)


def sequential_run():

    vare = jnp.logspace(-8, -3, 16)
    
    for i in [0, ]:
        target, _, num_steps = targets[i]
            
        def f(vare):
            sampler = SequentialSampler(target, diagonal_preconditioning= False, varEwanted= vare)
            return sampler.sample(num_steps, num_chains = 8, output = 'ess')
        
        ess = jax.pmap(f)(vare)
        
        plt.plot(vare, ess, '.-')
        plt.xscale("log")
        plt.xlabel('Var[E]/d')
        plt.ylabel('ESS')
        plt.savefig('plots/tst_ensemble/sequential/' + target.name + '.png')
        plt.close()
        
        print(target.name + ': ' + str(np.max(ess)) + ', at energy error Var[E]/d  = ' + str(vare[np.argmax(ess)]))
        


if __name__ == '__main__':
    x= np.swapaxes(np.load('x.npy'), 0, 1)
    grads = ess_cross_chain(x, 128, 30)
    print(grads)

    
    
    #ensemble_run()
    
    #sequential_run()