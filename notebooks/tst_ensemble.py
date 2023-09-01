import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd
import os
 
import tensorflow as tf
import tensorflow_probability as tfp 


#num_cores = 4 #specific to my PC
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

from sampling.ensemble import Sampler as EnsembleSampler
from sampling.sampler import Sampler as SequentialSampler


from benchmarks.benchmarks_mchmc import *
from benchmarks.german_credit import Target as GermanCredit
from benchmarks.brownian import Target as Brownian
from benchmarks.IRT import Target as IRT


num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)


targets = [[Banana(prior = 'prior'), 100, 30000],
        [IllConditionedGaussianGamma(prior = 'prior'), 1000, 100000],
        [GermanCredit(), 400, 100000],
        [Brownian(), 500, 100000],
        [IRT(), 700, 100000],
        [StochasticVolatility(), 1500, 50000]]


def ensemble_run():
    
    chains = 2048

    for i in [0, ]:
        target, num_steps, _ = targets[i]
        print(target.name)
        sampler = EnsembleSampler(target, chains, '.')
        x = sampler.sample(num_steps)
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

    ensemble_run()
    
    #sequential_run()