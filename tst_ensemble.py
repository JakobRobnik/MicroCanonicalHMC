import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

from sampling.ensemble import Sampler as EnsembleSampler

from benchmarks.benchmarks_mchmc import *
from benchmarks.german_credit import Target as GermanCredit
from benchmarks.brownian import Target as Brownian
from benchmarks.IRT import Target as IRT


num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)


targets = [[Banana(prior = 'prior'), 100],
        [IllConditionedGaussianGamma(prior = 'prior'), 1000],
        [GermanCredit(), 1000],
        [Brownian(), 1000],
        [IRT(), 1000],
        [StochasticVolatility(), 2000]]

def chain_pairs(y, x):

    moments1 = np.average(np.square(sampler.Target.transform(y)), axis = 0)
    moments2 = np.average(np.square(sampler.Target.transform(x)), axis = 0)
    var = np.std(np.square(y), axis=0)**2

    bias_d = np.square(moments1 - moments2) / var
    bias_avg, bias_max = np.average(bias_d, -1), np.max(bias_d, -1)
    
    t = np.arange(0, 2* len(bias_max), 2)
    plt.plot(t, bias_avg, color = 'tab:blue', label= 'average')
    plt.plot(t, bias_max, color = 'tab:red', label= 'max')
    plt.plot(t, np.ones(len(bias_max)) * 1e-2, '--', color = 'black')
    plt.legend()
    plt.xlabel('# gradient evaluations')
    plt.ylabel(r'$\mathrm{bias}^2$')
    plt.yscale('log')
    plt.savefig('plots/tst_ensemble/' + sampler.Target.name + '_pairs.png')
    plt.tight_layout()
    plt.close()



if __name__ == '__main__':
    chains = 4096

    for i in [3, ]:
        target, num_steps = targets[i]
        print(target.name)
        x0 = jnp.zeros(shape = (chains, target.d))
        sampler = EnsembleSampler(target, chains, diagonal_preconditioning = False, alpha = 1.)
        #sampler.isotropic_u0 = True
        x = sampler.sample(num_steps)#, x_initial = x0)
        
        y = x[:, ::2, :]
        print(y.shape)
        
        sampler.eps_factor = 2.0
        x = sampler.sample(num_steps//2)#, x_initial = x0)
      
        print(x.shape)
        
        chain_pairs(y, x)