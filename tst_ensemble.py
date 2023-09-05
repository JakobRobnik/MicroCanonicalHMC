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
        [IllConditionedGaussianGamma(prior = 'prior'), 2000],
        [GermanCredit(), 400],
        [Brownian(), 500],
        [IRT(), 700],
        [StochasticVolatility(), 1000]]


if __name__ == '__main__':
    chains = 2048

    for i in [0, 1, 2, 3, 4, 5]:
        target, num_steps = targets[i]
        print(target.name)
        sampler = EnsembleSampler(target, chains, debug= True, plotdir= 'plots/tst_ensemble/')
        # key = jax.random.PRNGKey(42)
        # keys_all = jax.random.split(key, sampler.chains + 1)
        # x = sampler.Target.prior_draw(keys_all[1:])
        # key = keys_all[0]
        # l, g = sampler.Target.grad_nlogp(x)
        # print(l.shape, g.shape)

        x = sampler.sample(num_steps)
        

# def sequential_run():

#     vare = jnp.logspace(-8, -3, 16)
    
#     for i in [0, ]:
#         target, _, num_steps = targets[i]
            
#         def f(vare):
#             sampler = SequentialSampler(target, diagonal_preconditioning= False, varEwanted= vare)
#             return sampler.sample(num_steps, num_chains = 8, output = 'ess')
        
#         ess = jax.pmap(f)(vare)
        
#         plt.plot(vare, ess, '.-')
#         plt.xscale("log")
#         plt.xlabel('Var[E]/d')
#         plt.ylabel('ESS')
#         plt.savefig('plots/tst_ensemble/sequential/' + target.name + '.png')
#         plt.close()
        
#         print(target.name + ': ' + str(np.max(ess)) + ', at energy error Var[E]/d  = ' + str(vare[np.argmax(ess)]))
        