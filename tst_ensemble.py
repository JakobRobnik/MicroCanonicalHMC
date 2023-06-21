import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

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

def ensemble_run():
    chains = 4096

    for i in [5, ]:
        target, num_steps, _ = targets[i]
        print(target.name)
        sampler = EnsembleSampler(target, chains)
        x = sampler.sample(num_steps)


def sequential_run():
    
    for i in [5, ]:
        target, _, num_steps = targets[i]
        sampler = SequentialSampler(target, diagonal_preconditioning= False, varEwanted= 5e-5)
        grads_for_es = sampler.sample(num_steps, num_chains = 5, output = 'ess')
        print(target.name + ': ' + str(grads_for_es))
        


if __name__ == '__main__':
    ensemble_run()
    #sequential_run()