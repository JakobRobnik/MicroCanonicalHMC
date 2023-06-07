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



if __name__ == '__main__':
    chains = 4096

    for i in [0, ]:
        target, num_steps = targets[i]
        print(target.name)
        x0 = jnp.zeros(shape = (chains, target.d))
        sampler = EnsembleSampler(target, chains, diagonal_preconditioning = False, alpha = 1.)
        #sampler.isotropic_u0 = True
        x = sampler.sample(num_steps)#, x_initial = x0)
