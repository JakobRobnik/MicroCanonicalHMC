import os

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from mclmc.sampler import Sampler
from mclmc.benchmark_targets import *
from mclmc.grid_search import search_wrapper



target = IllConditionedGaussian(d = 100, condition_number= 10**6, numpy_seed= None, prior = 'posterior')

sampler = Sampler(target, integrator= 'MN')
#sampler.varEwanted = 1e-3
sampler.diagonal_preconditioning = True
#sampler.frac_tune2 = 1.0
num_samples = 100000
X = sampler.sample(num_samples, 1, output = 'normal')
exit()

n = jnp.arange(1, num_samples+1)
second_moments = jnp.cumsum(jnp.square(X), axis=1) / n[None, :, None]


# intermediate definiton
bias = jnp.average(jnp.square(second_moments - target.second_moments[None, None, :]) / target.variance_second_moments[None, None, :], axis = -1)
for i in range(10):
    plt.plot(n, bias[i, :], '-', color = 'tab:orange', alpha = 0.5)
plt.plot(n, jnp.average(bias, 0), '-', lw = 3, color = 'tab:orange', label = 'averaged')


# Hoffman definition
bias = jnp.max(jnp.square(second_moments - target.second_moments[None, None, :]) / target.variance_second_moments[None, None, :], axis = -1)
for i in range(10):
    plt.plot(n, bias[i, :], '-', color = 'tab:red', alpha = 0.5)
plt.plot(n, jnp.average(bias, 0), '-', lw = 3, color = 'tab:red', label = 'max')

plt.plot([1, 1e5], [0.01, 0.01], '--', color = 'black', alpha = 0.5)

plt.legend()
plt.xlabel('n')
plt.ylabel(r'$b^2$')
plt.xscale('log')
plt.yscale('log')
plt.savefig('without preconditioning.png')
plt.show()
