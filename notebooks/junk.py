import jax
import jax.numpy as jnp

from mclmc.sampler import Sampler
from benchmarks.benchmarks_mchmc import *
from mclmc.boundary import Boundary



target = StandardNormal(d= 100)



boundary = Boundary(target.d, where_positive= jnp.array([0, ]))
sampler = Sampler(target, boundary = boundary)

x = sampler.sample(100000)



import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.hist(x[:, 0], bins = 30)
plt.subplot(1, 2, 2)
plt.hist(x[:, 1], bins = 30)
plt.show()

