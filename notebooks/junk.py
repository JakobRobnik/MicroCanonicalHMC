import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw

from scipy.integrate import odeint

from mclmc.sampler import Sampler
from benchmarks.benchmarks_mchmc import StandardNormal


target = StandardNormal(d = 100)

sampler = Sampler(target)

x = sampler.sample(10000)

print(sampler.hyp)

plt.hist(x[:, 0], bins = 20)
plt.savefig('neki.png')
plt.close()
