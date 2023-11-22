import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw

from scipy.integrate import odeint

from mclmc.sampler import Sampler
from benchmarks.benchmarks_mchmc import *


target = IllConditionedGaussian(d = 100, condition_number= 1000.)

sampler = Sampler(target, diagonal_preconditioning= True)

x = sampler.sample(100000)


print(sampler.hyp['sigma'] / jnp.sqrt(target.second_moments))

