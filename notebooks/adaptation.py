import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

from mclmc.vanilla import Sampler
from mclmc import dynamics
from benchmarks.benchmarks_mchmc import *



target= StandardNormal(d = 100)

sampler = Sampler(target, 5, 0.1, integrator= dynamics.leapfrog, hmc= False, adjust = True, full_refreshment= True)


x, hyp = sampler.adaptation_dual_averaging(100)
