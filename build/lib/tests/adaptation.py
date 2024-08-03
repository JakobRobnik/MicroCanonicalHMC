import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

from mclmc.vanilla import Sampler
from mclmc import dynamics
from benchmarks.targets import *


target= StandardNormal(d = 100)
target.prior_draw = lambda key: jax.random.normal(key, (target.d,))

sampler = Sampler(target, 3, 10., integrator= dynamics.leapfrog, hmc= False, adjust = True, full_refreshment= True)

x, hyp = sampler.adaptation_predictor(100)
