import jax
import blackjax
import numpy as np
import jax.numpy as jnp

from benchmarks.inference_models import *
from mclmc import run_mclmc


eevpd = 4e-6


model = Funnel_with_Data()
samples = run_mclmc(model, 10**5, transform= lambda x: x[[0, 10]], desired_energy_var= eevpd)


