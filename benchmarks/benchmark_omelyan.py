
import sys

import numpy as np

sys.path.append("./")
from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Brownian, Gaussian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility

import os
import jax 

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()

models = {Gaussian(d, condition_number=1., eigenvalues='linear'): {'mclmc': 20000, 'adjusted_mclmc': 20000, 'nuts': 20000}
    for d in np.ceil(np.logspace(1, 5, num=10))
    }

run_benchmarks(batch_size=128, models=models, key_index=21, do_grid_search=False, do_non_grid_search=True, return_ess_corr=False)
