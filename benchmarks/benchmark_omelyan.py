
import sys

import numpy as np

sys.path.append("./")
sys.path.append("../blackjax")
from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Brownian, Gaussian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility

import os
import jax 

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
# num_cores = jax.local_device_count()

models = {Gaussian(d, condition_number=1., eigenvalues='linear'): {'mclmc': 10000, 'adjusted_mclmc': 5000, 'nuts': 5000}
    for d in np.ceil(np.logspace(2, 5, num=10)).astype(int)
    }

run_benchmarks(batch_size=1, models=models, key_index=20, do_grid_search=True, do_non_grid_search=False, return_ess_corr=False, integrators = ["omelyan", "mclachlan", "velocity_verlet"])

run_benchmarks(batch_size=1, models=models, key_index=20, do_grid_search=False, do_non_grid_search=True, return_ess_corr=False, integrators = ["omelyan", "mclachlan", "velocity_verlet"])