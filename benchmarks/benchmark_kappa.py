
import sys

import numpy as np

sys.path.append("./")
sys.path.append("../blackjax")
from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Brownian, Gaussian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility

import os
import jax 

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()

models = {
    Gaussian(100, k, eigenvalues=eigenval_type): {'mclmc': 100000, 'adjusted_mclmc': 40000, 'nuts': 40000}
    for k in np.ceil(np.logspace(1, 5, num=10)).astype(int) for eigenval_type in ["log", "outliers"]
    }

run_benchmarks(batch_size=128, models=models, key_index=25, do_grid_search=True, do_non_grid_search=True, return_ess_corr=False)