
import sys

import numpy as np

sys.path.append("./")
sys.path.append("../blackjax")
from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Brownian, Gaussian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility

import os
import jax 

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
# num_cores = jax.local_device_count()

first_list = np.array([2,3,4,5,6,7,8,9])
second_list = np.ceil(np.logspace(2, 5, num=10)).astype(int)
full_list = np.concatenate((first_list, second_list))

models = {model(d): {'mclmc': 1000, 'adjusted_mclmc': 1000, 'nuts': 1000}
    for d in first_list for model in [lambda dim: Gaussian(dim, condition_number=1., eigenvalues='linear'), lambda dim : Gaussian(dim, condition_number=1000, eigenvalues='log'), 
                                    #    lambda d: Rosenbrock(d)
                                       ]
    }

run_benchmarks(batch_size=128, models=models, key_index=20, do_grid_search=False, do_non_grid_search=True, do_fast_grid_search=True, return_ess_corr=False, integrators = ["omelyan", "mclachlan", "velocity_verlet"])

# run_benchmarks(batch_size=4, models=models, key_index=20, do_grid_search=False, do_non_grid_search=True, return_ess_corr=False, integrators = ["omelyan", "mclachlan", "velocity_verlet"])