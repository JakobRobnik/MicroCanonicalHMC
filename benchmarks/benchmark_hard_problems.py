
import sys

sys.path.append("./")
sys.path.append("../blackjax")

from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Brownian, Funnel, Gaussian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility

import os
import jax 

batch_size = 4

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
# num_cores = jax.local_device_count()

models = {

        ItemResponseTheory(): {'mclmc': 40000, 'adjusted_mclmc': 10000, 'adjusted_mchmc': 10000, 'adjusted_hmc': 40000, 'nuts': 40000},
        StochasticVolatility(): {'mclmc': 40000, 'adjusted_mclmc': 40000, 'adjusted_mchmc': 40000, 'adjusted_hmc': 40000, 'nuts': 40000},}

models2 = {
        Gaussian(100): {"mclmc": 10000, 'adjusted_mclmc': 10000, 'adjusted_mchmc': 10000, 'adjusted_hmc': 10000, "nuts": 10000},
        Brownian(): {"mclmc": 20000, 'adjusted_mclmc': 20000, 'adjusted_mchmc': 20000, 'adjusted_hmc': 20000, "nuts": 20000},
        GermanCredit(): {'mclmc': 40000, 'adjusted_mclmc': 40000, 'adjusted_mchmc': 40000, 'adjusted_hmc': 40000, 'nuts': 40000},
        Rosenbrock(): {'mclmc': 80000, 'adjusted_mclmc': 80000, 'adjusted_mchmc': 80000, 'adjusted_hmc': 40000, 'nuts': 80000},
        Funnel(): {'mclmc': 200000, 'adjusted_mclmc': 200000, 'adjusted_mchmc': 200000, 'adjusted_hmc': 200000, 'nuts': 200000},
    }

run_benchmarks(batch_size=batch_size, models=models, key_index=26, do_grid_search=False, do_fast_grid_search=False, do_non_grid_search=True, return_ess_corr=False, integrators = ["mclachlan", "velocity_verlet"])
