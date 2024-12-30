
import sys

sys.path.append("./")
sys.path.append("../blackjax")

from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Brownian, Funnel, Gaussian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility

import os
import jax 

batch_size = 128

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

models = {

        StochasticVolatility(): {'mclmc': 60000, 'adjusted_mclmc': 60000, 'adjusted_mchmc': 60000, 'adjusted_hmc': 60000, 'nuts': 60000}
        ,

        ItemResponseTheory(): {'mclmc': 40000, 'adjusted_mclmc': 10000, 'adjusted_mchmc': 10000, 'adjusted_hmc': 40000, 'nuts': 40000},
        }

models2 = {
        GermanCredit(): {'mclmc': 40000, 'adjusted_mclmc': 40000, 'adjusted_mchmc': 40000, 'adjusted_hmc': 40000, 'nuts': 40000},
        Brownian(): {"mclmc": 20000, 'adjusted_mclmc': 20000, 'adjusted_mchmc': 20000, 'adjusted_hmc': 20000, "nuts": 20000},
        # Gaussian(100): {"mclmc": 1000, 'adjusted_mclmc': 1000, 'adjusted_mchmc': 1000, 'adjusted_hmc': 1000, "nuts": 1000},
        Rosenbrock(): {'mclmc': 80000, 'adjusted_mclmc': 80000, 'adjusted_mchmc': 80000, 'adjusted_hmc': 80000, 'nuts': 80000},
        # Funnel(): {'mclmc': 200000, 'adjusted_mclmc': 200000, 'adjusted_mchmc': 200000, 'adjusted_hmc': 200000, 'nuts': 200000},
    }

run_benchmarks(batch_size=batch_size, models=models, key_index=27, do_grid_search=False, do_fast_grid_search=False, do_non_grid_search=True, return_ess_corr=False, integrators = ["mclachlan", "velocity_verlet"], pvmap=jax.pmap, num_tuning_steps=5000)
