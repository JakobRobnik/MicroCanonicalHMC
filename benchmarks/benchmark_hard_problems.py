
import sys

sys.path.append("./")
sys.path.append("../blackjax")

from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Brownian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility

import os
import jax 

batch_size = 128

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

models = {

        # Brownian(): {"mclmc": 40000, "adjusted_mclmc": 40000, "nuts": 40000},
        GermanCredit(): {'mclmc': 80000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
        # ItemResponseTheory(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
        # Rosenbrock(): {'mclmc': 80000, 'adjusted_mclmc' : 80000, 'nuts': 80000},
        # StochasticVolatility(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
    }

run_benchmarks(batch_size=batch_size, models=models, key_index=24, do_grid_search=False, do_fast_grid_search=False, do_non_grid_search=True, integrators = ["omelyan", "mclachlan", "velocity_verlet"])
