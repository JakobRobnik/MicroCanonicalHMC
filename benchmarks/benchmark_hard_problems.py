
import sys

sys.path.append("./")
sys.path.append("../blackjax")

from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Banana, Brownian, Funnel, Gaussian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility, rng_inference_gym_icg

import os
import jax 

batch_size = 128

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

models = {

        

        }

models2 = {
        # Banana(): {"mclmc": 20000, 'adjusted_mclmc': 1000, 'adjusted_mchmc': 1000, 'adjusted_hmc': 1000, "nuts": 20000},
        # Rosenbrock(): {'mclmc': 400000, 'adjusted_mclmc': 40000, 'adjusted_mchmc': 80000, 'adjusted_hmc': 40000, 'nuts': 10000},
        # Brownian(): {"mclmc": 20000, 'adjusted_mclmc': 40000, 'adjusted_mchmc': 20000, 'adjusted_hmc': 20000, "nuts": 10000},
    
        # GermanCredit(): {'mclmc': 40000, 'adjusted_mclmc': 40000, 'adjusted_mchmc': 40000, 'adjusted_hmc': 40000, 'nuts': 10000},
        # Gaussian(100, condition_number=1,): {"mclmc": 10000, 'adjusted_mclmc': 10000, 'adjusted_mchmc': 1000, 'adjusted_hmc': 1000, "nuts": 2000},
        Gaussian(ndims=100, eigenvalues='Gamma', numpy_seed= rng_inference_gym_icg): {"mclmc": 10000, 'adjusted_mclmc': 30000, 'adjusted_mchmc': 1000, 'adjusted_hmc': 1000, "nuts": 2000},
        # ItemResponseTheory(): {'mclmc': 40000, 'adjusted_mclmc': 40000, 'adjusted_mchmc': 40000, 'adjusted_hmc': 40000, 'nuts': 10000},
        # StochasticVolatility(): {'mclmc': 400000, 'adjusted_mclmc': 40000, 'adjusted_mchmc': 120000, 'adjusted_hmc': 120000, 'nuts': 10000},
        # Funnel(): {'mclmc': 200000, 'adjusted_mclmc': 10000000, 'adjusted_mchmc': 200000, 'adjusted_hmc': 200000, 'nuts': 100000},
    }

run_benchmarks(batch_size=batch_size, models=models2, key_index=51, do_grid_search=False, do_fast_grid_search=False, do_non_grid_search=True, return_ess_corr=False, integrators = ["velocity_verlet"], pvmap=jax.pmap, num_tuning_steps=20000, do_nuts=True, do_adjusted_mclmc=False, do_adjusted_mclmc_with_nuts_tuning=True, do_unadjusted_mclmc=False)
