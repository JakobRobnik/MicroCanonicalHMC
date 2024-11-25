
import sys

sys.path.append("./")
sys.path.append("../blackjax")

from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import Brownian, Gaussian, GermanCredit, ItemResponseTheory, Rosenbrock, StochasticVolatility

import os
import jax 

batch_size = 50

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

# models = {

#         # Brownian(): {"mclmc": 40000, "adjusted_mclmc": 40000, "nuts": 40000},
#         GermanCredit(): {'mclmc': 80000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
#         ItemResponseTheory(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
#         Rosenbrock(): {'mclmc': 80000, 'adjusted_mclmc' : 80000, 'nuts': 80000},
#         StochasticVolatility(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
#     }

models = {Gaussian(5) : {'mclmc': 2000, 'adjusted_mclmc': 2000, 'nuts': 2000}}

    # Gaussian(d, condition_number=1., eigenvalues='linear'): {'mclmc': 20000, 'adjusted_mclmc': 20000, 'nuts': 20000}
    # for d in [2,3,4,5,6,7,8,9,10]
    # }

    # models = {

    #     Brownian(): {"mclmc": 40000, "adjusted_mclmc": 40000, "nuts": 40000},
    #     GermanCredit(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
    #     ItemResponseTheory(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
    #     Rosenbrock(): {'mclmc': 80000, 'adjusted_mclmc' : 80000, 'nuts': 80000},
    #     StochasticVolatility(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
    # }


run_benchmarks(batch_size=1, models=models, key_index=21, do_grid_search=False, do_non_grid_search=False, do_fast_grid_search=True)
    # benchmark(batch_size=128, models=models, key_index=21, do_grid_search=True, do_non_grid_search=False)

# run_benchmarks(batch_size=batch_size, models=models, key_index=23, do_grid_search=False, do_fast_grid_search=True, do_non_grid_search=True)
