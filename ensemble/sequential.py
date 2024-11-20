
import sys

sys.path.append("./")
sys.path.append("../blackjax")

from benchmarks.benchmark import run_benchmarks
from benchmarks.inference_models import*

import os
import jax 

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()


steps = lambda n1=40000, n2=10000: {"mclmc": n1,  'adjusted_mclmc': n2, "nuts": n2}

models = {
        Banana(): steps(),
        Gaussian(ndims=100, eigenvalues='Gamma', numpy_seed= rng_inference_gym_icg): steps(200000, 30000),
        GermanCredit(): steps(100000, 30000),
        Brownian(): steps(),
        ItemResponseTheory(): steps(),
        StochasticVolatility(): steps()
    }[-2:]


results = run_benchmarks(batch_size=128, models=models, key_index=0, do_grid_search=False)
