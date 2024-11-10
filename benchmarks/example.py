import sys
sys.path.append("./")
import os

import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()

from metrics import benchmark
from benchmarks.sampling_algorithms import (

    nuts,
    unadjusted_mclmc,
)
from benchmarks.inference_models import (
    Gaussian,
)


num_chains = 128
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=Gaussian(100,100),
    sampler=unadjusted_mclmc(integrator_type="mclachlan", preconditioning=False),
    key=jax.random.PRNGKey(1), 
    n=10000,
    batch=num_chains,
)

print(f"\nGradient calls for MCLMC to reach RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=Gaussian(100,100),
    sampler=nuts(integrator_type="velocity_verlet", preconditioning=False),
    key=jax.random.PRNGKey(1), 
    n=10000,
    batch=num_chains,
)

print(f"\nGradient calls for NUTS to reach RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")