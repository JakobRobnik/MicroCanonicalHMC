import sys
sys.path.append("./")
import os

import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()

from metrics import benchmark
from benchmarks.sampling_algorithms import (

    adjusted_hmc,
    adjusted_mclmc,
    nuts,
    unadjusted_mclmc,
)
from benchmarks.inference_models import (
    Brownian,
    Gaussian,
    GermanCredit,
)

model = Gaussian(ndims=31,condition_number=1)
# model = GermanCredit()
num_chains = 20
# ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
#     model=model,
#     sampler=unadjusted_mclmc(integrator_type="mclachlan", preconditioning=False, num_windows=2,),
#     key=jax.random.PRNGKey(1), 
#     n=20000,
#     batch=num_chains,  
# )

# print(f"\nGradient calls for unadjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
# print(f'ess {ess_avg}')


ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=adjusted_mclmc(integrator_type="velocity_verlet", preconditioning=False, num_windows=2,max='avg'),
    key=jax.random.PRNGKey(1), 
    n=20000,
    batch=num_chains,  
)


print(f"\nGradient calls for adjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
print(f'ess {ess_avg}, L = {params.L.mean()}')

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=adjusted_hmc(integrator_type="velocity_verlet", preconditioning=False, num_windows=2,max='avg'),
    key=jax.random.PRNGKey(1), 
    n=20000,
    batch=num_chains,  
)

print(f"\nGradient calls for adjusted HMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
print(f'ess {ess_avg}, L = {params.L.mean()}')

raise Exception

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=adjusted_mclmc(integrator_type="mclachlan", preconditioning=False, num_windows=2,max='max_svd', frac_tune3=0.1),
    key=jax.random.PRNGKey(1), 
    n=20000,
    batch=num_chains,  
)


print(f"\nGradient calls for adjusted MCLMC (max svd tuning) to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
print(f'ess {ess_avg}, L = {params.L.mean()}')

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=nuts(integrator_type="velocity_verlet", preconditioning=False),
    key=jax.random.PRNGKey(1), 
    n=10000,
    batch=num_chains,
)

print(f"\nGradient calls for NUTS to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
print(f'ess {ess_avg}')