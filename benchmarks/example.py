import itertools
import sys
sys.path.append("./")
import os

import jax
import jax.numpy as jnp

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
    Rosenbrock,
)

# model = Gaussian(ndims=30,condition_number=1)
# model = GermanCredit()
model = Brownian()
# model = Rosenbrock()
n = 20000
num_chains = 128

# ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
#     model=model,
#     sampler=unadjusted_mclmc(integrator_type="mclachlan", preconditioning=False, num_windows=2,),
#     key=jax.random.PRNGKey(1), 
#     n=2000,
#     batch=num_chains,  
# )

# print(f"\nGradient calls for unadjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
# print(f'ess {ess_avg}')
# print(f'ess {ess_avg}, L = {params.L.mean()}, step size = {params.step_size.mean()}')  

for integrator_type, max, (L_proposal_factor, random_trajectory_length), frac_tune3, in itertools.product(['mclachlan', 'velocity_verlet'], ['avg', 'max'], [(jnp.inf, True), (1.25, False)], [0.0]):


    # print(f"\nGradient calls for adjusted MCLMC with {integrator_type} and max to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
    print(f'integrator type {integrator_type}, max {max}, L_proposal_factor {L_proposal_factor}, random_trajectory_length {random_trajectory_length}, frac_tune3 {frac_tune3}')
    
    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
        model=model,
        sampler=adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, num_windows=2,max=max, frac_tune3=frac_tune3, tuning_factor=1.3,target_acc_rate=0.9, L_proposal_factor=L_proposal_factor, random_trajectory_length=random_trajectory_length),
        key=jax.random.PRNGKey(1), 
        n=n,
        batch=num_chains,  
    )
    print(f'ess avg {ess_avg}, ess max {ess} L = {jnp.nanmean(params.L)}, step size = {jnp.nanmean(params.step_size)}')  
    
    # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    #     model=model,
    #     sampler=adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, num_windows=2,max='max_svd', frac_tune3=0.0, tuning_factor=1.3,target_acc_rate=0.9),
    #     key=jax.random.PRNGKey(1), 
    #     n=n,
    #     batch=num_chains,  
    # )

    # print(f"\nGradient calls for adjusted MCLMC with {integrator_type} and max svd (stage 2) to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
    # print(f'ess avg {ess_avg}, ess max {ess} L = {jnp.nanmean(params.L)}, step size = {jnp.nanmean(params.step_size)}')  
    
    # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    #     model=model,
    #     sampler=adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, num_windows=2,max='max_svd', frac_tune3=0.1, tuning_factor=1.3,target_acc_rate=0.9),
    #     key=jax.random.PRNGKey(1), 
    #     n=n,
    #     batch=num_chains,  
    # )

    # print(f"\nGradient calls for adjusted MCLMC with {integrator_type} and max svd (stage 3) to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
    # print(f'ess avg {ess_avg}, ess max {ess} L = {jnp.nanmean(params.L)}, step size = {jnp.nanmean(params.step_size)}')  

   

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=nuts(integrator_type="velocity_verlet", preconditioning=False),
    key=jax.random.PRNGKey(1), 
    n=10000,
    batch=num_chains,
)

print(f"\nGradient calls for NUTS to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
print(f'ess {ess_avg}, ess max {ess}')

raise Exception



raise Exception

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=adjusted_mclmc(integrator_type="mclachlan", preconditioning=False, num_windows=2,max='max_svd', frac_tune3=0.1,tuning_factor=1.3),
    key=jax.random.PRNGKey(1), 
    n=20000,
    batch=num_chains,  
)


print(f"\nGradient calls for adjusted MCLMC (max svd tuning) to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
print(f'ess {ess_avg}, L = {params.L.mean()}, step size = {params.step_size.mean()}')

raise Exception

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=adjusted_hmc(integrator_type="velocity_verlet", preconditioning=False, num_windows=2,max='avg'),
    key=jax.random.PRNGKey(1), 
    n=20000,
    batch=num_chains,  
)

print(f"\nGradient calls for adjusted HMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
print(f'ess {ess_avg}, L = {params.L.mean()}')


