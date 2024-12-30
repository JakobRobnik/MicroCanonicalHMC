import itertools
import sys
import time
sys.path.append("./")
sys.path.append("../blackjax")
import os

import jax
import jax.numpy as jnp
import blackjax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()

from metrics import benchmark
from benchmarks.sampling_algorithms import (

    adjusted_hmc,
    adjusted_mclmc,
    adjusted_mclmc_no_tuning,
    nuts,
    unadjusted_mclmc,
    unadjusted_mclmc_no_tuning,
    unadjusted_underdamped_langevin,
    unadjusted_underdamped_langevin_no_tuning,
)
from benchmarks.inference_models import (
    Brownian,
    Gaussian,
    GermanCredit,
    Rosenbrock,
    StochasticVolatility,
)

# model = Gaussian(ndims=10,condition_number=100)
# model = GermanCredit()
# model = StochasticVolatility()
# model = Gaussian(ndims=10)
# model = Rosenbrock()
n = 40000
num_chains = 64

# print(jnp.sqrt(jnp.mean(model.E_x2)*model.ndims))
# raise Exception

init_state_key, init_pos_key = jax.random.split(jax.random.PRNGKey(0))


# initial_position = model.sample_init(init_pos_key)
# initial_state = blackjax.mcmc.underdamped_langevin.init(
#             position=initial_position,
#             logdensity_fn=model.logdensity_fn,
#             rng_key=init_state_key,
#         )

# ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
#     model=model,
#     sampler=unadjusted_underdamped_langevin_no_tuning(initial_state=initial_state, integrator_type="velocity_verlet", L=jnp.sqrt(model.ndims), step_size=jnp.sqrt(model.ndims)/5,num_tuning_steps=1 , sqrt_diag_cov=(jnp.ones(model.ndims))  ),
#     key=jax.random.PRNGKey(1), 
#     n=n,
#     batch=num_chains,  
# )



# tic = time.time()
# ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
#     model=model,
#     sampler=unadjusted_mclmc(integrator_type="mclachlan", preconditioning=False, num_windows=2,),
#     key=jax.random.PRNGKey(1), 
#     n=n,
#     batch=num_chains, 
#     pvmap=jax.vmap 
# )
# toc = time.time()
# print(f"Time elapsed {toc-tic}")



# print(f"\nGradient calls for unadjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
# print(f'ess {ess_avg}')
# print(f'ess {ess_avg}, ess max {ess}, L = {params.L.mean()}, step size = {params.step_size.mean()}')  

for model, integrator_type, (max, tuning_factor, frac_tune3), (L_proposal_factor, random_trajectory_length) in itertools.product([Gaussian(ndims=10,condition_number=100), Brownian(), Rosenbrock()],['mclachlan'], [
    ('max',1.0, 0.0), 
    ('avg', 1.3, 0.0), 
    ('max_svd', 0.5, 0.0),
    ('max_svd', 0.5, 0.15),
    ], [(jnp.inf, True)], ):


    # print(f"\nGradient calls for adjusted MCLMC with {integrator_type} and max to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
    print("\n\n")
    print(f'integrator type {integrator_type}, max {max}, L_proposal_factor {L_proposal_factor}, random_trajectory_length {random_trajectory_length}, frac_tune3 {frac_tune3}, tuning_factor {tuning_factor}, model {model.name}')
    
    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
        model=model,
        sampler=adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, num_windows=2,max=max, frac_tune3=frac_tune3, tuning_factor=tuning_factor,target_acc_rate=0.9, L_proposal_factor=L_proposal_factor, random_trajectory_length=random_trajectory_length, num_tuning_steps=2000),
        key=jax.random.PRNGKey(3), 
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

# raise Exception

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=nuts(integrator_type="velocity_verlet", preconditioning=False, num_tuning_steps=5000),
    key=jax.random.PRNGKey(0), 
    n=n//10,
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


