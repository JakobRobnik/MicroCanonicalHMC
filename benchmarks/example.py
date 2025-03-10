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
from benchmarks.lattice import Phi4
from benchmarks.sampling_algorithms import (

    adjusted_hmc,
    adjusted_mclmc,
    adjusted_mclmc_no_tuning,
    adjusted_mclmc_tuning,
    nuts,
    unadjusted_mclmc,
    unadjusted_mclmc_no_tuning,
    unadjusted_mclmc_tuning,
    unadjusted_underdamped_langevin,
    unadjusted_underdamped_langevin_no_tuning,
)
from benchmarks.inference_models import (
    Banana,
    Brownian,
    Funnel,
    Gaussian,
    GermanCredit,
    Rosenbrock,
    StochasticVolatility,
)

model = Banana()
# model = Gaussian(ndims=10,condition_number=1)
# model = GermanCredit()
# model = Phi4()
# model = StochasticVolatility()
# model = Gaussian(ndims=100, condition_number=10)
# model = Gaussian(10)
model = Funnel()
# model = Brownian()
# model = Brownian()
n = 10000000
num_chains = 128

# print(Phi4(100,1.0).E_x2 )
# # raise Exception

# tic = time.time()
# ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_max, _,bias = benchmark(
#     model=model,
#     sampler=unadjusted_mclmc(integrator_type="mclachlan", preconditioning=False, num_windows=1,),
#     key=jax.random.PRNGKey(1), 
#     n=n,
#     batch=num_chains, 
#     pvmap=jax.pmap 
# )
# toc = time.time()
# print(f"Time elapsed {toc-tic}")

# print(f"\nGradient calls for unadjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_max} (avg over {num_chains} chains and dimensions)")


tic = time.time()
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_max, _,bias, _ = benchmark(
    model=model,
    sampler=adjusted_mclmc(target_acc_rate=0.99, integrator_type="velocity_verlet", preconditioning=True, num_windows=2,num_tuning_steps=100000),
    key=jax.random.PRNGKey(1), 
    n=n,
    batch=num_chains, 
    pvmap=jax.pmap 
)
toc = time.time()
print(f"Time elapsed {toc-tic}")

print(f"\nacc rate 0.9, lprop inf, Gradient calls for adjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_max} (avg over {num_chains} chains and dimensions)")

print(f"L and step size {params.L.mean()} {params.step_size.mean()}")

raise Exception

tic = time.time()
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_max, _,bias, _ = benchmark(
    model=model,
    sampler=adjusted_mclmc(target_acc_rate=0.99, integrator_type="velocity_verlet", preconditioning=True, num_windows=2,L_proposal_factor=5.0, num_tuning_steps=100000),
    key=jax.random.PRNGKey(1), 
    n=n,
    batch=num_chains, 
    pvmap=jax.pmap 
)
toc = time.time()
print(f"Time elapsed {toc-tic}")
print(f"\n acc rate 0.99, lprop 5, Gradient calls for adjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_max} (avg over {num_chains} chains and dimensions)")

print(f"L and step size {params.L.mean()} {params.step_size.mean()}")


tic = time.time()
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_max, _,bias, _ = benchmark(
    model=model,
    sampler=adjusted_mclmc(target_acc_rate=0.9, integrator_type="velocity_verlet", preconditioning=True, num_windows=2,L_proposal_factor=5.0, num_tuning_steps=100000),
    key=jax.random.PRNGKey(1), 
    n=n,
    batch=num_chains, 
    pvmap=jax.pmap 
)
toc = time.time()
print(f"Time elapsed {toc-tic}")

print(f"\n acc rate 0.9, lprop 5, Gradient calls for adjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_max} (avg over {num_chains} chains and dimensions)")

print(f"L and step size {params.L.mean()} {params.step_size.mean()}")

tic = time.time()
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_max, _,bias, _ = benchmark(
    model=model,
    sampler=adjusted_mclmc(target_acc_rate=0.9, integrator_type="velocity_verlet", preconditioning=True, num_windows=2, num_tuning_steps=100000),
    key=jax.random.PRNGKey(1), 
    n=n,
    batch=num_chains, 
    pvmap=jax.pmap 
)
toc = time.time()
print(f"Time elapsed {toc-tic}")

print(f"\n acc rate 0.9, lprop inf, Gradient calls for adjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_max} (avg over {num_chains} chains and dimensions)")

print(f"L and step size {params.L.mean()} {params.step_size.mean()}")

raise Exception

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_max, _,_, _ = benchmark(
        model=model,
        sampler=nuts(
            integrator_type="velocity_verlet", 
            preconditioning=True, 
            num_tuning_steps=100000,
            target_acc_rate=0.99),
        key=jax.random.PRNGKey(0), 
        n=n,
        batch=num_chains,
    )
print(f"\nGradient calls for nuts to reach standardized RMSE of X^2 of 0.1: {grads_to_low_max} (avg over {num_chains} chains and dimensions)")

raise Exception

# print(jnp.sqrt(jnp.mean(model.E_x2)*model.ndims))
# raise Exception

# init_state_key, init_pos_key, tune_key = jax.random.split(jax.random.PRNGKey(1),3)

# init_pos = model.sample_init(init_pos_key)

# blackjax_state_after_tuning, params, = unadjusted_mclmc_tuning(initial_position=init_pos,
#     num_steps=n,
#     rng_key=tune_key,
#     logdensity_fn=model.logdensity_fn,
#     diagonal_preconditioning=True,
#     frac_tune3=0.0,
#     num_windows=1,
#     num_tuning_steps=5000,
#     integrator_type="mclachlan",

#     )

# # initial_state = 

# initial_state = blackjax.mcmc.adjusted_mclmc.init(
#             position=blackjax_state_after_tuning.position,
#             logdensity_fn=model.logdensity_fn,
#             # rng_key=init_sta`te_key,
#             random_generator_arg=init_state_key,
#         )

# ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
#             model=model,
#             sampler=adjusted_mclmc_no_tuning(initial_state,integrator_type='mclachlan',
#                                              L=params.L, 
#                                             #  L=1.96471682, 
#                                              L_proposal_factor=jnp.inf,
#                                             #  L_proposal_factor=3.99,
#                                              step_size=params.step_size, 
#                                             #  step_size=0.34388438, 
#                                              inverse_mass_matrix=params.inverse_mass_matrix, random_trajectory_length=True, 
#                                              num_tuning_steps=1000,
#                                              ),
#             key=jax.random.PRNGKey(5), 
#             n=n,
#             batch=num_chains,  
#         )

# print(f'ess avg {ess_avg}, ess max {ess} L = {jnp.nanmean(params.L)}, step size = {jnp.nanmean(params.step_size)}, acceptance rate = {acceptance_rate}') 


# raise Exception

# initial_position = model.sample_init(init_pos_key)
# initial_state = blackjax.mcmc.underdamped_langevin.init(
#             position=initial_position,
#             logdensity_fn=model.logdensity_fn,
#             rng_key=init_state_key,
#         )

# ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
#     model=model,
#     sampler=unadjusted_underdamped_langevin_no_tuning(initial_state=initial_state, integrator_type="velocity_verlet", L=jnp.sqrt(model.ndims), step_size=jnp.sqrt(model.ndims)/5,num_tuning_steps=1 , inverse_mass_matrix=(jnp.ones(model.ndims))  ),
#     key=jax.random.PRNGKey(1), 
#     n=n,
#     batch=num_chains,  
# )





# print(f"\nGradient calls for unadjusted MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
# print(f'ess {ess_avg}')
# print(f'ess {ess_avg}, ess max {ess}, L = {params.L.mean()}, step size = {params.step_size.mean()}')  

for model in [
    # Gaussian(10, condition_number=100),
    # Brownian(),
    # Funnel(),
    # Phi4(100,1.0),
    model
    ]:
    print(f"\nModel {model.name}\n")
    for integrator_type, (max, tuning_factor, frac_tune3), (L_proposal_factor, random_trajectory_length) in itertools.product(['velocity_verlet'], [
        # ('max',1.0, 0.0), 
        ('avg', 1.3, 0.0), 
        ('avg', 1.3, 0.1), 
        # ('max_svd', 0.5, 0.0),
        # ('max_svd', 0.5, 0.1),
        ], [(jnp.inf, True)], ):


        # print(f"\nGradient calls for adjusted MCLMC with {integrator_type} and max to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
        print("\n\n")
        print(f'integrator type {integrator_type}, max {max}, L_proposal_factor {L_proposal_factor}, random_trajectory_length {random_trajectory_length}, frac_tune3 {frac_tune3}, tuning_factor {tuning_factor}')
        
        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
            model=model,
            sampler=adjusted_mclmc(integrator_type=integrator_type, preconditioning=True, num_windows=1,max=max, frac_tune3=frac_tune3, tuning_factor=tuning_factor,target_acc_rate=0.9, L_proposal_factor=L_proposal_factor, random_trajectory_length=random_trajectory_length, num_tuning_steps=2000),
            key=jax.random.PRNGKey(5), 
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

    # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    #     model=model,
    #     sampler=nuts(
    #         integrator_type="velocity_verlet", 
    #         preconditioning=True, 
    #         num_tuning_steps=5000,
    #         target_acc_rate=0.99),
    #     key=jax.random.PRNGKey(0), 
    #     n=n//10,
    #     batch=num_chains,
    # )

    # print(f"\nGradient calls for NUTS to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
    # print(f'ess {ess_avg}, ess max {ess}')

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


