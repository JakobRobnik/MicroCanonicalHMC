import itertools
import sys

sys.path.append("./")
sys.path.append("../blackjax")
import os
from benchmarks.benchmark import grid_search_only_L

import jax
import jax.numpy as jnp

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()


from metrics import benchmark, grid_search, grid_search_langevin_mams
from benchmarks.sampling_algorithms import (

    adjusted_mclmc_no_tuning,
    adjusted_mclmc_tuning,
    nuts,
    unadjusted_mclmc,
    unadjusted_mclmc_no_tuning,
)
from benchmarks.inference_models import (
    Brownian,
    Gaussian,
    GermanCredit,
)
import blackjax

# model = Gaussian(ndims=2,condition_number=1)
# model = GermanCredit()

init_pos_key, fast_tune_key_adjusted, key_for_fast_grid = jax.random.split(jax.random.PRNGKey(1), 3)






# L, step_size, ess, ess_avg, ess_corr_avg, rate, edge = grid_search_only_L(
#     model=model,
#     sampler='adjusted_mclmc',
#     num_steps=10000,
#     num_chains=128,
#     integrator_type='mclachlan',
#     key=key_for_fast_grid,
#     grid_size=10,
#     # z=blackjax_adjusted_mclmc_sampler_params.L*2,
#     # delta_z=blackjax_adjusted_mclmc_sampler_params.L*2-1.0,
#     # state=blackjax_adjusted_state_after_tuning,
#     grid_iterations=2,
#     opt='max',
#     L_proposal_factor=1.25
# )

# print(f"fast grid search edge {edge}")
# print(f"fast grid search L {L}, step_size {step_size}")
# print(f"fast grid search ess {ess}, ess_avg {ess_avg}, ess_corr_avg {ess_corr_avg}, rate {rate}")


num_chains = 128
num_steps = 100000
model = GermanCredit()
# model = Brownian()

results, edge, state = grid_search_langevin_mams(
    model=model,
    key=jax.random.PRNGKey(1),
    grid_size=6,
    num_iter=2,
    integrator_type='mclachlan',
    num_steps=num_steps,
    num_chains=num_chains,
    pvmap=jax.pmap
)
print(results[0], results[1])
print(results[2])
print(results[3])



# init_state_key, init_pos_key = jax.random.split(jax.random.PRNGKey(1))


# initial_position = model.sample_init(init_pos_key)
# initial_state = blackjax.mcmc.adjusted_mclmc.init(
#             position=initial_position,
#             logdensity_fn=model.logdensity_fn,
#             # rng_key=init_sta`te_key,
#             random_generator_arg=jax.random.PRNGKey(0),
#         )

for integrator_type, (max, tuning_factor, frac_tune3), (L_proposal_factor, random_trajectory_length) in itertools.product(['mclachlan'], [
        # ('max',1.0, 0.0), 
        ('avg', 1.3, 0.0), 
        # ('max_svd', 0.5, 0.0),
        # ('max_svd', 0.5, 0.1),
        ], 
            [(results[1][0], False)], 
            # [(None, False)], 
                ):


        # print(f"\nGradient calls for adjusted MCLMC with {integrator_type} and max to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
        print("\n\n")
        print(f'integrator type {integrator_type}, max {max}, L_proposal_factor {L_proposal_factor}, random_trajectory_length {random_trajectory_length}, frac_tune3 {frac_tune3}, tuning_factor {tuning_factor}')
        
        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
            model=model,
            sampler=adjusted_mclmc_no_tuning(state,integrator_type=integrator_type, 
                                             L=results[0][0], 
                                            #  L=1.96471682, 
                                             L_proposal_factor=L_proposal_factor,
                                            #  L_proposal_factor=3.99,
                                             step_size=results[3][1], 
                                            #  step_size=0.34388438, 
                                             sqrt_diag_cov=1., random_trajectory_length=random_trajectory_length, num_tuning_steps=0,
                                             ),
            key=jax.random.PRNGKey(5), 
            n=num_steps,
            batch=num_chains,  
        )
        print(f'ess avg {ess_avg}, ess max {ess} L = {jnp.nanmean(params.L)}, step size = {jnp.nanmean(params.step_size)}')  

raise Exception

# initial_position = model.sample_init(jax.random.PRNGKey(1))
# initial_state = blackjax.mcmc.mclmc.init(
#             position=initial_position,
#             logdensity_fn=model.logdensity_fn,
#             rng_key=jax.random.PRNGKey(1),
#         )

# ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark(
#                 model=model,
#                 sampler=unadjusted_mclmc_no_tuning(
#                     integrator_type='mclachlan',
#                     initial_state=initial_state,
#                     sqrt_diag_cov=1.,
#                     L=2.,
#                     step_size=1.,
#                     # L_proposal_factor=L_proposal_factor,
#                     return_ess_corr=False,
#                     num_tuning_steps=1000, # doesn't matter what is passed here
#                 ),
#                 key=jax.random.PRNGKey(1),
#                 n=1000,
#                 batch=2,
#                 pvmap=jax.pmap,
#             )

# print(ess)
# raise Exception






