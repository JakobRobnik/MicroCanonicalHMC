import sys

sys.path.append("./")
sys.path.append("../blackjax")
import os
from benchmarks.benchmark import grid_search_only_L

import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()


from metrics import benchmark, grid_search, grid_search_langevin_mams
from benchmarks.sampling_algorithms import (

    adjusted_mclmc_tuning,
    nuts,
    unadjusted_mclmc,
    unadjusted_mclmc_no_tuning,
)
from benchmarks.inference_models import (
    Brownian,
    Gaussian,
)
import blackjax

# model = Gaussian(ndims=2,condition_number=1)
model = Brownian()

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

results, edge, _ = grid_search_langevin_mams(
    model=model,
    key=jax.random.PRNGKey(1),
    grid_size=2,
    num_iter=2,
    integrator_type='mclachlan',
    num_steps=1000,
    num_chains=128,
    pvmap=jax.pmap
)
print(results[0], results[1])
print(results[2])
raise Exception









init_pos_key, fast_tune_key_adjusted, key_for_fast_grid = jax.random.split(jax.random.PRNGKey(1), 3)






L, step_size, ess, ess_avg, ess_corr_avg, rate, edge = grid_search_only_L(
    model=model,
    sampler='adjusted_mclmc',
    num_steps=2000,
    num_chains=128,
    integrator_type='mclachlan',
    key=key_for_fast_grid,
    grid_size=10,
    # z=blackjax_adjusted_mclmc_sampler_params.L*2,
    # delta_z=blackjax_adjusted_mclmc_sampler_params.L*2-1.0,
    # state=blackjax_adjusted_state_after_tuning,
    grid_iterations=2,
    opt='avg'
)

print(f"fast grid search edge {edge}")
print(f"fast grid search L {L}, step_size {step_size}")
print(f"fast grid search ess {ess}, ess_avg {ess_avg}, ess_corr_avg {ess_corr_avg}, rate {rate}")