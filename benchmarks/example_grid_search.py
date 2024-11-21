import sys

sys.path.append("./")
sys.path.append("../blackjax")
import os
from benchmarks.benchmark import grid_search_only_L

import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()


from metrics import benchmark
from benchmarks.sampling_algorithms import (

    adjusted_mclmc_tuning,
    nuts,
    unadjusted_mclmc,
)
from benchmarks.inference_models import (
    Gaussian,
)

model = Gaussian(ndims=10,condition_number=1)

init_pos_key, fast_tune_key_adjusted, key_for_fast_grid = jax.random.split(jax.random.PRNGKey(1), 3)






L, step_size, ess, ess_avg, ess_corr_avg, rate, edge = grid_search_only_L(
    model=model,
    sampler='mclmc',
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