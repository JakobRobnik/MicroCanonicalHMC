import itertools
import sys
import time
sys.path.append("./")
sys.path.append("../blackjax")
import os

import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc.adjusted_mclmc import rescale
from benchmarks.sampling_algorithms import (
    adjusted_hmc_no_tuning,
    adjusted_mclmc_tuning,
    calls_per_integrator_step,
    integrator_order,
    map_integrator_type_to_integrator,
    adjusted_mclmc,
    adjusted_mclmc_no_tuning,
    nuts,
    unadjusted_mclmc_no_tuning,
    target_acceptance_rate_of_order,
    unadjusted_mclmc,
    unadjusted_mclmc_tuning,
)
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
# num_cores = jax.local_device_count()

from metrics import benchmark
from benchmarks.sampling_algorithms import (

    adjusted_hmc,
    adjusted_mclmc,
    adjusted_mclmc_no_tuning,
    adjusted_mclmc_tuning,
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
import wandb

wandb.login()

# model = Gaussian(ndims=10,condition_number=1)
model = Brownian()
num_chains = 128
n = 10000
# 1: Define objective/training function


initial_position = model.sample_init(jax.random.PRNGKey(0))
initial_state = blackjax.mcmc.adjusted_mclmc.init(
            position=initial_position,
            logdensity_fn=model.logdensity_fn,
            random_generator_arg=jax.random.PRNGKey(0),
        )

# Todo: q for wilka: different key

def objective(config):

    tune_key, key = jax.random.split(jax.random.PRNGKey(0), 2)

    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark(
                            model=model,
                            sampler=adjusted_mclmc_no_tuning(
                                integrator_type=config.integrator_type,
                                initial_state=initial_state,
                                inverse_mass_matrix=1.,
                                L=config.L,
                                random_trajectory_length=False,
                                step_size=config.step_size,
                                L_proposal_factor=2.0,
                                return_ess_corr=False,
                                num_tuning_steps=0, # doesn't matter what is passed here
                            ),
                            key=key,
                            n=n,
                            batch=num_chains,
                            pvmap=jax.vmap,
                        )


    score = ess
    return score

name = "brownian"

def main():
    wandb.init(project=name)
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: Define the search space
sweep_configuration = {
    "method": "random",

    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "L": {"max": jnp.sqrt(model.ndims).item() * 4, "min": jnp.sqrt(model.ndims).item() / 4},
        "step_size": {"max": jnp.sqrt(model.ndims).item(), "min": jnp.sqrt(model.ndims).item() / 16},
        
        "integrator_type": {"values": ["mclachlan"]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=name)

wandb.agent(sweep_id, function=main, count=40)