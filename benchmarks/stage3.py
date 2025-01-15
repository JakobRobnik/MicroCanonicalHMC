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
    map_integrator_type_to_integrator,
)
from benchmarks.inference_models import (
    Banana,
    Brownian,
    Funnel,
    Gaussian,
    GermanCredit,
    ItemResponseTheory,
    Rosenbrock,
    StochasticVolatility,
)

# model = Gaussian(ndims=500,condition_number=100)
# model = ItemResponseTheory()
# model = GermanCredit()
# model = Phi4()
# model = StochasticVolatility()
model = Funnel()
# model = Gaussian(ndims=100, condition_number=10)
# model = Rosenbrock()
# model = Brownian()
# model = Brownian()
n = 50000
num_chains = 64



tune_key, unadjusted_key = jax.random.split(jax.random.PRNGKey(1))
warmup = blackjax.window_adaptation(
        blackjax.nuts, model.logdensity_fn, integrator=map_integrator_type_to_integrator["hmc"]['velocity_verlet']
    )
initial_position = model.sample_init(tune_key)

(state, unadjusted_params), _ = warmup.run(unadjusted_key, initial_position, 1000)

from blackjax.mcmc.adjusted_mclmc import rescale

random_trajectory_length = True
if random_trajectory_length:
            integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps))
else:
    integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)

kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc.build_kernel(
integrator=map_integrator_type_to_integrator["mclmc"]['velocity_verlet'],
integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
inverse_mass_matrix=inverse_mass_matrix,
)(
    rng_key=rng_key,
    state=state,
    step_size=step_size,
    logdensity_fn=model.logdensity_fn,
    L_proposal_factor=jnp.inf,
)


init_key = jax.random.split(tune_key, 1)[0]
state = blackjax.mcmc.adjusted_mclmc.init(
    position=initial_position,
    logdensity_fn=model.logdensity_fn,
    random_generator_arg=init_key,
)

from blackjax.util import pytree_size, store_only_expectation_values
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.adaptation.adjusted_mclmc_adaptation import adjusted_mclmc_make_L_step_size_adaptation, adjusted_mclmc_make_adaptation_L


dim = pytree_size(initial_position)
params = MCLMCAdaptationState(
    jnp.sqrt(dim), jnp.sqrt(dim) * 0.2, 
    inverse_mass_matrix=jnp.sqrt(unadjusted_params["inverse_mass_matrix"]),
    # inverse_mass_matrix=unadjusted_params.inverse_mass_matrix,
    )

jax.debug.print("params {x}", x=params.L)
(
    blackjax_state_after_tuning,
    blackjax_mclmc_sampler_params, _) = adjusted_mclmc_make_L_step_size_adaptation(
        kernel=kernel,
    dim=dim,
    frac_tune1=1.0,
    frac_tune2=0.0,
    target=0.9,
    diagonal_preconditioning=True,
    max=max,
    tuning_factor=1.0,
    logdensity_grad_fn=None,
    fix_L_first_da=True,
    )(
        state, params, n, tune_key
    )

jax.debug.print("params 2 {x}", x=blackjax_mclmc_sampler_params.L)

stage3_key = jax.random.split(tune_key, 1)[0]
blackjax_state_after_tuning, blackjax_mclmc_sampler_params = adjusted_mclmc_make_adaptation_L(
                kernel, frac=0.3, Lfactor=0.3, max='avg', eigenvector=None,
            )(blackjax_state_after_tuning, blackjax_mclmc_sampler_params, n, stage3_key)

(
    blackjax_state_after_tuning,
    blackjax_mclmc_sampler_params, _) = adjusted_mclmc_make_L_step_size_adaptation(
        kernel=kernel,
    dim=dim,
    frac_tune1=1.0,
    frac_tune2=0.0,
    target=0.9,
    diagonal_preconditioning=True,
    max=max,
    tuning_factor=1.0,
    logdensity_grad_fn=None,
    fix_L_first_da=True,
    )(
        blackjax_state_after_tuning, blackjax_mclmc_sampler_params, n, tune_key
    )

jax.debug.print("params 3 {x}", x=blackjax_mclmc_sampler_params.L)

sampler = adjusted_mclmc_no_tuning(
    integrator_type="velocity_verlet",
    initial_state=blackjax_state_after_tuning,
    inverse_mass_matrix=jnp.sqrt(unadjusted_params["inverse_mass_matrix"]),
    L=blackjax_mclmc_sampler_params.L,
    random_trajectory_length=random_trajectory_length,
    step_size=blackjax_mclmc_sampler_params.step_size,
    L_proposal_factor=jnp.inf,
    return_ess_corr=False,
    num_tuning_steps=0,
)

ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
            model=model,
            sampler=sampler,
            key=jax.random.PRNGKey(5), 
            n=n,
            batch=num_chains,  
        )

print(ess)