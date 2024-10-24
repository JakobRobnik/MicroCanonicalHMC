from typing import Callable, Union
from chex import PRNGKey
import jax
import jax.numpy as jnp
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

# from blackjax.adaptation.window_adaptation import da_adaptation
from blackjax.mcmc.integrators import (
    generate_euclidean_integrator,
    generate_isokinetic_integrator,
    mclachlan,
    yoshida,
    velocity_verlet,
    omelyan,
    isokinetic_mclachlan,
    isokinetic_velocity_verlet,
    isokinetic_yoshida,
    isokinetic_omelyan,
)
from blackjax.util import run_inference_algorithm
import blackjax
from blackjax.util import pytree_size, store_only_expectation_values
from blackjax.adaptation.step_size import (
    dual_averaging_adaptation,
)
from blackjax.mcmc.adjusted_mclmc import rescale
from jax.flatten_util import ravel_pytree

from blackjax.diagnostics import effective_sample_size


def calls_per_integrator_step(c):
    if c == "velocity_verlet":
        return 1
    if c == "mclachlan":
        return 2
    if c == "yoshida":
        return 3
    if c == "omelyan":
        return 5

    else:
        raise Exception("No such integrator exists in blackjax")


def integrator_order(c):
    if c == "velocity_verlet":
        return 2
    if c == "mclachlan":
        return 2
    if c == "yoshida":
        return 4
    if c == "omelyan":
        return 4

    else:
        raise Exception("No such integrator exists in blackjax")


target_acceptance_rate_of_order = {2: 0.65, 4: 0.8}


def da_adaptation(
    rng_key: PRNGKey,
    initial_position,
    algorithm,
    logdensity_fn: Callable,
    num_steps: int = 1000,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    progress_bar: bool = False,
    integrator=blackjax.mcmc.integrators.velocity_verlet,
):

    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    kernel = algorithm.build_kernel(integrator=integrator)
    init_kernel_state = algorithm.init(initial_position, logdensity_fn)
    inverse_mass_matrix = jnp.ones(pytree_size(initial_position))

    def step(state, key):

        adaptation_state, kernel_state = state

        new_kernel_state, info = kernel(
            key,
            kernel_state,
            logdensity_fn,
            jnp.exp(adaptation_state.log_step_size),
            inverse_mass_matrix,
        )

        new_adaptation_state = da_update(
            adaptation_state,
            info.acceptance_rate,
        )

        # jax.debug.print("acceptance rate {x}", x=info.acceptance_rate)

        return (
            (new_adaptation_state, new_kernel_state),
            (True),
        )

    keys = jax.random.split(rng_key, num_steps)
    init_state = da_init(initial_step_size), init_kernel_state
    (adaptation_state, kernel_state), _ = jax.lax.scan(
        step,
        init_state,
        keys,
    )
    return kernel_state, {
        "step_size": da_final(adaptation_state),
        "inverse_mass_matrix": inverse_mass_matrix,
    }


# blackjax doesn't export coefficients, which is inconvenient
map_integrator_type_to_integrator = {
    "hmc": {
        "mclachlan": mclachlan,
        "yoshida": yoshida,
        "velocity_verlet": velocity_verlet,
        "omelyan": omelyan,
    },
    "mclmc": {
        "mclachlan": isokinetic_mclachlan,
        "yoshida": isokinetic_yoshida,
        "velocity_verlet": isokinetic_velocity_verlet,
        "omelyan": isokinetic_omelyan,
    },
}


# produce a kernel that only stores the average values of the bias for E[x_2] and Var[x_2]
def with_only_statistics(model, alg, initial_state, key, num_steps):

    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm=alg,
        state_transform=lambda state: jnp.array([model.transform(state.position) ** 2]),
        incremental_value_transform=lambda x: jnp.array(
            [
                jnp.average(jnp.square(x - model.E_x2) / model.Var_x2),
                # jnp.sqrt(jnp.average(jnp.square(x - model.E_x2) / model.Var_x2)),
                # jnp.sqrt(jnp.average(jnp.square(x - model.E_x2) / (model.Var_x2))),
                jnp.max(jnp.square(x - model.E_x2) / model.Var_x2),
            ]
        ),
    )

    return run_inference_algorithm(
        rng_key=key,
        initial_state=memory_efficient_sampling_alg.init(initial_state),
        inference_algorithm=memory_efficient_sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=False,
    )[1]


def run_unadjusted_mclmc_no_tuning(initial_state, integrator_type, step_size, L, sqrt_diag_cov, return_ess_corr=False):

    def s(model, num_steps, initial_position, key):

        fast_key, slow_key = jax.random.split(key, 2)

        alg = blackjax.mclmc(
            model.logdensity_fn,
            L=L,
            step_size=step_size,
            sqrt_diag_cov=sqrt_diag_cov,
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        )

        expectations = with_only_statistics(model, alg, initial_state, fast_key, num_steps)[0]


        ess_corr = jax.lax.cond(not return_ess_corr, lambda: jnp.inf, lambda: jnp.mean(effective_sample_size(jax.vmap(lambda x: ravel_pytree(x)[0])(run_inference_algorithm(
            rng_key=slow_key,
            initial_state=initial_state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=lambda state, _: (model.transform(state.position)),
            progress_bar=True)[1])[None, ...]))/num_steps)

        return (
            MCLMCAdaptationState(L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov),
            calls_per_integrator_step(integrator_type),
            1.0,
            expectations, ess_corr
        )

    return s

def run_adjusted_mclmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    sqrt_diag_cov,
    L_proposal_factor=jnp.inf,
    return_ess_corr=False,
):

    def s(model, num_steps, initial_position, key):


        num_steps_per_traj = L / step_size
        alg = blackjax.adjusted_mclmc(
            logdensity_fn=model.logdensity_fn,
            step_size=step_size,
            integration_steps_fn=lambda k: jnp.ceil(jax.random.uniform(k) * rescale(num_steps_per_traj)),
            integrator= map_integrator_type_to_integrator["mclmc"][integrator_type],
            sqrt_diag_cov=sqrt_diag_cov,
            L_proposal_factor=L_proposal_factor,
        )

        fast_key, slow_key = jax.random.split(key, 2)

        results = with_only_statistics(model, alg, initial_state, fast_key, num_steps)
        expectations, info = results[0], results[1]


       
        ess_corr = jax.lax.cond(not return_ess_corr, lambda: jnp.inf, lambda: jnp.mean(effective_sample_size(jax.vmap(lambda x: ravel_pytree(x)[0])(run_inference_algorithm(
            rng_key=slow_key,
            initial_state=initial_state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=lambda state, _: (model.transform(state.position)),
            progress_bar=False)[1])[None, ...]))/num_steps)
        
        # ess_corr = lambda: jnp.mean(effective_sample_size(jax.vmap(lambda x: ravel_pytree(x)[0])(run_inference_algorithm(
        #     rng_key=slow_key,
        #     initial_state=initial_state,
        #     inference_algorithm=alg,
        #     num_steps=num_steps,
        #     transform=lambda state, _: (model.transform(state.position)),
        #     progress_bar=True)[1])[None, ...]))/num_steps
        
        # jax.debug.print("acceptance rate {x}", x=info.acceptance_rate)
        # jax.debug.print("acceptance rate direct {x}", x=info.acceptance_rate.mean())
        # jax.debug.print("acceptance rate indirect {x}", x=info.is_accepted.mean())

        return (
            MCLMCAdaptationState(L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov),
            num_steps_per_traj * calls_per_integrator_step(integrator_type),
            info.acceptance_rate.mean(),
            expectations, ess_corr
        )

    return s

def unadjusted_mclmc_tuning(initial_position, num_steps, rng_key, logdensity_fn, integrator_type, diagonal_preconditioning, frac_tune3=0.1):

    tune_key, init_key = jax.random.split(rng_key, 2)

    initial_state = blackjax.mcmc.mclmc.init(
            position=initial_position,
            logdensity_fn=logdensity_fn,
            rng_key=init_key,
        )

    kernel = lambda sqrt_diag_cov: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        sqrt_diag_cov=sqrt_diag_cov,
    )

    return blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=diagonal_preconditioning,
        frac_tune3=frac_tune3,
        
    )

def adjusted_mclmc_tuning(initial_position, num_steps, rng_key, logdensity_fn, integrator_type, diagonal_preconditioning, target_acc_rate, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1, L_proposal_factor=jnp.inf, params=None, max=False, num_windows=1):

    integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    init_key, tune_key = jax.random.split(rng_key, 2)

    initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )

    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
        integrator=integrator,
        integration_steps_fn=lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)
        ),
        sqrt_diag_cov=sqrt_diag_cov,
    )(
        rng_key=rng_key,
        state=state,
        step_size=step_size,
        logdensity_fn=logdensity_fn,
        L_proposal_factor=L_proposal_factor,
    )

    (
        blackjax_state_after_tuning,
        blackjax_adjusted_mclmc_sampler_params,
        params_history,
        final_da,
    ) = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=target_acc_rate,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        frac_tune3=frac_tune3,
        diagonal_preconditioning=diagonal_preconditioning,
        params=params,
        max=max,
        num_windows=num_windows,
    )

    return blackjax_state_after_tuning, blackjax_adjusted_mclmc_sampler_params


def run_unadjusted_mclmc(integrator_type, preconditioning, frac_tune3=0.1, return_ess_corr=False):

    def s(model, num_steps, initial_position, key):

        tune_key, run_key = jax.random.split(key, 2)

        

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = unadjusted_mclmc_tuning( initial_position, num_steps, tune_key, model.logdensity_fn, integrator_type, preconditioning, frac_tune3)

        return run_unadjusted_mclmc_no_tuning(
            blackjax_state_after_tuning,
            integrator_type,
            blackjax_mclmc_sampler_params.step_size,
            blackjax_mclmc_sampler_params.L,
            blackjax_mclmc_sampler_params.sqrt_diag_cov,
            return_ess_corr=return_ess_corr,
        )(model, num_steps, initial_position, run_key)

    return s


def run_adjusted_mclmc(
    integrator_type,
    preconditioning,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    L_proposal_factor=jnp.inf,
    target_acc_rate=None,
    params=None,
    return_ess_corr=False,
    max=False,
    num_windows=1,
):

    def s(model, num_steps, initial_position, key):

        tune_key, run_key = jax.random.split(key, 2)


        if target_acc_rate is None:
            new_target_acc_rate = target_acceptance_rate_of_order[
                integrator_order(integrator_type)
            ]
        else:
            new_target_acc_rate = target_acc_rate


        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params) = adjusted_mclmc_tuning( initial_position, num_steps, tune_key, model.logdensity_fn, integrator_type, preconditioning, new_target_acc_rate, frac_tune1, frac_tune2, frac_tune3, L_proposal_factor, params=params, max=max, num_windows=num_windows)


        return run_adjusted_mclmc_no_tuning(
            blackjax_state_after_tuning,
            integrator_type,
            blackjax_mclmc_sampler_params.step_size,
            blackjax_mclmc_sampler_params.L,
            blackjax_mclmc_sampler_params.sqrt_diag_cov,
            L_proposal_factor,
            return_ess_corr=return_ess_corr,
        )(model, num_steps, initial_position, run_key)

    return s

def run_nuts(integrator_type, preconditioning, return_ess_corr=False):

    def s(model, num_steps, initial_position, key):

        integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

        rng_key, warmup_key = jax.random.split(key, 2)

        if not preconditioning:
            state, params = da_adaptation(
                rng_key=warmup_key,
                initial_position=initial_position,
                algorithm=blackjax.nuts,
                integrator=integrator,
                logdensity_fn=model.logdensity_fn,
                num_steps=2000,
                target_acceptance_rate=0.8,
            )

        else:
            warmup = blackjax.window_adaptation(
                blackjax.nuts, model.logdensity_fn, integrator=integrator
            )
            (state, params), _ = warmup.run(warmup_key, initial_position, 2000)

        alg = blackjax.nuts(
            logdensity_fn=model.logdensity_fn,
            step_size=params["step_size"],
            inverse_mass_matrix=params["inverse_mass_matrix"],
            integrator=integrator,
        )

        fast_key, slow_key = jax.random.split(rng_key, 2)

        results = with_only_statistics(model, alg, state, fast_key, num_steps)
        expectations, info = results[0], results[1]

        ess_corr = jax.lax.cond(not return_ess_corr, lambda: jnp.inf, lambda: jnp.mean(effective_sample_size(jax.vmap(lambda x: ravel_pytree(x)[0])(run_inference_algorithm(
            rng_key=slow_key,
            initial_state=state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=lambda state, _: (model.transform(state.position)),
            progress_bar=True)[1])[None, ...]))/num_steps)
        
        

        return (
            params,
            info.num_integration_steps.mean()
            * calls_per_integrator_step(integrator_type),
            info.acceptance_rate.mean(),
            expectations, ess_corr
        )

    return s


samplers = {
    "nuts": run_nuts,
    "mclmc": run_unadjusted_mclmc,
    "adjusted_mclmc": run_adjusted_mclmc,
}
