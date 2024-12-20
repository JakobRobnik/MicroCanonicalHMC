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
def with_only_statistics(model, alg, initial_state, key, num_steps, incremental_value_transform=None):

    if incremental_value_transform is None:
        incremental_value_transform=lambda x: jnp.array(
                [
                    jnp.average(jnp.square(x[0] - model.E_x2) / model.Var_x2),
                    # jnp.sqrt(jnp.average(jnp.square(x - model.E_x2) / model.Var_x2)),
                    # jnp.sqrt(jnp.average(jnp.square(x - model.E_x2) / (model.Var_x2))),
                    jnp.max(jnp.square(x[0] - model.E_x2) / model.Var_x2),
                    
                    
                ]
            )

    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm=alg,
        state_transform=lambda state: jnp.array([model.transform(state.position) ** 2, model.transform(state.position)]),
        incremental_value_transform=incremental_value_transform,
    )

    return run_inference_algorithm(
        rng_key=key,
        initial_state=memory_efficient_sampling_alg.init(initial_state),
        inference_algorithm=memory_efficient_sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=False,
    )[1]


def unadjusted_mclmc_no_tuning(initial_state, integrator_type, step_size, L, sqrt_diag_cov, num_tuning_steps, return_ess_corr=False):

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
            progress_bar=False)[1])[None, ...]))/num_steps)

        return (
            MCLMCAdaptationState(L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov),
            calls_per_integrator_step(integrator_type),
            1.0,
            expectations, 
            ess_corr, 
            num_tuning_steps
        )

    return s

def adjusted_mclmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    sqrt_diag_cov,
    num_tuning_steps,
    L_proposal_factor=jnp.inf,
    return_ess_corr=False,
    random_trajectory_length=True,
):

    def s(model, num_steps, initial_position, key):

        num_steps_per_traj = L / step_size
        if random_trajectory_length:
            integration_steps_fn = lambda k: jnp.ceil(jax.random.uniform(k) * rescale(num_steps_per_traj))
        else:
            integration_steps_fn = lambda _: jnp.ceil(num_steps_per_traj)

        alg = blackjax.adjusted_mclmc(
            logdensity_fn=model.logdensity_fn,
            step_size=step_size,
            integration_steps_fn=integration_steps_fn,
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
        #     progress_bar=False)[1])[None, ...]))/num_steps
        
        # jax.debug.print("acceptance rate {x}", x=info.acceptance_rate)
        # jax.debug.print("acceptance rate direct {x}", x=info.acceptance_rate.mean())
        # jax.debug.print("acceptance rate indirect {x}", x=info.is_accepted.mean())

        return (
            MCLMCAdaptationState(L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov),
            num_steps_per_traj * calls_per_integrator_step(integrator_type),
            info.acceptance_rate.mean(),
            expectations, 
            ess_corr,
            num_tuning_steps,
        )

    return s

def adjusted_hmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    sqrt_diag_cov,
    num_tuning_steps,
    # integration_steps_fn,
    return_ess_corr=False,
):

    def s(model, num_steps, initial_position, key):

        num_steps_per_traj = L / step_size

        alg = blackjax.dynamic_hmc(
            logdensity_fn=model.logdensity_fn,
            step_size=step_size,
            inverse_mass_matrix=jnp.ones(pytree_size(initial_position)),
            integrator= map_integrator_type_to_integrator["hmc"][integrator_type],
            # integration_steps_fn=integration_steps_fn,
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
        #     progress_bar=False)[1])[None, ...]))/num_steps
        
        # jax.debug.print("acceptance rate {x}", x=info.acceptance_rate)
        # jax.debug.print("acceptance rate direct {x}", x=info.acceptance_rate.mean())
        # jax.debug.print("acceptance rate indirect {x}", x=info.is_accepted.mean())

        return (
            # todo fix
            MCLMCAdaptationState(L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov),
            num_steps_per_traj * calls_per_integrator_step(integrator_type),
            info.acceptance_rate.mean(),
            expectations, 
            ess_corr,
            num_tuning_steps,
        )

    return s

def unadjusted_mclmc_tuning(initial_position, num_steps, rng_key, logdensity_fn, integrator_type, diagonal_preconditioning, frac_tune3=0.1, num_windows=1, num_tuning_steps=500):

    tune_key, init_key = jax.random.split(rng_key, 2)

    frac_tune1 = num_tuning_steps / (2*num_steps)
    frac_tune2 = num_tuning_steps / (2*num_steps)

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
        frac_tune2=frac_tune2,
        frac_tune1=frac_tune1,
        num_windows=num_windows,
        
    )

def adjusted_mclmc_tuning(initial_position, num_steps, rng_key, logdensity_fn,  diagonal_preconditioning, target_acc_rate, kernel, frac_tune3=0.1, params=None, max='avg', num_windows=1,  tuning_factor=1.0, num_tuning_steps=500):


    init_key, tune_key = jax.random.split(rng_key, 2)

    initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )

    frac_tune1 = num_tuning_steps / (2*num_steps)
    frac_tune2 = num_tuning_steps / (2*num_steps)
    

    

    logdensity_grad_fn = jax.grad(logdensity_fn)

    (
        blackjax_state_after_tuning,
        blackjax_adjusted_mclmc_sampler_params,
        
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
        tuning_factor=tuning_factor,
        logdensity_grad_fn=logdensity_grad_fn,
    )

    return blackjax_state_after_tuning, blackjax_adjusted_mclmc_sampler_params


def unadjusted_mclmc(integrator_type, preconditioning, frac_tune3=0.1, return_ess_corr=False, num_windows=1):

    def s(model, num_steps, initial_position, key):

        tune_key, run_key = jax.random.split(key, 2)

        num_tuning_steps = 2000
        

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = unadjusted_mclmc_tuning( initial_position, num_steps, tune_key, model.logdensity_fn, integrator_type, preconditioning, frac_tune3, num_windows=num_windows, num_tuning_steps=num_tuning_steps)

        # num_tuning_steps = (0.1 + 0.1) * num_windows * num_steps + frac_tune3 * num_steps

        return unadjusted_mclmc_no_tuning(
            blackjax_state_after_tuning,
            integrator_type,
            blackjax_mclmc_sampler_params.step_size,
            blackjax_mclmc_sampler_params.L,
            blackjax_mclmc_sampler_params.sqrt_diag_cov,
            num_tuning_steps,
            return_ess_corr=return_ess_corr,
        )(model, num_steps, initial_position, run_key)

    return s


def adjusted_mclmc(
    integrator_type,
    preconditioning,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    L_proposal_factor=jnp.inf,
    target_acc_rate=None,
    params=None,
    return_ess_corr=False,
    max='avg',
    num_windows=1,
    random_trajectory_length=True,
    tuning_factor=1.0,
):
    
    

    def s(model, num_steps, initial_position, key):

        tune_key, run_key = jax.random.split(key, 2)

        integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

        if random_trajectory_length:
            integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps))
        else:
            integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
        integrator=integrator,
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        sqrt_diag_cov=sqrt_diag_cov,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=model.logdensity_fn,
            L_proposal_factor=L_proposal_factor,
        )


        if target_acc_rate is None:
            new_target_acc_rate = target_acceptance_rate_of_order[
                integrator_order(integrator_type)
            ]
        else:
            new_target_acc_rate = target_acc_rate


        num_tuning_steps = 2000
        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params) = adjusted_mclmc_tuning( initial_position, num_steps, tune_key, model.logdensity_fn, preconditioning, new_target_acc_rate, kernel, frac_tune3, params=params, max=max, num_windows=num_windows, tuning_factor=tuning_factor,num_tuning_steps=num_tuning_steps)

        # num_tuning_steps = (frac_tune1 + frac_tune2 ) * num_windows * num_steps + frac_tune3 * num_steps
        # jax.debug.print("num_tuning_steps {x}", x=num_tuning_steps)

        return adjusted_mclmc_no_tuning(
            blackjax_state_after_tuning,
            integrator_type,
            blackjax_mclmc_sampler_params.step_size,
            blackjax_mclmc_sampler_params.L,
            blackjax_mclmc_sampler_params.sqrt_diag_cov,
            num_tuning_steps,
            L_proposal_factor,
            return_ess_corr=return_ess_corr,
            random_trajectory_length=random_trajectory_length,
        )(model, num_steps, initial_position, run_key)

    return s

def adjusted_hmc(
    integrator_type,
    preconditioning,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    target_acc_rate=None,
    params=None,
    return_ess_corr=False,
    max='avg',
    num_windows=1,
    tuning_factor=1.0,
):

    def s(model, num_steps, initial_position, key):

        tune_key, run_key, init_key = jax.random.split(key, 3)


        if target_acc_rate is None:
            new_target_acc_rate = target_acceptance_rate_of_order[
                integrator_order(integrator_type)
            ]
        else:
            new_target_acc_rate = target_acc_rate

        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)).astype(jnp.int32)
       

  

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.dynamic_hmc.build_kernel(
        integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        )(
            rng_key=rng_key,
            state=state,
            logdensity_fn=model.logdensity_fn,
            step_size=step_size,
            inverse_mass_matrix=sqrt_diag_cov,
        )

        num_tuning_steps = 2000
      

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params) = adjusted_mclmc_tuning( initial_position, num_steps, tune_key, model.logdensity_fn, preconditioning, new_target_acc_rate, kernel, frac_tune3, params=params, max=max, num_windows=num_windows, tuning_factor=tuning_factor,num_tuning_steps=num_tuning_steps)
        
        # num_tuning_steps = (frac_tune1 + frac_tune2 ) * num_windows * num_steps + frac_tune3 * num_steps

        return adjusted_hmc_no_tuning(
            blackjax_state_after_tuning,
            integrator_type=integrator_type,
            # step_size=0.1,
            # L = 1.0,
            step_size=blackjax_mclmc_sampler_params.step_size,
            L=blackjax_mclmc_sampler_params.L,
            sqrt_diag_cov=jnp.ones(pytree_size(initial_position)),
            num_tuning_steps=num_tuning_steps,
            # integration_steps_fn=integration_steps_fn,
            return_ess_corr=return_ess_corr,
        )(model, num_steps, initial_position, run_key)

    return s

def nuts(integrator_type, preconditioning, return_ess_corr=False, return_samples=False,incremental_value_transform=None):


    def s(model, num_steps, initial_position, key):
        # num_tuning_steps = num_steps // 5
        num_tuning_steps = 2000

        integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

        rng_key, warmup_key = jax.random.split(key, 2)



        if not preconditioning:
            state, params = da_adaptation(
                rng_key=warmup_key,
                initial_position=initial_position,
                algorithm=blackjax.nuts,
                integrator=integrator,
                logdensity_fn=model.logdensity_fn,
                num_steps=num_tuning_steps,
                target_acceptance_rate=0.8,
            )

        else:
            warmup = blackjax.window_adaptation(
                blackjax.nuts, model.logdensity_fn, integrator=integrator
            )
            (state, params), _ = warmup.run(warmup_key, initial_position, num_tuning_steps)

        alg = blackjax.nuts(
            logdensity_fn=model.logdensity_fn,
            step_size=params["step_size"],
            inverse_mass_matrix=params["inverse_mass_matrix"],
            integrator=integrator,
        )

        fast_key, slow_key = jax.random.split(rng_key, 2)

        results = with_only_statistics(model, alg, state, fast_key, num_steps, incremental_value_transform=incremental_value_transform)
        expectations, info = results[0], results[1]


        ess_corr = jax.lax.cond(not return_ess_corr, lambda: jnp.inf, lambda: jnp.mean(effective_sample_size(jax.vmap(lambda x: ravel_pytree(x)[0])(run_inference_algorithm(
            rng_key=slow_key,
            initial_state=state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=lambda state, _: (model.transform(state.position)),
            progress_bar=False)[1])[None, ...]))/num_steps)

        


        if return_samples:
            expectations=run_inference_algorithm(
            rng_key=slow_key,
            initial_state=state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=lambda state, _: (model.transform(state.position)),
            progress_bar=False)[1]
      
        
        

        return (
            params,
            info.num_integration_steps.mean()
            * calls_per_integrator_step(integrator_type),
            info.acceptance_rate.mean(),
            expectations, 
            ess_corr,
            num_tuning_steps,
        )

    return s





def unadjusted_underdamped_langevin_no_tuning(initial_state, integrator_type, step_size, L, sqrt_diag_cov, num_tuning_steps, return_ess_corr=False):

    def s(model, num_steps, initial_position, key):

        fast_key, slow_key = jax.random.split(key, 2)

        alg = blackjax.underdamped_langevin(
            model.logdensity_fn,
            L=L,
            step_size=step_size,
            sqrt_diag_cov=sqrt_diag_cov,
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
        )

        expectations = with_only_statistics(model, alg, initial_state, fast_key, num_steps)[0]


        ess_corr = jax.lax.cond(not return_ess_corr, lambda: jnp.inf, lambda: jnp.mean(effective_sample_size(jax.vmap(lambda x: ravel_pytree(x)[0])(run_inference_algorithm(
            rng_key=slow_key,
            initial_state=initial_state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=lambda state, _: (model.transform(state.position)),
            progress_bar=False)[1])[None, ...]))/num_steps)

        return (
            MCLMCAdaptationState(L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov),
            calls_per_integrator_step(integrator_type),
            1.0,
            expectations, 
            ess_corr, 
            num_tuning_steps
        )

    return s

def unadjusted_underdamped_langevin_tuning(initial_position, num_steps, rng_key, logdensity_fn, integrator_type, diagonal_preconditioning, frac_tune3=0.1, num_windows=1, desired_energy_var=5e-4):

    tune_key, init_key = jax.random.split(rng_key, 2)

    initial_state = blackjax.mcmc.underdamped_langevin.init(
            position=initial_position,
            logdensity_fn=logdensity_fn,
            rng_key=init_key,
        )

    kernel = lambda sqrt_diag_cov: blackjax.mcmc.underdamped_langevin.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
        sqrt_diag_cov=sqrt_diag_cov,
    )

    return blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=diagonal_preconditioning,
        frac_tune3=frac_tune3,
        num_windows=num_windows,
        desired_energy_var=desired_energy_var,
        
    )

def unadjusted_underdamped_langevin(integrator_type, preconditioning, frac_tune3=0.1, return_ess_corr=False, num_windows=1, desired_energy_var=5e-4):

    def s(model, num_steps, initial_position, key):

        tune_key, run_key = jax.random.split(key, 2)

        

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = unadjusted_underdamped_langevin_tuning( initial_position, num_steps, tune_key, model.logdensity_fn, integrator_type, preconditioning, frac_tune3, num_windows=num_windows, desired_energy_var=desired_energy_var)

        num_tuning_steps = (0.1 + 0.1) * num_windows * num_steps + frac_tune3 * num_steps

        return unadjusted_underdamped_langevin_no_tuning(
            blackjax_state_after_tuning,
            integrator_type,
            blackjax_mclmc_sampler_params.step_size,
            blackjax_mclmc_sampler_params.L,
            blackjax_mclmc_sampler_params.sqrt_diag_cov,
            num_tuning_steps,
            return_ess_corr=return_ess_corr,
        )(model, num_steps, initial_position, run_key)

    return s


samplers = {
    "nuts": nuts,
    "mclmc": unadjusted_mclmc,
    "adjusted_mclmc": adjusted_mclmc,
    "adjusted_hmc": adjusted_hmc,
}
