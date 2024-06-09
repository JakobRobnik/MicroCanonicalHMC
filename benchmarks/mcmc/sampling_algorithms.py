

from typing import Callable, Union
from chex import PRNGKey
import jax
import jax.numpy as jnp
from benchmarks import mcmc
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
# from blackjax.adaptation.window_adaptation import da_adaptation
from blackjax.mcmc.integrators import generate_euclidean_integrator, generate_isokinetic_integrator, mclachlan, yoshida, velocity_verlet, omelyan, isokinetic_mclachlan, isokinetic_velocity_verlet, isokinetic_yoshida, isokinetic_omelyan
from blackjax.mcmc.adjusted_mclmc import rescale
from blackjax.util import run_inference_algorithm
import blackjax
from blackjax.util import pytree_size
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)

# __all__ = ["samplers"]


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
    integrator = blackjax.mcmc.integrators.velocity_verlet,
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
            inverse_mass_matrix)

        new_adaptation_state = da_update(
            adaptation_state,
            info.acceptance_rate,
        )

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
    # print(adaptation_state, "adaptation_state\n\n")
    return kernel_state, {"step_size" : da_final(adaptation_state), "inverse_mass_matrix" : inverse_mass_matrix}

# blackjax doesn't export coefficients, which is inconvenient
map_integrator_type_to_integrator = {
    'hmc': {
        "mclachlan" : mclachlan,
        "yoshida" : yoshida,
        "velocity_verlet" : velocity_verlet,
        "omelyan" : omelyan
    },
    'mclmc' : {
        "mclachlan" : isokinetic_mclachlan,
        "yoshida" : isokinetic_yoshida,
        "velocity_verlet" : isokinetic_velocity_verlet,
        "omelyan" : isokinetic_omelyan
    }
}

def run_nuts(
    integrator_type, logdensity_fn, num_steps, initial_position, transform, key, preconditioning):
    
    # integrator = generate_euclidean_integrator(coefficients)
    integrator = map_integrator_type_to_integrator['hmc'][integrator_type]
    # integrator = blackjax.mcmc.integrators.velocity_verlet # note: defaulted to in nuts

    rng_key, warmup_key = jax.random.split(key, 2)

    if not preconditioning:
        state, params = da_adaptation(
            rng_key=warmup_key, 
            initial_position=initial_position, 
            algorithm=blackjax.nuts,
            integrator=integrator,
            logdensity_fn=logdensity_fn)
    
    else:
        # print(params["inverse_mass_matrix"], "inv\n\n")
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn, integrator=integrator)
        (state, params), _ = warmup.run(warmup_key, initial_position, 2000)

    nuts = blackjax.nuts(logdensity_fn=logdensity_fn, step_size=params['step_size'], inverse_mass_matrix= params['inverse_mass_matrix'], integrator=integrator)

    final_state, state_history, info_history = run_inference_algorithm(
        rng_key=rng_key,
        initial_state=state,
        inference_algorithm=nuts,
        num_steps=num_steps,
        transform=lambda x: transform(x.position),
        progress_bar=True
    )

    return state_history, params, info_history.num_integration_steps.mean() * calls_per_integrator_step(integrator_type), info_history.acceptance_rate.mean(), None, None

def run_mclmc(integrator_type, logdensity_fn, num_steps, initial_position, transform, key, preconditioning, frac_tune3):

    integrator = map_integrator_type_to_integrator['mclmc'][integrator_type]

    init_key, tune_key, run_key = jax.random.split(key, 3)


    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    
    kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=integrator,
        sqrt_diag_cov=sqrt_diag_cov,
    )

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=preconditioning,
        frac_tune3=frac_tune3,
        # desired_energy_var= 1e-5
    )

    # jax.debug.print("params {x}", x=(blackjax_mclmc_sampler_params.L, blackjax_mclmc_sampler_params.step_size))
    # jax.debug.print("params {x}", x=blackjax_mclmc_sampler_params.sqrt_diag_cov**2)


    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
        sqrt_diag_cov=blackjax_mclmc_sampler_params.sqrt_diag_cov,
        integrator = integrator,

        # sqrt_diag_cov=jnp.ones((initial_position.shape[0],)),
    )

    # try doing low mem version twice

    # _, _, expectations = run_inference_algorithm(
    #     rng_key=run_key,
    #     initial_state=blackjax_state_after_tuning,
    #     inference_algorithm=sampling_alg,
    #     num_steps=num_steps,
    #     return_state_history=False,
    #     transform=lambda x: transform(x.position),
    #     # expectation=lambda x: jnp.array([x**2, x]),
    #     expectation=lambda x: x**2,
    #     progress_bar=True,
    # )

    # jax.debug.print("blahblah[1] {x}", x=blahblah)

    # ex2 = expectations[-1][0]
    # ex = expectations[-1][1]
    # var = expectations[:, 0] - ex[:, 1]**2


    _, samples, _ = run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda x: transform(x.position),
        progress_bar=True,
    )
    
    # def cumulative_avg(samples):
    #     return jnp.cumsum(samples, axis = 0) / jnp.arange(1, samples.shape[0] + 1)[:, None]


    # print(samples.mean(axis=0))
    # jax.debug.print("allclose {x}", x=jnp.allclose((samples**2).mean(axis=0), expectations[-1], atol=1e-2))

    
    # jax.debug.print("expectation {x}", x=expectations[-1])
    # jax.debug.print("samples {x}", x=(samples**2).mean(axis=0))
    # jax.debug.print("samples {x}", x=cumulative_avg(samples))
    # jax.debug.print("samples {x}", x=jnp.var(samples, axis=0))
    # jax.debug.print("comparison {x}", x=jnp.mean(expectations))

    # raise Exception

    acceptance_rate = 1.
    return samples, blackjax_mclmc_sampler_params, calls_per_integrator_step(integrator_type), acceptance_rate, None, None


def run_adjusted_mclmc(integrator_type, logdensity_fn, num_steps, initial_position, transform, key, preconditioning, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.0, target_acc_rate=None, L_proposal_factor=jnp.inf):
    integrator = map_integrator_type_to_integrator['mclmc'][integrator_type]

    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, random_generator_arg=init_key
    )


    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
                integrator=integrator,
                integration_steps_fn = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(avg_num_integration_steps)),
                sqrt_diag_cov=sqrt_diag_cov,
            )(
                rng_key=rng_key, 
                state=state, 
                step_size=step_size, 
                logdensity_fn=logdensity_fn,
                L_proposal_factor=L_proposal_factor)
    
    if target_acc_rate is None:
        target_acc_rate = target_acceptance_rate_of_order[integrator_order(integrator_type)]
        print("target acc rate")

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
        params_history,
        final_da
    ) = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=target_acc_rate,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        frac_tune3=frac_tune3,
        diagonal_preconditioning=preconditioning,
    )



    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L
    # jax.debug.print("params {x}", x=(blackjax_mclmc_sampler_params.step_size, blackjax_mclmc_sampler_params.L))


    alg = blackjax.adjusted_mclmc(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn = lambda key: jnp.ceil(jax.random.uniform(key) * rescale(L/step_size)) ,
        integrator=integrator,
        sqrt_diag_cov=blackjax_mclmc_sampler_params.sqrt_diag_cov,
        L_proposal_factor=L_proposal_factor
        

    )


    _, out, info = run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps, 
        transform=lambda x: transform(x.position), 
        progress_bar=True)
    


    return out, blackjax_mclmc_sampler_params, calls_per_integrator_step(integrator_type) * (L/step_size), info.acceptance_rate, params_history, final_da


samplers = {
    'nuts' : run_nuts,
    'mclmc' : run_mclmc, 
    'adjusted_mclmc': run_adjusted_mclmc, 
    }


# foo = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(20.56))

# print(jnp.mean(jax.vmap(foo)(jax.random.split(jax.random.PRNGKey(1), 10000000))))