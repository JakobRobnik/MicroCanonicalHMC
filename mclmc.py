import jax
import blackjax
import numpy as np
import jax.numpy as jnp



def _run_mclmc(logdensity_fn, num_steps, initial_position, transform= lambda x: x, key= jax.random.key(0), desired_energy_var= 5e-4, progress_bar= True):

    init_key, tune_key, run_key = jax.random.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    # build the kernel
    kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        sqrt_diag_cov=sqrt_diag_cov,
    )

    # find values for the hyperparameters: L (typical momentum decoherence length) and step_size
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        #diagonal_preconditioning=False,
        desired_energy_var=desired_energy_var
    )

    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
    )

    # run the sampler
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda x, info: transform(x.position),
        progress_bar=progress_bar,
    )

    return samples, blackjax_state_after_tuning, blackjax_mclmc_sampler_params


def run_mclmc(model, num_steps, transform= lambda x: x, rng_key= jax.random.key(0), desired_energy_var= 5e-4, progress_bar= True):
    key_init, key_sample = jax.random.split(rng_key)
    initial_position = model.sample_init(key_init)
    samples = _run_mclmc(model.logdensity_fn, num_steps, initial_position, transform= transform, key= key_sample, desired_energy_var= desired_energy_var, progress_bar= progress_bar)[0]
    return samples
