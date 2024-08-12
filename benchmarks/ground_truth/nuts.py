import jax
import blackjax

### compute the "ground truth" chains, i.e. very long NUTS chains



def run_nuts(model, num_steps, key= jax.random.key(0)):
    
    integrator = blackjax.mcmc.integrators.velocity_verlet

    rng_key, warmup_key, init_key = jax.random.split(key, 3)
    initial_position = model.sample_init(init_key)
    
    warmup = blackjax.window_adaptation(blackjax.nuts, model.logdensity_fn, integrator=integrator)
    (state, params), _ = warmup.run(warmup_key, initial_position, 2000)

    nuts = blackjax.nuts(logdensity_fn= model.logdensity_fn, step_size=params['step_size'], inverse_mass_matrix= params['inverse_mass_matrix'], integrator=integrator)

    _, state_history, _ = blackjax.util.run_inference_algorithm(
        rng_key=rng_key,
        initial_state=state,
        inference_algorithm=nuts,
        num_steps=num_steps,
        transform=lambda state, info: model.transform(state.position),
        progress_bar=True
    )

    return state_history
