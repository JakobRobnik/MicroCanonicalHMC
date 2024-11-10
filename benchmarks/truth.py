import jax
import jax.numpy as jnp
import blackjax
import numpy as np
import os
from benchmarks.inference_models import *


### compute the "ground truth" chains, i.e. very long NUTS chains

dir_ground_truth = os.path.dirname(os.path.realpath(__file__)) + '/ground_truth/'


def nuts(model, num_steps, key= jax.random.key(0)):
    
    integrator = blackjax.mcmc.integrators.velocity_verlet

    rng_key, warmup_key, init_key = jax.random.split(key, 3)
    initial_position = model.sample_init(init_key)
    
    warmup = blackjax.window_adaptation(blackjax.nuts, model.logdensity_fn, integrator=integrator, target_acceptance_rate= 0.95)
    (state, params), _ = warmup.run(warmup_key, initial_position, 2000)

    nuts = blackjax.nuts(logdensity_fn= model.logdensity_fn, step_size=params['step_size'], inverse_mass_matrix= params['inverse_mass_matrix'], integrator=integrator)

    _, state_history = blackjax.util.run_inference_algorithm(
        rng_key=rng_key,
        initial_state=state,
        inference_algorithm=nuts,
        num_steps=num_steps,
        transform=lambda state, info: model.transform(state.position),
        progress_bar=True
    )

    return state_history


def cov_matrix(model, samples):
    
    sample_mean = jnp.average(samples, axis= 0)
    x = samples - sample_mean[None, :]
    cov = (x.T @ x) / samples.shape[0]
    np.savez(dir_ground_truth + model.name + '/cov.npz', x_avg = sample_mean, cov= cov)
    


if __name__ == '__main__':
    
    model = Funnel_with_Data(d= 101, sigma= 1.)
    samples = nuts(model, num_steps= 10**7)
    cov_matrix(model, samples)
    
    