import jax 
import jax.numpy as jnp
import os
jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(128)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)
import blackjax
from benchmarks.mcmc.inference_models import *
from blackjax.mcmc.integrators import isokinetic_velocity_verlet, velocity_verlet
from blackjax.util import run_inference_algorithm, store_only_expectation_values
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

from benchmarks.mcmc.inference_models import *
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def frobenious(model):
    
    def func(cov):
        residual = jnp.eye(model.ndims) - model.hessian @ cov
        return jnp.average(jnp.diag(residual @ residual))
    
    return func
    


def sample(model, num_steps, step_size, key):
    init_key, state_key, run_key = jax.random.split(key, 3)
    
    # initialize
    initial_state = blackjax.mcmc.mclmc.init(
        position=model.sample_init(init_key), logdensity_fn=model.logdensity_fn, rng_key=state_key
    )

    # kernel
    sampling_alg = blackjax.mclmc(
        model.logdensity_fn,
        L= jnp.sqrt(jnp.sum(jnp.diag(model.cov))),
        step_size=step_size,
        integrator = isokinetic_velocity_verlet,
    ) 

    # transform the kernel to save memory
    state_transform = lambda state: jnp.outer(state.position, state.position) # expectation values to compute: E[x_i x_j] and E[energy change], E[energy change^2] 
    
    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm= sampling_alg,
        state_transform= state_transform,
        exp_vals_transform= frobenious(model) # convert the covariance matrix expectation values to the bias scalar (to save memory)
    )
    
    initial_state = memory_efficient_sampling_alg.init(initial_state)
    
    # run the algorithm
    b, info = run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=memory_efficient_sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )[1]
     
    eevpd = jnp.std(info.energy_change)**2 / model.ndims
    return b, eevpd


key = jax.random.PRNGKey(42)
num_eps, num_chains = 32, 4

Model = namedtuple('Model', ['model', 'num_steps', 'stepsize_bounds'])
models = [Model(StandardNormal(d=100), 10**7, [5., 12.]),
          Model(Rosenbrock(), 10**7, [0.1, 0.6]),
          Model(Brownian(), 10**7, []),
          Model(GermanCredit(), 10**7, [0.07, 0.35])]

if __name__ == '__main__':
    for i in [1, ]:
        m = models[i]
        keys = jax.random.split(key, num_chains)
        step_size = jnp.log10(*np.log10(m.stepsize_bounds), num_eps)
        b, eevpd = jax.pmap(lambda stepsize: jax.pmap(lambda k: sample(m.model, m.num_steps, stepsize, k))(keys))(step_size)

        np.savez('data/bias/' + m.model.name + '.npz', stepsize= step_size, bias= b, eevpd= eevpd)
        
    