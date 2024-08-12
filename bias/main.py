import jax 
import jax.numpy as jnp
import os
jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(128)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from collections import namedtuple
import numpy as np
import sys

import blackjax
from blackjax.mcmc.integrators import isokinetic_velocity_verlet
from blackjax.util import run_inference_algorithm, store_only_expectation_values, thinning

from benchmarks.inference_models import *



class Shift():
    """Shit the target to zero mean."""
    def __init__(self, target):
        self.ndims = target.ndims
        self.name = target.name
        self.cov = target.cov
        self.inv_cov = target.inv_cov
        self.logdensity_fn = lambda x: target.logdensity_fn(x + target.E_x)
        self.sample_init = lambda key: target.sample_init(key) - target.E_x



def mclmc(key, step_size, model, L):
    init_key, state_key = jax.random.split(key)
    
    # initialize
    initial_state = blackjax.mcmc.mclmc.init(
        position=model.sample_init(init_key), logdensity_fn=model.logdensity_fn, rng_key=state_key
    )

    # kernel
    sampling_alg = blackjax.mclmc(
        model.logdensity_fn,
        L= jnp.sqrt(jnp.sum(jnp.diag(model.cov))) * L,
        step_size= step_size,
        integrator= isokinetic_velocity_verlet,
    )
    
    return sampling_alg, initial_state
    

def hmc(key, mclmc_step_size, model, L):
    step_size = mclmc_step_size / jnp.sqrt(model.ndims)
    
    hmc = blackjax.uhmc(logdensity_fn= model.logdensity_fn, step_size= step_size, inverse_mass_matrix = jnp.ones(model.ndims), num_integration_steps= L)
    initial_state = hmc.init(model.sample_init(key))

    return hmc, initial_state


def _sample(model, sampling_alg, num_steps, burn_in_steps, num_thinning, step_size, key, L, mclmc):
    
    init_key, run_key = jax.random.split(key)
    
    alg, initial_state = sampling_alg(init_key, step_size, model, L)
    
    def frobenious(cov):
        residual = jnp.eye(model.ndims) - model.inv_cov @ cov
        return jnp.average(jnp.diag(residual @ residual))
    
    # transform the kernel to save memory
    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm= alg,
        state_transform= lambda state: jnp.outer(state.position, state.position), # expectation values to compute: E[x_i x_j] and E[energy change], E[energy change^2] 
        exp_vals_transform= frobenious, # directly convert the covariance matrix expectation values to the bias scalar (to save memory)
        burn_in = burn_in_steps if mclmc else (burn_in_steps // L)
    )
    
    thinned_memory_efficient_sampling_alg = thinning(memory_efficient_sampling_alg, num_thinning)
    
    
    initial_state = thinned_memory_efficient_sampling_alg.init(initial_state)
    
    nsteps = num_steps if mclmc else (num_steps // L)

    # run the algorithm
    b, info = run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=thinned_memory_efficient_sampling_alg,
        num_steps=nsteps,
        transform=transform,
        progress_bar=False,
    )[1]
    print(info.energy_change.shape)
    eevpd = jnp.std(info.energy_change)**2 / model.ndims
    return b, eevpd


def sample(m, sampling_alg, L):
    key = jax.random.PRNGKey(42)
    num_eps, num_chains = 32, 4
    num_thinning = 100
    
    keys = jax.random.split(key, num_chains)
    step_size = jnp.logspace(*np.log10(m.stepsize_bounds), num_eps)
    _sample(m.model, sampling_alg.alg, 10**4, m.burn_in, num_thinning, step_size[0], key, L, sampling_alg.name == 'mclmc')
    # b, eevpd = jax.pmap(lambda stepsize: jax.pmap(lambda k: _sample(m.model, sampling_alg.alg, m.num_steps, m.burn_in, num_thinning, stepsize, k, L, sampling_alg.name == 'mclmc'))(keys))(step_size)
    # np.savez('data/bias/'+ sampling_alg.name + str(L) + '_'+ m.model.name + '.npz', stepsize= step_size, bias= b[:, :, ::thinning], eevpd= eevpd)


SamplingAlg = namedtuple('Algorithm', ['alg', 'name'])
sampling_algs = [SamplingAlg(mclmc, 'mclmc'), SamplingAlg(hmc, 'hmc')]

Model = namedtuple('Model', ['model', 'num_steps', 'stepsize_bounds', 'burn_in'])
num_steps, num_burnin = 10**7, 10**4
models = [Model(StandardNormal(d=100), num_steps, [2.5, 20.], 0), # [2.5, 14.]
          Model(IllConditionedGaussian(d= 100, condition_number= 1000.), num_steps, [0.4, 7.], num_burnin),
          Model(Shift(Rosenbrock()), num_steps, [0.1, 1.5], num_burnin), #[0.1, 0.6]
          Model(Shift(Brownian()), num_steps, [0.06, 1.], num_burnin),
          Model(Shift(GermanCredit()), num_steps, [0.03, 0.4], num_burnin)]


if __name__ == '__main__':
    ialg, imodel, L = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sample(models[imodel], sampling_algs[ialg], L)
        
    