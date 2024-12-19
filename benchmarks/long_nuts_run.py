import itertools
import sys

sys.path.append("./")
sys.path.append("../blackjax")
import os

from benchmarks.lattice import Phi4
import jax
import jax.numpy as jnp
import blackjax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()

from metrics import benchmark
from benchmarks.sampling_algorithms import (

    adjusted_hmc,
    adjusted_mclmc,
    nuts,
    unadjusted_mclmc,
    unadjusted_mclmc_no_tuning,
    unadjusted_underdamped_langevin,
    unadjusted_underdamped_langevin_no_tuning,
)
from blackjax.diagnostics import potential_scale_reduction

from benchmarks.inference_models import (
    Brownian,
    Gaussian,
    GermanCredit,
    Rosenbrock,
)

# model = Gaussian(ndims=10,condition_number=1)
model = Phi4(L=2, lam=1)
# model = GermanCredit()
# model = Brownian()
# model = Rosenbrock()
n = 10000
num_chains = 3

def relative_fluctuations(E_x2):
      E_x2 = E_x2.T
      E_x2_median = jnp.median(E_x2, axis = 1)
      diff = jnp.abs((E_x2 - E_x2_median[:, None]) / E_x2_median[:, None])
      return jnp.max(diff)

def nuts_rhat(model):

    sampler=nuts(integrator_type="velocity_verlet", preconditioning=False, return_ess_corr=False, return_samples=False, incremental_value_transform=lambda x: x)


    key = jax.random.PRNGKey(1)
    key, init_key = jax.random.split(key, 2)
    keys = jax.random.split(key, num_chains)

    pvmap = jax.pmap

    init_keys = jax.random.split(init_key, num_chains)
    init_pos = pvmap(model.sample_init)(init_keys)  # [batch_size, dim_model]

    params, grad_calls_per_traj, acceptance_rate, expectation, ess_corr, num_tuning_steps = pvmap(
        lambda pos, key: sampler(
            model=model, num_steps=n, initial_position=pos, key=key, 
        )
    )(init_pos, keys)

    print(expectation.shape)

    e_x2 = expectation[:,:,0,:]
    e_x = expectation[:,:,1,:]


    print((potential_scale_reduction(e_x2)))
    print(relative_fluctuations(e_x2))

    e_x2_avg = (e_x2[:,-1,:].mean(axis=0))
    e_x_avg = (e_x[:,-1,:].mean(axis=0))

    print(f"x^2 is {e_x2_avg} and var_x2 = {e_x2_avg - e_x_avg**2}")

print(nuts_rhat(
    
    model=Gaussian(10)))
