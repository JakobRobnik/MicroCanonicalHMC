import itertools
import pickle
import sys

sys.path.append("./")
sys.path.append("../blackjax")
import os

from benchmarks.lattice import U1, Phi4
import jax
import jax.numpy as jnp
import blackjax
import time 

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()

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

# import os 
# cwd = os.getcwd()
# print(os.listdir("benchmarks/"), "fpp")
# raise Exception
# with open('./benchmarks/ground_truth/Phi4/e_x2.pkl', 'wb') as f:
#         pickle.dump(jnp.zeros(10,), f)

# raise Exception
# model = Gaussian(ndims=10,condition_number=1)
# model = Phi4(L=2, lam=1)
# model = GermanCredit()
# model = Brownian()
# model = Rosenbrock()
n = 100000
num_chains = 4


L = 200
lam = 1.0
model = Phi4(L=L,lam=lam)
# print(U1(Lt=20,Lx=20,).ndims)

def relative_fluctuations(E_x2):
      E_x2 = E_x2.T
      E_x2_median = jnp.median(E_x2, axis = 1)
      diff = jnp.abs((E_x2 - E_x2_median[:, None]) / E_x2_median[:, None])
      return jnp.max(diff)

def nuts_rhat(model):

    sampler=nuts(integrator_type="velocity_verlet", preconditioning=True, return_ess_corr=False, return_samples=False, incremental_value_transform=lambda x: x, return_history=False)


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

    # jax.debug.print("step size {x}", x=params["step_size"])
    # jax.debug.print("grad calls {x}", x=grad_calls_per_traj)

    # jax.debug.print("expectation shape {x}", x=expectation.shape)
    print(expectation.mean())
    # raise Exception
    # raise Exception
    e_x2 = expectation[:,0,:]
    e_x = expectation[:,1,:]
    e_x4 = expectation[:,2,:]


    print("potential scale reduction", (potential_scale_reduction(e_x2)))
    print("relative fluctuations", relative_fluctuations(e_x2))

    e_x2_avg = (e_x2.mean(axis=0))
    e_x_avg = (e_x.mean(axis=0))

    print(f"x^2 is {e_x2_avg} and var_x = {e_x2_avg - e_x_avg**2}")

    with open(f'./benchmarks/ground_truth/Phi4/e_x2_{model.L}_{model.lam}.pkl', 'wb') as f:
        pickle.dump(e_x2, f)
    with open(f'./benchmarks/ground_truth/Phi4/e_x4_{model.L}_{model.lam}.pkl', 'wb') as f:
        pickle.dump(e_x4, f)

toc = time.time()
(nuts_rhat(
    
    # model=U1(Lt=200,Lx=200,)
    model=model,
    # model=Gaussian(ndims=11,condition_number=1)
    # model=Brownian()
))
tic = time.time()
print(f"time: {tic-toc}")
