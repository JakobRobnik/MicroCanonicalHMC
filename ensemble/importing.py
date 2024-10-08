import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

from benchmarks.inference_models import Banana

import sys, os
sys.path.append('../blackjax/')
#sys.path.append('../blackjax/blackjax/')
from blackjax.adaptation.ensemble_umclmc import init
from blackjax.mcmc import mclmc
from blackjax.mcmc.integrators import isokinetic_leapfrog

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

print(len(jax.devices()))#, jax.lib.xla_bridge.get_backend().platform)
 #jax.extend.backend.get_backend().platform)

mesh = jax.sharding.Mesh(jax.devices(), 'chains')
p, pscalar = PartitionSpec('chains'), PartitionSpec()


target = Banana()

sequential_kernel = mclmc.build_kernel(logdensity_fn= target.logdensity_fn, integrator= isokinetic_leapfrog)

mclmc_kernel = shard_map(jax.vmap(sequential_kernel, (0, 0, None, None)), 
                            mesh= mesh, 
                            in_specs= (p, p, pscalar, pscalar), 
                            out_specs= (p, p)
                            )

key = jax.random.key(0)
num_chains = 4096
num_samples = 1000

key_init, key_sample = jax.random.split(key)

def split_keys(key):
    return jax.device_put(jax.random.split(key, num_chains), NamedSharding(mesh, p))
        

def initialize_position(key):
    keys = split_keys(key)
    return shard_map(jax.vmap(target.sample_init), mesh=mesh, in_specs=p, out_specs=p)(keys)


initial_position = initialize_position(key_init)
initial_state = init(initial_position, target.logdensity_fn, mesh)

keys = split_keys(key_sample)

def step(state, useless):
    new_state, info = mclmc_kernel(keys, state, 10., 0.1)
    return new_state, None

state, info = jax.lax.scan(step, init= initial_state, xs = None, length= num_samples)

#shifter --image=reubenharry/cosmo:1.0 python3 -m ensemble.importing




