import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'


def init(key):
    phi = jax.random.uniform(key) * 2 * jnp.pi
    return jnp.array([jnp.cos(phi), jnp.sin(phi)])


def sequential_kernel(x, key):
    """random walk on a circle"""
    phi = jnp.atan2(x[1], x[0])
    eps = jax.random.normal(key) * 0.1
    return jnp.array([jnp.cos(phi + eps), jnp.cos(phi + eps)])


mesh = jax.sharding.Mesh(jax.devices(), 'chains')
p = PartitionSpec('chains')

kernel = shard_map(jax.vmap(sequential_kernel), 
                   mesh=mesh, 
                   in_specs=(p, p), 
                   out_specs=p)

def step(state, key):
    keys = jax.random.split(key, num_chains)
    state = kernel(state, keys)
    return state, None


key = jax.random.key(0)
num_chains = 128 * 4
num_samples = 10

init_state= jnp.tile(jnp.array([0., 1.]), (num_chains, 1))

final_state, info = jax.lax.scan(step, init= init_state, xs = jax.random.split(key, num_samples))
print(final_state.shape)

