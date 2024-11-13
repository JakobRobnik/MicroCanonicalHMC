import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map

# import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

# Initializes distributed JAX
jax.distributed.initialize()

# Displays the devices accessible
verbose = (jax.process_index() == 0)
print(f"[{jax.process_index()}]: local devices: {len(jax.local_devices())}")
if verbose: print(f"Global devices: {len(jax.devices())}")


def func(L, rng_key):
    """function that we want to evaluate for different values of parameters"""    
    
    return L + 0.1 * jax.random.normal(rng_key)
    

# create the parameter grid
name_params = ('L', 'rng_key')
L = jnp.linspace(0., 1., 16)
key = jax.random.split(jax.random.key(0), 128)
_x, _y = jnp.meshgrid(key, L)

# distribute the grid across the devices
shard_grid_shape = (2, 4) # shape of the shard grid (can be anything, just make sure that total number of shards = number of devices)
#mesh = jax.sharding.Mesh(shard_grid_shape, name_params)
mesh = jax.make_mesh(shard_grid_shape, name_params)
p = PartitionSpec(*name_params)
x = jax.device_put(_x, NamedSharding(mesh, p))
y = jax.device_put(_y, NamedSharding(mesh, p))


# execute calculation on multiple devices
parallel_execute = shard_map(jax.vmap(func, ((0, 1), (0, 1))), 
                        mesh= mesh,
                        in_specs= (p, p), 
                        out_specs= p
                        )

results = parallel_execute(_x, _y)

#save the results in a single file
jnp.save('grid_results.npy', results)

