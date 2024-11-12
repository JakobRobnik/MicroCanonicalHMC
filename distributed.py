import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map

# This script only works for a GPU backend.


# Initializes distributed JAX
jax.distributed.initialize()
num_devices = jax.process_count() # = 4 x number of nodes
verbose = (jax.process_index() == 0) # let only the first device print things


# Creates local data (we will use different devices to do different realizations)
num_realizations = 64

global_key = jax.random.key(42)
local_key = jax.random.split(global_key, num_devices)[jax.process_index()] # each device gets its own key
local_size = num_realizations//num_devices
local_keys = jax.random.split(local_key, local_size) # the remaining random keys (here we assume that num_realizations is divisible by the number of devices)

# Put it on the global devices
mesh = jax.sharding.Mesh(jax.devices(), 'devices')
p = jax.sharding.PartitionSpec('devices')
global_keys = jax.make_array_from_single_device_arrays((num_realizations,),  jax.sharding.NamedSharding(mesh, p), [local_keys])


# parameter grid
x = jnp.linspace(0, 1, 10)
y = jnp.linspace(1, 2, 5)
Y, X = jnp.meshgrid(y, x)

def func(x, y, key):
    """function that we want to evaluate for different values of parameters x, y and different random keys"""
    return jnp.sum(jnp.power(y * jnp.abs(jax.random.normal(key, shape = (10,))), x))

in_specs = (0, 0, None)
func_vmap = jax.vmap(lambda key: jax.vmap(jax.vmap(func, in_specs), in_specs)(X, Y, key)) # we distribute over grid and the local_keys on a single gpu, using vmap


# Execute calculation
parallel_execute = shard_map(func_vmap, 
                        mesh= mesh,
                        in_specs= p, 
                        out_specs= p
                        )

results = parallel_execute(global_keys)

if verbose:
    print(results.shape)
