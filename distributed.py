import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.experimental.multihost_utils import process_allgather

from benchmark import setup

# This script only works for a GPU backend.


# Initializes distributed JAX
jax.distributed.initialize()
num_devices = jax.process_count() # = 4 x number of nodes
verbose = (jax.process_index() == 0) # let only the first device print things


# Creates local data (we will use different devices to do different realizations)
num_realizations = 64
assert (num_realizations % num_devices) == 0 # num_realizations should be divisible by the number of devices)

global_key = jax.random.key(42)
local_key = jax.random.split(global_key, num_devices)[jax.process_index()] # each device gets its own key
local_size = num_realizations//num_devices
local_keys = jax.random.split(local_key, local_size) # the remaining random keys 

# Put it on the global devices
mesh = jax.sharding.Mesh(jax.devices(), 'devices')
p = jax.sharding.PartitionSpec('devices')
global_keys = jax.make_array_from_single_device_arrays((num_realizations,),  jax.sharding.NamedSharding(mesh, p), [local_keys])


# Use the external setup() to determine what function(x, y, z, ..., key) do we want to evaluate for different values of the parameters x, y, z, ... and random keys.
# grid = (x, y, z, ...), where each parameter is a vector of different values. A full grid x \times y \times z ... will be computed.
grid, func, save_name = setup()

num_params = len(grid)
Grid = jnp.meshgrid(*grid, indexing = 'ij') # get the grid matrices


# we distribute over grid and the local_keys on a single gpu, using vmap (we could also do this part without ifs, using recursion, but not sure if neccessary)
# this following lines are general wrt num_params. For example for num_params = 2 they are equivalent to
# return jax.vmap(lambda key: jax.vmap(jax.vmap(func, in_specs), in_specs)(*Grid, key))
in_specs = (0, ) * num_params + (None, )    
f = func
for i in range(num_params):
    f = jax.vmap(f, in_specs)

if num_params == 1:
    func_vmap = jax.vmap(lambda key: f(Grid, key))
else:
    func_vmap = jax.vmap(lambda key: f(*Grid, key))


# parallelize over different devices
parallel_execute = shard_map(func_vmap, 
                        mesh= mesh,
                        in_specs= p, 
                        out_specs= p
                        )

# execute calculation
results = parallel_execute(global_keys)

# save results
jnp.save(save_name, process_allgather(results))
