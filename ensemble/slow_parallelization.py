import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from time import time
import os
# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"

jax.config.update('jax_platform_name', 'cpu') # we will use cpu here (because we can reserve so many), this line is only to avoid jax warning
jax.config.update("jax_enable_x64", True) # since we are on cpu we do not mind using 64 bits

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128' # we have a cpu node with 128 cores
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

p = jax.sharding.PartitionSpec('i')
mesh = jax.sharding.Mesh(jax.devices(), 'i')


params = jax.random.split(jax.random.key(0), num_cores)#jnp.linspace(1., 3., num_cores) # we want to evaluate func(params)

def func(key):
    """just some function that takes some time"""
    
    size = 1000 # detemines how costly func is
    
    # generate a random symmetric matrix
    key1, key2 = jax.random.split(key)
    D = jnp.diag(jax.random.normal(key1, shape= (size, ))) # eigenvalues
    R, _ = jnp.array(jnp.linalg.qr(jax.random.normal(key2, shape= (size, size))))  # random rotation
    M = R @ D @ R.T
    
    # compute its eigenvalues
    eigvals, _ = jnp.linalg.eigh(M)
    
    return jnp.power(jnp.sum(eigvals), 3.1)


def timeit(f):
    """given f, measure how long f(params) takes"""
    t0 = time()
    results = f(params)
    jax.block_until_ready(results)
    t1 = time()
    print(t1 - t0)



timeit(jax.vmap(func)) # few seconds
timeit(jax.pmap(func)) # a minute
timeit(shard_map(jax.vmap(func), mesh= mesh, in_specs= p, out_specs= p)) # a minute

