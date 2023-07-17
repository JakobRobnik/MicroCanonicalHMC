import jax
import jax.numpy as jnp
from benchmarks.benchmarks_sg import *
from sampling.sampler import Sampler

key = jax.random.PRNGKey(0)
key, key_data = jax.random.split(key)

num_data, num_batch = 1000, 10
target = LinearRegression(num_batch, key_data, num_data)
#
# sampler= Sampler(target, diagonal_preconditioning= False, frac_tune3= 0.1, frac_tune2= 0.1, frac_tune1=0.0, integrator= 'LF')
# z, E, L, eps = sampler.sample(10000, output = 'detailed')
# print(L, eps)

Lfull, epsfull = jnp.inf, 0.015936224742675004*0.1
sampler= Sampler(target, diagonal_preconditioning= False, frac_tune3= 0.0, frac_tune2= 0.0, frac_tune1=0.0, L= Lfull, eps= epsfull, sg= True)

z = sampler.sample(10000)

target.plot_posterior(z, 'position_0.1')
