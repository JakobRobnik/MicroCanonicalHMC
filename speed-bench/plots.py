import cProfile
import sys

import numpy as np


sys.path.insert(0, './')

import mclmc.sampling

from mclmc.sampling.sampler import Sampler, Target  
from mclmc.sampling.dynamics import update_momentum
import jax 

import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt

nlogp = lambda x: 0.5*jnp.sum(jnp.square(x))

def test(num_steps, d):
    sampler = Sampler(Target(d = d, nlogp=nlogp), frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0)
    sampler.sample(num_steps, x_initial = jax.random.normal(shape=(d,), key=jax.random.PRNGKey(0))).block_until_ready()
  
import timeit

if __name__ == '__main__':
    
    n = 10
    
    vals = []
    ran =  np.linspace(100,500000, 10)
    for num_steps in ran:
      d = 50
      time = (timeit.timeit("test(num_steps, d)", globals=locals(), number=n) / n)
      print(f'Benchmark took {time/num_steps} seconds per step')
      vals.append(time)
    
    vals_2 = []
    ran_2 =  np.linspace(100,100000, 50)
    for d in ran_2:
      num_steps = 1000
      time = (timeit.timeit("test(num_steps, int(d))", globals=locals(), number=n) / n)
      print(f'Benchmark took {time/num_steps} seconds per step')
      vals_2.append(time)

    sns.lineplot(x=ran, y=vals)
    plt.xlabel('Number of steps (100 dimensions)')
    plt.ylabel('Time in seconds')
    plt.savefig("img/speed_bench_steps.png")

    plt.figure()

    sns.lineplot(x=ran_2, y=vals_2)
    plt.xlabel('Number of dimensions (1000 steps)')
    plt.ylabel('Time in seconds')
    plt.savefig("img/speed_bench_dim.png")