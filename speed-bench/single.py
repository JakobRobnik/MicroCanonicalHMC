import cProfile
import sys

import numpy as np


sys.path.insert(0, './')

import sampling

from sampling.sampler import Sampler, Target  
from sampling.dynamics import update_momentum
import jax 

import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt


n = 10
d = 1000
num_steps = 100000
sampler = Sampler(Target(d = d, nlogp=lambda x: 0.5*jnp.sum(jnp.square(x))), frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0)

def test(num_steps, d):
    sampler.sample(num_steps, x_initial = jax.random.normal(shape=(d,), key=jax.random.PRNGKey(0))).block_until_ready()
  
import timeit

if __name__ == '__main__':
    
    time = (timeit.timeit("test(num_steps, d)", globals=locals(), number=n) / n)
    print(f'Benchmark took {time/num_steps} seconds per step')
    
  