import cProfile
import sys

sys.path.insert(0, './')

import sampling

from sampling.sampler import Sampler  
from sampling.dynamics import update_momentum
import jax 

import jax.numpy as jnp

nlogp = lambda x: 0.5*jnp.sum(jnp.square(x))
value_grad = jax.value_and_grad(nlogp)

class StandardGaussian():

  def __init__(self, d):
    self.d = d

  def grad_nlogp(self, x):
    """should return nlogp and gradient of nlogp"""
    return value_grad(x)

  def transform(self, x):
    return x[:2]
    #return x

  def prior_draw(self, key):
    """Args: jax random key
       Returns: one random sample from the prior"""

    return jax.random.normal(key, shape = (self.d, ), dtype = 'float64') * 4

target = StandardGaussian(d = 1000)
sampler = Sampler(target, varEwanted = 5e-4, frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0)

def test():
    sampler.sample(500, 1)

if __name__ == '__main__':
    import timeit
    n = 40
    print(f'Benchmark took {timeit.timeit("test()", globals=locals(), number=n) / n} seconds')