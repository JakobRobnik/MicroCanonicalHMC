import sys  
sys.path.insert(0, '../../')
sys.path.insert(0, './')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from sampling.sampler import Sampler, Target

nlogp = lambda x: 0.5*jnp.sum(jnp.square(x))

class StandardGaussian(Target):

  def __init__(self, d, nlogp):
    Target.__init__(self,d,nlogp)

  def transform(self, x):
    return x[:2]
  
  def prior_draw(self, key):
    """Args: jax random key
       Returns: one random sample from the prior"""

    return jax.random.normal(key, shape = (self.d, ), dtype = 'float64') * 4

target = StandardGaussian(d = 10, nlogp=nlogp)
sampler = Sampler(target, varEwanted = 5e-4)

target_simple = Target(d = 10, nlogp=nlogp)

def test_mclmc():
    samples1 = sampler.sample(100, 1, random_key=jax.random.PRNGKey(0))
    samples2 = sampler.sample(100, 1, random_key=jax.random.PRNGKey(0))
    samples3 = sampler.sample(100, 1, random_key=jax.random.PRNGKey(1))
    assert jnp.array_equal(samples1,samples2), "sampler should be pure"
    assert not jnp.array_equal(samples1,samples3), "this suggests that seed is not being used"
    # run with multiple chains
    sampler.sample(100, 3)

    Sampler(target_simple).sample(100, x_initial = jax.random.normal(shape=(10,), key=jax.random.PRNGKey(0)))
