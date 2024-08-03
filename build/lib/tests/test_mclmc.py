
from pytest import raises

from mclmc.dynamics import leapfrog
from mclmc.sampler import OutputType, Sampler, Target

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


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





def test_mclmc():
    target = StandardGaussian(d = 10, nlogp=nlogp)
    sampler = Sampler(target, varEwanted = 5e-4)

    samples1 = sampler.sample(100, 1, random_key=jax.random.PRNGKey(0))
    samples2 = sampler.sample(100, 1, random_key=jax.random.PRNGKey(0))
    samples3 = sampler.sample(100, 1, random_key=jax.random.PRNGKey(1))
    assert jnp.array_equal(samples1,samples2), "sampler should be pure"
    assert not jnp.array_equal(samples1,samples3), "this suggests that seed is not being used"
    # run without key
    sampler.sample(20)
    # run with multiple chains
    sampler.sample(20, 3)
    # run with different output types
    sampler.sample(20, 1, output=OutputType.expectation)
    sampler.sample(20, 1, output=OutputType.detailed)
    sampler.sample(20, 1, output=OutputType.normal)

    with raises(AttributeError) as excinfo:
        sampler.sample(20, 1, output=OutputType.ess)

    # run with leapfrog
    sampler = Sampler(target, varEwanted = 5e-4, integrator=leapfrog)
    sampler.sample(20)
    # run without autotune
    sampler = Sampler(target, varEwanted = 5e-4, frac_tune1 = 0.1, frac_tune2 = 0.1, frac_tune3 = 0.1,)
    sampler.sample(20,)

    # with a specific initial point
    sampler.sample(20, x_initial=jax.random.normal(shape=(10,), key=jax.random.PRNGKey(0)))
    
    # running with wrong dimensions causes TypeError
    with raises(TypeError) as excinfo:  
        sampler.sample(20, x_initial=jax.random.normal(shape=(11,), key=jax.random.PRNGKey(0)))
    
    # multiple chains
    sampler.sample(20, 3)
    
    # simple target
    target_simple = Target(d = 10, nlogp=nlogp)
    Sampler(target_simple).sample(100, x_initial = jax.random.normal(shape=(10,), key=jax.random.PRNGKey(0)))

# speed
def gaussian():
    d = 1000
    target_simple = Target(d = d, nlogp=nlogp)
    samples = Sampler(target_simple).sample(100000, x_initial = jax.random.normal(shape=(d,), key=jax.random.PRNGKey(0)))
    return samples

def test_speed(benchmark):
    result = benchmark(gaussian)
    assert jnp.abs(jnp.mean(result))<1e-3