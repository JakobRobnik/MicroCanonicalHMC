import sys

from pytest import raises
import pytest  
sys.path.insert(0, '../../')
sys.path.insert(0, './')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from sampling.sampler import Sampler

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



def test_mclmc():
    target = StandardGaussian(d = 10)
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
    # sampler.sample(20, 3, output='ess')
    # sampler.sample(20, 3, output='expectation')
    sampler.sample(20, 3, output='detailed')
    sampler.sample(20, 3, output='normal')

    # run with leapfrog
    sampler = Sampler(target, varEwanted = 5e-4, integrator='LF')
    sampler.sample(20)
    # run without autotune
    sampler = Sampler(target, varEwanted = 5e-4, frac_tune1 = 0.1, frac_tune2 = 0.1, frac_tune3 = 0.1,)
    sampler.sample(20,)

    # with a specific initial point
    sampler.sample(20, x_initial=jax.random.normal(shape=(10,), key=jax.random.PRNGKey(0)))
    
    # running with wrong dimensions causes TypeError
    with raises(TypeError) as excinfo:  
        sampler.sample(20, x_initial=jax.random.normal(shape=(11,), key=jax.random.PRNGKey(0)))
