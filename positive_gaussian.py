import os

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from sampling.sampler import Sampler


class PositiveConstraint:

    def __init__(self, target, positive):
        
        self.positive = positive

        # get the attributes from the previous target
        self.d = target.d
        self.transform = self.reflection

        self.nlogp = lambda x: target.nlogp(self.reflection(x)) #we extend the domain by reflection p(-x_i) = p(x_i) for parameters which need to be positive
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        
        if hasattr(target, 'transform'):
            self.transform = lambda x: target.transform(self.reflection(x)) # at the end we reflect the samples back to the original domain
        else:
            self.transform = self.reflection

        self.prior_draw = target.prior_draw


    def reflection(self, x):
        return x * jnp.sign(jnp.sign(x) + 2 - 2 * self.positive)
    


class StandardNormal():
    """Standard Normal distribution in d dimensions"""

    def __init__(self, d):
        self.d = d
        self.variance = jnp.ones(d)
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * jnp.sum(jnp.square(x), axis= -1)


    def transform(self, x):
        return x

    def prior_draw(self, key):
        return jnp.abs(jax.random.normal(key, shape = (self.d, ), dtype = 'float64'))


d = 100
num_positive = 50
positive = jnp.concatenate((jnp.ones(num_positive), jnp.zeros(d-num_positive)))
target= PositiveConstraint(StandardNormal(d = 100), positive)

sampler = Sampler(target, 10.0, 5.0, integrator= 'LF')
x, burnin = sampler.sample(10000)
x= x[burnin:]
#x = jax.vmap(target.prior_draw)(jax.random.split(jax.random.PRNGKey(0), 1000))
