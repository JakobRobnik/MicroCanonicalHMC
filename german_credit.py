#Sparse logistic regression fitted to the German credit data
#We use the version implemented in the inference-gym: https://pypi.org/project/inference-gym/
#In some part we directly use their tutorial: https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

import ESH
import parallel

import inference_gym.using_jax as gym
import functools

import jax
from jax import lax
import jax.numpy as jnp
import arviz as az


import ESH
import parallel





class Target():
    """the target class used for the MCHMC"""

    def __init__(self):
        self.d = 51
        identity_fn = target.sample_transformations['identity']
        self.variance = identity_fn.ground_truth_standard_deviation
        self.gaussianization_available = True

    def nlogp(self, x):
        return map_objective_fn(x)

    def grad_nlogp(self, x):
        return map_objective_grad_fn(x)

    def gaussianize(self, x):
        """This is not a gaussianization in this case, but a transformation to the coordinates in which the ground truth variance is known"""
        return target.default_event_space_bijector(x)



def check(samples):
    num_steps = len(samples)
    x_chain = target.default_event_space_bijector(samples)

    identity_fn = target.sample_transformations['identity']
    x_chain_transformed = identity_fn(x_chain)
    x_mean = x_chain_transformed[num_steps // 2:].mean((0, 1))
    x_stddev = x_chain_transformed[num_steps // 2:].std((0, 1))
    plt.figure()
    plt.title('Mean')
    plt.bar(jnp.arange(51), x_mean, 0.5, label='MCHMC estimate')
    plt.bar(0.5 + jnp.arange(51), identity_fn.ground_truth_mean, 0.5, label='Ground Truth')
    plt.ylabel('Value')
    plt.xlabel('Coordinate')
    plt.legend()

    plt.figure()
    plt.title('Standard Deviation')
    plt.bar(jnp.arange(51), x_stddev, 0.5, label='MCHMC estimate')
    plt.bar(0.5 + jnp.arange(51), identity_fn.ground_truth_standard_deviation, 0.5,
            label='Ground Truth')
    plt.ylabel('Value')
    plt.xlabel('Coordinate')
    plt.legend()
    plt.show()


def setup():
    jax.config.update('jax_enable_x64', True)

    # setup
    target = gym.targets.GermanCreditNumericSparseLogisticRegression()

    nested_zeros = lambda shape, dtype: jax.tree_multimap(lambda d, s: jnp.zeros(s, d), dtype, shape)

    z = jax.tree_multimap(lambda d, b, s: nested_zeros(b.inverse_event_shape(s), d), target.dtype,
                          target.default_event_space_bijector, target.event_shape)
    x = jax.tree_multimap(lambda z, b: b(z), z, target.default_event_space_bijector)

    target = gym.targets.VectorModel(target, flatten_sample_transformations=True)

    def map_objective_fn(z):
      x = target.default_event_space_bijector(z)
      return -target.unnormalized_log_prob(x)

    map_objective_grad_fn = jax.grad(map_objective_fn)

    return map_objective_fn, map_objective_grad_fn

if __name__ == '__main__':

    eps = 0.1
    d = 51
    L = 1.5 * np.sqrt(d)
    esh = ESH.Sampler(Target= Target(), eps=eps)
    np.random.seed(0)
    x0 = np.zeros(51)
    bias = esh.sample(x0, L, max_steps= 10000, prerun_steps = 1000, track= 'FullBias')

    plt.plot(bias, '.')

    plt.xlabel('log')
    plt.ylabel('log')
    plt.ylabel('bias')
    plt.xlabel('gradient evaluations')
    plt.show()