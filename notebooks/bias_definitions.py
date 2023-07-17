import os

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from sampling.sampler import Sampler
from sampling.sampler import find_crossing
from sampling.benchmark_targets import *


def definitions():
    target = IllConditionedGaussian(d = 100, condition_number= 10**4)

    sampler = Sampler(target, integrator= 'LF')
    #sampler.varEwanted = 1e-3
    sampler.diagonal_preconditioning = False

    num_samples = 100000
    X = sampler.sample(num_samples, 12, output = 'normal')


    n = jnp.arange(1, num_samples+1)
    second_moments = jnp.cumsum(jnp.square(X), axis=1) / n[None, :, None]

    # our definition
    # bias = jnp.average(jnp.square(1 - (second_moments/target.second_moment[None, None, :])), axis = -1) * 0.5
    # for i in range(10):
    #     plt.plot(n, bias[i, :], color = 'tab:blue', alpha = 0.5)
    # plt.plot(n, jnp.average(bias, 0), '-', lw = 3, color = 'tab:blue', label = 'our / 2')

    # intermediate definiton
    bias = jnp.average(jnp.square(second_moments - target.second_moment[None, None, :]) / target.variance_second_moment[None, None, :], axis = -1)
    for i in range(10):
        plt.plot(n, bias[i, :], '-', color = 'tab:orange', alpha = 0.5)
    plt.plot(n, jnp.average(bias, 0), '-', lw = 3, color = 'tab:orange', label = 'intermediate')


    # Hoffman definition
    bias = jnp.max(jnp.square(second_moments - target.second_moment[None, None, :]) / target.variance_second_moment[None, None, :], axis = -1)
    for i in range(10):
        plt.plot(n, bias[i, :] / jnp.log(target.d), '-', color = 'tab:red', alpha = 0.5)
    plt.plot(n, jnp.average(bias, 0) /  7.717980949522242, '-', lw = 3, color = 'tab:red', label = 'Hoffman / log d')

    plt.plot([1, 1e5], [0.01, 0.01], '--', color = 'black', alpha = 0.5)

    plt.legend()
    plt.xlabel('n')
    plt.ylabel(r'$b^2$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('definitions adjusted.png')
    plt.show()



def lee():
    repeat = 10000

    dimensions = [1, 10, 100, 1000, 10000]
    xmax = []
    for d in dimensions:

        xmax.append(np.average(np.max(np.square(np.random.normal(size=(repeat, d))), axis=1)))

    print(xmax)
    plt.plot(dimensions, xmax, 'o', label=r'$\mathrm{max}_i \{z_i^2\}_{i=1}^d$')
    plt.plot(dimensions, 2 * np.log(dimensions), color='black', label='2 log d')

    plt.xscale('log')
    plt.xlabel('d')
    plt.legend()
    plt.savefig('LEE.png')
    plt.show()