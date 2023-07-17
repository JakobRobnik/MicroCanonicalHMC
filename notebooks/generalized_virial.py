import os

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from sampling.sampler import Sampler
from benchmarks.benchmarks_mchmc import *


M = 300

from benchmarks.german_credit import Target as GermanCredit

d = 100
target = IllConditionedGaussian(d = d, condition_number= 100.0, numpy_seed= 1)
x = jax.vmap(target.prior_draw)(jax.random.split(jax.random.PRNGKey(0), M))
g = jax.vmap(lambda x: target.grad_nlogp(x)[1])(x)


V = jnp.array([[jnp.average(x[:, i] * g[:, j]) for j in range(target.d)] for i in range(target.d)])
shift = jnp.eye(target.d) - V
Bsq = jnp.average(jnp.diagonal(shift.T @ shift))


Cov = jnp.array([[jnp.average(x[:, i] * x[:, j]) for j in range(target.d)] for i in range(target.d)])
shift = jnp.eye(target.d) - Cov@target.Hessian
bsq = jnp.average(jnp.diagonal(shift.T @ shift))

print(Bsq/bsq)

def hutchinson(x, g, repeat, key):
    M, d = x.shape
    z = jax.random.rademacher(key, (repeat, d))
    X = z - (g @ z.T).T @ x / M
    return jnp.average(jnp.square(X))


def hutchinson(x, g, repeat, key):
    M, d = x.shape
    z = jax.random.rademacher(key, (repeat, d))
    X = z - (g @ z.T).T @ x / M
    return jnp.cumsum(jnp.average(jnp.square(X), axis = 1)) / jnp.arange(1, 1+repeat)
    #return jnp.average(jnp.square(X))


def hutchinson_gauss(x, g, repeat, key):
    M, d = x.shape
    z = jax.random.normal(key, (repeat, d))
    X = z - (g @ z.T).T @ x / M
    return jnp.cumsum(jnp.average(jnp.square(X), axis = 1)) / jnp.arange(1, 1+repeat)
    #return jnp.average(jnp.square(X))

repeat= 30000
b= hutchinson(x, g, repeat, jax.random.PRNGKey(0))
bg = hutchinson_gauss(x, g, repeat, jax.random.PRNGKey(0))

n = jnp.arange(1, 1+len(b))
plt.plot(n, jnp.abs(b-Bsq)/Bsq, '.-', label = 'Rademacher')
plt.plot(n, jnp.abs(bg-Bsq)/Bsq, '.-', label = 'Gauss')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('# z realizations')
plt.ylabel('relative error')
plt.show()


