import os

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from mclmc.sampler import Sampler
from mclmc.sampler import find_crossing
from mclmc.benchmark_targets import *
from mclmc.grid_search import search_wrapper


num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)


target = Funnel(d=20)

#eps = jnp.logspace(jnp.log10(0.25), jnp.log10(2.5), 10)
eps = jnp.logspace(jnp.log10(2), jnp.log10(7), 10)



def f(e):
    sampler = Sampler(target, 4.0 * np.sqrt(target.d), e)
    return sampler.sample(240000, 120, tune= 'none', output= 'ess')

def g(e):
    sampler = Sampler(target, 4.0 * np.sqrt(target.d), e)
    return sampler.sample(240000, 120, tune= 'none', output= 'ess funnel')

def h(vare):
    sampler = Sampler(target, 4.0 * np.sqrt(target.d), 1.0)
    sampler.varEwanted = vare
    return sampler.sample(240000, 120, tune= 'none', output= 'ess', adaptive = True)


#ess = jax.vmap(g)(eps)
# vare = jnp.logspace(-4.5, -1.5, 10)
# ess = jax.vmap(h)(vare)
#
# plt.plot(vare, ess, 'o-')
# plt.xscale('log')
# plt.savefig('adaptive_stepsize_alpha=4.png')
# plt.show()

sampler = Sampler(target, 4.0 * np.sqrt(target.d), 1.0)
sampler.varEwanted = 1e-3

# plt.subplot(1, 2, 1)
# X, E, L, eps = sampler.sample(100000, tune= 'cheap', output= 'detailed', adaptive = False)
# theta = X[:-1, -1]
# vare = jnp.log10(jnp.square(E[1:]-E[:-1]) / target.d)
# plt.hexbin(theta, vare)
# plt.ylim(0, -11)
#
# plt.subplot(1, 2, 2)
# X, W, E, L = sampler.sample(100000, tune= 'cheap', output= 'detailed', adaptive = True)
# theta = X[:-1, -1]
# vare = jnp.log10(jnp.square(E[1:]-E[:-1]) / target.d)
# plt.hexbin(theta, vare)
# plt.ylim(0, -11)
# plt.show()

X, W, E, L = sampler.sample(100000, tune= 'cheap', output= 'detailed', adaptive = True)
theta = X[:, -1]
#vare = jnp.log10(jnp.square(E[1:]-E[:-1]) / target.d)
plt.hexbin(np.exp(0.5 * theta), W)
plt.savefig('stepsize_adaptive')
plt.show()
#print(funnel_ess(1.0, 0.1))
#search_wrapper(funnel_ess, 0.5, 3.0, 0.05, 1.5, save_name='show')
