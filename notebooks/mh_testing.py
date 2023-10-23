import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
plt.style.use(['seaborn-v0_8-talk', 'img/style.mplstyle'])

num_cores = 128 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

import sys,os
home = os.getcwd() + '/../'
sys.path.append(home)

from sampling.mh import Sampler, OutputType
from benchmarks.benchmarks_mchmc import *
from sampling.dynamics import leapfrog, minimal_norm



def grid(f, x, y, vmapable = False):
    
    ### do the grid search ###
    if vmapable:
        z, a = jax.vmap(lambda _x: jax.vmap(lambda _y: f(_x, _y))(y))(x)
    else:
        results = jnp.array([[f(_x, _y) for _y in y] for _x in x])
        z, a = results[:, :, 0], results[:, :, 1]
    
    ### optimal parameters ###
    I = np.argmax(z)
    i, j = I // len(y), I % (len(y))
    best = (x[i], y[j], np.max(z), a[i, j])
    
    return z, a, best


def stn(d, hmc):
    target = StandardNormal(d= d)

    samples = (int)(1000 * np.power(d/1000, 0.25))
    
    def ess(N, eps):
        sampler = Sampler(target, N, eps, integrator= leapfrog, hmc= hmc)
        e, a = sampler.sample(samples, 128, output = OutputType.ess)
        return e, jnp.average(a)

    if hmc:
        x = jnp.arange(1, 7)
        if d >= 3000:
            x = jnp.arange(5, 10)
        y = jnp.logspace(jnp.log10(0.2), jnp.log10(0.32), 15) * jnp.power(d/1000., -0.25)
    else:
        x = jnp.arange(1, 6)
        if d > 3000:
            x = jnp.arange(3, 8)
        y = jnp.logspace(jnp.log10(7.), jnp.log10(23.), 15) * jnp.power(d/1000., 0.25)
    
    z, a, best = grid(ess, x, y)
    
    name = ('hmc' if hmc else 'mchmc') + '_stn_'+ str(d) + 'd'
    np.savez('data/dimensions/mh/' + name + '.npz', ess = z, acc = a, best = best)
    
    img(x, y, z, a, best, name)



def img(x, y, z, a, best, name):
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(r'ESS = {2:.3f}, acc = {3:.3f} ($N$ = {0}, $\epsilon$ = {1:.2f})'.format(*(best)), fontsize = 26)

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(['ESS', 'acceptance rate'][i], fontsize = 25)    
        ax = plt.gca()
        Z = z if i == 0 else a
        cax = ax.matshow(Z, cmap = [plt.cm.Blues, plt.cm.Greys_r][i])
        plt.colorbar(cax)
        
        ax.set_xticks(np.arange(0, len(y), 2), [str(_y)[:4] for _y in y[::2]])
        ax.set_yticks(np.arange(0, len(x), 1), [str(_x)[:4] for _x in x[::1]])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(r'$\epsilon$', fontsize = 22)
        ax.set_ylabel(r'$N = L / \epsilon$', fontsize = 22)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('img/dimensions/'+name+'.png')
    plt.show()
    
    
    
from time import time


stn(10000, True)
print('done 3')

# for d in [100, 300, 1000, 3000]:
#     t0 = time()
#     stn(d, True)
#     print((time() - t0)/60.)