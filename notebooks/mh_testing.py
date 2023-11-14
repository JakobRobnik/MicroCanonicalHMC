import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
plt.style.use(['seaborn-v0_8-talk', 'img/style.mplstyle'])

num_cores = 128
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

import sys,os
home = os.getcwd() + '/../'
sys.path.append(home)

from mclmc.mh import Sampler, OutputType
from benchmarks.benchmarks_mchmc import *
from mclmc.dynamics import leapfrog, minimal_norm



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
    if i == 0 or i == len(x)-1 or j == 0 or j == len(y)-1:
        print("Warning: optimum achieved at the grid border")
    best = (x[i], y[j], np.max(z), a[i, j])
    
    return z, a, best


def predict_optimal(d, hmc, adjust):
    """predict optimal N and eps"""
       
    if hmc:
        params = (0.25, -0.25, 5, 0.253) if adjust else (0., 0., 3, 0.45)
    else:
        params = (0.25, 0.25, 3, 12.69) if adjust else (0., 0.5, 2, 21.5)
    
    N = jnp.rint(params[2] * np.power(d/1000., params[0])).astype(int)
    eps = params[3] * jnp.power(d/1000., params[1])
    
    return N, eps
    


def stn(d, hmc, adjust):
    target = StandardNormal(d= d)

    samples = (int)(1500 * (np.power(d/1000, 0.25) if adjust else 1))
    
    def ess(N, eps):
        sampler = Sampler(target, N, eps, integrator= leapfrog, hmc= hmc, adjust = adjust)
        e, a = sampler.sample(samples, 128, output = OutputType.ess)
        return e, jnp.average(a)

    # set the grid
    Nopt, epsopt = predict_optimal(d, hmc, adjust)
    nmin = max(Nopt-2, 0)
    x = jnp.arange(nmin, nmin + 5)
    y = jnp.logspace(-jnp.log10(1.5), jnp.log10(1.5), 10) * epsopt
    
    x = jnp.arange(2, 2 + 5)
    y = jnp.logspace(jnp.log10(0.8), jnp.log10(1.4), 10)
    
    # do the computation
    z, a, best = grid(ess, x, y)
    
    # save the results    
    name = ('' if adjust else 'u') + ('hmc' if hmc else 'mchmc') + '_stn_'+ str(d) + 'd'
    np.savez('data/dimensions/mh/' + name + '.npz', ess = z, acc = a, best = best)
    
    # plot the grid results
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
    plt.close()
    

stn(2, False, False)
print('done')

#[2, 3, 4, 5, 10, 30, 100, 300, 1000, 3000, 10000]

# for d in [3000]:
#     for hmc in [True, False]:
#         for adjust in [False, ]:
#             print(d, adjust)
#             stn(d, hmc, adjust)
