import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import os
import time

#num_cores = 6 #specific to my PC
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from applications.lattice_field_theories.theories import phi4
from sampling.sampler import Sampler
from sampling.sampler import find_crossing
from sampling.grid_search import search_wrapper

dir = os.path.dirname(os.path.realpath(__file__))
#params_critical_line = pd.read_csv(dir + '/theories/phi4_parameters.csv')



def parallel_run(function, values):
    parallel_function= jax.pmap(jax.vmap(function))
    results = jnp.array(parallel_function(values.reshape(num_cores, len(values) // num_cores)))
    return results.reshape([len(values), ] + [results.shape[i] for i in range(2, len(results.shape))])



def get_params(side):
    """parameters from https://arxiv.org/pdf/2207.00283.pdf"""
    return np.array(params_critical_line[params_critical_line['L'] == side][['lambda']])[0][0] # = lambda



def ground_truth():
    
    index = 3
    side = ([8, 16, 32, 64, 128])[index]

    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    keys = jax.random.split(jax.random.PRNGKey(42), 8)  # we run 8 independent chains
            
    folder = dir + '/phi4results/mchmc/ground_truth/chi/'
    
    def f(i_lam, i_repeat):
        return sample_chi(side, lam[i_lam], 1000000, keys[i_repeat], 1.0, 0.15)


    
    fvmap = lambda x, y: jax.pmap(jax.vmap(lambda xx: jax.vmap(f, (None, 0))(xx, y)))(x.reshape(4, 4))

    data = fvmap(jnp.arange(len(lam)), jnp.arange(8))

    np.save(folder+'L' + str(side) + '.npy', data.reshape(16, 8))

    

def sample(side, lam, num_samples, key, alpha, beta):
    
    target = phi4.Theory(side, lam)
    sampler = Sampler(target, L=jnp.sqrt(target.d) * alpha, eps= jnp.sqrt(target.d) * beta, integrator='MN')

    phi, E, burnin = sampler.sample(num_samples, output = 'full')
    burnin = 1000

    phi_reshaped = phi.reshape(num_samples, target.L, target.L)[burnin:]
    
    P = jax.vmap(target.psd)(phi_reshaped)
    Pchain = jnp.cumsum(P, axis= 0) / jnp.arange(1, 1 + num_samples-burnin)[:, None, None]
    return Pchain


def sample_chi(side, lam, num_samples, key, alpha, beta):
    
    target = phi4.Theory(side, lam)
    sampler = Sampler(target, L=jnp.sqrt(target.d) * alpha, eps= jnp.sqrt(target.d) * beta, integrator='MN')

    phibar, burnin = sampler.sample(num_samples, output = 'normal')
    burnin = 1000
    return phi4.reduce_chi(target.susceptibility2(phibar[burnin:, 0]), side)
    
    

def grid_search():
    
    alpha = jnp.logspace(jnp.log10(0.4), jnp.log10(3.0), 6)
    beta = jnp.logspace(jnp.log10(0.1), jnp.log10(0.5), 6)
    Alpha, Beta = jnp.meshgrid(alpha, beta)
    index = 2
    side = ([8, 16, 32, 64])[index]
    folder = dir + '/phi4results/mchmc/ess/psd/'
    
    num_samples= 10000
    burnin = 1000

    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    PSD0 = jnp.array(np.median(np.load(dir + '/phi4results/hmc/ground_truth/psd/L' + str(side) + '.npy').reshape(len(lam), 8, side, side), axis =1))

    #We run multiple independent chains to average ESS over them. Each of the 4 GPUs simulatanously runs repeat1 chains for each lambda
    #This is repeated sequentially repeat2 times. In total we therefore get 4 * repeat1 * repeat2 chains.
    repeat1 = ([100, 100, 1, 10])[index]
    repeat2 = ([10, 10, 10, 100])[index]
    keys = jax.random.split(jax.random.PRNGKey(42), repeat1*repeat2)


    def f(i_lam, i_repeat):
        PSD = jax.vmap(jax.vmap(lambda a, b: sample(side, lam[i_lam], num_samples, keys[i_repeat], a, b)))(Alpha.T, Beta.T)
        b2_sq = jnp.average(jnp.square(1 - (PSD / PSD0[None, None, i_lam, :, :])), axis=(-2, -1))  # shape = (n_samples,)
        return b2_sq
    
    fvmap = lambda x, y: jax.pmap(jax.vmap(lambda xx: jax.vmap(f, (None, 0))(xx, y)))(x.reshape(4, 4))

    b2_sq = jnp.zeros((len(lam), num_samples-burnin, len(alpha), len(beta)))
    

    for r2 in range(repeat2):
        _b2_sq = fvmap(jnp.arange(len(lam)), jnp.arange(r2*repeat1, (r2+1)*repeat1))
        b2_sq += jnp.average(_b2_sq.reshape(len(lam), repeat1, num_samples-burnin, len(alpha), len(beta)), axis=1)

    b2_sq /= repeat2

    num_steps = jnp.argmax(b2_sq < 0.01, axis=1)
    ess = (200.0 / (num_steps)) * (num_steps != 0)
  
    np.save(folder + 'L' + str(side) + '.npy', ess)


# def grid_search2():
    
#     alpha = 1.0
#     beta = jnp.logspace(0.1, 0.5, 6)
#     index = 0
#     side = ([8, 16, 32, 64])[index]
#     folder = dir + '/phi4results/mchmc/ess/psd/'
    
#     num_samples= 10000
#     burnin = 1000

#     lam = phi4.unreduce_lam(phi4.reduced_lam, side)
#     PSD0 = jnp.array(np.median(np.load(dir + '/phi4results/hmc/ground_truth/psd/L' + str(side) + '.npy').reshape(len(lam), 8, side, side), axis =1))

#     #We run multiple independent chains to average ESS over them. Each of the 4 GPUs simulatanously runs repeat1 chains for each lambda
#     #This is repeated sequentially repeat2 times. In total we therefore get 4 * repeat1 * repeat2 chains.
#     repeat1 = ([1000, 1000, 1000, 1000])[index]
#     repeat2 = ([1, 1, 1, 1])[index]
#     keys = jax.random.split(jax.random.PRNGKey(42), repeat1*repeat2)

#     def f(i_lam, i_repeat):
#         PSD = jax.vmap(lambda b: sample(side, lam[i_lam], num_samples, keys[i_repeat], alpha, b))(beta)
#         b2_sq = jnp.average(jnp.square(1 - (PSD / PSD0[None, i_lam, :, :])), axis=(-2, -1))  # shape = (n_samples,)
#         return b2_sq

#     fvmap = lambda x, y: jax.pmap(jax.vmap(lambda xx: jax.vmap(f, (None, 0))(xx, y)))(x.reshape(4, 4))

#     b2_sq = jnp.zeros((len(lam), num_samples-burnin, len(beta)))
    
#     for r2 in range(repeat2):
#         _b2_sq = fvmap(jnp.arange(len(lam)), jnp.arange(r2*repeat1, (r2+1)*repeat1))
#         b2_sq += jnp.average(_b2_sq.reshape(16, repeat1, num_samples-burnin, len(beta)), axis=1)

#     b2_sq /= repeat2

#     num_steps = jnp.argmax(b2_sq < 0.01, axis=1)
#     ess = (200.0 / (num_steps)) * (num_steps != 0)
  
#     np.save(folder + 'L' + str(side) + '.csv', ess)

    
# def compute_ess():
    
#     alpha = 1.0
#     beta = 0.2
#     index = 0
#     side = ([8, 16, 32, 64])[index]
#     folder = dir + '/phi4results/mchmc/ess/psd/'
    
#     num_samples= 10000
#     burnin = 1000

#     lam = phi4.unreduce_lam(phi4.reduced_lam, side)
#     PSD0 = jnp.array(np.median(np.load(dir + '/phi4results/hmc/ground_truth/psd/L' + str(side) + '.npy').reshape(len(lam), 8, side, side), axis =1))

#     #We run multiple independent chains to average ESS over them. Each of the 4 GPUs simulatanously runs repeat1 chains for each lambda
#     #This is repeated sequentially repeat2 times. In total we therefore get 4 * repeat1 * repeat2 chains.
#     repeat1 = ([1000, 1000, 1000, 1000])[index]
#     repeat2 = ([1, 1, 1, 1])[index]
#     keys = jax.random.split(jax.random.PRNGKey(42), repeat1*repeat2)

#     def f(i_lam, i_repeat):
#         PSD = sample(side, lam[i_lam], num_samples, keys[i_repeat], alpha, beta)
#         b2_sq = jnp.average(jnp.square(1 - (PSD / PSD0[i_lam, :, :])), axis=(1, 2))  # shape = (n_samples,)
#         return b2_sq

#     fvmap = lambda x, y: jax.pmap(jax.vmap(lambda xx: jax.vmap(f, (None, 0))(xx, y)))(x.reshape(4, 4))

#     b2_sq = jnp.zeros((len(lam), num_samples-burnin))
    
#     for r2 in range(repeat2):
#         _b2_sq = fvmap(jnp.arange(len(lam)), jnp.arange(r2*repeat1, (r2+1)*repeat1))
#         b2_sq += jnp.average(_b2_sq.reshape(16, repeat1, num_samples-burnin), axis=1)

#     b2_sq /= repeat2

#     num_steps = jnp.argmax(b2_sq < 0.01, axis=1)
#     ess = (200.0 / (num_steps)) * (num_steps != 0)
  
#     np.save(folder + 'L' + str(side) + '_'+str(alpha)+'_'+str(beta)+'.csv', ess)

    
    

t0 = time.time()

grid_search()#compute_ess()
#ground_truth()

t1 = time.time()
print(time.strftime('%H:%M:%S', time.gmtime(t1 - t0)))

