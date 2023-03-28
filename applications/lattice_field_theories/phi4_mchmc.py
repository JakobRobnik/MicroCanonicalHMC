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
    side = ([8, 16, 32, 64])[index]

    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    keys = jax.random.split(jax.random.PRNGKey(42), 8)  # we run 8 independent chains
            
    folder = dir + '/phi4results/mchmc/ground_truth/chi/'
    
    def f(i_lam, i_repeat):
        return sample_chi(side, lam[i_lam], 5000000, keys[i_repeat], 0.6, 0.05)


    
    fvmap = lambda x, y: jax.pmap(jax.vmap(lambda xx: jax.vmap(f, (None, 0))(xx, y)))(x.reshape(4, 4))

    data = fvmap(jnp.arange(len(lam)), jnp.arange(8))

    np.save(folder+'L' + str(side) + '.npy', data.reshape(16, 8))

    

# def sample(side, lam, num_samples, key, alpha, beta, integrator):
    
#     target = phi4.Theory(side, lam)
#     target.transform = lambda x: x
#     sampler = Sampler(target, L=jnp.sqrt(target.d) * alpha, eps= jnp.sqrt(target.d) * beta, integrator= integrator)

#     phi = sampler.sample(num_samples, tune = 'none', random_key = key)
#     burnin = 1000
#     phi_reshaped = phi.reshape(num_samples, target.L, target.L)[burnin:]
    
#     P = jax.vmap(target.psd)(phi_reshaped)
#     Pchain = jnp.cumsum(P, axis= 0) / jnp.arange(1, 1 + num_samples-burnin)[:, None, None]
#     return Pchain



def sample_chi(side, lam, num_samples, key, alpha, beta):
    
    target = phi4.Theory(side, lam)
    sampler = Sampler(target, L=jnp.sqrt(target.d) * alpha, eps= jnp.sqrt(target.d) * beta, integrator='MN')

    phibar = sampler.sample(num_samples)
    burnin = 1000
    return phi4.reduce_chi(target.susceptibility2(phibar[burnin:, 0]), side)
    
    

# def grid_search():
    
#     index = 1
#     side = ([8, 16, 32, 64])[index]
#     folder = dir + '/phi4results/mchmc/ess/psd/grid_search/'
    
#     integrator = 'MN'
#     grads_per_step = 2.0 if integrator == 'MN' else 1.0
    
#     num_samples= 2000
#     burnin = 500

#     lam = phi4.unreduce_lam(phi4.reduced_lam, side)
#     PSD0 = jnp.array(np.median(np.load(dir + '/phi4results/nuts/ground_truth/psd/L' + str(side) + '.npy').reshape(len(lam), 8, side, side), axis =1))

#     #We run multiple independent chains to average ESS over them. Each of the 4 GPUs simulatanously runs repeat1 chains for each lambda
#     #This is repeated sequentially repeat2 times. In total we therefore get 4 * repeat1 * repeat2 chains.
#     repeat1 = ([20, 10, 100, 100])[index]
#     keys = jax.random.split(jax.random.PRNGKey(42), repeat1)


#     def f(i_lam, i_repeat):
#         PSD = jax.vmap(jax.vmap(lambda a, b: sample(side, lam[i_lam], num_samples, keys[i_repeat], a, b, integrator)))(Alpha.T, Beta.T)
#         b2_sq = jnp.average(jnp.square(1 - (PSD / PSD0[None, None, i_lam, :, :])), axis=(-2, -1))  # shape = (n_samples,)
#         return b2_sq
    
#     fvmap = lambda x, y: jax.pmap(jax.vmap(lambda xx: jax.vmap(f, (None, 0))(xx, y)))(x.reshape(4, 4))

#     _b2_sq = fvmap(jnp.arange(len(lam)), jnp.arange(repeat1)).reshape(len(lam), repeat1, len(alpha), len(beta), num_samples-burnin)
#     b2_sq = jnp.average(_b2_sq, axis=1)
#     num_steps = np.argmax(b2_sq < 0.01, axis = -1)
        
#     ess = (200.0 / (grads_per_step * num_steps)) * (num_steps != 0)
#     np.save(folder + 'L' + str(side) + '.npy', ess)


def load_data(index):
    side = ([8, 16, 32, 64])[index]
    integrator = 'MN'
    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    PSD0 = jnp.array(np.median(np.load(dir + '/phi4results/nuts/ground_truth/psd/L' + str(side) + '.npy').reshape(len(lam), 8, side, side), axis =1))
    
    return side, integrator, lam, PSD0


def sample(lamb, side, tune, alpha, beta, integrator, num_samples, remove, chains, PSD0):
    
    grads_per_step = 2.0 if integrator == 'MN' else 1.0
    target = phi4.Theory(side, lamb)
    target.transform = lambda x: x
    sampler = Sampler(target, alpha * jnp.sqrt(target.d), beta * jnp.sqrt(target.d), integrator= integrator)
    sampler.frac_num1 = 0.1
    sampler.frac_num1 = 0.1
    sampler.varEwanted = 3e-2#1e-3 #targeted energy variance Var[E]/d

    phi1, E, L, eps = sampler.sample(num_samples, chains, tune = tune, output = 'detailed')

    vare= jnp.average(jnp.square(E[:, 1:]-E[:, :-1]), axis = 1)/target.d
    varavg, varstd = jnp.average(vare), jnp.std(vare)
    Lavg, Lstd, epsavg, epsstd = jnp.average(L), jnp.std(L), jnp.average(eps), jnp.std(eps)

    phi = phi1.reshape(chains, num_samples, side, side)[:, remove:, :, :]
    P = jax.vmap(jax.vmap(target.psd))(phi)
    PSD = jnp.cumsum(P, axis= 1) / jnp.arange(1, 1 + num_samples-remove)[None, :, None, None] #shape = (chains, samples, L, L)

    b2_sq = jnp.average(jnp.square(1 - (PSD / PSD0[None, None, :, :])), axis=(0, 2, 3))  # shape = (samples,)
    num_steps = jnp.argmax(b2_sq < 0.01)
    return (200.0 / (num_steps * grads_per_step)) * (num_steps != 0), Lavg, Lstd, epsavg, epsstd, varavg, varstd

    
def tuning_free():
    
    index = 2
    side, integrator, lam, PSD0 = load_data(index)
    
    tune = 'cheap'
    
    chains = ([100, 100, 12, 4])[index]

    if tune == 'cheap':
        num_samples= 5000
        remove= 2000
    else:
        num_samples = 3000
        remove = 1000
    
    if index < 2:
        data_tuple = jax.vmap(jax.vmap(lambda i_lam: sample(lam[i_lam], side, tune, 0.6, 0.3, integrator, num_samples, remove, chains, PSD0[i_lam])))(jnp.arange(16).reshape(4, 4))
        data = np.array([np.concatenate(dd) for dd in data_tuple]).T
        
    else: #there is not enough space on the GPU, we do lambda sequentially
        data = [sample(lam[i_lam], side, tune, 0.6, 0.3, integrator, num_samples, remove, chains, PSD0[i_lam]) for i_lam in range(16)]
            
    df = pd.DataFrame(data, columns = ['ESS', 'L', 'L err', 'eps', 'eps err', 'Var[E]/d', 'Var[E]/d err'])
    df.to_csv(dir + '/phi4results/mchmc/ess/psd/autotune/' + tune + '_LL' + str(side) + '.csv', index =False)

    

def grid_search():
    
    index = 3
    side, integrator, lam, PSD0 = load_data(index)
    
    alpha = jnp.logspace(jnp.log10(0.4), jnp.log10(3.0), 6)
    beta = jnp.logspace(jnp.log10(0.2), jnp.log10(0.5), 6)
    Alpha, Beta = jnp.meshgrid(alpha, beta)
    tune = 'none'
    
    chains = ([100, 24, 12, 12])[index]

    num_samples = 3000
    remove = 1000
    if index< 2:
        data = [jax.vmap(jax.vmap(lambda a, b: sample(lam[i_lam], side, tune, a, b, integrator, num_samples, remove, chains, PSD0[i_lam])[0]))(Alpha.T, Beta.T) for i_lam in range(16)]
    elif index == 2:
        data = [[jax.vmap(lambda b: sample(lam[i_lam], side, tune, a, b, integrator, num_samples, remove, chains, PSD0[i_lam])[0])(beta) for a in alpha] for i_lam in range(16)]
    else:
        data = [[[sample(lam[i_lam], side, tune, a, b, integrator, num_samples, remove, chains, PSD0[i_lam])[0] for b in beta] for a in alpha] for i_lam in range(16)]
        
        
    np.save(dir + '/phi4results/mchmc/ess/psd/grid_search/L' + str(side) + '.npy', data)
    
    
def timing():
    side = 2**12
    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    target = phi4.Theory(side, lam[-1])
    sampler = Sampler(target, L=jnp.sqrt(target.d) * 0.6, eps= jnp.sqrt(target.d) * 0.3, integrator= 'MN')
    phi = sampler.sample(250, tune = 'none', output = 'expectation')
    
    
t0 = time.time()

#timing()
tuning_free()
#grid_search()
#compute_ess()
#ground_truth()

t1 = time.time()
print(time.strftime('%H:%M:%S', time.gmtime(t1 - t0)))

