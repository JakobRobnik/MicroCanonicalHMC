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



def get_params(side):
    """parameters from https://arxiv.org/pdf/2207.00283.pdf"""
    return np.array(params_critical_line[params_critical_line['L'] == side][['lambda']])[0][0] # = lambda



def load_data(index):
    side = ([8, 16, 32, 64])[index]
    integrator = 'MN'
    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    PSD0 = jnp.array(np.median(np.load(dir + '/phi4results/nuts/ground_truth/psd/L' + str(side) + '.npy').reshape(len(lam), 8, side, side), axis =1))
    
    return side, integrator, lam, PSD0



# def sample_chi(lamb, side, num_samples, chains, x):
    
#     target = phi4.Theory(side, lamb)
#     target.transform = lambda x: x
#     sampler = Sampler(target, 0.6 * jnp.sqrt(target.d), 0.3 * jnp.sqrt(target.d))
#     sampler.varEwanted = 1e-2 #targeted energy variance Var[E]/d

#     phi = sampler.sample(num_samples, chains, random_key = jax.random.PRNGKey(11), x_initial= x, tune = True, output = 'normal')
#     phibar = jnp.average(phi, axis = -1)[len(phibar)//3, :]
#     chi = jax.vmap(target.susceptibility2)(phibar)
#     chi_reduced = phi4.reduce_chi(chi, side)

#     return phi[:, -1, :], chi_reduced


def sample_chi_2(lamb, side, num_samples, chains):
    
    target = phi4.Theory(side, lamb)
    sampler = Sampler(target, 0.6 * jnp.sqrt(target.d), 0.02 * jnp.sqrt(target.d))
    sampler.varEwanted = 1e-2 #targeted energy variance Var[E]/d

    phibar = sampler.sample(num_samples, chains, random_key = jax.random.PRNGKey(0), tune = False, output = 'normal')[:, num_samples//4:, 0]
    chi = jax.vmap(target.susceptibility2)(phibar)
    chi_reduced = phi4.reduce_chi(chi, side)

    return chi_reduced
                      
# def ground_truth():
    
#     index = 2

#     side = ([8, 16, 32, 64])[index]
#     lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    
#     chains = ([12, 12, 4, 12])[index]
#     num_samples= ([50000, 50000, 200000, 50000])[index]

#     data = np.empty((16, chains))
    
#     #draw from the prior
#     x = jax.vmap(phi4.Theory(side, lam[15]).prior_draw)(jax.random.split(jax.random.PRNGKey(42), chains))
    
#     #do the high temperature (like a burn-in)
#     x = sample_chi(lam[15], side, num_samples, chains, x)[0] 
    
#     #annealing: use the final state as an initial state at the next temperature level (temperature = lambda)
#     for i_lam in range(15, -1, -1):
#         x, data[i_lam] = sample_chi(lam[i_lam], side, num_samples, chains, x)
     
#     np.save(dir+'/phi4results/mchmc/ground_truth/chi/L'+str(side)+'.npy', data)

                          
def ground_truth():
    
    index = 3

    side = ([8, 16, 32, 64])[index]
    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    
    chains = ([12, 12, 12, 12])[index]
    num_samples= ([2000000, 2000000, 2000000, 8000000])[index]

    data = np.empty((16, chains))
    
    #draw from the prior
    
    #do the high temperature (like a burn-in)
    data = jax.vmap(lambda lamb: sample_chi_2(lamb, side, num_samples, chains))(lam)
    
     
    np.save(dir+'/phi4results/mchmc/ground_truth/chi/LL'+str(side)+'.npy', data)

                      
                      
                      
def sample(lamb, side, tune, alpha, beta, integrator, num_samples, remove, chains, x, PSD0):
    
    grads_per_step = 2.0 if integrator == 'MN' else 1.0
    target = phi4.Theory(side, lamb)
    target.transform = lambda x: x
    sampler = Sampler(target, alpha * jnp.sqrt(target.d), beta * jnp.sqrt(target.d), integrator= integrator)
    sampler.frac_num1 = 0.1
    sampler.frac_num1 = 0.1
    sampler.varEwanted = 2e-2 #targeted average single step squared energy error per dimension Var[E]/d

    phi, E, L, eps = sampler.sample(num_samples, chains, x_initial= x, tune = tune, output = 'detailed')

    vare= jnp.average(jnp.square(E[:, 1:]-E[:, :-1]), axis = 1)/target.d
    varavg, varstd = jnp.average(vare), jnp.std(vare)
    Lavg, Lstd, epsavg, epsstd = jnp.average(L), jnp.std(L), jnp.average(eps), jnp.std(eps)

    phi_reshaped = phi.reshape(chains, num_samples, side, side)[:, remove:, :, :]
    P = jax.vmap(jax.vmap(target.psd))(phi_reshaped)
    PSD = jnp.cumsum(P, axis= 1) / jnp.arange(1, 1 + num_samples-remove)[None, :, None, None] #shape = (chains, samples, L, L)

    b2_sq = jnp.average(jnp.square(1 - (PSD / PSD0[None, None, :, :])), axis=(0, 2, 3))  # shape = (samples,)
    num_steps = jnp.argmax(b2_sq < 0.01)
    
    final_state = phi[:, -1, :]
    props = np.array([(200.0 / (num_steps * grads_per_step)) * (num_steps != 0), Lavg, Lstd, epsavg, epsstd, varavg, varstd])
    
    return final_state, props

    
def tuning_free():
    
    index = 2
    side, integrator, lam, PSD0 = load_data(index)
    
    chains = ([100, 100, 24, 4])[index]
    tune = True
    num_samples= 2500
    remove= 0

    data = np.empty((16, 7))
    
    #draw from the prior
    target = phi4.Theory(side, lam[15])
    x = jax.vmap(target.prior_draw)(jax.random.split(jax.random.PRNGKey(123), chains))
    
    #do the high temperature (like a burn-in)
    x = sample(lam[15], side, tune, 0.6, 0.3, integrator, num_samples, remove, chains, x, PSD0[15])[0] 
    
    #annealing: use the final state as an initial state at the next temperature level (temperature = lambda)
    for i_lam in range(15, -1, -1):
        x, data[i_lam] = sample(lam[i_lam], side, tune, 0.6, 0.3, integrator, num_samples, remove, chains, x, PSD0[i_lam])
            
    df = pd.DataFrame(data, columns = ['ESS', 'L', 'L err', 'eps', 'eps err', 'Var[E]/d', 'Var[E]/d err'])
    df.to_csv(dir + '/phi4results/mchmc/ess/psd/autotune/L' + str(side) + '.csv', index =False)

    

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
    
    
def large_lattice():
    side = 2**10
    lamb = phi4.unreduce_lam(phi4.reduced_lam, side)[-1]
    
    target = phi4.Theory(side, lamb)
    sampler = Sampler(target, 0.6 * jnp.sqrt(target.d), 0.3 * jnp.sqrt(target.d))
    sampler.varEwanted = 5e-3 #targeted energy variance Var[E]/d

    phibar = sampler.sample(num_samples, 4, tune = tune, output = 'normal')
                      
    chi = phi4.reduce_chi(jax.vmap(target.susceptibility2_full, (0, None))(phibar, side))
    
    np.save(dir + '/phi4results/mchmc/large_lattice/L' + str(side) + '.npy', chi)

    
t0 = time.time()

#timing()
ground_truth()

#grid_search()
#compute_ess()
#ground_truth()

t1 = time.time()
print(time.strftime('%H:%M:%S', time.gmtime(t1 - t0)))

