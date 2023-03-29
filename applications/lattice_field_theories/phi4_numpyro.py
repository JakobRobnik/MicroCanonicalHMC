import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import jax
import jax.numpy as jnp
import time

# num_cores = 6 #specific to my PC
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

import numpyro
from numpyro.infer import MCMC, NUTS, HMC

numpyro.set_platform("gpu")

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from applications.lattice_field_theories.theories import phi4
from HMC.mchmc_to_numpyro import mchmc_target_to_numpyro

dir = os.path.dirname(os.path.realpath(__file__))

phi4_model = mchmc_target_to_numpyro(phi4.Theory)


def sample(L, lam, num_samples, key, num_warmup=500, thinning=1, full=True, psd=True, nuts = True):
    # setup
    theory = phi4.Theory(L, lam)
    
    if nuts:
        hmc_setup = NUTS(phi4_model, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)
    else:
        hmc_setup = HMC(phi4_model, num_steps = 20, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)
    
    sampler = MCMC(hmc_setup, num_warmup=num_warmup, num_samples=num_samples, num_chains=1,
                   progress_bar=False, thinning=thinning)

    key, prior_key = jax.random.split(key)
    x0 = theory.prior_draw(prior_key)

    # run
    sampler.warmup(key, L, lam, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    burn_in_steps = jnp.sum(jnp.array(sampler.get_extra_fields()['num_steps'], dtype=int))

    sampler.run(key, L, lam, extra_fields=['num_steps'])

    phi = jnp.array(sampler.get_samples()['x'])

    steps = jnp.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    if psd:
        
        PSD = theory.psd(phi.reshape(num_samples // thinning, L, L))

        if full:
            return burn_in_steps, steps, jnp.cumsum(PSD, axis=0) / jnp.arange(1, 1 + num_samples // thinning)[:, None, None]
        else:
            return jnp.average(PSD, axis=0)

    else:
        phi_bar = jnp.average(phi, axis=1)

        if full:
            chi = phi4.reduce_chi(theory.susceptibility2_full(phi_bar), L)
            return burn_in_steps, steps, chi

        else:
            return phi4.reduce_chi(theory.susceptibility2(phi_bar), L)


        
def ground_truth():
    
    psd = False
    index = 1
    side = ([8, 16, 32, 64, 128])[index]
    thinning = ([50, 20, 10, 10, 1])[index] #100, 30, 10, 10

    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    keys = jax.random.split(jax.random.PRNGKey(42), 8)  # we run 8 independent chains

            
    folder = dir + '/phi4results/nuts/ground_truth/'+('psd' if psd else 'chi')+'/'
    
    def f(i_lam, i_repeat):
        return sample(L=side, lam=lam[i_lam], num_samples= 10000* thinning, key=keys[i_repeat],
                      num_warmup=2000, thinning=thinning, full=False, psd=psd)


    
    if index <100:
        fvmap = lambda x, y: jax.pmap(jax.vmap(lambda xx: jax.vmap(f, (None, 0))(xx, y)))(x.reshape(4, 4))

        data = fvmap(jnp.arange(len(lam)), jnp.arange(8))
        
        if psd:
            np.save(folder+'L' + str(side) + '.npy', data.reshape(16, 8, side, side))
    
        else:
            np.save(folder+'L' + str(side) + '.npy', data.reshape(16, 8))
    
    else:
        fvmap = lambda x, y: jax.pmap(lambda xx: jax.vmap(f, (None, 0))(xx, y))(x)

        for I_lam in range(4):

            data = fvmap(jnp.arange(I_lam * 4, (I_lam+1)*4), jnp.arange(8))

            np.save(folder + 'L' + str(side) + '_'+str(I_lam)+'.npy', data)
    
        #join the files
        np.save(folder + 'L'+str(side)+'.npy', np.concatenate([np.load(folder + 'L'+str(side)+'_'+str(i)+'.npy') for i in range(4)]))

    
def compute_ess():
    nuts = True
    index = 2
    side = ([8, 16, 32, 64])[index]
    thinning = ([1, 1, 1, 1])[index]
    num_samples= 5000

    #We run multiple independent chains to average ESS over them. Each of the 4 GPUs simulatanously runs repeat1 chains for each lambda
    #This is repeated sequentially repeat2 times. In total we therefore get 4 * repeat1 * repeat2 chains.
    repeat1 = ([120, 120, 60, 10])[index]
    repeat2 = ([1, 1, 1, 1])[index]
    
    
    lam = phi4.unreduce_lam(phi4.reduced_lam, side)
    folder = dir + '/phi4results/'+('nuts' if nuts else 'hmc')+'/ess/psd/'   
    PSD0 = jnp.array(np.median(np.load(dir + '/phi4results/nuts/ground_truth/psd/L' + str(side) + '.npy').reshape(len(lam), 8, side, side), axis =1))

    keys = jax.random.split(jax.random.PRNGKey(42), repeat1*repeat2)

    def f(i_lam, i_repeat):
        burn, _steps, PSD = sample(L=side, lam=lam[i_lam], num_samples=num_samples * thinning,                                                                            key=keys[i_repeat], num_warmup=500, thinning=thinning, full=True, psd=True, nuts = nuts)
        steps = jnp.cumsum(_steps)  # shape = (n_samples, )
        b2_sq = jnp.average(jnp.square(1 - (PSD / PSD0[i_lam, :, :])), axis=(1, 2))  # shape = (n_samples,)
        return burn, steps, b2_sq

    fvmap = lambda x, y: jax.pmap(jax.vmap(lambda xx: jax.vmap(f, (None, 0))(xx, y)))(x.reshape(4, 4))

    burn = jnp.zeros(len(lam))
    steps = jnp.zeros((len(lam), num_samples))
    b2_sq = jnp.zeros((len(lam), num_samples))
    

    for r2 in range(repeat2):
        _burn, _steps, _b2_sq = fvmap(jnp.arange(len(lam)), jnp.arange(r2*repeat1, (r2+1)*repeat1))
        burn += jnp.average(_burn.reshape(16, repeat1), axis=1)
        steps += jnp.average(_steps.reshape(16, repeat1, num_samples), axis=1)
        b2_sq += jnp.average(_b2_sq.reshape(16, repeat1, num_samples), axis=1)

    burn /= repeat2
    steps /= repeat2
    b2_sq /= repeat2

    index = jnp.argmax(b2_sq < 0.01, axis=1)
    num_steps = steps[jnp.arange(len(lam)), index]
    ess = (200.0 / (num_steps)) * (index != 0)
    ess_with_warmup = (200.0 / (num_steps + burn)) * (index != 0)
    df = pd.DataFrame(np.array([phi4.reduced_lam, ess, ess_with_warmup]).T, columns=['reduced lam', 'ESS', 'ESS (with warmup)'])
    df.to_csv(folder + '/LL' + str(side) + '.csv')


def large_lattice():
    side = 2**10
    lam = phi4.unreduce_lam(phi4.reduced_lam, side)[-1]
    thin = 10
    sample(side, lam, 500*thin, jax.random.PRNGKey(0), num_warmup=500, thinning=thin, full=True, psd= True, nuts = False)



t0 = time.time()

#ground_truth()
compute_ess()
#large_lattice()

t1 = time.time()
print(time.strftime('%H:%M:%S', time.gmtime(t1 - t0)))




# side = 8
# i_lam = -1
# lam = phi4.unreduce_lam(phi4.reduced_lam, side)
# burn, _steps, PSD = sample(side, lam[i_lam], 1000, jax.random.PRNGKey(0), num_warmup=500, thinning=1, full=True, psd=True, nuts = False)

# steps = jnp.cumsum(_steps)  # shape = (n_samples, )
# PSD0 = jnp.array(np.median(np.load(dir + '/phi4results/nuts/ground_truth/psd/L' + str(side) + '.npy').reshape(len(lam), 8, side, side), axis =1))

# b2_sq = jnp.average(jnp.square(1 - (PSD / PSD0[i_lam, :, :])), axis=(1, 2))  # shape = (n_samples,)

# plt.plot(steps, np.sqrt(b2_sq))
# plt.yscale("log")
# plt.savefig('krnkei.png')
# plt.close()