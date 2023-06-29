from sampler import Sampler as Sampler 
from old_sampler import Sampler as OldSampler 
from annealing import Sampler as AnnealingSampler 
from old_annealing import Sampler as OldAnnealingSampler 

import jax.numpy as nnp
import jax

nlogp = lambda x : (x**2).mean()/2
energy_fn = lambda x : nlogp(x)
value_grad = jax.jit(jax.value_and_grad(energy_fn))

class MD():


    def __init__(self, d):
        self.d = d
        self.nbrs = None

    def grad_nlogp(self, x):
        return value_grad(x)

    def transform(self, x):
        return x

    def prior_draw(self, key):
        return nnp.zeros((self.d,)) + 1
    

target = MD(d = 2064)

def run_sequential(L_factor, chain_length, old):

    eps = 0.1

    if old:
        sampler = OldSampler(target, integrator = 'MN', frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, L = L_factor*eps,
                  eps=eps)    
    else:
        sampler = Sampler(target, integrator = 'MN', frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, L = L_factor*eps, eps=eps)
    num_chains = 1
    samples, energy, L, _ = sampler.sample(chain_length, num_chains, output= 'detailed', random_key=jax.random.PRNGKey(0))

    return samples,energy, L, eps


def run_annealing(old, temps, chain_length, num_chains):

    if old:
        sampler = OldAnnealingSampler(target)
    else: sampler = AnnealingSampler(target)

    eps = 0.1
    sampler.L = 30*eps
    sampler.eps_initial = eps

    if not old:
        samples, energy = sampler.sample(steps_at_each_temp=chain_length, temp_schedule=nnp.array(temps), num_chains=num_chains, tune_steps=0, random_key=jax.random.PRNGKey(0))
    else:
        samples = sampler.sample(steps_at_each_temp=chain_length, temp_schedule=nnp.array(temps), num_chains=num_chains, tune_steps=0, random_key=jax.random.PRNGKey(0))
        energy = 0

    
    return samples, energy

if __name__=="__main__":

    ## make sure that the refactored sequential sampler gives same result as old one
    old_samples,energy,L,eps = run_sequential(L_factor=30, chain_length=5, old=True)
    samples,energy,L,eps = run_sequential(L_factor=30, chain_length=5, old=False)
    assert old_samples.mean() == samples.mean()

    ## make sure that the refactored annealing sampler gives same result as old one
    annealing_samples,energy = run_annealing(old=False, temps=[1.0,2.0], chain_length=5, num_chains=10)
    old_annealing_samples,energy = run_annealing(old=True, temps=[1.0,2.0], chain_length=5, num_chains=10)

    assert nnp.allclose(old_annealing_samples, annealing_samples[-1,-1,:,:])

    annealing_samples,energy = run_annealing(old=False, temps=[1.0], chain_length=5, num_chains=1)
    ## the annealing sampler, in the limiting case of a single chain, a single temperature, and no tuning, should of course reproduce the results of the sequential sampler
    assert nnp.allclose(annealing_samples[0,:,0,:], samples[:5, :])
    