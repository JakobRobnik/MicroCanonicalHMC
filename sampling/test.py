from sampler import Sampler as Sampler 
from old_sampler import Sampler as OldSampler 
from annealing import Sampler as AnnealingSampler 
from old_annealing import Sampler as OldAnnealingSampler 

import jax.numpy as nnp
import math
import jax
import scipy

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
    
shift_fn = lambda x, y : x + y 

target = MD(d = 2064)

def run_sequential(L_factor, chain_length, old):


    eps = 0.1

    if old:
        sampler = OldSampler(target, frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, L = L_factor*eps,
                  eps=eps)    
    else:
        sampler = Sampler(target, shift_fn=shift_fn, masses=nnp.ones(2064,) , frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, L = L_factor*eps, eps=eps)
        # sampler = Sampler(target, frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, L = L_factor*eps, eps=eps)
        # jax.numpy.tile(mol.masses,3)
    num_chains = 1
    samples, energy, L, _ = sampler.sample(chain_length, num_chains, output= 'detailed', random_key=jax.random.PRNGKey(0))



    # name = 'mclmc' + str(eps) + str(L) + str(num_chains)
    # trajectory.save_pdb('./data/prod_alanine_dipeptide_amber/traj'+name+'.pdb')

    return samples,energy, L, eps


def run_annealing(old):


  
    # sampler = OAS(target, shift_fn=shift_fn)
    if old:
        sampler = OldAnnealingSampler(target) # , masses=jax.numpy.tile(mol.masses,3))
    else: sampler = AnnealingSampler(target, shift_fn=shift_fn, masses=nnp.ones(2064,)) # masses=jax.numpy.tile(mol.masses,3))

    eps = 0.1
    # L = eps * 30

    sampler.L = 30*eps
    sampler.eps_initial = eps

    
                         
                #          masses = jax.numpy.tile(mol.masses,3), frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, L = L_factor*eps,
                #   eps=eps)
    if not old:
        samples, energy = sampler.sample(steps_at_each_temp=5, temp_schedule=nnp.array([1.0]), num_chains=10, tune_steps=0, random_key=jax.random.PRNGKey(0))
    else:
        samples = sampler.sample(steps_at_each_temp=5, temp_schedule=nnp.array([1.0]), num_chains=10, tune_steps=0, random_key=jax.random.PRNGKey(0))
        energy = 0
    # , x_initial=x_initial[::100])   

    
    return samples, energy

# Energy error:  0.10286242477917347
# samples,energy,L,eps = run_sequential(T=4000, dt=2, L_factor=30, chain_length=10, old=False)
# print(f"mean of samples: {samples.mean()}")
# print(f"energy of shape {energy.shape}")
# print(f"energy: {(nnp.square(energy[1:]-energy[:-1])/2064).mean()}")
# print(f"mean of samples: {samples.mean()}")
# print(f"energy: {(nnp.square(energy[1:]-energy[:-1])/2064).mean()}")



# print(samples)

# print("Annealing\n\n")
# samples, energy = run_annealing(old=False)
# print(f"energy of shape {energy[0,:,0].shape}")
# energy_error = (nnp.square(energy[:, 1:, :]-energy[:, :-1, :])/2064).mean()
# print(f"error: {energy_error}")
# print(f"mean of samples: {samples.mean()}")
# print("Annealing, old:\n\n")
# samples, energy = run_annealing(old=True)
# print(f"mean of samples: {samples.mean()}")

if __name__=="__main__":
    # assert 4==5

    ## make sure that the refactored sequential sampler gives same result
    old_samples,energy,L,eps = run_sequential(L_factor=30, chain_length=10, old=True)
    samples,energy,L,eps = run_sequential(L_factor=30, chain_length=10, old=False)
    assert old_samples.mean() == samples.mean()


    annealing_samples,energy = run_annealing(old=False)
    old_annealing_samples,energy = run_annealing(old=True)

    assert nnp.allclose(old_annealing_samples, annealing_samples[0,-1,:,:])

    # assert nnp.allclose(annealing_samples[0,:,0,:], samples[:5, :])
    