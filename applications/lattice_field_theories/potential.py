import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import os

num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from applications.lattice_field_theories.theories import gauge_theory as u1
from sampling.sampler import Sampler

dir = os.path.dirname(os.path.realpath(__file__))


class PotentialComputation:
    
    def __init__(self, Lt, Lx, beta):

        self.Lt, self.Lx, self.beta = Lt, Lx, beta



    def mchmc(self):

        target = u1.Theory(self.Lt, self.Lx, self.beta, observable= 'Wilson loop', Oparams = 'all')
        alpha = 1.0
        beta_eps= 0.1
        sampler = Sampler(target, L= np.sqrt(target.d) * alpha, eps= np.sqrt(target.d) * beta_eps, integrator='LF', frac_tune1= 0., frac_tune2= 0., frac_tune3= 0., diagonal_preconditioning= False)

        samples = 100000
        burnin = samples//10
        chains = num_cores
        W, E, L, eps = sampler.sample(samples, num_chains= chains, output= 'detailed')
        
        print(np.average(np.square(E)) / target.d)
        Wavg = jnp.average(W, axis = 1) # path integral average
        self.logW = jnp.log(Wavg.reshape(chains, self.Lt-1, self.Lx//2))
        

    def linfit(self):
        V, errV = jax.vmap(self.linfit_single, 1)(self.logW)
        


    def linfit_single(self, _y):
        """Seljak notes, lecture 2"""
        
        nonans = jnp.isfinite(_y)
        x = jnp.arange(1, self.Lt)[nans]
        y = _y[nonans]
        S = len(x)
        Sx = jnp.sum(x)
        Sy = jnp.sum(y)
        Sxx= jnp.sum(jnp.square(x))
        Sxy = jnp.sum(x*y)
        delta = S * Sxx - Sx**2
        #zmle = jnp.array([(Sxx * Sy - Sx * Sxy) / delta, (S * Sxy - Sx*Sy) / delta]) #constant, potential
        #Cov_lik = jnp.array([[Sxx, -Sx], [-Sx, S]]) / delta
        return -(S * Sxy - Sx*Sy) / delta, np.sqrt(S / delta) #potential and its error



def linfit(_x, _y):
    nonans = jnp.isfinite(_y)
    x, y = _x[nonans], _y[nonans]
    S = len(x)
    Sx = jnp.sum(x)
    Sy = jnp.sum(y)
    Sxx= jnp.sum(jnp.square(x))
    Sxy = jnp.sum(x*y)
    delta = S * Sxx - Sx**2
    zmle = jnp.array([(Sxx * Sy - Sx * Sxy) / delta, (S * Sxy - Sx*Sy) / delta]) #constant, potential
    Cov_lik = jnp.array([[Sxx, -Sx], [-Sx, S]]) / delta
    return zmle, Cov_lik




Lt, Lx, beta = 16, 16, 4.

Pot = PotentialComputation(Lt, Lx, beta)
Pot.mchmc()


y = Pot.logW[0, :, 3]
x = jnp.arange(1, Lt)

plt.title('Euclidean time decay \nnx = 4, beta = 4, lattice = 16x16')
plt.plot(x, y, 'o')

z, Cov = linfit(x, y)
plt.plot(x, z[0] + z[1] * x, alpha = 0.7, color = 'black')

plt.xlabel('nt')
plt.ylabel('-log W')
plt.savefig(dir + '/Euclidean time decay.png')
plt.show()
