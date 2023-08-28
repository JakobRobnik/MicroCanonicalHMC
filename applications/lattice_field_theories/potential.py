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
        

    def plot_potential(self):
        
        chains, _, num_nx = self.logW.shape
        
        # compute the linear fits to obtain the potential
        Vdata = jnp.array([[self.linfit(self.logW[i, :, j]) for j in range(num_nx)] for i in range(chains)])
        
    
        # plot the potential
        ff = 24
        plt.rcParams['xtick.labelsize'] = ff
        plt.rcParams['ytick.labelsize'] = ff
        plt.figure(figsize= (14, 5))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.title('lattice = 16x16, beta = 4', fontsize = ff)

        nx = np.arange(num_nx) + 1
        for chain in range(chains):
            plt.errorbar(nx, Vdata[chain, :, 0], yerr = Vdata[chain, :, 1], fmt = 'o', capsize = 3)
        
        
        plt.xlabel(r'$n_x$', fontsize = ff)
        plt.ylabel(r'$a V(a n_x)$', fontsize = ff)
        plt.savefig(dir + '/potential_averaging.png')
        plt.xlim(0, 8.5)
        plt.ylim(0, 0.7)
        plt.tight_layout()
        plt.show()




    def linfit(self, _y):
        """Seljak notes, lecture 2"""
        
        nonans = jnp.isfinite(_y)
        x = jnp.arange(1, self.Lt)[nonans]
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
Pot.plot_potential()
