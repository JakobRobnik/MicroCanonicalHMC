import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import os
from jaxopt import GaussNewton, LevenbergMarquardt
from scipy.optimize import minimize

# num_cores = 6 #specific to my PC
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from applications.lattice_field_theories.theories import gauge_theory as u1
from sampling.sampler import Sampler

dirr = os.path.dirname(os.path.realpath(__file__)) + '/'


class PotentialComputation:
    
    def __init__(self, Lt, Lx, beta):

        self.Lt, self.Lx, self.beta = Lt, Lx, beta

        #self.Vdata = np.load(dirr + 'data_potential/0.npy')


    def mchmc(self, vare, name):
        
        key = jax.random.PRNGKey(42)
        
        target = u1.Theory(self.Lt, self.Lx, self.beta, observable= 'Polyakov loop')
        
        alpha = 1.0
        beta_eps= 0.1
        sampler = Sampler(target, L= np.sqrt(target.d) * alpha, eps= np.sqrt(target.d) * beta_eps, integrator='LF', 
                          frac_tune1= 0.05, frac_tune2= 0., frac_tune3= 0., diagonal_preconditioning= False, varEwanted= vare)
    
        samples = 2*10**5
        #burnin = samples//10
        chains = num_cores
        P, E, L, eps = sampler.sample(samples, num_chains= chains, output= 'detailed', random_key = key)
        print(E)
        plt.plot(E)
        plt.savefig(dirr + 'energy.png')
        plt.close()
        autocorr = jnp.average(P, axis = 1)
        self.Vdata = -jnp.log(autocorr) / self.Lt

        np.save(dirr + 'data_potential/'+name+'.npy', self.Vdata)
        
        print(np.average(np.square(E)) / target.d)
        print(eps)
        # plt.plot(E, '.')
        # plt.savefig(dirr + '/energy.png')
        # plt.close()
    

    def plot_potential(self):
    
        # plot the potential
        ff = 24
        plt.rcParams['xtick.labelsize'] = ff
        plt.rcParams['ytick.labelsize'] = ff
        plt.figure(figsize= (14, 8))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ### data ###
        chains, num_nx = self.Vdata.shape
        nx = np.arange(num_nx)+1
        V1, V2 = jnp.average(self.Vdata, axis = 0), jnp.average(jnp.square(self.Vdata), axis = 0)
        V, Verr = V1, jnp.sqrt(V2 - jnp.square(V1))
        plt.errorbar(nx, V, yerr = Verr, fmt = 'o', capsize = 3, color= 'black')
        
        #### fit ###
        #params = self.fit(V, Verr/jnp.sqrt(chains-1))
        params = self.linfit(V, 4)
        r = jnp.linspace(0, 8, 100)
        plt.plot(r, params[0] + params[1] * r, color = 'tab:blue', label = 'best linear fit ($\kappa = $' +str(np.round(params[1], 3))+')')
        
        plt.plot(r, self.V0(r, params), color = 'tab:orange', label = 'finite lattice potential (same parameters)' )
     
        plt.xlabel(r'$n$', fontsize = ff)
        plt.ylabel(r'$a V(a n)$', fontsize = ff)
        #plt.xlim(0, 8.5)
        #plt.ylim(0, 0.5)
        plt.suptitle('U(1) potential with Polyakov loops', fontsize = ff + 2, fontweight="bold") 
        plt.tight_layout()

        plt.savefig(dirr + 'potential.png')
        plt.close()


    def V0(self, n, z):
        a = jnp.cosh(z[1] * self.Lt * (self.Lx//2  - n))
        b = jnp.cosh(z[1] * self.Lt * (self.Lx//2))
        return z[0] - jnp.log(a / b) / self.Lt 
        
        
    def fit(self, V, Verr):
        n = jnp.arange(self.Lx//2) + 1
        params0 = np.array([0.04, 0.07])
        res = lambda z: jnp.square((self.V0(n, z) - V) / Verr)
        gn = LevenbergMarquardt(residual_fun= res, maxiter = 200)
        #gn = GaussNewton(residual_fun= res, maxiter = 200)
        return gn.run(params0).params
        
        #loss = lambda z: 0.5 * jnp.sum(jnp.square((self.V0(n, z) - V) / Verr))
        #grad= jax.value_and_grad(loss)
        
        #opt = minimize(grad, jac = True, x0 = params0, method = 'CG')
        #print(opt)
        #return opt['x']
    
    
    def linfit(self, V, cut):
        """z = [intercept, slope]"""
        x = (jnp.arange(self.Lx//2) + 1)[:cut]
        y = V[:cut]
        
        S = len(x)
        Sx = jnp.sum(x)
        Sy = jnp.sum(y)
        Sxx= jnp.sum(jnp.square(x))
        Sxy = jnp.sum(x*y)
        delta = S * Sxx - Sx**2
        zmle = jnp.array([(Sxx * Sy - Sx * Sxy) / delta, (S * Sxy - Sx*Sy) / delta])
        return zmle
        
        

Lt, Lx, beta = 16, 16, 7.

Pot = PotentialComputation(Lt, Lx, beta)
vare = [1e-4, ]#, 5e-4, 1e-3]
for i in range(len(vare)):
    print('--------------')
    print(i)
    Pot.mchmc(vare[i], str(i))



#Pot.plot_potential()