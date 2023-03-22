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
from sampling.grid_search import search_wrapper

dir = os.path.dirname(os.path.realpath(__file__))



def plot_mixing():
    """show the charge as a function of number of gradient calls"""

    side = 16
    beta = 7.0
    target = u1.Theory(side, beta)
    alpha = 1.0
    beta_eps= 0.2
    sampler = Sampler(target, L= np.sqrt(target.d) * alpha, eps= np.sqrt(target.d) * beta_eps, integrator='LF')
    #sampler.tune_hyperparameters(dialog= True)

    Q, E, burnin = sampler.sample(100000, output= 'energy')
    Q = Q[burnin:]
    E = E[burnin:]

    #topo_charge = jax.vmap(target.topo_charge)(x)
    print(np.std(E)**2/target.d)
    plt.plot(Q, '.')
    plt.xlabel("gradient evaluations")
    plt.ylabel('topological charge')
    #plt.savefig('albergo_fig1.png')
    plt.show()


def topo_sus(side, beta):
    """compute topological susceptibility"""

    target = u1.Theory(side, beta)
    alpha = 1.0
    beta_eps= 0.03
    sampler = Sampler(target, L= np.sqrt(target.d) * alpha, eps= np.sqrt(target.d) * beta_eps, integrator='LF')
    #sampler.tune_hyperparameters(dialog= True)

    samples = 1000000
    Q, burnin = sampler.sample(samples, num_chains= num_cores, output='normal')
    mask = burnin < samples//2 #reliable chains
    if np.sum(mask) < num_cores//2: #most chains are unreliable
        raise ValueError('Too short sampling')

    burnin= np.max(burnin[mask])
    Q = Q[mask, burnin:, 0]
    # charge = np.concatenate(Q)
    # q_range = [q for q in range(-5, 6)]
    # prob = np.array([np.sum(np.abs(charge-q) < 1e-3) for q in q_range]) / len(charge)
    # plt.plot(q_range, prob, 'o')
    # plt.show()
    return np.average(np.square(Q), axis = 1) / side**2


def plot_topo_sus():
    """show topological susceptibility as a function of beta"""

    chi0 = np.loadtxt('theories/topo_susceptibility_ground_truth_L8.csv')
    beta = np.arange(1, 11)

    chi = np.array([topo_sus(8, b) for b in beta])

    plt.errorbar(beta, np.average(chi, axis = 1), yerr = np.std(chi, axis = 1), fmt = 'o-', capsize= 2, label= 'MCHMC')
    plt.plot(beta, chi0, 'o-', label= 'ground truth')
    plt.plot(beta, u1.thermodynamic_ground_truth(beta), ':',  color = 'black', alpha = 0.5, label = 'TD limit')

    plt.xlabel(r'$\beta$')
    plt.ylabel('topological susceptibility')
    plt.legend()
    plt.savefig('topo_sus_sin_definition.png')
    plt.show()



side = 16
beta = 7.0
target = u1.Theory(side, beta)
alpha = 1.0
beta_eps= 0.1
sampler = Sampler(target, L= np.sqrt(target.d) * alpha, eps= np.sqrt(target.d) * beta_eps, integrator='LF')
#sampler.tune_hyperparameters(dialog= True)

Q, E, burnin = sampler.sample(100000, output= 'energy')
Q = Q[burnin:]
E = E[burnin:]

plt.hist(Q, bins = 10)
plt.show()



#plot_topo_sus()
plot_mixing()
#print(np.average(topo_sus(8, 1.0)))