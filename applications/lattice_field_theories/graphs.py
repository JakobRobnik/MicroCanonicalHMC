import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from applications.lattice_field_theories.theories import phi4

plt.rcParams.update({'font.size': 22, 'axes.spines.right': False, 'axes.spines.top': False})
dir = os.path.dirname(os.path.realpath(__file__))
tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def gerdes_tabel1():

    side= 12
    lam = get_params(side)
    target = phi4.Theory(side, lam)

    sampler = Sampler(target, L = np.sqrt(target.d)*1.0, eps = np.sqrt(target.d) * 0.01, integrator= 'LF')

    phi, E = sampler.sample(10000, output = 'full', random_key= jax.random.PRNGKey(0))

    L = jax.vmap(target.nlogp)(phi)
    from sampling.sampler import burn_in_ending
    print(burn_in_ending(L))
    plt.plot(L)
    plt.show()


    phibar, E = sampler.sample(100000, output = 'energy', random_key= jax.random.PRNGKey(0))

    burnin = 1000
    phi, E = phibar[burnin:], E[burnin:]
    plt.plot(E)
    plt.show()
    print(np.std(E)**2/target.d)

    plt.plot(target.susceptibility2_full(phibar))
    plt.show()


def quartiles(data, axis):

    val = np.expand_dims(np.median(data, axis), axis)
    shape = np.array(np.shape(data), dtype = int)
    shape[axis] = shape[axis]//2

    upper = np.median(data[data - val > 0].reshape(shape), axis = axis)
    lower = np.median(data[data - val < 0].reshape(shape), axis = axis)

    return val, lower, upper


def gerdes_fig3():
    sides = [6, 8, 10, 12, 14]

    data = np.load(dir + '/phi4results/hmc/ground_truth/all.npy')

    chi, lower, upper = quartiles(data, axis= 2)
    #plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20, 10))

    for i in range(len(sides)):
        plt.plot(phi4.reduced_lam, chi[i], 'o', color = tab_colors[i], label = 'L = '+str(sides[i]))
        plt.fill_between(phi4.reduced_lam, lower[i], upper[i], alpha = 0.9, color = tab_colors[i])

    plt.xlabel(r'reduced $\lambda$')
    plt.ylabel(r'reduced $\chi$')
    plt.legend()
    plt.show()



def ess():
    sides = [6, ]#8, 10, 12, 14]
    plt.figure(figsize=(20, 10))

    for i in range(len(sides)):
        data = np.load(dir + '/phi4results/hmc/ess/L6.npy')
        plt.plot(phi4.reduced_lam, data[:, 0], 'o-', color=tab_colors[i], label='L = ' + str(sides[i]))

    plt.xlabel(r'reduced $\lambda$')
    plt.ylabel('ESS')
    plt.legend()
    plt.show()




def visualize_field(phi, side):

    field = np.reshape(phi, (side, side))

    plt.matshow(field)
    plt.colorbar()
    plt.show()



ess()