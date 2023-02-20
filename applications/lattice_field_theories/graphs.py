import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from applications.lattice_field_theories.theories import phi4

plt.rcParams.update({'font.size': 22, 'axes.spines.right': False, 'axes.spines.top': False})
dir = os.path.dirname(os.path.realpath(__file__))
tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def gerdes_fig3():
    sides = [6, 8, 10, 12]

    data = np.load(dir + '/phi4results/ground_truth.npy')
    print(data[-1, 0, :])
    chi = np.median(data, axis = 2)
    upper = np.median((data - chi[:, :, None] > 0), axis = 2) + chi
    lower = np.median((data - chi[:, :, None] < 0), axis = 2) + chi

    print(upper - lower)

    #plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20, 10))

    for i in range(len(sides)):
        plt.plot(phi4.reduced_lam, chi[i], 'o-', color = tab_colors[i], label = 'L = '+str(sides[i]))
        plt.fill_between(phi4.reduced_lam, lower[i], upper[i], alpha = 0.5, color = tab_colors[i])

    plt.xlabel(r'reduced $\lambda$')
    plt.ylabel(r'reduced $\chi$')
    plt.legend()
    plt.show()

def ess():
    plt.subplot(2, 1, 2)
    ESS = np.load('ESS_L'+str(side) + '.npy')
    plt.plot(data['reduced lam'], ESS, 'o:')
    plt.yscale('log')
    plt.show()



def visualize_field(phi, side):

    field = np.reshape(phi, (side, side))

    plt.matshow(field)
    plt.colorbar()
    plt.show()


gerdes_fig3()