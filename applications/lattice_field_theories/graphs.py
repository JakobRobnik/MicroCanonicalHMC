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

    chi_hmc = np.median(np.load(dir + '/phi4results/hmc/ground_truth/chi/all.npy'), axis= 2)
    chi_mchmc = [np.load(dir + '/phi4results/mchmc/ground_truth/chi/L'+str(s)+'.npy') for s in sides]
    #plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20, 10))

    for i in range(len(sides)):
        plt.plot(phi4.reduced_lam, chi_hmc[i], '-', lw = 3, color = tab_colors[i], label = 'L = '+str(sides[i]))
        #plt.fill_between(phi4.reduced_lam, lower[i], upper[i], alpha = 0.9, color = tab_colors[i])

        plt.plot(phi4.reduced_lam, chi_mchmc[i], 'o', markersize= 10, color=tab_colors[i])

    plt.plot([], [], 'o', markersize = 10, color ='grey', label = 'MCHMC')
    plt.plot([], [], '-', lw = 3, color ='grey', label= 'NUTS')

    plt.xlabel(r'$L^{1/\nu}(\lambda - \lambda_c) / \lambda_c$')
    plt.ylabel(r'$L^{-\gamma/\nu} \chi$')
    plt.legend(ncol= 3)
    plt.savefig('gerdes_fig3.png')
    plt.show()



def ess():
    sides = [6, 8,]# 10, 12]#, 14]
    plt.figure(figsize=(20, 10))

    for i in range(len(sides)):
        side = sides[i]
        data = np.load(dir + '/phi4results/hmc/ess/psd/L'+str(side)+'.npy')
        plt.plot(phi4.reduced_lam, data[:, 0], 'o-', color=tab_colors[i], label='L = ' + str(side))

    plt.xlabel(r'reduced $\lambda$')
    plt.ylabel('ESS')
    plt.legend()
    plt.savefig(dir+'/phi4results/ess.png')
    plt.show()




def visualize_field(phi, side):

    field = np.reshape(phi, (side, side))

    plt.matshow(field)
    plt.colorbar()
    plt.show()


def check_ground_truth():

    sides = [6, 8, 10, 12]
    side = 12
    psd = np.median(np.load(dir + '/phi4results/hmc/ground_truth/psd/L' + str(side) + '.npy'), axis =1)
    M = np.max(psd, axis = (1, 2))
    m = np.min(psd, axis=(1, 2))

    plt.figure(figsize= (15, 10))
    plt.plot(M)
    plt.yscale('log')
    plt.show()

    plt.figure(figsize= (15, 10))
    plt.plot(m)
    plt.yscale('log')
    plt.show()

    plt.figure(figsize= (15, 10))
    plt.plot(np.array(M)/np.array(m))
    plt.yscale('log')
    plt.show()
    exit()
    data = np.sort(PSD0, axis=1)
    val = (data[:, 2, :, :] + data[:, 3, :, :]) *0.5
    #lower, upper = data[:, 1, :, :], data[:, 4, :, :]
    lower, upper = data[:, 0, :, :], data[:, 5, :, :]


    rel = 0.5*(upper - lower)/val
    print(np.max(rel))
    # plt.imshow(rel[0, :, :], origin= 'lower')
    # plt.colorbar()
    # plt.show()



def grid_search_results():

    df = pd.read_csv(dir+'/phi4results/mchmc/grid_search/L6/all.npy')

    plt.figure(figsize = (14, 7))

    #plt.subplot(3, 1, 1)
    plt.title('MCHMC (L = 6)')
    plt.plot(df['reduced lambda'], df['ess'], 'o-', markersize = 7)
    plt.ylabel('ESS')
    plt.xlabel('reduced lambda')
    plt.savefig('ess_mchmc.png')
    plt.show()

    plt.subplot(3, 1, 2)
    plt.plot(df['reduced lambda'], df['alpha'], 'o-')
    plt.ylabel('alpha')

    plt.subplot(3, 1, 3)
    plt.plot(df['reduced lambda'], df['beta'], 'o-')
    plt.ylabel('beta')
    plt.xlabel('reduced lambda')
    plt.savefig(dir+'/phi4results/grid_search.png')
    plt.show()


#check_ground_truth()
grid_search_results()
#gerdes_fig3()
#ess()