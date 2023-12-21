import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt

from benchmarks.benchmarks_mchmc import random_walk
#from NUTS import sample_nuts



def ground_truth(key_num):
    
    from HMC.mchmc_to_numpyro import mchmc_target_to_numpyro

    key = jax.random.PRNGKey(key_num)
    mchmc_target = Target()
    numpyro_target = mchmc_target_to_numpyro(Target)
    samples, steps, steps_warmup = sample_nuts(numpyro_target, mchmc_target, None, 10000, 100000, 20, random_key=key, progress_bar= True)

    x = np.array(samples['x'])
    xsq = jnp.square(jax.vmap(mchmc_target.transform)(x))

    second_moments = jnp.average(xsq, axis = 0)
    variance_second_moments = jnp.std(xsq, axis = 0)**2

    np.save('benchmarks/ground_truth/'+name+'/ground_truth_'+str(key_num) +'.npy', [second_moments, variance_second_moments])
    np.save('benchmarks/ground_truth/'+name+'/chain_'+str(key_num) +'.npy', x)


def join_ground_truth():
    data = np.array([np.load('benchmarks/ground_truth/'+name+'/ground_truth_'+str(i)+'.npy') for i in range(3)])

    truth = np.median(data, axis = 0)
    np.save('benchmarks/ground_truth/'+name+'/ground_truth.npy', truth)

    for i in range(3):
        bias_d = np.square(data[i, 0] - truth[0]) / truth[1]
        print(np.average(bias_d), np.max(bias_d))


def plot_hierarchical():
    x= np.load('ground_truth/'+name+'/chain_1.npy')
    print(x.shape)
    sigi = np.exp(x[:, 0])
    sigo = np.exp(x[:, 1])
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 10))
    plt.hexbin(sigi, sigo, cmap = 'cividis')
    plt.plot([0.1, ], [0.15, ], '*', color = 'gold', markersize = 20)
    plt.xlim(0.04, 0.25)
    plt.ylim(0.04, 0.25)
    plt.title('Hyper parameters')
    plt.xlabel(r'$\sigma_{\mathrm{rw}}$')
    plt.ylabel(r'$\sigma_{\mathrm{obs}}$')
    plt.xticks([0.05, 0.1, 0.15, 0.2, 0.25])
    plt.yticks([0.05, 0.1, 0.15, 0.2, 0.25])
    plt.savefig('hierarchical_posterior.png')
    plt.show()


def plot_walk():
    x = np.sort(np.load('ground_truth/' + name + '/chain_1.npy')[:, 2:], axis = 0)
    n = len(x)
    xavg = x[n//2]
    xp, xm = x[3 * n // 4], x[n // 4]

    plt.plot(Target().data, 'o', color='tab:red', label = 'data')

    plt.plot(xavg, color = 'tab:blue', label = 'posterior')
    plt.fill_between(np.arange(len(xm)), xm, xp, color = 'tab:blue', alpha = 0.3)

    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.savefig('walk_posterior.png')
    plt.show()


def map():
    chains = 10
    from optimization.adam import optimize_adam
    from scipy.optimize import minimize
    t = Target()
    def store(x):
        X.append(x[0])
        Y.append(x[1])

    x0 = jax.vmap(t.prior_draw)(jax.random.split(jax.random.PRNGKey(0), chains))
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 10))

    for i in range(chains):
        X = []
        Y = []
        #opt = minimize(t.grad_nlogp, jac = True, x0 = x0[i], method = 'BFGS', callback = store, options = {'maxiter': 5000})
        #opt = minimize(t.grad_nlogp, jac = True, x0 = x0[i], method = 'L-BFGS-B', callback = store, options = {'maxiter': 1000, 'maxcor': 50})
        opt = minimize(t.grad_nlogp, jac = True, x0 = x0[i], method = 'Newton-CG', callback = store, options = {'maxiter': 1000})

        print(len(X))
        plt.plot(X, Y, '.-', color = 'black', alpha = 0.5)
        plt.plot(X[0], Y[0], 'o', color='tab:red')
        plt.plot(X[-1], Y[-1], 'o', color='tab:blue')


    plt.plot(jnp.log(jnp.array([0.1, ])), jnp.log(jnp.array([0.15, ])), '*', color='gold', markersize=20)
    plt.xlabel(r'$\log \sigma_{\mathrm{rw}}$')
    plt.ylabel(r'$\log \sigma_{\mathrm{obs}}$')
    plt.show()


if __name__ == '__main__':
    #plott()
    #mchmc()
    #ground_truth(2)
    #plot_hierarchical()
    join_ground_truth()