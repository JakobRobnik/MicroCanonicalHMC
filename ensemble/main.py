import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import pandas as pd
import os, sys
sys.path.append('../blackjax/')
from blackjax.adaptation.ensemble_mclmc import emaus
from blackjax.mcmc.integrators import velocity_verlet_coefficients, mclachlan_coefficients, omelyan_coefficients
from benchmarks.inference_models import *
from ensemble.grid_search import do_grid
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'
#print(len(jax.devices()), jax.lib.xla_bridge.get_backend().platform)
mesh = jax.sharding.Mesh(jax.devices(), 'chains')


# models to solve
targets = [[Banana(), 100, 300],
            [Gaussian(ndims=100, eigenvalues='Gamma', numpy_seed= rng_inference_gym_icg), 500, 500],
            [GermanCredit(), 500, 500],
            [Brownian(), 500, 500],
            [ItemResponseTheory(), 500, 500],
            [StochasticVolatility(), 1000, 1000]]



def find_crossing(n, bias, cutoff):
    """the smallest M such that bias[m] < cutoff for all m >= M. Returns n[M]"""

    indices = jnp.argwhere(bias > cutoff)
    M= jnp.max(indices)+1
    
    if M == len(bias):
        return jnp.inf
    else: 
        return n[M]

def plot_convergence_metrics(steps1, info1, file_name):
    
    ### entropy ###
    H = info1['entropy']
    H -= H[-1]
    
    mask = H < 0.
    plt.plot(steps1[~mask], H[~mask], '.-', color= 'tab:red', label= 'positive')
    plt.plot(steps1[mask], -H[mask], '.-', color= 'tab:green', label= 'negative')
    
    plt.yscale('log')
    plt.ylabel('|Entropy|')    
    plt.xlabel('# gradient evaluations')
    
    plt.savefig(file_name+ '_entropy.png')
    plt.close()
    

def plot_trace(info1, info2, model, grads_per_step, acc_prob, dir):
            

    n1 = info1['step_size'].shape[0]
    
    ntotal = n1 + grads_per_step * jnp.sum(info2['steps_per_sample'])
    
    steps1 = jnp.arange(1, n1+1)
    steps2 = jnp.cumsum(info2['steps_per_sample']) * grads_per_step + n1
    steps = np.concatenate((steps1, steps2))

    plot_convergence_metrics(steps1, info1, dir + model.name)
    
    
    plt.figure(figsize= (15, 5))

    def end_stage1():
        ax = plt.gca()
        ylim = ax.get_ylim()
        lw = ax.spines['bottom'].get_linewidth()
        color = ax.spines['bottom'].get_edgecolor()
        plt.plot((n1+1) * np.ones(2), ylim, color= color, lw= lw)
        plt.ylim(*ylim)
    
    ### bias ###
    plt.subplot(1, 2, 1)
    #plt.title('Bias')
    
    # true
    bias = np.concatenate((info1['bias'], info2['bias']))
    n = [find_crossing(steps, bias[:, i], 0.01) for i in range(2)]
    plt.plot(steps, bias[:, 1], color = 'tab:blue', label= 'average')
    plt.plot(steps, bias[:, 0], color = 'tab:red', label = 'max')

    # equipartition
    plt.plot(steps1, info1['equi_diag'], '.', color = 'tab:olive', label = 'diagonal equipartition')
    plt.plot(steps1, info1['equi_full'], '.', color = 'tab:green', label = 'full rank equipartition')
    plt.plot(steps2, info2['equi_diag'], '.-', color = 'tab:olive', alpha= 0.3)
    plt.plot(steps2, info2['equi_full'], '.-', color = 'tab:green', alpha= 0.3)
    
    
    # relative fluctuations
    plt.plot(steps1, info1['r_avg'], '--', color = 'tab:blue')
    plt.plot(steps1, info1['r_max'], '--', color = 'tab:red', label = 'fluctuations')
    
    # pathfinder
    pf= pd.read_csv('ensemble/submission/pathfinder_convergence.csv', sep= '\t')
    pf_grads_all = np.array(pd.read_csv('ensemble/submission/pathfinder_cost.csv', sep= '\t')[model.name])
    pf_grads = np.max(pf_grads_all) # in an ensemble setting we have to wait for the slowest chain

    pf = pf[pf['name'] == model.name]
    pf_bavg, pf_bmax = pf[['bavg', 'bmax']].to_numpy()[0]

    if pf_bavg > 2 * np.max([np.max(bias), np.max(info1['equi_full']), np.max(info1['equi_diag'])]): # pathfinder has not converged
        plt.plot([], [], '*', color= 'grey', label= 'Pathfinder: not converged')
    else:
        plt.plot([pf_grads, ], [pf_bavg, ], '*', color= 'tab:blue')
        plt.plot([pf_grads, ], [pf_bmax, ], '*', color= 'tab:red')
        plt.plot([], [], '*', color= 'grey', label= 'Pathfinder')
    
    plt.plot([0, ntotal], jnp.ones(2) * 1e-2, '-', color = 'black')
    plt.legend()
    plt.ylabel(r'$\mathrm{bias}^2$')
    plt.xlabel('# gradient evaluations')

    #plt.ylim(1e-4, 1e2)

    plt.yscale('log')
    end_stage1()
    
    ### stepsize tuning ###
    
    plt.subplot(3, 2, 2)
    #plt.title('Hyperparameters')
    plt.plot(steps1, info1['EEVPD'], '.', color='teal', label= 'observed')
    plt.plot(steps1, info1['EEVPD_wanted'], '-', color='black', alpha = 0.5, label = 'targeted')

    plt.legend(loc=4)
    plt.ylabel("EEVPD")
    plt.yscale('log')
    
    ylim = plt.gca().get_ylim()
    end_stage1()
    
    ax = plt.gca().twinx()  # instantiate a second axes that shares the same x-axis
    ax.spines['right'].set_visible(True)
    ax.plot(steps2, info2['acc_prob'], '.', color='teal')
    ax.plot([steps1[-1], steps2[-1]], acc_prob * np.ones(2), '-', alpha= 0.5, color='black')    
    ax.set_ylabel('acc prob')
    ax.tick_params(axis='y')
    
    
    plt.subplot(3, 2, 4)
    plt.plot(steps1, info1['step_size'], '.', color='teal')
    plt.plot(steps2, info2['step_size'], '.', color='teal')
    plt.ylabel(r"step size")
    plt.yscale('log')
    end_stage1()
    
    ### L tuning ###
    plt.subplot(3, 2, 6)
    L0 = jnp.sqrt(jnp.sum(model.E_x2))
    plt.plot(steps1, info1['L'], '.', color='chocolate')
    plt.plot(steps2, info2['L'], '.', color='chocolate')
    #plt.plot([0, ntotal], L0 * jnp.ones(2), '-', color='black')
    end_stage1()
    plt.ylabel("L")
    plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    
    
    plt.tight_layout()
    plt.savefig(dir + model.name + '.png')
    plt.close()

    return n


def _main(dir,
          chains= 4096, 
          alpha = 1., bias_type= 3, C= 0.1, power= 3./8., early_stop=1, 
          diagonal_preconditioning= 1, integrator= 2, steps_per_sample= 10, acc_prob= 0.9):
    
    # algorithm settings
    key = jax.random.key(42)
    integrator_coefficients= [velocity_verlet_coefficients, mclachlan_coefficients, omelyan_coefficients][integrator]
    grads_per_step = len(integrator_coefficients) // 2

    results = {}
    for t in targets:
        target, num_steps1, num_steps2 = t
        #print(target.name)
        info1, info2 = emaus(target, num_steps1, num_steps2, chains, mesh, key, 
                             alpha= alpha, bias_type= bias_type, C= C, power= power, early_stop= early_stop,
                             diagonal_preconditioning= diagonal_preconditioning, integrator_coefficients= integrator_coefficients, steps_per_sample= steps_per_sample, acc_prob= acc_prob) # run the algorithm
        
        
        result = plot_trace(info1, info2, target, grads_per_step, acc_prob, dir) # do plots and compute the results
        results['grads_to_low_bmax_' + target.name] = result[0]
        results['grads_to_low_bavg_' + target.name] = result[1]
    
    return results


mylogspace = lambda a, b, num, decimals=3: np.round(np.logspace(np.log10(a), np.log10(b), num), decimals)

grid = lambda params, fixed_params= None: do_grid(_main, params, fixed_params=fixed_params)



if __name__ == '__main__':
    
    _main('ensemble/img/')
    # grid({'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
    #       'power': [3./4., 3./8.]})
    
    #grid({'alpha': mylogspace(0.1, 10., 5)})
    

    #shifter --image=reubenharry/cosmo:1.0 python3 -m ensemble.main
    

# TODO for package release:
# - make stage 1 a while loop
# - equi full should only be computed optionally
# - remove true bias calculations
# - test on multiple nodes
# - test on other targets
# - nan handling (check also that initial condition is not nan)