
import os, sys
sys.path.append('../blackjax/')
sys.path.append('../')
sys.path.append('./')
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import pandas as pd
import os, sys
sys.path.append('../blackjax/')
from blackjax.adaptation.ensemble_mclmc import emaus
from blackjax.mcmc.integrators import velocity_verlet_coefficients, mclachlan_coefficients, omelyan_coefficients
from benchmarks.inference_models import *
from ensemble.grid_search import do_grid
from ensemble.extract_image import imported_plot, third_party_methods
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'
#print(len(jax.devices()), jax.lib.xla_bridge.get_backend().platform)
mesh = jax.sharding.Mesh(jax.devices(), 'chains')

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.size'] = 16


rng_key_int = int(sys.argv[1])

targets = [[Banana(), 100, 300],
            [Gaussian(ndims=100, eigenvalues='Gamma', numpy_seed= rng_inference_gym_icg), 500, 500],
            [GermanCredit(), 500, 400],
            [Brownian(), 500, 500],
            [ItemResponseTheory(), 500, 3000],#500], # change to 3000 for M dependence plot
            [StochasticVolatility(), 800, 3000]#1500]# change to 3000 for M dependence plot
            ][:1]

annotations = False
    
def find_crossing(n, bias, cutoff):
    """the smallest M such that bias[m] < cutoff for all m >= M. Returns n[M]"""

    indices = jnp.argwhere(bias < cutoff)
    #M= jnp.max(indices)+1
    if len(indices) == 0:
        return jnp.inf
    else:
        return n[jnp.min(indices)]


def plot_trace_sv(info1, info2, grads_per_step):
            
    
    n1 = info1['step_size'].shape[0]
    
    ntotal = n1 + grads_per_step * jnp.sum(info2['steps_per_sample'])
    
    steps1 = jnp.arange(1, n1+1)
    steps2 = jnp.cumsum(info2['steps_per_sample']) * grads_per_step + n1
    steps = np.concatenate((steps1, steps2))

    bias = np.concatenate((info1['bias'], info2['bias']))
    n = [find_crossing(steps, bias[:, i], 0.01) for i in range(2)]
        
    plt.figure(figsize= (8, 6))

    def end_stage1():
        ax = plt.gca()
        ylim = ax.get_ylim()
        lw = ax.spines['bottom'].get_linewidth()
        color = ax.spines['bottom'].get_edgecolor()
        plt.plot((n1+1) * np.ones(2), ylim, color= color, lw= lw)
        plt.ylim(*ylim)
    
    ### bias ###
    
    # true
    #plt.plot(steps, bias[:, 1], color = 'tab:blue')
    plt.plot(steps, bias[:, 0], lw= 3, color = 'tab:red', label= 'EMAUS')
    
    for (method, color) in third_party_methods:
        spline_loc = 'ensemble/third_party/' + method + '.npz'
        plt.plot(steps+100, imported_plot(steps, spline_loc), '-',  color= 'tab:'+color, label= method)
        plt.ylim(1e-3, 1e4)
    
    plt.plot([0, ntotal], jnp.ones(2) * 1e-2, '-', color = 'black')
    plt.legend(loc=3)
    plt.ylabel(r'$\mathrm{bias}^2_{\mathrm{max}}$')
    plt.xlabel('# gradient evaluations')
    plt.xlim(0, ntotal)
    plt.yscale('log')
    end_stage1()
    
    plt.tight_layout()
    plt.savefig('ensemble/submission/StochasticVolatility.png')
    plt.close()




def plot_trace(info1, info2, model, grads_per_step, acc_prob, dir):
            
    
    n1 = info1['step_size'].shape[0]
    
    ntotal = n1 + grads_per_step * jnp.sum(info2['steps_per_sample'])
    
    steps1 = jnp.arange(1, n1+1)
    steps2 = jnp.cumsum(info2['steps_per_sample']) * grads_per_step + n1
    steps = np.concatenate((steps1, steps2))

    bias = np.concatenate((info1['bias'], info2['bias']))
    n = [find_crossing(steps, bias[:, i], 0.01) for i in range(2)]
        
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
    plt.plot([], [], '-', color= 'tab:red', label= 'max')
    plt.plot([], [], '-', color= 'tab:blue', label= 'average')
    
    # true
    plt.plot(steps, bias[:, 1], color = 'tab:blue')
    plt.plot(steps, bias[:, 0], lw= 3, color = 'tab:red')
    plt.plot([], [], color='tab:gray', label= r'Second moments $b_t^2[x_i^2]$')

    # equipartition
    plt.plot(steps1, info1['equi_diag'], '.', color = 'tab:blue', alpha= 0.4)
    plt.plot([], [], '.', color= 'tab:gray', alpha= 0.4, label = r'Equipartition $B_t^2$')
    #plt.plot(steps1, info1['equi_full'], '.', color = 'tab:green', alpha= 0.4, label = 'full rank equipartition')
    #plt.plot(steps2, info2['equi_diag'], '.', color = 'tab:blue', alpha= 0.3)
    #plt.plot(steps2, info2['equi_full'], '.-', color = 'tab:green', alpha= 0.3)
    
    
    # relative fluctuations
    plt.plot(steps1, info1['r_avg'], '--', color = 'tab:blue')
    plt.plot(steps1, info1['r_max'], '--', color = 'tab:red')
    plt.plot([], [], '--', color = 'tab:gray',label = r'Fluctuations $\delta_t^2[x_i^2]$')

    # pathfinder
    pf= pd.read_csv('ensemble/submission/pathfinder_convergence.csv', sep= '\t')
    pf_grads_all = np.array(pd.read_csv('ensemble/submission/pathfinder_cost.csv', sep= '\t')[model.name])
    pf_grads = np.max(pf_grads_all) # in an ensemble setting we have to wait for the slowest chain

    pf = pf[pf['name'] == model.name]
    pf_bavg, pf_bmax = pf[['bavg', 'bmax']].to_numpy()[0]

    if pf_bavg < 2 * np.max([np.max(bias), np.max(info1['equi_full']), np.max(info1['equi_diag'])]): # pathfinder has not converged
        
        plt.plot([pf_grads, ], [pf_bavg, ], '*', color= 'tab:blue')
        plt.plot([pf_grads, ], [pf_bmax, ], '*', color= 'tab:red')
        plt.plot([], [], '*', color= 'tab:gray', label= 'Pathfinder')
    
    if annotations:
        plt.text(steps1[len(steps1)//2], 7e-4, 'Unadjusted', horizontalalignment= 'center')
        plt.text(steps2[len(steps2)//2], 7e-4, 'Adjusted', horizontalalignment= 'center')
        
  
    plt.ylim(2e-4, 2e2)
    
    plt.plot([0, ntotal], jnp.ones(2) * 1e-2, '-', color = 'black')
    plt.legend(fontsize= 12)
    plt.ylabel(r'$\mathrm{bias}^2$')
    plt.xlabel('# gradient evaluations')
    plt.xlim(0, ntotal)
    plt.yscale('log')
    end_stage1()
    
    ### stepsize tuning ###
    
    plt.subplot(3, 2, 2)
    #plt.title('Hyperparameters')
    plt.plot(steps1, info1['EEVPD'], '.', color='tab:orange')
    plt.plot([], [], '-', color= 'tab:orange', label= 'observed')
    plt.plot(steps1, info1['EEVPD_wanted'], '-', color='black', alpha = 0.5, label = 'targeted')

    plt.legend(loc=4, fontsize=10)
    plt.ylabel("EEVPD")
    plt.yscale('log')
    
    ylim = plt.gca().get_ylim()
    end_stage1()
    
    ax = plt.gca().twinx()  # instantiate a second axes that shares the same x-axis
    ax.spines['right'].set_visible(True)
    ax.plot(steps2, info2['acc_prob'], '.', color='tab:orange')
    ax.plot([steps1[-1], steps2[-1]], acc_prob * np.ones(2), '-', alpha= 0.5, color='black')    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('acc prob')
    ax.tick_params(axis='y')
        
    plt.subplot(3, 2, 4)
    plt.plot(steps1, info1['step_size'], '.', color='tab:orange')
    plt.plot(steps2, info2['step_size'], '.', color='tab:orange')
    plt.ylabel(r"step size")
    #plt.yscale('log')
    end_stage1()
    
    ### L tuning ###
    plt.subplot(3, 2, 6)
    L0 = jnp.sqrt(jnp.sum(model.E_x2))
    plt.plot(steps1, info1['L'], '.', color='tab:green')
    plt.plot(steps2, info2['L'], '.', color='tab:green')
    #plt.plot([0, ntotal], L0 * jnp.ones(2), '-', color='black')
    end_stage1()
    plt.ylabel("L")
    #plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    
    
    plt.tight_layout()
    plt.savefig(dir + model.name + '.png')
    plt.close()

    return n


def _main(dir,
          chains= 4096, 
          alpha = 1.9, bias_type= 3, C= 0.1, power= 3./8., # unadjusted parameters
          early_stop=1, r_end= 1e-2, # switch parameters
          diagonal_preconditioning= 1, integrator= 0, steps_per_sample= 15, acc_prob= None # adjusted parameters
          ):
    
    # algorithm settings
    key = jax.random.split(jax.random.key(42), 100)[rng_key_int]
    integrator_coefficients= [None, velocity_verlet_coefficients, mclachlan_coefficients, omelyan_coefficients][integrator]

    results = {}
    for t in targets:
        target, num_steps1, num_steps2 = t
        #print(target.name)
        #vec = (target.R.T)[[0, -1], :]
        
        
        info1, info2, grads_per_step, _acc_prob = emaus(target, num_steps1, num_steps2, chains, mesh, key, 
                             alpha= alpha, bias_type= bias_type, C= C, power= power, early_stop= early_stop, r_end= r_end,
                             diagonal_preconditioning= diagonal_preconditioning, integrator_coefficients= integrator_coefficients, steps_per_sample= steps_per_sample, acc_prob= acc_prob,
                             ensemble_observables= lambda x: x
                             #ensemble_observables = lambda x: vec @ x
                             ) # run the algorithm
        
        X = np.concatenate((info1[1], info2[1]))
        np.save('ensemble/movie/samples_' + target.name + '.npy', X)
        
        
        result = plot_trace(info1, info2, target, grads_per_step, _acc_prob, dir) # do plots and compute the results
        #plot_trace_sv(info1, info2, grads_per_step)
        #return
    
        results['grads_to_low_bmax_' + target.name] = result[0]
        results['grads_to_low_bavg_' + target.name] = result[1]
    
    return results


mylogspace = lambda a, b, num, decimals=3: np.round(np.logspace(np.log10(a), np.log10(b), num), decimals)

grid = lambda params, fixed_params= None, verbose= True, extra_word= '': do_grid(_main, params, fixed_params=fixed_params, verbose= verbose, extra_word= extra_word)



if __name__ == '__main__':
    
    results = _main('ensemble/img/')
    # print(results)
    # print('C_power')
    # grid({'C': mylogspace(0.001, 3, 6),
    #        'power': [3./4., 3./8.]})
    
    # grid({'integrator': [2, 3],
    #        'steps_per_sample': np.logspace(np.log10(5), np.log10(30), 10).astype(int)})
    
    # print('alpha')
    # grid({'alpha': mylogspace(1, 4., 6)})
    
    
    grid({'chains': [2**k for k in range(6, 13)]}, extra_word= str(rng_key_int))
    
    # print('r_end')
    # grid({'r_end': mylogspace(1e-3, 1e-1, 6)})
    
    #grid({'steps_per_sample': np.logspace(np.log10(5), np.log10(30), 10).astype(int)})
    

