import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os, sys
sys.path.append('../blackjax/')
from blackjax.adaptation.ensemble_mclmc import emaus
from benchmarks.inference_models import *


os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'
#print(len(jax.devices()), jax.lib.xla_bridge.get_backend().platform)

mesh = jax.sharding.Mesh(jax.devices(), 'chains')


def plot_trace(info1, info2, model, mclachlan):
        
    grads_per_step = 2 if mclachlan else 1 # in stage 2
    
    plt.figure(figsize= (15, 5))

    n1 = info1['step_size'].shape[0]
    ntotal = n1 + grads_per_step * jnp.sum(info2['steps_per_sample'])
    
    steps1 = jnp.arange(1, n1+1)
    steps2 = jnp.cumsum(info2['steps_per_sample']) * grads_per_step + n1
    
    def end_stage1(alpha = 0.4):
        ylim = plt.gca().get_ylim()
        plt.plot((n1+1) * np.ones(2), ylim, color = 'black', alpha = alpha)
        plt.ylim(*ylim)
    
    ### bias ###
    plt.subplot(1, 2, 1)
    plt.title('Bias')
    
    # true
    plt.plot(steps1, info1['contracted_exp_vals'][:, 1], color = 'tab:blue', label= 'average')
    plt.plot(steps1, info1['contracted_exp_vals'][:, 0], color = 'tab:red', label = 'max')
    plt.plot(steps2, info2['contracted_exp_vals'][:, 1], color = 'tab:blue')
    plt.plot(steps2, info2['contracted_exp_vals'][:, 0], color = 'tab:red')

    # equipartition
    plt.plot(steps1, info1['equi_diag'], color = 'tab:olive', label = 'diagonal equipartition')
    plt.plot(steps1, info1['equi_full'], color = 'tab:green', label = 'full rank equipartition')
    plt.plot(steps2, info2['equi_diag'], color = 'tab:olive', alpha= 0.15)
    plt.plot(steps2, info2['equi_full'], color = 'tab:green', alpha= 0.15)
    
    plt.plot([0, ntotal], jnp.ones(2) * 1e-2, '-', color = 'black')
    plt.legend()
    plt.ylabel(r'$\mathrm{bias}^2$')
    plt.xlabel('# gradient evaluations')

    #plt.ylim(1e-4, 1e2)

    plt.yscale('log')
    end_stage1()
    
    ### stepsize tuning ###
    
    plt.subplot(3, 2, 2)
    plt.title('Hyperparameters')
    plt.plot(steps1, info1['EEVPD'], '.', color='teal', label= 'observed')
    plt.plot(steps1, info1['EEVPD_wanted'], '-', color='black', alpha = 0.5, label = 'targeted')

    plt.legend()
    plt.ylabel("EEVPD")
    plt.yscale('log')
    
    ylim = plt.gca().get_ylim()
    end_stage1(1.)
    
    ax = plt.gca().twinx()  # instantiate a second axes that shares the same x-axis
    ax.plot(steps2, info2['acc_prob'], '.', color='teal')
    ax.plot(steps2, 0.7 * np.ones(steps2.shape), '-', alpha= 0.5, color='black')    
    ax.set_ylabel('acceptance probability')
    ax.tick_params(axis='y')
    
    
    plt.subplot(3, 2, 4)
    plt.plot(steps1, info1['step_size'], '.-', color='teal')
    plt.plot(steps2, info2['step_size'], '.-', color='teal')
    plt.ylabel(r"step size")
    plt.yscale('log')
    end_stage1()
    
    ### L tuning ###
    plt.subplot(3, 2, 6)
    L0 = jnp.sqrt(jnp.sum(model.E_x2))
    plt.plot(steps1, info1['L'], '.-', color='chocolate')
    plt.plot(steps2, info2['L'], '.-', color='chocolate')
    plt.plot([0, ntotal], L0 * jnp.ones(2), '-', color='black')
    end_stage1()
    plt.ylabel("L")
    plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    
    
    plt.tight_layout()
    plt.savefig('ensemble/img/' + model.name + '.png')
    plt.close()



def mainn():

    targets = [[Banana(), 100, 100],
                [Gaussian(ndims=100, eigenvalues='Gamma', numpy_seed= rng_inference_gym_icg), 500, 500],
                [GermanCredit(), 500, 500],
                [Brownian(), 500, 500],
                [ItemResponseTheory(), 500, 500],
                [StochasticVolatility(), 1000, 1000]]

    chains = 4096
    mclachlan= True
    
    key = jax.random.key(42)
    
    for i in [0, 1, 2, 3, 4, 5]:
        target, num_steps1, num_steps2 = targets[i]
        print(target.name)
        info1, info2 = emaus(target, num_steps1, num_steps2, chains, mesh, key, mclachlan= mclachlan)
        
        plot_trace(info1, info2, target, mclachlan)

   

if __name__ == '__main__':

    mainn()
    
    #shifter --image=reubenharry/cosmo:1.0 python3 -m ensemble.main