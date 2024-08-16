import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
p = PartitionSpec('chains')
mesh = jax.sharding.Mesh(jax.devices(), 'chains')

from blackjax.adaptation.ensemble_mclmc import algorithm as emclmc
from benchmarks.inference_models import *

print(len(jax.devices()), jax.lib.xla_bridge.get_backend().platform)


def plot_trace(info1, info2, model, mclachlan):
        
    grads_per_step = 2 if mclachlan else 1 # in stage 2
    
    plt.figure(figsize= (15, 5))

    n1 = info1['stepsize'].shape[0]
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
    plt.plot(steps1, info1['summary'][:, 1], color = 'tab:blue', label= 'average')
    plt.plot(steps1, info1['summary'][:, 0], color = 'tab:red', label = 'max')
    plt.plot(steps2, info2['summary'][:, 1], color = 'tab:blue')
    plt.plot(steps2, info2['summary'][:, 0], color = 'tab:red')

    # equipartition
    plt.plot(steps1, info1['equi diag'], color = 'tab:olive', label = 'diagonal equipartition')
    plt.plot(steps1, info1['equi full'], color = 'tab:green', label = 'full rank equipartition')
    plt.plot(steps2, info2['equi diag'], color = 'tab:olive', alpha= 0.15)
    plt.plot(steps2, info2['equi full'], color = 'tab:green', alpha= 0.15)
    
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
    plt.plot(steps1, info1['eevpd observed'], '.', color='teal', label= 'observed')
    plt.plot(steps1, info1['eevpd wanted'], '-', color='black', alpha = 0.5, label = 'targeted')

    plt.legend()
    plt.ylabel("EEVPD")
    plt.yscale('log')
    
    ylim = plt.gca().get_ylim()
    end_stage1(1.)
    
    ax = plt.gca().twinx()  # instantiate a second axes that shares the same x-axis
    ax.plot(steps2, info2['acc prob'], '.', color='teal')
    ax.plot(steps2, 0.7 * np.ones(steps2.shape), '-', alpha= 0.5, color='black')    
    ax.set_ylabel('acc prob')
    ax.tick_params(axis='y')
    
    
    plt.subplot(3, 2, 4)
    plt.plot(steps1, info1['stepsize'], '.-', color='teal')
    plt.plot(steps2, info2['stepsize'], '.-', color='teal')
    plt.ylabel(r"$\epsilon$")
    plt.yscale('log')
    end_stage1()
    
    ### L tuning ###
    plt.subplot(3, 2, 6)
    L0 = jnp.sqrt(jnp.sum(model.E_x2))
    plt.plot(steps1, info1['L'], '.-', color='chocolate')

    plt.plot(steps2, info2['steps_per_sample'] * info2['stepsize'], '.-', color='chocolate')
    plt.plot([0, ntotal], L0 * jnp.ones(2), '-', color='black')
    end_stage1()
    plt.ylabel("L")
    plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    
    
    plt.tight_layout()
    plt.savefig('img/' + model.name + '.png')
    plt.close()



def run(model, num_steps, num_chains, key):
    
    mclachlan = True
    key_sampling, key_init = jax.random.split(key)
    
    sample_init = shard_map(jax.vmap(model.sample_init), mesh=mesh, in_specs=p, out_specs=p)
    transform = shard_map(jax.vmap(model.transform), mesh=mesh, in_specs= p, out_specs=p)

    initial_position = sample_init(jax.random.split(key_init, num_chains))
    
    def observables(x):
        f = jnp.average(jnp.square(transform(x)), axis = 0)
        bsq = jnp.square(f - model.E_x2) / model.Var_x2
        return jnp.array([jnp.max(bsq), jnp.average(bsq)])
    
    info1, info2 = emclmc(model.logdensity_fn, num_steps, 
                          initial_position, key_sampling,
                          num_steps_per_sample= 3,
                          mclachlan= mclachlan, 
                          observables= observables, 
                          mesh= mesh
                          )
    
    plot_trace(info1, info2, model, mclachlan)
    


def mainn():

    targets = [[Banana(prior = 'prior'), 100],
                [IllConditionedGaussianGamma(prior = 'prior'), 1000],
                [GermanCredit(), 1000],
                [Brownian(), 1000],
                [ItemResponseTheory(), 1000],
                [StochasticVolatility(), 2000]]

    chains = 256
    
    key = jax.random.key(42)
    
    for i in [0,]:
        target, num_steps = targets[i]
        print(target.name)
        run(target, num_steps, chains, key)

   

if __name__ == '__main__':

    mainn()