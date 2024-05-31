import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

from blackjax.adaptation.ensemble_mclmc import algorithm as emclmc
from blackjax.adaptation.ensemble_umclmc import Parallelization
from benchmarks.targets import *

num_cores = 1#jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

parallelization = Parallelization(pmap_chains= num_cores, vmap_chains= 1024//num_cores)



def plot_trace(info1, info2, target, mclachlan):
        
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
    L0 = jnp.sqrt(jnp.sum(target.second_moments))
    plt.plot(steps1, info1['L'], '.-', color='chocolate')

    plt.plot(steps2, info2['steps_per_sample'] * info2['stepsize'], '.-', color='chocolate')
    plt.plot([0, ntotal], L0 * jnp.ones(2), '-', color='black')
    end_stage1()
    plt.ylabel("L")
    plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    
    
    plt.tight_layout()
    plt.savefig('img/ensemble/adjusted/' + target.name + '.png')
    plt.close()




def run(target, num_steps, num_steps_per_sample, key):
    
    mclachlan = True
    key_sampling, key_init = jax.random.split(key)
    keys_init = jax.random.split(key_init, parallelization.num_chains).reshape(parallelization.shape)
    x_init = parallelization.pvmap(target.prior_draw)(keys_init)
    transform = parallelization.pvmap(target.transform)
    
    def observables(x):
        f = jnp.average(jnp.square(transform(x)), axis = 0)
        bsq = jnp.square(f - target.second_moments) / target.variance_second_moments    
        return jnp.array([jnp.max(bsq), jnp.average(bsq)])

    info1, info2 = emclmc(lambda x: -target.nlogp(x), num_steps, parallelization,
                          x_init, key_sampling, 
                          num_steps_per_sample, mclachlan, 
                          observables)
    

    #grads, success = error.grads_to_low_error(bias[0])
    
    plot_trace(info1, info2, target, mclachlan)
    
    # if success:
    #    print(grads.astype(int))




def mainn():

    targets = [[Banana(prior = 'prior'), 100],
                [IllConditionedGaussianGamma(prior = 'prior'), 1000],
                [GermanCredit(), 1000],
                [Brownian(), 1000],
                [ItemResponseTheory(), 1000],
                [StochasticVolatility(), 2500]]

    key = jax.random.key(42)
    num_steps_per_sample = 30
    
    for i in [5,]:
        target, num_steps = targets[i]
        print(target.name)
        run(target, num_steps, num_steps_per_sample, key)


if __name__ == '__main__':

    mainn()


