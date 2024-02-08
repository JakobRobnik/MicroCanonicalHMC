import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from mclmc.ensemble import Sampler
from blackjax.adaptation.ensemble_mclmc import algorithm as emclmc

from benchmarks.targets import *
from benchmarks import error


#num_cores = jax.local_device_count()
#print(num_cores, jax.lib.xla_bridge.get_backend().platform)



def plot_trace(info, bias, target):
        
    end_stage1 = lambda: None#plt.plot(steps1 * np.ones(2), plt.gca().get_ylim(), color = 'grey', alpha = 0.2)
    
    plt.figure(figsize= (15, 5))

    n1 = info['L'].shape[0]
    #steps = jnp.concatenate((jnp.arange(n1) * self.n, n1 * self.n + jnp.arange(len(eps)-n1) * self.n * self.grad_evals_per_step))
    steps = jnp.arange(1, n1+1)
    
    
    ### bias ###
    plt.subplot(1, 2, 1)
    plt.title('bias')
    
    # true
    plt.plot(steps, bias[1], color = 'tab:blue', label= 'average')
    plt.plot(steps, bias[0], color = 'tab:red', label = 'max')
    

    # equipartition
    plt.plot(steps, info['equi diag'], color = 'tab:olive', label = 'diagonal equipartition')
    plt.plot(steps, info['equi full'], '--', color = 'tab:green', label = 'full rank equipartition')
    
    plt.plot(steps, jnp.ones(steps.shape) * 1e-2, '-', color = 'black')
    plt.legend()
    plt.ylabel(r'$\mathrm{bias}^2$')
    plt.ylim(1e-4, 1e2)
    plt.yscale('log')
    end_stage1()
    
    ### stepsize tuning ###
    plt.subplot(3, 2, 2)
    plt.plot(steps, info['eevpd observed'], '.', color='tab:blue', alpha = 0.5, label= 'observed')
    plt.plot(steps, info['eevpd wanted'], '.-', color='purple', label = 'targeted')
    plt.ylabel("EEVPD")
    plt.yscale('log')
    end_stage1()
    plt.legend()
    
    plt.subplot(3, 2, 4)
    plt.plot(steps, info['stepsize'], '.-', color='royalblue')
    plt.ylabel(r"$\epsilon$")
    plt.yscale('log')
    end_stage1()
    
    ### L tuning ###
    plt.subplot(3, 2, 6)
    L0 = jnp.sqrt(jnp.sum(target.second_moments))
    plt.plot(steps, info['L'], '.-', color='tab:orange')
    plt.plot(steps, L0 * jnp.ones(steps.shape), '-', color='black')
    end_stage1()
    plt.ylabel("L")
    #plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    plt.tight_layout()
    plt.savefig('img/ensemble/adjusted/' + target.name + '.png')
    plt.close()



def run(target, num_steps, chains, key):
    
    key_sampling, key_init = jax.random.split(key)
    
    x_init = jax.vmap(target.prior_draw)(jax.random.split(key_init, chains))
    observables = lambda x: jnp.square(target.transform(x))
    info = emclmc(lambda x: -target.nlogp(x), num_steps, x_init, chains, key_sampling, observables)    
    
    return 
    bias = [error.err(target.second_moments, target.variance_second_moments, contract)(info['expected vals']) for contract in [jnp.max, jnp.average]]
    
    grads, success = error.grads_to_low_error(bias[0])
    
    plot_trace(info, bias, target)
    
    if success:
       print(grads.astype(int))




def run_old(target, num_steps, chains, key):
    
    sampler = Sampler(target, chains, diagonal_preconditioning = False, alpha = 1.)
    x = sampler.sample(num_steps)
    

def mainn():


    targets = [[Banana(prior = 'prior'), 100],
                [IllConditionedGaussianGamma(prior = 'prior'), 500],
                [GermanCredit(), 400],
                [Brownian(), 500],
                [ItemResponseTheory(), 700],
                [StochasticVolatility(), 2000]]

    chains = 1024

    key = jax.random.PRNGKey(42)
    
    for i in [0, ]:
        target, num_steps = targets[i]
        print(target.name)
        run(target, num_steps, chains, key)

   
    

if __name__ == '__main__':
    
    mainn()

