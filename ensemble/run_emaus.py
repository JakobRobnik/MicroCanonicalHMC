import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import pandas as pd
import os, sys
sys.path.append('../blackjax/')
sys.path.append('../')
sys.path.append('./')
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


rng_key_int = 1 # int(sys.argv[1])

targets = [[Banana(), 100, 300],
            # [Gaussian(ndims=100, eigenvalues='Gamma', numpy_seed= rng_inference_gym_icg), 500, 500],
            # [GermanCredit(), 500, 400],
            # [Brownian(), 500, 500],
            # [ItemResponseTheory(), 500, 3000],#500], # change to 3000 for M dependence plot
            # [StochasticVolatility(), 800, 3000]#1500]# change to 3000 for M dependence plot
            ]

annotations = False
    


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
        
        # X = np.concatenate((info1[1], info2[1]))
        # print(info1[0].shape)
        X = info2[1]
        print(X.shape)
        print(X[0][0], X[-1][-1], "result")

        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        
        # np.save('ensemble/movie/samples_' + target.name + '.npy', X)
        
        # plot the results
        import seaborn as sns
        sns.scatterplot(x= X[:, 0], y= X[:, 1], alpha= 0.1)
        # save plot
        plt.savefig(dir + target.name + '.png')
        # saved to
        print(dir + target.name + '.png')
        
        # result = plot_trace(info1, info2, target, grads_per_step, _acc_prob, dir) # do plots and compute the results
        # #plot_trace_sv(info1, info2, grads_per_step)
        # #return
    
        # results['grads_to_low_bmax_' + target.name] = result[0]
        # results['grads_to_low_bavg_' + target.name] = result[1]
    
    return results


mylogspace = lambda a, b, num, decimals=3: np.round(np.logspace(np.log10(a), np.log10(b), num), decimals)

grid = lambda params, fixed_params= None, verbose= True, extra_word= '': do_grid(_main, params, fixed_params=fixed_params, verbose= verbose, extra_word= extra_word)



if __name__ == '__main__':
    
    results = _main('ensemble/img/')


# TODO for package release:
# - update re main branch
# - make stage 1 a while loop
# - remove true bias calculations
# - test on multiple nodes
# - test on other targets, initializations
# - nan handling (check also that initial condition is not nan)