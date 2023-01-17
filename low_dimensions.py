import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pandas as pd
import os

from sampling.sampler import Sampler as MCHMC
from sampling.sampler import ess_cutoff_crossing
from sampling.benchmark_targets import *
from sampling import grid_search



num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)


def table():


    def ESS(alpha, eps1, chains, samples):
        sampler.set_hyperparameters(alpha * np.sqrt(sampler.Target.d), eps1 * np.sqrt(sampler.Target.d))
        X, W = sampler.parallel_sample(chains, samples)

        b2_all = sampler.full_b(X, W)
        b2 = jnp.median(b2_all, axis= 1)

        no_nans = 1 - jnp.any(jnp.isnan(b2))
        cutoff_reached = b2[-1] < 0.1

        # plt.plot(bias, '.')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        return ess_cutoff_crossing(b2) * no_nans * cutoff_reached / sampler.grad_evals_per_step  # return 0 if there are nans, or if the bias cutoff was not reached



    dimensions = [2, 3, 4, 5, 10, 25, 50]
    L1, eps1 = 1.0, 0.6

            #alpha_min, alpha_max, eps(1)_min, eps(1)_max
    borders = np.array([[0.8, 5.0, 1.0, 4.0], [0.8, 5.0, 1.0, 4.0], [0.8, 4.0, 1.0, 3.0], [0.8, 4.0, 0.9, 3.0], [0.8, 3.0, 0.8, 3.0], [0.7, 1.5, 0.6, 2.0], [0.7, 1.5, 0.6, 2.0]])
                 #chains, samples
    num_samples = [[300, 300], [300, 500], [300, 600], [300, 1000], [300, 2000], [100, 2000], [50, 2000]]
    results = np.empty((len(dimensions), 3))
    for i in range(len(dimensions)):
        sampler = MCHMC(StandardNormal(d = dimensions[i]), integrator= 'LF', generalized= True)
        res = np.array(grid_search.search_wrapper(lambda a, e: ESS(a, e, num_samples[i][0],  num_samples[i][1]), L1 * borders[i, 0], L1 * L1 * borders[i, 1], eps1 * L1 * borders[i, 2], eps1 * L1 * borders[i, 3]))
        print(res)
        results[i, :] = res

    df = pd.DataFrame({'d': dimensions, 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps(1)': results[:, 2]})

    df.to_csv('data/STN_low_dimensions.csv', index=False)


dimensions = [2, 3, 4, 5, 10, 25, 50]

df = pd.read_csv('data/STN_low_dimensions.csv')

df2 = pd.read_csv('data/dimensions_dependence/kappa1g.csv')

plt.title('Standard Normal')
plt.plot(dimensions, df['ESS'], 'o', color = 'tab:orange')
plt.plot(df2['Target '], df2['ESS'], 'o', color = 'tab:blue')
plt.xlabel('d')
plt.ylabel('ESS')
plt.xscale('log')
plt.ylim(0, 5)
plt.xlim(1, 1e4)
#plt.yscale('log')
plt.savefig('ess_low_dimensions.png')
plt.show()


plt.plot(dimensions, df['alpha'] * np.sqrt(dimensions), 'o', color = 'tab:orange')
plt.plot(df2['Target '], df2['alpha'] * np.sqrt(df2['Target ']), 'o', color = 'tab:blue')
plt.xlabel('d')
plt.ylabel(r'$L(\nu)$')
plt.xscale('log')
#plt.ylim(0, 5)
plt.xlim(1, 1e4)
plt.yscale('log')
plt.savefig('L_low_dimensions.png')
plt.show()



plt.plot(dimensions, df['eps(1)'] * np.sqrt(dimensions), 'o', color = 'tab:orange')
plt.plot(df2['Target '], df2['eps'] , 'o', color = 'tab:blue')
plt.xlabel('d')
plt.ylabel(r'$\epsilon$')
plt.xscale('log')
plt.yscale('log')
#plt.ylim(0, 5)
plt.xlim(1, 1e4)
#plt.yscale('log')
plt.savefig('epsilon_low_dimensions.png')
plt.show()

