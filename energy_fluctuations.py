import os
num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mchmc
from benchmark_targets import *
import german_credit

import jax
import jax.numpy as jnp

tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def moving_variance(x, w, width):
    """smoothing by a moving average method:
        smooth[i] = average(x[i - half_width : i + half_width + 1])
    """

    xsq = jnp.square(x)

    def move_by_one(state, useless):
        W, F1, F2, add, drop = state
        W_bare = W - w[drop]
        W_new = W_bare + w[add]
        x1 = (W * F1 + w[add] * x[add] - w[drop] * x[drop]) / W_new
        x2 = (W * F2 + w[add] * xsq[add] - w[drop] * xsq[drop]) / W_new

        var = x2 - jnp.square(x1)
        return (W_new, F1, F2, add + 1, drop + 1), var

    Wi = jnp.sum(w[:width])
    F1i = jnp.average(x[:width], weights = w[:width])
    F2i = jnp.average(xsq[:width], weights=w[:width])
    var = jax.lax.scan(move_by_one, init = (Wi, F1i, F2i, width, 0), xs = None, length= len(x) - width)[1]

    return var



def variance_with_weights(x, w):

    avg = np.average(x, weights= w)
    return np.average(np.square(x - avg), weights= w)


def energy_variance(target, L, eps):
    burn_in, samples = 1000, 1000
    width = 61
    sampler = mchmc.Sampler(target, L, eps, integrator='LF', generalized=True)

    Xall, Wall, Eall = sampler.parallel_sample(num_cores, burn_in + samples, monitor_energy=True, num_cores= num_cores)
    var_chains= np.empty(num_cores)
    fullvar_chains= np.empty(num_cores)

    for chain in range(num_cores):
        X, W, E = Xall[chain, burn_in:, :], Wall[chain, burn_in:], Eall[chain, burn_in:]
        var_t = moving_variance(E, W, 61)
        plt.plot(var_t)
        var_chains[chain] = np.median(var_t)
        fullvar_chains[chain] = variance_with_weights(E, W)
    plt.show()
    varE = np.median(var_chains) / target.d
    fullvarE = np.median(fullvar_chains) / target.d

    return varE, fullvarE



def benchmarks():

    # targets
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(d=50, mu1=0.0, mu2=8.0, sigma1=1.0, sigma2=1.0, f=0.2), Rosenbrock(d=36), Funnel(d=20), german_credit.Target(), StochasticVolatility()]

    #optimal settings
    file = 'submission/Table generalized_LF_q=0.csv'
    results = pd.read_csv(file)
    eps_all = np.array(results['eps'])
    alpha_all = np.array(results['alpha'])

    i_target = 0
    target = targets[i_target]
    eps_opt, L_opt = eps_all[i_target], alpha_all[i_target] * np.sqrt(target.d)

    eps = eps_opt * np.logspace(-1, np.log10(2), 3)
    var, fullvar = np.empty(len(eps)), np.empty(len(eps))
    energy_variance(target, L_opt, eps_opt)

    for i in range(len(eps)):
        print(i)
        var[i], fullvar[i] = energy_variance(target, L_opt, eps[i])

    plt.plot(eps / eps_opt, var, 'o:')
    plt.plot(eps / eps_opt, fullvar, 'o:')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\epsilon / \widehat{\epsilon}$')
    plt.ylabel('Var[E] / d')
    plt.show()


    #results['energy variance'] = varE
    #results.to_csv('submission/Table generalized_LF_q=0_energy.csv', index = False)



def epsilon_dependence():
    num_target = 5
    name = (['STN', 'STN1000', 'ICG', 'rosenbrock', 'funnel', 'german'])[num_target]
    target = ([StandardNormal(d = 100), StandardNormal(d = 1000), IllConditionedGaussian(100, 100.0), Rosenbrock(d = 36), Funnel(d=20), german_credit.Target()])[num_target]
    epsilon = ([np.logspace(np.log10(1), np.log10(15), 15),
                np.logspace(np.log10(1), np.log10(15), 15) * np.sqrt(10),
                np.logspace(np.log10(0.1), np.log10(5), 15),
                np.logspace(np.log10(0.01), np.log10(1.2), 15) / np.sqrt(3),
                np.logspace(np.log10(0.01), np.log10(3), 15),
                np.logspace(np.log10(0.01), np.log10(1.2), 15)])[num_target]

    num_steps, burn_in = 3000, 2000
    Evar = np.empty((len(epsilon), 3))
    L = 1.0 * jnp.sqrt(target.d) * np.sqrt(np.average(target.variance))
    #L = np.inf
    sampler = mchmc.Sampler(target, L, 1.0, 'LF', True)

    for i in range(len(epsilon)):
        print(i)
        sampler.eps = epsilon[i]
        #bias[i, :] = sampler.sample('prior', num_steps, L, key, generalized= False, integrator= 'LF', ess= True)
        X, W, E = sampler.parallel_sample(10, num_steps, monitor_energy=True)
        E = E[:, burn_in:]
        W = W[:, burn_in:]

        var = np.average(np.square(E - np.average(E, weights= W, axis = 1)[:, None]), weights=W, axis = 1) / target.d
        med = np.median(var)
        lower_quart, upper_quart = np.median(var[var < med]), np.median(var[var > med])
        Evar[i] = [med, lower_quart, upper_quart]

    df = pd.DataFrame({'eps': epsilon, 'varE': Evar[:, 0], 'low err varE': Evar[:, 1], 'high err varE': Evar[:, 2]})
    df.to_csv('Tests/data/energy/'+name+'.csv', index= False)




def dimension_dependence():

    dimensions = [100, 300, 1000, 3000, 10000]
    name = ['kappa1', 'kappa100', 'Rosenbrock']
    targets = [lambda d: StandardNormal(d), lambda d: IllConditionedGaussian(d, 100.0), lambda d: Rosenbrock(d)]
    DF = [pd.read_csv('Tests/data/dimensions_dependence/'+nam+'g.csv') for nam in name]
    L = np.array([np.array(df['alpha']) * np.sqrt(dimensions) for df in DF])
    eps = np.array([np.array(df['eps']) for df in DF])

    num_steps, burn_in = 3000, 2000
    Evar = np.empty((len(targets), len(dimensions), 3))


    for num in range(len(targets)):
        print(name[num])

        for i in range(len(dimensions)):
            sampler = mchmc.Sampler(targets[num](dimensions[i]), L[num, i], eps[num, i], 'LF', True)

            X, w, E = sampler.parallel_sample(10, num_steps, monitor_energy=True)
            E = E[:, burn_in:]
            w = w[:, burn_in:]

            var = np.average(np.square(E - np.average(E, weights=w, axis=1)[:, None]), weights=w, axis=1) / dimensions[i]
            med = np.median(var)
            lower_quart, upper_quart = np.median(var[var < med]), np.median(var[var > med])
            Evar[num, i] = [med, lower_quart, upper_quart]

    np.save('Tests/data/energy/dimension_scaling.npy', Evar)

    #for d in df['d']df['eps']



if __name__ == '__main__':

    # from line_profiler import LineProfiler
    #
    # lprofiler = LineProfiler()
    # lprofiler.add_function(energy_variance)  # add functions that you would like to time
    #
    # lp_wrapper = lprofiler(benchmarks)  # the main function which will run the program
    # lp_wrapper()  # run program
    # lprofiler.print_stats()  # print times

    benchmarks()
    #dimension_dependence()
    #epsilon_dependence()