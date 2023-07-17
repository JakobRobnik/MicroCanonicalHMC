import os
num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mchmc
from benchmark_targets import *
import german_credit
from jump_identification import remove_jumps

import jax
import jax.numpy as jnp

tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']



def plot_power_lines(power):

    xlims = plt.gca().get_xlim()
    ylims = plt.gca().get_ylim()
    log_x_range = np.log(xlims)
    initital_value = np.linspace(np.log(ylims[0])- power * (log_x_range[1] - log_x_range[0]), np.log(ylims[1]) , 20) if power > 0 else np.linspace(np.log(ylims[0]), np.log(ylims[1])- power * (log_x_range[1] - log_x_range[0]), 20)

    for log_y0 in initital_value:
        plt.plot(np.exp(log_x_range), np.exp(log_y0 + power * (log_x_range - log_x_range[0])), ':', color='black', alpha=0.15)

    plt.xlim(*xlims)
    plt.ylim(*ylims)



def benchmarks():

    # targets
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(), Rosenbrock(), Funnel(), german_credit.Target(), StochasticVolatility()]

    #optimal settings
    file = 'submission/Table generalized_LF_q=0.csv'
    results = pd.read_csv(file)
    eps_all = np.array(results['eps'])
    alpha_all = np.array(results['alpha'])

    i_target = len(targets) - 2
    name = names[i_target]
    target = targets[i_target]
    eps_opt = eps_all[i_target]
    L_opt = np.sqrt(target.d) * alpha_all[i_target]

    eps = eps_opt * np.logspace(-1, np.log10(2), 10)
    var, fullvar = np.empty(len(eps)), np.empty(len(eps))

    for i in range(len(eps)):
        print(i)
        var[i], fullvar[i] = energy_variance(target, L_opt, eps[i])

    plt.title(name)
    plt.plot(eps, var, 'o-')
    plt.plot(eps, fullvar, 'o-')
    plt.plot(np.ones(2) * eps_opt, [1e-5, 1.0], color = 'black', alpha = 0.5)
    plt.plot(eps, np.ones(len(eps)) * 1e-3, color='black', alpha=0.5)

    plt.yscale('log')
    plt.xscale('log')
    plot_power_lines(4.0)

    plt.xlabel(r'$\epsilon$')
    plt.ylabel('Var[E] / d')
    #plt.savefig(name + '_stepsize.png')
    plt.show()




def epsilon_dependence():

    burn_in, samples = 2000, 1000
    chains = num_cores * 2

    names = ['STN', 'STN1000', 'ICG', 'rosenbrock', 'funnel', 'german', 'stochastic volatility']
    targets = [StandardNormal(d=100), StandardNormal(d=1000), IllConditionedGaussian(100, 100.0), Rosenbrock(), Funnel(), german_credit.Target(), StochasticVolatility()]
    num_eps = [20, 20, 20, 10, 10, 10, 10]

    file = 'submission/Table generalized_LF_q=0.csv'
    results = pd.read_csv(file)
    eps_all = np.array(results['eps'])[[0, 2, 3, 4, 5]]
    alpha_all = np.array(results['alpha'])[[0, 2, 3, 4, 5]]
    eps_all = np.insert(eps_all, [0, 0], [6.799491, 24.023774])
    alpha_all = np.insert(alpha_all, [0, 0], [0.775923, 0.847204])


    for num_target in range(3):

        Evar = np.empty((num_eps[num_target], 3))
        name, target = names[num_target], targets[num_target]
        eps_opt, L_opt = eps_all[num_target], np.sqrt(target.d) * alpha_all[num_target]
        print(name)

        if num_target<3:
            epsilon = eps_opt * np.logspace(-1, np.log10(10), len(Evar))
        else:
            epsilon = eps_opt * np.logspace(-1, np.log10(2), len(Evar))


        sampler = mchmc.Sampler(target, L_opt, eps_opt, 'LF', True)

        # burn-in
        x_init = sampler.parallel_sample(chains, burn_in, random_key= jax.random.PRNGKey(42), final_state=True, num_cores=num_cores)

        for i in range(len(epsilon)):
            print(i)
            sampler.eps = epsilon[i]
            #bias[i, :] = sampler.sample('prior', num_steps, L, key, generalized= False, integrator= 'LF', ess= True)
            _, W_all, E_all = sampler.parallel_sample(chains, samples, x_initial= x_init, monitor_energy=True, num_cores= num_cores)

            Evar[i] = get_var(E_all, W_all) / target.d

        df = pd.DataFrame({'eps': epsilon, 'varE': Evar[:, 0], 'low err varE': Evar[:, 1], 'high err varE': Evar[:, 2]})
        df.to_csv('data/energy/'+name+'.csv', index= False)



def dimension_dependence():

    dimensions = [100, 300, 1000, 3000, 10000]
    name = ['kappa1', 'kappa100', 'Rosenbrock']
    targets = [lambda d: StandardNormal(d), lambda d: IllConditionedGaussian(d, 100.0), lambda d: Rosenbrock(d, 0.5)]
    DF = [pd.read_csv('data/dimensions_dependence/'+nam+'g.csv') for nam in name]
    L = np.array([np.array(df['alpha']) * np.sqrt(dimensions) for df in DF])
    eps = np.array([np.array(df['eps']) for df in DF])

    burn_in, samples = 2000, 1000
    Evar = np.empty((len(targets), len(dimensions), 3))


    for num in range(2, len(targets)):
        print(name[num])

        for i in range(len(dimensions)):
            sampler = mchmc.Sampler(targets[num](dimensions[i]), L[num, i], eps[num, i] * 0.95, 'LF', True)

            E = sampler.sample(samples+burn_in, 10, output= 'energy')[1]

            Evar[num, i] = get_var(E[:, burn_in:]) / dimensions[i]

    np.save('data/energy/dimension_scaling_0.95.npy', Evar)

    #for d in df['d']df['eps']


if __name__ == '__main__':

    #power_spectrum()
    dimension_dependence()
    #epsilon_dependence()
    #benchmarks()
    #benchmark_weights()
    #dimension_dependence()
    #epsilon_dependence()