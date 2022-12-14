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


def moving_median_slow(x, w, width, f):

    smooth = np.empty(len(x) - width+1)

    for i in range(len(smooth)):
        smooth[i] = f(x[i: i+width], w[i: i+width])

    return smooth



def moving_variance(x, w, width):
    """returns moving window average and variance"""
    xsq = jnp.square(x)

    def move_by_one(state, useless):
        W, F1, F2, add, drop = state
        W_bare = W - w[drop]
        W_new = W_bare + w[add]
        F1 = (W * F1 + w[add] * x[add] - w[drop] * x[drop]) / W_new
        F2 = (W * F2 + w[add] * xsq[add] - w[drop] * xsq[drop]) / W_new

        var = F2 - jnp.square(F1)
        return (W_new, F1, F2, add + 1, drop + 1), (F1, var)

    Wi = jnp.sum(w[:width])
    F1i = jnp.average(x[:width], weights = w[:width])
    F2i = jnp.average(xsq[:width], weights=w[:width])
    results = jax.lax.scan(move_by_one, init = (Wi, F1i, F2i, width, 0), xs = None, length= len(x) - width)[1]

    return results



def variance_with_weights(x, w):

    avg = np.average(x, weights= w)
    return np.average(np.square(x - avg), weights= w)



def energy_variance(target, L, eps):
    burn_in, samples = 2000, 1000
    width = 61
    sampler = mchmc.Sampler(target, L, eps, integrator='LF', generalized=True)

    sampler.eps = 0.4
    x_init = sampler.parallel_sample(num_cores, burn_in, random_key= jax.random.PRNGKey(0), final_state= True, num_cores= num_cores)
    sampler.eps = eps
    Xall, Wall, Eall = sampler.parallel_sample(num_cores, samples, x_initial= x_init, random_key= jax.random.PRNGKey(1), monitor_energy=True, num_cores= num_cores)

    # Xall, Wall, Eall = sampler.parallel_sample(num_cores, samples + burn_in, random_key= jax.random.PRNGKey(42), monitor_energy=True, num_cores=num_cores)
    # Wall, Eall = Wall[:, burn_in:], Eall[:, burn_in:]

    var_chains= np.empty(num_cores)
    fullvar_chains= np.empty(num_cores)

    for chain in range(num_cores):
        W, E = Wall[chain], Eall[chain]
        # plt.subplot(2, 1, 1)
        # plt.plot(E)
        E, W = remove_jumps(E, W)
        # plt.subplot(2, 1, 2)
        # plt.plot(E)
        # plt.show()
        avg_t, var_t = moving_variance(E, W, width)
        #plt.plot(E[width // 2: len(E) - width // 2 - 1] - avg_t, '.', color=tab_colors[chain])
        var_chains[chain] = np.median(var_t)
        fullvar_chains[chain] = variance_with_weights(E, W)


    varE = np.median(var_chains) / target.d
    fullvarE = np.median(fullvar_chains) / target.d

    return varE, fullvarE


def importance_weights(target, L, eps):
    burn_in, samples = 2000, 1000
    sampler = mchmc.Sampler(target, L, eps, integrator='LF', generalized=True)

    sampler.eps = 0.4
    x_init = sampler.parallel_sample(num_cores, burn_in, random_key= jax.random.PRNGKey(0), final_state= True, num_cores= num_cores)
    sampler.eps = eps
    X, W = sampler.parallel_sample(num_cores, samples, x_initial= x_init, random_key= jax.random.PRNGKey(1), num_cores= num_cores)

    iw = jnp.square(jnp.average(W, axis = 1)) / jnp.average(jnp.square(W), axis = 1)
    print(iw)
    return jnp.average(iw)


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


def benchmark_weights():

    # targets
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(), Rosenbrock(), Funnel(), german_credit.Target(), StochasticVolatility()]

    #optimal settings
    file = 'submission/Table generalized_LF_q=0.csv'
    results = pd.read_csv(file)
    eps_all = np.array(results['eps'])
    alpha_all = np.array(results['alpha'])

    for i_target in range(len(targets)):
        name = names[i_target]
        target = targets[i_target]
        eps_opt, L_opt = eps_all[i_target], alpha_all[i_target] * np.sqrt(target.d)

        eps = eps_opt * np.logspace(-1, np.log10(2), 10)
        iw = np.empty(len(eps))

        for i in range(len(eps)):
            print(i)
            iw[i] = importance_weights(target, L_opt, eps[i])

        plt.title(name)
        plt.plot(eps, iw, 'o-')
        plt.plot(np.ones(2) * eps_opt, [0, 1], color = 'black', alpha = 0.5)
        plt.plot(eps, np.ones(len(eps)), color='black', alpha=0.5)

        plt.xscale('log')

        plt.xlabel(r'$\epsilon$')
        plt.ylabel(r'$(\sum w_i)^2 / \sum w_i^2$')
        plt.savefig(name + 'weights_stepsize.png')
        plt.show()

    #results['energy variance'] = varE
    #results.to_csv('submission/Table generalized_LF_q=0_energy.csv', index = False)


def power_spectrum():

    # targets
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(), Rosenbrock(), Funnel(), german_credit.Target(), StochasticVolatility()]

    #optimal settings
    file = 'submission/Table generalized_LF_q=0.csv'
    results = pd.read_csv(file)
    eps_all = np.array(results['eps'])
    alpha_all = np.array(results['alpha'])
    burn_in = 2000
    plt.figure(figsize= (15, 10))
    plt.rcParams.update({'font.size': 30})

    i_target = 4
    name = names[i_target]
    target = targets[i_target]
    eps_opt, L_opt = eps_all[i_target], alpha_all[i_target] * np.sqrt(target.d)
    sampler = mchmc.Sampler(target, L_opt, eps_opt, 'LF', True)
    x, w, E = sampler.sample(3000, monitor_energy=True)
    plt.plot(E[burn_in:] - np.average(E[burn_in:]), '.')
    plt.show()

    for i_target in range(len(targets)):
        name = names[i_target]
        target = targets[i_target]
        eps_opt, L_opt = eps_all[i_target], alpha_all[i_target] * np.sqrt(target.d)

        sampler = mchmc.Sampler(target, L_opt, eps_opt, 'LF', True)
        X, w, E = sampler.parallel_sample(num_cores * 2, 3000, monitor_energy = True, num_cores = num_cores)
        successful = np.all(np.isfinite(E), axis = 1)
        X, w, E = X[successful, burn_in:, :], w[successful, burn_in:], E[successful, burn_in:]

        sigmas = np.empty(len(X))
        for chain in range(len(X)):
            x1 = np.average(X[chain], weights=w[chain], axis = 0)
            x2 = np.average(np.square(X[chain]), weights=w[chain], axis = 0)
            xvar = x2 - np.square(x1)
            sigmas[chain] = np.sqrt(np.average(xvar))
        sigma = np.median(sigmas)
        print(sigma)

        psd = np.square(np.abs(np.fft.rfft(E, axis = 1)))
        psd_avg = np.median(psd, axis = 0) / target.d
        freq = np.arange(len(psd_avg)) / (len(E) * sampler.eps / sigma)
        if i_target != 4:
            plt.plot(freq[1:], psd_avg[1:] / len(psd_avg), label = name)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k \sigma$')
    plt.ylabel('P(k) / d')
    plt.legend(fontsize = 25)

    plt.savefig('plots/energy/PSD/all.png')
    plt.show()



def get_var(E_all, W_all):
    num_infs = 0
    chains = len(E_all)
    var = []
    for chain in range(chains):
        if not np.all(np.isfinite(E_all[chain])):
            num_infs += 1
        else:
            E, W = remove_jumps(E_all[chain], W_all[chain])
            var.append(np.average(np.square(E - np.average(E, weights= W)), weights=W))

    if chains - num_infs < 3:
        return np.array([-1, -1, -1])

    else:
        var = np.array(var)
        med = np.median(var)
        lower_quart, upper_quart = np.median(var[var < med]), np.median(var[var > med])
        return np.array([med, lower_quart, upper_quart])



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

            _, W, E = sampler.parallel_sample(10, samples + burn_in, monitor_energy=True)

            Evar[num, i] = get_var(E[:, burn_in:], W[:, burn_in:]) / dimensions[i]

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