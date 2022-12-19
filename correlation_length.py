import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from scipy.stats import linregress
from scipy.fftpack import next_fast_len


import mchmc
from benchmark_targets import *




def ICG_correlations():
    data= pd.read_csv('submission/Table_ICG_LF.csv')

    kappa = np.array(data['Condition number'])
    alpha = np.array(data['alpha'])
    eps = np.array(data['eps'])
    targets= [IllConditionedGaussian(d = 100, condition_number= kap) for kap in kappa]
    num_samples, num_chains = 200, 3
    ess = np.empty(len(kappa))
    for i in range(len(kappa)):
        print(i)
        sampler = mchmc.Sampler(targets[i], alpha[i] * np.sqrt(targets[i].d), eps[i], 'LF', False)
        samples = sampler.parallel_sample(num_chains, num_samples)[0]
        es = np.array(effective_sample_size(torch.from_numpy(np.array(samples)))/ (num_chains * num_samples))
        ess[i] = 1.0 / np.average(1.0 / es)

    np.save('ess_correlations_'+str(num_samples)+'samples.npy', ess)



def ICG_energy():
    method = ['none', 'bounces', 'generalized'][0]
    data= pd.read_csv('submission/Table_ICG_LF'+('_g+' if method == 'generalized' else '') + '.csv')

    kappa = np.array(data['Condition number'])
    alpha = np.array(data['alpha'])
    if method == 'none':
        alpha[:] = np.inf
    eps = np.array(data['eps'])
    targets= [IllConditionedGaussian(d = 100, condition_number= kap) for kap in kappa]
    num_samples, num_chains = 2000, 10
    energy_std = np.empty((len(kappa), 2))
    for i in range(len(kappa)):
        print(i)
        sampler = mchmc.Sampler(targets[i], alpha[i] * np.sqrt(targets[i].d), eps[i], 'LF', method == 'generalized')
        X, W, E = sampler.parallel_sample(num_chains, num_samples, monitor_energy= True)
        # avgE = np.average(E, weights=w)
        # stdE = np.sqrt(np.average(np.square(E - avgE), weights=w))

        std = np.std(E[:, 1000:], axis = 1) / targets[i].d
        energy_std[i] = [np.average(std), np.std(std)]

    np.save('data/ICG/energy_'+method+'.npy', energy_std)



def plot_correlations():
    data= pd.read_csv('submission/Table_ICG_LF.csv')
    kappa = np.array(data['Condition number'])

    ess5000 = np.load('ess_correlations_'+str(5000)+'samples.npy')
    ess200 = np.load('ess_correlations_' + str(200) + 'samples.npy')

    plt.plot(kappa, ess5000, 'o-', label = 'correlation ESS (5000 x 3 samples)')
    plt.plot(kappa, ess200, 'o-', label='correlation ESS (200 x 3 samples)')
    plt.plot(kappa, data['ESS'], 'o-', label = 'second moment error ESS')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('condition number')
    plt.ylabel('ESS')
    plt.legend()
    plt.savefig('ESS_from_correlations.png')
    plt.show()


def plot_energy():
    method = ['none', 'bounces', 'generalized'][0]

    std = np.load('data/ICG/energy_'+method+'.npy')

    data= pd.read_csv('submission/Table_ICG_LF.csv')
    kappa = np.array(data['Condition number'])

    plt.plot(kappa, std[:, 0], color = 'tab:purple')
    plt.fill_between(kappa, std[:, 0] - std[:, 1], std[:, 0] + std[:, 1], color = 'tab:purple', alpha = 0.5)

    plt.xscale('log')
    plt.savefig('ICG_energy.png')
    plt.show()


def ICG_alpha_eps():
    names = ['alpha', 'eps']

    methods = ['bounces', 'generalized']
    data = [pd.read_csv('submission/Table_ICG_LF.csv'), pd.read_csv('submission/Table_ICG_LF_g.csv')]
    kappa = data[0]['Condition number']
    colors = ['tab:blue', 'tab:orange']

    std = np.sqrt([np.average(IllConditionedGaussian(d=100, condition_number=kap).variance) for kap in kappa])
    print(std)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.subplots(1, 2)
    plt.rcParams.update({'font.size': 25})

    for j in range(2):  # alpha and eps
        print(names[j])
        for i in range(2):  # bounces and generalized
            ax[j].plot(kappa, data[i][names[j]] / std, marker='.', color=colors[i], label=methods[i])

            reg = linregress(np.log(kappa), np.log(data[i][names[j]] / std))
            print(methods[i] + ': ' + str(reg.slope))
            ax[j].plot(kappa, np.power(kappa, reg.slope) * np.exp(reg.intercept), alpha=0.5, color=colors[i])

        ax[j].set_ylabel(names[j])
        plt.xlabel('condition number')

        ax[j].set_xscale('log')
        ax[j].set_yscale('log')
        print('---------------------')

    plt.legend()
    plt.savefig('alpha_eps.png')
    plt.show()


def ICG_ess():
    methods = ['bounces', 'generalized']
    data = [pd.read_csv('submission/Table_ICG_LF.csv'), pd.read_csv('submission/Table_ICG_LF_g.csv')]
    kappa = data[0]['Condition number']
    colors = ['tab:blue', 'tab:orange']


    fig = plt.figure(figsize=(15, 10))
    ax = fig.subplots(1, 1)
    plt.rcParams.update({'font.size': 25})

    for i in range(2):  # bounces and generalized
        ess_predicted = 1 / (data[i]['alpha'] * np.sqrt(100) / data[i]['eps'])
        ax.plot(kappa, data[i]['ESS'] / ess_predicted, '-o', color=colors[i], label=methods[i])


    ax.set_ylabel('ESS * L / eps')
    ax.set_xlabel('condition number')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig('ess.png')
    plt.show()




def ess_corr(x):
    """Taken from: https://blackjax-devs.github.io/blackjax/diagnostics.html
        shape(x) = (num_samples, d)"""

    input_array = jnp.array([x, ])

    num_chains = 1#input_array.shape[0]
    num_samples = input_array.shape[1]

    mean_across_chain = input_array.mean(axis=1, keepdims=True)
    # Compute autocovariance estimates for every lag for the input array using FFT.
    centered_array = input_array - mean_across_chain
    m = next_fast_len(2 * num_samples)
    ifft_ary = jnp.fft.rfft(centered_array, n=m, axis=1)
    ifft_ary *= jnp.conjugate(ifft_ary)
    autocov_value = jnp.fft.irfft(ifft_ary, n=m, axis=1)
    autocov_value = (
        jnp.take(autocov_value, jnp.arange(num_samples), axis=1) / num_samples
    )
    mean_autocov_var = autocov_value.mean(0, keepdims=True)
    mean_var0 = (jnp.take(mean_autocov_var, jnp.array([0]), axis=1) * num_samples / (num_samples - 1.0))
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    weighted_var = jax.lax.cond(
        num_chains > 1,
        lambda _: weighted_var+ mean_across_chain.var(axis=0, ddof=1, keepdims=True),
        lambda _: weighted_var,
        operand=None,
    )

    # Geyer's initial positive sequence
    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(mean_autocov_var, jnp.arange(1, num_samples_even), axis=1)
    rho_hat = jnp.concatenate([jnp.ones_like(mean_var0), 1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,], axis=1,)

    rho_hat = jnp.moveaxis(rho_hat, 1, 0)
    rho_hat_even = rho_hat[0::2]
    rho_hat_odd = rho_hat[1::2]

    mask0 = (rho_hat_even + rho_hat_odd) > 0.0
    carry_cond = jnp.ones_like(mask0[0])
    max_t = jnp.zeros_like(mask0[0], dtype=int)

    def positive_sequence_body_fn(state, mask_t):
        t, carry_cond, max_t = state
        next_mask = carry_cond & mask_t
        next_max_t = jnp.where(next_mask, jnp.ones_like(max_t) * t, max_t)
        return (t + 1, next_mask, next_max_t), next_mask

    (*_, max_t_next), mask = jax.lax.scan(
        positive_sequence_body_fn, (0, carry_cond, max_t), mask0
    )
    indices = jnp.indices(max_t_next.shape)
    indices = tuple([max_t_next + 1] + [indices[i] for i in range(max_t_next.ndim)])
    rho_hat_odd = jnp.where(mask, rho_hat_odd, jnp.zeros_like(rho_hat_odd))
    # improve estimation
    mask_even = mask.at[indices].set(rho_hat_even[indices] > 0)
    rho_hat_even = jnp.where(mask_even, rho_hat_even, jnp.zeros_like(rho_hat_even))

    # Geyer's initial monotone sequence
    def monotone_sequence_body_fn(rho_hat_sum_tm1, rho_hat_sum_t):
        update_mask = rho_hat_sum_t > rho_hat_sum_tm1
        next_rho_hat_sum_t = jnp.where(update_mask, rho_hat_sum_tm1, rho_hat_sum_t)
        return next_rho_hat_sum_t, (update_mask, next_rho_hat_sum_t)

    rho_hat_sum = rho_hat_even + rho_hat_odd
    _, (update_mask, update_value) = jax.lax.scan(
        monotone_sequence_body_fn, rho_hat_sum[0], rho_hat_sum
    )

    rho_hat_even_final = jnp.where(update_mask, update_value / 2.0, rho_hat_even)
    rho_hat_odd_final = jnp.where(update_mask, update_value / 2.0, rho_hat_odd)

    # compute effective sample size
    ess_raw = num_chains * num_samples
    tau_hat = (
        -1.0
        + 2.0 * jnp.sum(rho_hat_even_final + rho_hat_odd_final, axis=0)
        - rho_hat_even_final[indices]
    )

    tau_hat = jnp.maximum(tau_hat, 1 / np.log10(ess_raw))
    ess = ess_raw / tau_hat

    ### my part (combine all dimensions): ###
    neff = ess.squeeze() / num_samples
    return 1.0 / jnp.average(1 / neff)



def convergence():

    burn_in = 2000
    import german_credit
    names = ['STN', 'STN1000', 'ICG', 'rosenbrock', 'funnel', 'german', 'stochastic volatility']
    targets = [StandardNormal(d=100), StandardNormal(d=1000), IllConditionedGaussian(100, 100.0), Rosenbrock(), Funnel(), german_credit.Target(), StochasticVolatility()]

    file = 'submission/Table generalized_LF_q=0.csv'
    results = pd.read_csv(file)
    eps_all = np.array(results['eps'])[[0, 2, 3, 4, 5]]
    alpha_all = np.array(results['alpha'])[[0, 2, 3, 4, 5]]
    ESS = np.array(results['ESS'])[[0, 2, 3, 4, 5]]
    eps_all = np.insert(eps_all, [0, 0], [6.799491, 24.023774])
    alpha_all = np.insert(alpha_all, [0, 0], [0.775923, 0.847204])
    ESS = np.insert(ESS, [0, 0], [0.295046, 0.298640])

    num_steps = 30.0 / ESS
    num_steps[num_steps > 3000] = 3000

    points = 20
    ess = np.empty((len(names), points))

    for num_target in range(len(targets)):
        print(names[num_target])
        sampler = mchmc.Sampler(targets[num_target], np.sqrt(targets[num_target].d) * alpha_all[num_target], eps_all[num_target], 'LF', True)

        X = sampler.sample(burn_in+num_steps[num_target], monitor_energy= True)[0]
        X = X[burn_in:, :]

        n = np.linspace(10, num_steps[num_target], points).astype(int)
        a = np.array([ess_corr(X[:nn, :]) for nn in n])
        ess[num_target, :] = a

    np.save('data/ess_corr.npy', ess)


def plot_convergence():


    ess = np.load('data/ess_corr.npy')
    names = ['STN', 'STN1000', 'ICG', 'rosenbrock', 'funnel', 'german', 'stochastic volatility']
    colors= ['tab:blue', 'darkblue', 'tab:orange', 'tab:red', 'tab:green', 'purple', 'saddlebrown']

    file = 'submission/Table generalized_LF_q=0.csv'
    results = pd.read_csv(file)
    eps_all = np.array(results['eps'])[[0, 2, 3, 4, 5]]
    alpha_all = np.array(results['alpha'])[[0, 2, 3, 4, 5]]
    ESS = np.array(results['ESS'])[[0, 2, 3, 4, 5]]
    eps_all = np.insert(eps_all, [0, 0], [6.799491, 24.023774])
    alpha_all = np.insert(alpha_all, [0, 0], [0.775923, 0.847204])
    ESS = np.insert(ESS, [0, 0], [0.295046, 0.298640])

    num_steps = 30.0 / ESS
    num_steps[num_steps > 3000] = 3000
    points = 20

    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 30})

    for num_target in range(len(names)):
        n = np.linspace(10, num_steps[num_target], points).astype(int)

        plt.plot(n * ESS[num_target], ess[num_target] / ESS[num_target], 'o-', color= colors[num_target], label = names[num_target])

    plt.legend()
    plt.xlabel(r'$n_{\mathrm{eff}}$')
    plt.ylabel(r'$ESS_{\mathrm{corr}} / ESS_{b_2}$')
    plt.plot([0, 30], [1, 1], color = 'black')
    plt.ylim(0, 5)
    plt.savefig('correlation length convergence.png')
    plt.show()



if __name__ == '__main__':
    #convergence()
    plot_convergence()