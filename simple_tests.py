import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
import os

import ESH
import CTV
import parallel
from targets import *
import bias

import jax
import jax.numpy as jnp




num_cores = 6
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

def parallel_run(function, values):
    parallel_function= jax.pmap(jax.vmap(function))
    results = jnp.array(parallel_function(values.reshape(num_cores, len(values) // num_cores)))
    return results.reshape([len(values), ] + [results.shape[i] for i in range(2, len(results.shape))])



def bounce_frequency(d, alpha):

    key = jax.random.PRNGKey(0)

    length = alpha * jnp.sqrt(d)
    sampler = ESH.Sampler(Target= IllConditionedGaussian(d=d, condition_number=100.0), eps=1.0)

    ess = parallel_run(lambda L: sampler.sample_multiple_chains(5, 300000, L, key, ess= True, prerun=0), length)

    return jnp.average(ess, 1), jnp.std(ess, 1)

    #eta = (2.93 * np.power(d, -0.78) * np.logspace(-0.8, 0.8, 24))[n]

    #length = 1.5 * np.sqrt(d)
    #sampler = CTV.Sampler(Target = IllConditionedGaussian(d= d, condition_number=100), eps= 3)
    #sampler = ESH.Sampler(Target= Rosenbrock(d= d), eps= 0.5)
    #a= np.sqrt(np.concatenate((np.ones(d//2) * 2.0, np.ones(d//2) * 10.498957879911487)))
    #sampler = ESH.Sampler(Target= DiagonalPreconditioned(Rosenbrock(d= d), a), eps= 0.5)
    #return [ess, length, sampler.eps, d]



def bounce_frequency_full_bias(n, d):
    length = [2, 5, 10, 30, 50, 75, 80, 90, 100, 1000, 10000, 10000000]
    L = length[n]

    sampler = ESH.Sampler(Target= StandardNormal(d= d), eps=1)
    np.random.seed(1)
    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    bias = sampler.sample(x0, L, track= 'FullBias')

    return bias



def compute_energy(n):
    eps_arr = [0.05, 0.1, 0.5, 1, 2]
    eps = eps_arr[n]
    d = 50
    total_num = 1000000
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps= eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    t, X, P, E = esh.trajectory(x0, total_num)
    np.save('Tests/energy/E'+str(n)+'.npy', E)



def bimodal_mixing(n):
    mu = np.arange(1, 9)[n]

    d = 50
    eps, L = 1.0, 1.5 * np.sqrt(d)
    esh = ESH.Sampler(Target= BiModalEqual(d, mu), eps=eps)
    np.random.seed(1)
    x0 = np.random.normal(size= d)
    x0[0] += mu * 0.5

    avg_island_size = esh.sample(x0, L, prerun_steps= 500, track= 'ModeMixing')

    return [avg_island_size, mu, L, eps, d]



def ill_conditioned():
    d = 100
    L = 1.5 * np.sqrt(d)

    def f(n):
        kappa = jnp.logspace(0, 5, 18)[n]
        eps = jnp.array(10 * [2.0, ] + 2 * [1.5, ] + 3 * [1.0, ] + [0.7, ] + 2 * [0.5])[n]

        key = jax.random.PRNGKey(0)

        sampler = ESH.Sampler(Target=IllConditionedGaussian(d=d, condition_number=kappa), eps=eps)

        ess = sampler.sample_multiple_chains(5, 300000, L, key, ess=True, prerun=0)

        return jnp.array([jnp.average(ess, 1), jnp.std(ess, 1)])

    results = parallel_run(f, np.arange(18, dtype = int))

    print(np.shape(results))

    np.save('Tests/data/ill_conditioned_no_tuning.npy', results)




def funnel_plot():

    eps = 0.1
    free_time = 6
    free_steps = (int)(free_time / eps)
    d = 20
    esh = ESH.Sampler(Target= Funnel(d=d), eps=eps)
    np.random.seed(0)
    x0 = np.zeros(d)
    samples, w = esh.sample(x0, free_steps, 5000000, track= 'FullTrajectory')
    np.savez('Tests/data/funnel_free'+str(free_time) + '_eps'+str(eps), z = samples[:, :-1], theta= samples[:, -1], w = w)


def funnel_ess(n):
    length = (1 * np.sqrt(d) * np.logspace(-0.4, 0.4, 12))[n]
    sampler = ESH.Sampler(Target=Funnel(d =20), eps= 0.1)

    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    ess = sampler.sample(x0, length, prerun_steps=500, track= 'ESS')

    return [ess, length, sampler.eps, d]



def rosenbrock():

    eps = 0.001
    free_steps = (int)(1 / eps)
    d = 36
    esh = ESH.Sampler(Target= Rosenbrock(d=d), eps=eps)
    np.random.seed(0)
    x0 = np.zeros(d)
    samples, w = esh.sample(x0, free_steps, max_steps= 10000000, prerun_steps= 500, track= 'FullTrajectory')
    np.savez('Tests/data/rosenbrock3', samples = samples[::10, :], w = w[::10])


def my_hist(bins, count):
    probability = count / np.sum(count)
    print('probability outside of bins', probability[-1])

    for i in range(len(bins)):
        density = probability[i] / (bins[i][1] - bins[i][0])
        plt.fill_between(bins[i], np.zeros(2), density * np.ones(2), alpha = 0.5, color = 'tab:blue')



def bimodal():

    xmax = 3.5 #how many sigma away from the mean of the gaussians do we want to have bins

    def get_bins(mu, sigma, num_bins_per_mode):
        bins_mode = np.array([[- xmax + i * 2 * xmax / num_bins_per_mode, - xmax + (i+1) * 2 * xmax / num_bins_per_mode] for i in range(num_bins_per_mode)])

        bins = np.concatenate(( (bins_mode * sigma[0]) + mu[0], (bins_mode * sigma[1]) + mu[1]  ))

        return bins

    d = 50
    mu1, sigma1= 0.0, 1.0 # the first Gaussian
    mu2, sigma2, f = 7.0, 0.5, 0.2 #the second Gaussian

    name = 'sep'+str(mu2)+'_f'+str(f)+'_sigma'+str(sigma2)
    load = False

    eps, L = 1.0, 1.5 * np.sqrt(d)
    bins_per_mode = 20
    bins = get_bins([mu1, mu2], [sigma1, sigma2], bins_per_mode)

    sampler = ESH.Sampler(Target= BiModal(d=d, mu1= mu1, mu2 = mu2, sigma1= sigma1, sigma2 = sigma2, f= f), eps=eps)

    #X = esh.Target.draw(1000)[:, 0]
    np.random.seed(0)
    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    if load:
        P = np.load('Tests/data/bimodal_marginal'+name+'.npy')
    else:
        P = sampler.sample(x0, L, prerun_steps=5000, max_steps= 50000000, track='Marginal1d', bins= bins)

    my_hist(bins, P)
    f1, f2 = np.sum(P[:bins_per_mode]), np.sum(P[bins_per_mode : 2 * bins_per_mode])
    print('f = ' + str(f2 / (f1 + f2)) + '  (true f = ' + str(f) + ')')

    t = np.linspace(-xmax*sigma1+mu1-0.5, xmax*sigma2 + mu2 + 0.5, 1000)

    plt.plot(t, (1- f)*norm.pdf(t, loc = mu1, scale = sigma1) + f * norm.pdf(t, loc = mu2, scale = sigma2), color = 'black')
    plt.xlim(t[0], t[-1])
    plt.ylim(0, 0.4)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$p(x_1)$')

    plt.savefig('Tests/'+name+'.png')
    np.save('Tests/data/bimodal_marginal'+name+'.npy', P)

    plt.show()



def dimension_dependence():

    dimensions = [50, 100, 200, 500, 1000]#, 3000, 10000]
    alpha = (1.5 * jnp.logspace(-0.8, 0.8, 24))
    #condition_numbers = np.logspace(0, 5, 18)
    dict = {'alpha': alpha}
    for d in dimensions:
        print(d)
        avg, std = bounce_frequency(d, alpha)
        dict.update({'ess (d='+str(d)+')': avg, 'err ess (d='+str(d)+')': std})
    df = pd.DataFrame.from_dict(dict)
    df.to_csv('Tests/data/dimensions/Kappa100.csv', sep='\t', index=False)


def billiard_bimodal():

    d = 2
    mu1, sigma1= 0.0, 1.0 # the first Gaussian
    mu2, sigma2, f = 5.0, 1.0, 0.5 #the second Gaussian
    xmax = 10.0
    typical_set = np.sqrt(d)
    border_lim = typical_set + 1.0


    sampler = ESH.Sampler(Target= BiModal(d=d, mu1= mu1, mu2 = mu2, sigma1= sigma1, sigma2 = sigma2, f= f), eps= 0.001)
    #sampler = ESH.Sampler(Target=StandardNormal(d = d), eps=0.1)

    key = jax.random.PRNGKey(1)

    results = sampler.billiard_trajectory(400000, key, 30.0, border_lim, xmax, mu2 - mu1)
    x, y, w, region = results
    # i = 0
    # radia = jnp.sqrt(jnp.square(x) +jnp.square(y))
    # while not np.isnan(x[i]):
    #     i+=1
    # print(radia[i-5:i+1])
    #mask =  > xmax


    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color=  'tab:red', label = 'trajectory')

    phi = np.linspace(0, 2 * np.pi, 200)
    plt.plot(xmax * np.cos(phi), xmax * np.sin(phi), color=  'black', label = 'prior range')
    plt.plot(1 * np.cos(phi), 1 * np.sin(phi), color='tab:blue')
    plt.plot(2 * np.cos(phi), 2 * np.sin(phi), color='tab:blue', alpha = 0.5)

    plt.plot(1 * np.cos(phi) + mu2, 1 * np.sin(phi), color='tab:blue')
    plt.plot(2 * np.cos(phi) + mu2, 2 * np.sin(phi), color='tab:blue', alpha = 0.5)

    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()
    plt.xlim(-xmax, xmax)
    plt.ylim(-xmax, xmax)
    #plt.savefig('no bounces')
    plt.show()

def bimodal_explore():
    mu = 7.0

    d = 10
    eps, L = 1.0, 20 * np.sqrt(d)
    esh = ESH.Sampler(Target=BiModalEqual(d=d, mu=mu), eps=eps)
    np.random.seed(1)
    x0 = np.random.normal(size=d)
    x0[0] += mu * 0.5

    X, W = esh.sample(x0, L, prerun_steps=500, track='FullTrajectory')

    plt.subplot(2, 1, 1)
    plt.plot(X[:, 0])
    plt.ylabel('x1')
    plt.xlim(22000, 25000)

    plt.subplot(2, 1, 2)
    plt.plot([2 * esh.Target.nlogp(x) for x in X])
    plt.ylabel('- 2 log p')
    plt.xlabel('steps')
    plt.xlim(22000, 25000)
    plt.savefig('no_jump_in_logp')
    plt.show()


def energy_fluctuations():

    eps = (1.0 * np.logspace(np.log10(0.05), np.log10(4.0), 18))

    key = jax.random.PRNGKey(0)

    def f(e):
        esh = ESH.Sampler(Target=IllConditionedGaussian(d=100, condition_number=10000.0), eps=e)
        ess_chains = esh.sample_multiple_chains(3, 300000, 1.5 * jnp.sqrt(esh.Target.d), key, prerun=0, ess=True)
        ess, err_ess = jnp.average(ess_chains), jnp.std(ess_chains)
        energy_chains = esh.sample_multiple_chains(3, 300000, 1.5 * jnp.sqrt(esh.Target.d), key, prerun=0, energy_track = True)
        #print(np.shape(energy_chains))
        relative_energy_fluctuations = energy_chains[1]/energy_chains[0]
        Eavg, Estd = jnp.average(relative_energy_fluctuations), jnp.std(relative_energy_fluctuations)
        return jnp.array([ess, err_ess, Eavg, Estd])


    results = parallel_run(f, eps)
    #f(1.0)
    #print(results)
    plt.subplot(2, 1, 1)
    plt.errorbar(eps, results[:, 0], yerr = results[:, 1], capsize= 1.5, fmt = 'o:')
    plt.ylabel('ESS')
    plt.subplot(2, 1, 2)
    print(results[:, 2])
    plt.errorbar(eps, results[:, 2], yerr = results[:, 3], capsize= 1.5, fmt = 'o:')

    plt.ylabel('energy fluctuations')
    plt.xlabel('eps')

    plt.show()

def autocorr():

    key = jax.random.PRNGKey(0)

    sampler = ESH.Sampler(Target=IllConditionedGaussian(d=100, condition_number=1.0), eps=0.01)

    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(key, shape = (sampler.Target.d, ), dtype = 'float64')

    X, W = sampler.sample(x0, 1000000, 1.5 * jnp.sqrt(sampler.Target.d), key)
    print(np.shape(X))
    autocorr = np.fft.irfft(np.abs(np.square(np.fft.rfft(X[:, 0]))), len(X)) / len(X)
    plt.plot(np.arange(len(X)//2) * sampler.eps, autocorr[:len(X)//2] )
    plt.xscale('log')
    plt.show()



if __name__ == '__main__':

    ill_conditioned()