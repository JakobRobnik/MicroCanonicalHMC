import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import ESH
import standardKinetic
from benchmark_targets import *
import grid_search

import jax
import jax.numpy as jnp

num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

### Runs the bencmark problems. """


def parallel_run(function, values):
    parallel_function= jax.pmap(jax.vmap(function))
    results = jnp.array(parallel_function(values.reshape(num_cores, len(values) // num_cores)))
    return results.reshape([len(values), ] + [results.shape[i] for i in range(2, len(results.shape))])


def bounce_frequency(d, alpha, generalized = False):
    """For scaning over the L parameter"""

    key = jax.random.PRNGKey(1)

    length = alpha * jnp.sqrt(d)
    sampler = ESH.Sampler(Target= StandardNormal(d=d), eps= 7.3* jnp.sqrt(d / 100.0))
    #sampler = standardKinetic.Sampler(Target= StandardNormal(d=d), eps=2.0)

    #sampler = ESH.Sampler(Target= IllConditionedGaussian(d=d, condition_number=100.0), eps=1.0)

    ess = parallel_run(lambda L: sampler.parallel_sample(10, 30000, L, key, generalized= generalized, ess= True), length)

    return jnp.average(ess, 1), jnp.std(ess, 1)

    #eta = (2.93 * np.power(d, -0.78) * np.logspace(-0.8, 0.8, 24))[n]

    #length = 1.5 * np.sqrt(d)
    #sampler = standardKinetic.Sampler(Target = IllConditionedGaussian(d= d, condition_number=100), eps= 3)
    #sampler = ESH.Sampler(Target= Rosenbrock(d= d), eps= 0.5)
    #a= np.sqrt(np.concatenate((np.ones(d//2) * 2.0, np.ones(d//2) * 10.498957879911487)))
    #sampler = ESH.Sampler(Target= DiagonalPreconditioned(Rosenbrock(d= d), a), eps= 0.5)
    #return [ess, length, sampler.eps, d]



def full_bias():
    length = [2, 5, 30, 90, 1e20]
    d= 200

    bias = np.empty((len(length), 1000000))
    for n in range(len(length)):
        print(n)
        sampler = ESH.Sampler(Target= StandardNormal(d= d), eps=1.0)
        key = jax.random.PRNGKey(1)
        key, subkey = jax.random.split(key)
        x0 = sampler.Target.prior_draw(subkey)
        bias[n, :] = sampler.sample(x0, 1000000, length[n], key, ess= True)

    np.save('Tests/data/full_bias.npy', bias)




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

    avg_island_size = esh.sample(x0, L, track= 'ModeMixing')

    return [avg_island_size, mu, L, eps, d]



def ill_conditioned_workhorse(alpha, generalized):
    d = 100
    L = alpha * np.sqrt(d)

    def f(n):
        kappa = jnp.logspace(0, 5, 18)[n]
        eps = jnp.array(10 * [2.0, ] + 2 * [1.5, ] + 3 * [1.0, ] + [0.7, ] + 2 * [0.5, ])[n]

        key = jax.random.PRNGKey(0)

        sampler = ESH.Sampler(Target=IllConditionedGaussian(d=d, condition_number=kappa), eps=eps)

        ess = sampler.parallel_sample(10, 300000, L, key, generalized = generalized, ess=True)

        return jnp.array([jnp.average(ess), jnp.std(ess)])

    results = parallel_run(f, np.arange(18, dtype = int))

    return results


def ill_conditioned(tunning, generalized):
    word = '_l' if generalized else ''

    if not tunning:
        results = ill_conditioned_workhorse(1.0, generalized)
        np.save('Tests/data/kappa/no_tuning'+word+'.npy', results)

    else:
        alpha = (1.5 * jnp.logspace(-0.8, 0.8, 24))

        results= np.empty((len(alpha), 18, 2))

        for i in range(len(alpha)):
            print(str(i)+ '/' + str(len(alpha)))
            results[i] = ill_conditioned_workhorse(alpha[i], generalized)

        np.save('Tests/data/kappa/fine_tuned_full_results'+word+'.npy', results)
        np.save('Tests/data/kappa/fine_tuned'+word+'.npy', results[np.argmax(results[:, :, 0], axis= 0), np.arange(18, dtype = int), :])




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


def dimension_dependence():

    dimensions = [100, 300, 1000, 3000]
    alpha = (1.5 * jnp.logspace(-0.8, 0.8, 12))
    #condition_numbers = np.logspace(0, 5, 18)
    dict = {'alpha': alpha}
    generalized = False
    for d in dimensions:
        print(d)
        avg, std = bounce_frequency(d, alpha, generalized)
        dict.update({'ess (d='+str(d)+')': avg, 'err ess (d='+str(d)+')': std})
    df = pd.DataFrame.from_dict(dict)
    df.to_csv('Tests/data/dimensions/StandardNormal'+('_g' if generalized else '')+'_eps4.csv', sep='\t', index=False)



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

    X, W = esh.sample(x0, L, track='FullTrajectory')

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
        ess_chains = esh.parallel_sample(3, 300000, 1.5 * jnp.sqrt(esh.Target.d), key, ess=True)
        ess, err_ess = jnp.average(ess_chains), jnp.std(ess_chains)
        energy_chains = esh.parallel_sample(3, 300000, 1.5 * jnp.sqrt(esh.Target.d), key, energy_track = True)
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


def table1():
    """For generating Table 1 in the paper"""

    #version of the sampler
    q = 0 #choice of the Hamiltonian (q = 0 or q = 2)
    generalized = False #choice of the momentum decoherence mechanism
    alpha = -1 #bounce frequency (1.0 for generalized, 1.6 for bounces, something very large if no bounces). If -1, alpha is tuned by a grid search.
    integrator = 'LF' #integrator (Leapfrog (LF) or Minimum Norm (MN))
    parallel = False


    #name of the version
    if alpha > 1e10:
        generalized_string= 'no-bounces_'
        alpha_string = ''
    else:
        generalized_string = 'generalized_' if generalized else 'bounces_'
        alpha_string = '_tuning-free' if (alpha > 0) else ''
    parallel_string = '_parallel' if parallel else ''
    name_sampler = generalized_string + integrator + '_q=' + str(q) + alpha_string + parallel_string
    print(name_sampler)

    #targets
    import german_credit
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(d=50, mu1=0.0, mu2=8.0, sigma1=1.0, sigma2=1.0, f=0.2), Rosenbrock(d= 36), Funnel(d= 20), german_credit.Target(), StochasticVolatility()]

    # dimensions = [100, 300, 1000, 3000, 10000]
    # names= [str(d) for d in dimensions]
    # #targets= [StandardNormal(d) for d in dimensions]
    # #targets = [IllConditionedGaussian(d, 100.0) for d in dimensions]
    # targets= [Rosenbrock(d) for d in dimensions]
    #
    key = jax.random.PRNGKey(0)


    if q == 2:

        def ess_ctv_function(alpha, eps, target, num_steps=300000):
            return jnp.average(standardKinetic.Sampler(Target=target, eps=eps).parallel_sample(10, num_steps, alpha * np.sqrt(target.d), key))


        def tuning_ctv(target, eps_min=0.5, eps_max=5.0, num_steps=300000):
            return grid_search.search_wrapper(lambda a, e: ess_ctv_function(a, e, target, num_steps), 0.3, 20, eps_min, eps_max, original_esh)

        borders_ctv = [[0.5, 5.0], [2.0, 9.0], [0.1, 5.0], [0.0001, 0.005], [5000, 10000], [0.004, 0.02]]
        num_steps_ctv = [300000, 300000, 3000000, 3000000, 300000]
        i = -1
        #6.769621324307172e-05
        #standardKinetic.Sampler(Target=targets[i], eps=10000.0).sample(jnp.zeros(targets[i].d), 300000, 1.5 * jnp.sqrt(targets[i].d), key)

        tuning_ctv(targets[i], borders_ctv[i][0], borders_ctv[i][1])
        #results = np.array([np.array(tuning_ctv(targets[i], borders_ctv[i][0], borders_ctv[i][1], num_steps= num_steps_ctv[i])) for i in range(4)])
        #
        # df = pd.DataFrame({'Target ': names[:4], 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})
        # df.to_csv('submission/TablestandardKinetic.csv', sep='\t', index=False)
        # print(df)


    else:

        def ess_function(alpha, eps, target, num_samples): #Run independent chains in parallel to average ESS over the initial conditions (does not give very different results as just runing one chain).
            return jnp.average(ESH.Sampler(Target=target, eps=eps).parallel_sample(10, num_samples, alpha* np.sqrt(target.d), key, generalized=generalized, integrator= integrator, ess=True))

        def ess_function_parallel(alpha, eps, target, num_samples): #Run the parallel version in which all of the chains are used to compute the expectation values. Repeat this many times to do the average.
            return jnp.average(jnp.array([ESH.Sampler(Target=target, eps=eps).parallel_bias(6000, 500, alpha* np.sqrt(target.d), k, integrator) for k in jax.random.split(key, 3)]))

        ESS = ess_function_parallel if parallel else ess_function

        #1.0 for Ross, 5.6 for kappa 1, 2.5 for kappa 100
        #borders_eps = 1.0* np.array([[0.5 * np.sqrt(d/100.0), 2 * np.sqrt(d/100.0)] for d in dimensions])
        #num_samples= [100000 for d in dimensions]
        borders_eps = np.array([[1.0, 4.0], [0.5, 10.0], [0.1, 1.0], [0.1, 1.0], [0.1, 1.0], [0.1, 1.0]])
        num_samples= [30000, 300000, 500000, 300000, 300000, 10000]

        if integrator == 'MN':
            borders_eps *= np.sqrt(10.9)

        if alpha == -1: #do a grid scan over alpha and epsilon
            alpha_min, alpha_max = 4, 15 #for Rosenbrock
            results = np.array([grid_search.search_wrapper(lambda a, e: ESS(a, e, targets[i], num_samples[i]), alpha_min, alpha_max, borders_eps[i][0], borders_eps[i][1]) for i in range(len(targets))])

            df = pd.DataFrame({'Target ': names, 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})


        else: #do a grid scan over epsilon
            results = np.array([grid_search.search_wrapper_1d(lambda e: ESS(alpha, e, targets[i], num_samples[i]), borders_eps[i][0], borders_eps[i][1]) for i in range(2, len(targets))])
            df = pd.DataFrame({'Target ': names, 'ESS': results[:, 0], 'eps': results[:, 1]})

    #df.to_csv('Tests/data/dimensions_dependence/Rossenbrockg.csv', index=False)

    df.to_csv('submission/Table ' + name_sampler + '.csv', sep='\t', index=False)
    print(df)


def energy_fluctuations():

    results = pd.read_csv('submission/Table MCHMC q = 0.csv', sep = '\t')
    print(results)
    num_steps=  np.rint(400 / np.array(results['ESS']))
    alpha = np.array(results['alpha'])
    eps = np.array(results['eps'])

    # targets
    import german_credit
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(d=50, mu1=0.0, mu2=8.0, sigma1=1.0, sigma2=1.0, f=0.2),
               Rosenbrock(d=36), Funnel(d=20), german_credit.Target(), StochasticVolatility()]

    key = jax.random.PRNGKey(0)

    for i in range(len(targets)):
        target = targets[i]

        sampler = ESH.Sampler(target, eps= eps[i])
        L = alpha[i] * jnp.sqrt(target.d)
        key, subkey = jax.random.split(key)
        x0 = target.prior_draw(subkey)
        x, w, E = sampler.sample(x0, num_steps[i], L, key, generalized=False, integrator='LF', monitor_energy=True)

        avgE = np.average(E, weights=w)
        stdE = np.sqrt(np.average(np.square(E - avgE), weights=w))
        E1 = 0.5 * target.d * (1 + np.log(target.d))
        E2 = 0.5 * target.d
        print(stdE / E1, stdE / E2)


def stochastic_volatility():

    target = StochasticVolatility()
    eps = 0.63

    sampler = ESH.Sampler(target, eps)

    key = jax.random.PRNGKey(0)
    key, key_prior = jax.random.split(key)
    x0 = target.prior_draw(key_prior)
    #x0 = jnp.zeros(target.d)
    L = 1.61 * jnp.sqrt(target.d)
    #L = 1e20 * eps


    X, W = sampler.sample(x0, 300000, L, key, generalized= True, ess= False)

    thin = 10
    X= X[::thin, :]
    W = W[::thin]
    print('done sampling')
    print(np.sum(X[:, -2] * W) / np.sum(W))
    print(np.sum(X[:, -1] * W) / np.sum(W))


    def posterior_band(R, W):

        percentiles = [0.25, 0.5, 0.75]
        band = np.empty((len(percentiles), len(R[0])))
        for i in range(len(R[0])):
            perm = np.argsort(R[:, i])
            Ri = R[perm, i]
            Wi = W[perm]

            P = np.cumsum(Wi)
            P /= P[-1]

            band[:, i] = Ri[[np.argmin(np.abs(P - frac)) for frac in percentiles]]

        return band

    band = posterior_band(np.exp(X[:, :-2]), W)
    np.save('Tests/data/stochastic_volatility/MCHMC_posterior_band.npy', band)


    #np.savez('Tests/data/stochastic_volatility/MCHMC_samples.npz', s= X[:, :-2], sigma = X[:, :-2], nu= X[:, :-1], w = W)




def run_problem():
    """Code for runing a generic problem"""


    target = IllConditionedGaussian(d = 100, condition_number= 100.0)
    #target = StandardNormal(d = 100)
    #grid_search.search_wrapper_1d(lambda e: jnp.average(ESH.Sampler(Target=target, eps=e).parallel_sample(10, 300000, 1.6 * np.sqrt(target.d), jax.random.PRNGKey(0), ess=True)), 0.1, 1.5)

    sampler = ESH.Sampler(target, eps= 1.0)
    L = 1e20#1.6 * jnp.sqrt(target.d)
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x0 = target.prior_draw(subkey)
    x, w, E = sampler.sample(x0, 10000, L, key, generalized= False, integrator= 'LF', monitor_energy= True)

    avgE = np.average(E, weights= w)
    stdE = np.sqrt(np.average(np.square(E - avgE), weights= w))

    print(stdE, stdE / avgE, avgE, 0.5 * target.d *(1 + np.log(target.d)))
    # X, W = sampler.parallel_sample(5, 10000, L, key, generalized= False, integrator='LF')
    #
    # for i in range(5):
    #     plt.plot(W[i])
    # plt.show()


def esh_not_converging():

    target = IllConditionedESH()
    bounces = False
    L = 1.6 * jnp.sqrt(target.d) if bounces else 1e20
    eps = 1.0 #if bounces else 0.1
    num_chains = 500
    num_steps = 10000

    sampler = ESH.Sampler(target, eps)

    key = jax.random.PRNGKey(1)
                    # time  coordinate   chain
    results = np.empty((4, target.d, num_chains))

    for i in range(num_chains):
        key, key_prior, key_bounces = jax.random.split(key, 3)
        x0 = target.prior_draw(key_prior)#jax.random.normal(key_prior, shape= (target.d, ), dtype = 'float64')
        X, w = sampler.sample(x0, num_steps, L, key_bounces)

        results[:, :, i] = X[[0, 100, 1000, 10000], :]
        results[0, :, i] = x0

    np.save('ESH_not_converging/data/ESHexample_eps1_'+('MCHMC' if bounces else 'ESH')+'.npy', results)





def divergence():

    sampler = ESH.Sampler(IllConditionedGaussian(d = 100, condition_number=1000), eps = 3.0)
    key = jax.random.PRNGKey(0)
    key, prior_key = jax.random.split(key)
    x0 = sampler.Target.prior_draw(prior_key)
    X, w = sampler.sample(x0, 10000, 1e20, key, ess = True)
    print(np.any(np.isnan(w)))
    
    
def epsilon_dimension_dependence():

    generalized= False
    key = jax.random.PRNGKey(0)
    dimensions = [100, 300, 1000, 3000]
    #dimensions = [3000, ]

    def ess_function(eps, d):
        L = 1.6 * np.sqrt(d)
        target = IllConditionedGaussian(d= d, condition_number= 100.0)
        return jnp.average(ESH.Sampler(Target=target, eps=eps).parallel_sample(10, 100000, L, key, generalized= generalized, integrator= 'LF', ess=True))

    #5.6 for kappa = 1,

    def scan(d):
        print(d)
        eps_expected= 2.3 * np.sqrt(d / 100.0)
        epsilon = jnp.logspace(np.log10(0.5 *eps_expected), np.log10(2 * eps_expected), 24)

        ess = parallel_run(lambda e: ess_function(e, d), epsilon)

        j = jnp.argmax(ess)
        ess_best = ess[j]
        eps = epsilon[j]

        # plt.plot(epsilon, ess, '.:')
        # plt.show()

        cf_mask = ess > ess_best * 0.9
        cf_low, cf_high = np.min(epsilon[cf_mask]), np.max(epsilon[cf_mask])

        return np.array([d, ess_best, eps, cf_low, cf_high, ess_best])


    np.save('Tests/data/epsilon_scaling_kappa100.npy', [scan(d) for d in dimensions])


if __name__ == '__main__':

    #stochastic_volatility()
    #esh_not_converging()
    #table1()
    #energy_fluctuations()
    #run_problem()
    #epsilon_dimension_dependence()

    #full_bias()
    dimension_dependence()

    #ill_conditioned(tunning=False, generalized=False)
    #ill_conditioned(tunning=True)

