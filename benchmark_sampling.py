import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import ESH
import standardKinetic
from benchmark_targets import *
import grid_search
import myHMC

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
    steps = 1000000
    reduced_steps = ESH.point_reduction(steps, 100)
    bias = np.empty((len(length), len(reduced_steps)))
    for n in range(len(length)):
        print(n)
        sampler = ESH.Sampler(Target= StandardNormal(d= d), eps=1.0)
        key = jax.random.PRNGKey(1)
        key, subkey = jax.random.split(key)
        x0 = sampler.Target.prior_draw(subkey)
        bias[n, :] = sampler.sample(x0, steps, length[n], key, integrator='LF', generalized=False, ess= True)

    np.save('Tests/data/full_bias.npy', bias)


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



def table1():
    """For generating Table 1 in the paper"""

    #version of the sampler
    q = 0 #choice of the Hamiltonian (q = 0 or q = 2)
    generalized = True #choice of the momentum decoherence mechanism
    alpha = -1 #bounce frequency (1.0 for generalized, 1.6 for bounces, something very large if no bounces). If -1, alpha is tuned by a grid search.
    integrator = 'LF' #integrator (Leapfrog (LF) or Minimum Norm (MN))
    parallel = False
    HMC = False

    #name of the version
    if alpha > 1e10:
        generalized_string= 'no-bounces_'
        alpha_string = ''
    else:
        generalized_string = 'generalized_' if generalized else 'bounces_'
        alpha_string = '_tuning-free' if (alpha > 0) else ''
    parallel_string = '_parallel' if parallel else ''
    name_sampler = generalized_string + integrator + '_q=' + str(q) + alpha_string + parallel_string
    
    if HMC:
        name_sampler = 'HMC'
    
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

    if HMC:
        
        def ESS(length, eps, target, num_samples): #sequential mode. Only runs a handful of chains to average ESS over the initial conditions
            return jnp.average(myHMC.Sampler(Target=target, eps=eps).parallel_sample(3, num_samples, length, key, generalized= generalized, integrator= integrator, ess=True))

        eps = np.array([[0.01, 1], [0.01, 1.0], [0.01, 1.0], [0.01, 1.0], [0.01, 1.0], [0.002, 0.05]])
        L = np.array([[0.05, 5], [0.5, 3.0], [3, 25.0], [0.1, 5.0], [0.1, 5.0], [0.1, 3]])

        num_samples= [30000, 300000, 500000, 300000, 300000, 100000]

        results = np.array([grid_search.search_wrapper(lambda a, e: ESS(a, e, targets[i], num_samples[i]), L[i][0], L[i][1], eps[i][0], eps[i][1]) for i in range(5, len(targets))])

        df = pd.DataFrame({'Target ': names, 'ESS': results[:, 0], 'L': results[:, 1], 'eps': results[:, 2]})


    elif q == 2:

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


    else:

        def ess_function(alpha, eps, target, num_samples):  #sequential mode. Only runs a handful of chains to average ESS over the initial conditions
            return jnp.average(ESH.Sampler(Target=target, eps=eps).parallel_sample(10, num_samples, alpha* np.sqrt(target.d), key, generalized=generalized, integrator= integrator, ess=True))

        def ess_function_parallel(alpha, eps, target, num_samples): #Run the parallel version in which all of the chains are used to compute the expectation values. Repeat this a few times to do the average.
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
            alpha_min, alpha_max = 0.3, 20
            results = np.array([grid_search.search_wrapper(lambda a, e: ESS(a, e, targets[i], num_samples[i]), alpha_min, alpha_max, borders_eps[i][0], borders_eps[i][1]) for i in range(len(targets))])

            df = pd.DataFrame({'Target ': names, 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})


        else: #do a grid scan over epsilon
            results = np.array([grid_search.search_wrapper_1d(lambda e: ESS(alpha, e, targets[i], num_samples[i]), borders_eps[i][0], borders_eps[i][1]) for i in range(2, len(targets))])
            df = pd.DataFrame({'Target ': names, 'ESS': results[:, 0], 'eps': results[:, 1]})



    #df.to_csv('Tests/data/dimensions_dependence/Rossenbrockg.csv', index=False)

    df.to_csv('submission/Table_adjoint_ ' + name_sampler + '.csv', sep='\t', index=False)
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



def esh_not_converging_debug():

    target = IllConditionedESH()
    bounces = False
    L = 1.6 * jnp.sqrt(target.d) if bounces else 1e20
    eps = 1.0  # 0.1 was used in ESH, but 1 gives better results
    num_chains = 100
    num_steps = 1000

    sampler = ESH.Sampler(target, eps)

    key = jax.random.PRNGKey(1)

    thin = 10
                    # time  coordinate   chain
    results = np.empty((1000//thin, target.d, num_chains))

    for i in range(num_chains):
        key, key_prior, key_bounces = jax.random.split(key, 3)
        x0 = target.prior_draw(key_prior)#jax.random.normal(key_prior, shape= (target.d, ), dtype = 'float64')
        X, w = sampler.sample(x0, num_steps, L, key_bounces)

        results[1:, :, i] = X[thin::thin, :]
        results[0, :, i] = x0

    np.save('ESH_not_converging/data/ESHexample_eps1_'+('MCHMC' if bounces else 'ESH')+'.npy', results)



def full_bias_eps():
    target = Rosenbrock(d = 100)
    epsilon = np.logspace(np.log10(0.01), np.log10(1.5), 15)

    #target = Rosenbrock(d = 100)
    #epsilon = np.linspace(0.05, 1.5, 10)
    num_steps = 100000
    num_saved_steps = len(ESH.point_reduction(num_steps, 100))
    bias = np.empty((len(epsilon), num_saved_steps))
    Estd = np.empty(len(epsilon))
    importance_weight_factor = np.empty(len(epsilon))
    sampler = ESH.Sampler(target, eps= 1.0)
    L = 15 * jnp.sqrt(target.d)
    key = jax.random.PRNGKey(0)

    for i in range(len(epsilon)):
        print(i)
        sampler.eps = epsilon[i]
        #bias[i, :] = sampler.sample('prior', num_steps, L, key, generalized= False, integrator= 'LF', ess= True)
        X, W, E = sampler.sample('prior', num_steps, L, key, generalized= False, integrator= 'LF', monitor_energy=True)
        Estd[i] = np.sqrt(np.average(np.square(E - np.average(E, weights= W)), weights=W)) / target.d
        importance_weight_factor[i] = np.average(W)**2 / np.average(np.square(W))

    np.save('Tests/data/bias_variance/energy_weights.npy', [Estd, importance_weight_factor])

    #np.save('Tests/data/bias_variance/rosenbrock_b.npy', bias)






if __name__ == '__main__':

    #full_bias()
    #esh_not_converging_debug()
    #dimension_dependence()
    full_bias_eps()
    #plot_full_bias()

    #table1()
    #run_problem()


