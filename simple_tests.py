import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import ESH
import CTV
from targets import *
import grid_search

import jax
import jax.numpy as jnp


num_cores = 6
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

def parallel_run(function, values):
    parallel_function= jax.pmap(jax.vmap(function))
    results = jnp.array(parallel_function(values.reshape(num_cores, len(values) // num_cores)))
    return results.reshape([len(values), ] + [results.shape[i] for i in range(2, len(results.shape))])



def bounce_frequency(d, alpha, generalized = False, prerun = 0):

    key = jax.random.PRNGKey(1)

    length = alpha * jnp.sqrt(d)
    sampler = ESH.Sampler(Target= StandardNormal(d=d), eps=1.0)
    #sampler = CTV.Sampler(Target= StandardNormal(d=d), eps=2.0)

    #sampler = ESH.Sampler(Target= IllConditionedGaussian(d=d, condition_number=100.0), eps=1.0)

    ess = parallel_run(lambda L: sampler.sample_multiple_chains(10, 300000, L, key, generalized= generalized, ess= True, prerun=prerun), length)

    return jnp.average(ess, 1), jnp.std(ess, 1)

    #eta = (2.93 * np.power(d, -0.78) * np.logspace(-0.8, 0.8, 24))[n]

    #length = 1.5 * np.sqrt(d)
    #sampler = CTV.Sampler(Target = IllConditionedGaussian(d= d, condition_number=100), eps= 3)
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

    avg_island_size = esh.sample(x0, L, prerun_steps= 500, track= 'ModeMixing')

    return [avg_island_size, mu, L, eps, d]



def ill_conditioned_workhorse(alpha, generalized, prerun):
    d = 100
    L = alpha * np.sqrt(d)

    def f(n):
        kappa = jnp.logspace(0, 5, 18)[n]
        eps = jnp.array(10 * [2.0, ] + 2 * [1.5, ] + 3 * [1.0, ] + [0.7, ] + 2 * [0.5, ])[n]

        key = jax.random.PRNGKey(0)

        sampler = ESH.Sampler(Target=IllConditionedGaussian(d=d, condition_number=kappa), eps=eps)

        ess = sampler.sample_multiple_chains(10, 300000, L, key, generalized = generalized, ess=True, prerun= prerun)

        return jnp.array([jnp.average(ess), jnp.std(ess)])

    results = parallel_run(f, np.arange(18, dtype = int))

    return results


def ill_conditioned(tunning, generalized, prerun = 0):
    word = '_l' if generalized else ''
    if prerun != 0:
        word += '_t'

    if not tunning:
        results = ill_conditioned_workhorse(1.0, generalized, prerun)
        np.save('Tests/data/kappa/no_tuning'+word+'.npy', results)

    else:
        alpha = (1.5 * jnp.logspace(-0.8, 0.8, 24))

        results= np.empty((len(alpha), 18, 2))

        for i in range(len(alpha)):
            print(str(i)+ '/' + str(len(alpha)))
            results[i] = ill_conditioned_workhorse(alpha[i], generalized, prerun)

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


def dimension_dependence():

    dimensions = [50, 100, 200, 500, 1000, 3000, 10000]
    alpha = (15 * jnp.logspace(-0.8, 0.8, 24))
    #alpha = (1.5 * jnp.logspace(-0.5, 0.5, 12))
    #condition_numbers = np.logspace(0, 5, 18)
    dict = {'alpha': alpha}
    generalized, prerun = True, 0
    for d in dimensions:
        print(d)
        avg, std = bounce_frequency(d, alpha, generalized, prerun)
        dict.update({'ess (d='+str(d)+')': avg, 'err ess (d='+str(d)+')': std})
    df = pd.DataFrame.from_dict(dict)
    df.to_csv('Tests/data/dimensions/StandardNormal'+('_l' if generalized else '')+'.csv', sep='\t', index=False)


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


def table1():

    esh, generalized = True, False
    key = jax.random.PRNGKey(0)
    import inferencegym

    #targets
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(d=50, mu1=0.0, mu2=8.0, sigma1=1.0, sigma2=1.0, f=0.2), Rosenbrock(d= 36), Funnel(d= 20), inferencegym.Target('German Credit')]


    if esh:


        def ess_function(alpha, eps, target, generalized, prerun):
            return jnp.average(ESH.Sampler(Target=target, eps=eps).sample_multiple_chains(10, 1000000, alpha* np.sqrt(target.d), key, generalized=generalized, ess=True, prerun=prerun))


        def tuning(target, generalized, prerun, eps_min = 0.1, eps_max = 1.0):
            return grid_search.search_wrapper(lambda a, e: ess_function(a, e, target, generalized, prerun), 0.3, 20.0, eps_min, eps_max)


        borders_esh = [[1.0, 4.0], [0.5, 3.0], [0.1, 1.0], [0.1, 1.0], [0.1, 1.0]]
        results = np.array([np.array(tuning(targets[i], generalized, 0, borders_esh[i][0], borders_esh[i][1])) for i in range(len(targets))])

        df = pd.DataFrame({'Target ': names, 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})
        #df.to_csv('submission/TableESH_generalized.csv', sep='\t', index=False)
        print(df)


    else:

        def ess_ctv_function(alpha, eps, target, num_steps=300000):
            return jnp.average(
                CTV.Sampler(Target=target, eps=eps).sample_multiple_chains(10, num_steps, alpha * np.sqrt(target.d), key))


        def tuning_ctv(target, eps_min=0.5, eps_max=5.0, num_steps=300000):
            return grid_search.search_wrapper(lambda a, e: ess_ctv_function(a, e, target, num_steps), 0.3, 20, eps_min, eps_max)

        borders_ctv = [[0.5, 5.0], [2.0, 9.0], [0.1, 5.0], [0.0001, 0.005], [5000, 10000]]
        num_steps_ctv = [300000, 300000, 3000000, 3000000, 300000]
        i = 4
        #6.769621324307172e-05
        #CTV.Sampler(Target=targets[i], eps=10000.0).sample(jnp.zeros(targets[i].d), 300000, 1.5 * jnp.sqrt(targets[i].d), key)

        tuning_ctv(targets[i], borders_ctv[i][0], borders_ctv[i][1])
        #results = np.array([np.array(tuning_ctv(targets[i], borders_ctv[i][0], borders_ctv[i][1], num_steps= num_steps_ctv[i])) for i in range(4)])
        #
        # df = pd.DataFrame({'Target ': names[:4], 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})
        # df.to_csv('submission/TableCTV.csv', sep='\t', index=False)
        # print(df)


def run_problem():
    target = Rosenbrock(d= 32)
    eps = 0.1

    sampler = ESH.Sampler(target, eps)

    key = jax.random.PRNGKey(0)
    key, key_prior = jax.random.split(key)
    x0 = target.prior_draw(key_prior)
    L = 20 * jnp.sqrt(target.d)
    #L = 1e20 * eps
    ess = sampler.sample(x0, 1000000, L, key, generalized= False, ess= True)

    print(ess)


def divergence():

    sampler = ESH.Sampler(IllConditionedGaussian(d = 100, condition_number=1000), eps = 3.0)
    key = jax.random.PRNGKey(0)
    key, prior_key = jax.random.split(key)
    x0 = sampler.Target.prior_draw(prior_key)
    X, w = sampler.sample(x0, 10000, 10000000000, key, ess = True)
    print(np.any(np.isnan(w)))



if __name__ == '__main__':

    #run_problem()

    full_bias()
    #dimension_dependence()

    #ill_conditioned(tunning=False, generalized=False, prerun=1000)
    #ill_conditioned(tunning=True)
