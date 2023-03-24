import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp

num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)


from sampling import sampler as mchmc
from sampling import standardKinetic
from sampling.benchmark_targets import *
from sampling import grid_search
from HMC import myHMC
from sampling import german_credit

### Runs the bencmark problems. """

def parallel_run(function, values):
    parallel_function= jax.pmap(jax.vmap(function))
    results = jnp.array(parallel_function(values.reshape(num_cores, len(values) // num_cores)))
    return results.reshape([len(values), ] + [results.shape[i] for i in range(2, len(results.shape))])


def bounce_frequency(d, alpha, generalized = False):
    """For scaning over the L parameter"""


    length =alpha * jnp.sqrt(d)
    sampler = mchmc.Sampler(StandardNormal(d=d), 'LF', generalized)
    eps= 7.3 * jnp.sqrt(d / 100.0)

    def f(l):
        sampler.set_hyperparameters(l, eps)
        return sampler.sample(30000, 10, output= 'ess')

    ess = parallel_run(f, length)

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
    reduced_steps = mchmc.point_reduction(steps, 100)
    bias = np.empty((len(length), len(reduced_steps)))
    for n in range(len(length)):
        print(n)
        sampler = mchmc.Sampler(StandardNormal(d= d), length[n], 1.0, 'LF', False)
        bias[n, :] = sampler.sample(steps, output= 'ess')

    np.save('data/full_bias.npy', bias)



def ill_conditioned():
    condition_numbers = jnp.logspace(0, 5, 18)
    integrator= 'MN'
    generalized = True
    name_sampler = integrator + ('_g' if generalized else '')

    targets = [IllConditionedGaussian(d= 100, condition_number= kappa) for kappa in condition_numbers]
    num_samples = [2000 * (int)(np.power(kappa, 0.3)) for kappa in condition_numbers]


    def ESS(alpha, eps, target, num_samples):  #sequential mode. Only runs a handful of chains to average ESS over the initial conditions
        sampler = mchmc.Sampler(target, alpha* np.sqrt(target.d), eps, integrator, generalized)
        return jnp.average(sampler.sample(num_samples, 12, output = 'ess', tune= 'none'))

    def std_ESS(alpha, eps, target, num_samples):  #sequential mode. Only runs a handful of chains to average ESS over the initial conditions
        sampler = mchmc.Sampler(target, alpha * np.sqrt(target.d), eps, integrator, generalized)
        return jnp.std(sampler.sample(num_samples, 12, output= 'ess', tune = 'none'))


    borders_eps = np.array([[7.0 / np.power(kappa, 0.25) / 1.5, 7.0 / np.power(kappa, 0.25) * 1.5] for kappa in condition_numbers])

    if integrator == 'MN':
        borders_eps *= np.sqrt(10.9)

    borders_alpha = np.array([[0.5 * np.power(kappa, 0.1), 2 * np.power(kappa, 0.1)] for kappa in condition_numbers])
    results = np.array([grid_search.search_wrapper(lambda a, e: ESS(a, e, targets[i], num_samples[i]), borders_alpha[i, 0], borders_alpha[i, 1], borders_eps[i, 0], borders_eps[i, 1], save_name = 'kappa_num'+str(i)) for i in range(len(targets))])

    ess_errors = np.array([std_ESS(results[i, 1], results[i, 2], targets[i], num_samples[i]) for i in range(len(condition_numbers))])

    df = pd.DataFrame({'Condition number': condition_numbers, 'ESS': results[:, 0], 'err ESS': ess_errors, 'alpha': results[:, 1], 'eps': results[:, 2]})

    df.to_csv('submission/MCHMC/ICG/New__' + name_sampler + '.csv', index=False)
    print(df)



def ill_conditioned_tuning_free():
    condition_numbers = jnp.logspace(0, 5, 18)
    integrator= 'LF'
    generalized = True

    targets = [IllConditionedGaussian(d= 100, condition_number= kappa) for kappa in condition_numbers]
    num_samples = [2000 * (int)(np.power(kappa, 0.4)) for kappa in condition_numbers]
    print([k//40 for k in num_samples])

    def ESS(target, num_samples):  #sequential mode. Only runs a handful of chains to average ESS over the initial conditions
        sampler = mchmc.Sampler(target, integrator= integrator, generalized= generalized)
        sampler.num1, sampler.num2 = 40, 40
        ess = sampler.sample(num_samples, 12, output= 'ess', tune= 'cheap')
        return jnp.average(ess), jnp.std(ess)

    results = np.array([ESS(targets[i], num_samples[i]) for i in range(len(targets))])

    df = pd.DataFrame({'Condition number': condition_numbers, 'ESS': results[:, 0], 'err ESS': results[:, 1]})

    df.to_csv('submission/MCHMC/ICG/NewTable_ICG_tuning_free'+('_g' if generalized else '')+'.csv', index=False)
    print(df)



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
    df.to_csv('data/dimensions/StandardNormal'+('_g' if generalized else '')+'_eps4.csv', sep='\t', index=False)



def table1():
    """For generating Table 1 in the paper"""

    #version of the sampler
    q = 0 #choice of the Hamiltonian (q = 0 or q = 2)
    generalized = True #choice of the momentum decoherence mechanism
    alpha = -1.0 #bounce frequency (1.0 for generalized, 1.6 for bounces, something very large if no bounces). If -1, alpha is tuned by a grid search.
    integrator = 'LF' #integrator (Leapfrog (LF) or Minimum Norm (MN))
    HMC = False

    #name of the version
    if alpha > 1e10:
        generalized_string= 'no-bounces_'
        alpha_string = ''
    else:
        generalized_string = 'generalized_' if generalized else 'bounces_'
        alpha_string = '_tuning-free' if (alpha > 0) else ''
    #parallel_string = '_parallel' if parallel else ''
    name_sampler = generalized_string + integrator + '_q=' + str(q) + alpha_string #parallel_string
    
    if HMC:
        name_sampler = 'HMC' + '_' + integrator
    
    print(name_sampler)

    #targets
    indexes = [0, 1, 2, 3, 4, 5]
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(), Rosenbrock(), Funnel(), german_credit.Target(), StochasticVolatility()]


    #dimensions = [100, 300, 1000, 3000, 10000]
    #names= [str(d) for d in dimensions]
    #targets= [StandardNormal(d) for d in dimensions]
    #targets = [IllConditionedGaussian(d, 100.0) for d in dimensions]
    #targets= [Rosenbrock(d) for d in dimensions]

    key = jax.random.PRNGKey(0)

    if HMC:
        
        def ESS(length, eps, target, num_samples): #sequential mode. Only runs a handful of chains to average ESS over the initial conditions
            return jnp.average(myHMC.Sampler(Target=target, L = length, eps=eps, integrator= integrator).parallel_sample(10, num_samples, random_key= key, ess=True))

        eps = np.array([[0.02, 0.6], [0.01, 1.0], [0.01, 1.0], [0.01, 1.0], [0.01, 1.0], [0.002, 0.05]])
        L = np.array([[0.3, 2], [0.5, 3.0], [3, 25.0], [0.1, 5.0], [0.1, 5.0], [0.1, 3]])

        num_samples= [100000, 300000, 500000, 300000, 300000, 100000]


        results = np.array([grid_search.search_wrapper(lambda a, e: ESS(a, e, targets[i], num_samples[i]), L[i][0], L[i][1], eps[i][0], eps[i][1]) for i in indexes])

        df = pd.DataFrame({'Target ': [names[i] for i in indexes], 'ESS': results[:, 0], 'L': results[:, 1], 'eps': results[:, 2]})


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

        def ESS(alpha, eps, target, num_samples):  #sequential mode. Only runs a handful of chains to average ESS over the initial conditions
            sampler = mchmc.Sampler(target, alpha * np.sqrt(target.d), eps, integrator, generalized)
            return jnp.average(sampler.sample(num_samples, 12, output= 'ess'))

        def ESS_tf(target, num_samples):  #tuning-free sequential mode. Only runs a handful of chains to average ESS over the initial conditions
            sampler = mchmc.Sampler(target, integrator= integrator, generalized= generalized)
            sampler.tune_hyperparameters()
            ess= jnp.average(sampler.sample(num_samples, 12, output= 'ess'))
            print(ess)
            return ess, sampler.L / np.sqrt(target.d), sampler.eps

        #1.0 for Ross, 5.6 for kappa 1, 2.5 for kappa 100
        #borders_eps = 1.0* np.array([[0.5 * np.sqrt(d/100.0), 2 * np.sqrt(d/100.0)] for d in dimensions])
        #num_samples= [100000 for d in dimensions]
        borders_eps = np.array([[1.0, 4.0], [0.5, 10.0], [0.1, 1.0], [0.1, 1.5], [0.1, 1.0], [0.1, 1.0]])
        borders_alpha = np.array([[0.3, 3], [0.3, 3], [10, 40], [0.3, 10], [0.3, 3], [0.3, 3]])


        num_samples= [30000, 100000, 300000, 100000, 100000, 10000]

        #
        # print(ESS(0.7, 0.23, targets[-3], num_samples[-3]))
        # print(ESS(0.7, 0.15, targets[-3], num_samples[-3]))
        # exit()

        # df = pd.read_csv('submission/Table generalized_LF_q=0.csv')
        # df_tf = pd.read_csv('submission/Table generalized_LF_q=0_tuning-free3.csv')
        # L = 0.4 * np.array(df_tf['eps'] / df_tf['ESS'])
        # eps = np.array(df_tf['eps'])
        # for i_target in range(5, 6):
        #     print(names[i_target] + ': ' +str(ESS(L[i_target] / np.sqrt(targets[i_target].d), eps[i_target], targets[i_target], num_samples[i_target])))
        #
        # exit()
        #print(ESS(0.19, 0.6, targets[-1], 30000))


        if integrator[:2] == 'MN':
            print('is minimal norm')
            borders_eps *= np.sqrt(10.9)

        if alpha < 0: #do a grid scan over alpha and epsilon
            results = np.array([grid_search.search_wrapper(lambda a, e: ESS(a, e, targets[i], num_samples[i]), borders_alpha[i][0], borders_alpha[i][1], borders_eps[i][0], borders_eps[i][1]) for i in indexes])
            print(results)
            df = pd.DataFrame({'Target ': [names[i] for i in indexes], 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})


        else:

            results = np.array([ESS_tf(targets[i], num_samples[i]) for i in indexes])
            #results = np.array([grid_search.search_wrapper_1d(lambda e: ESS(alpha * sigma[i], e, targets[i], num_samples[i]), borders_eps[i][0], borders_eps[i][1]) for i in range(len(targets))])
            df = pd.DataFrame({'Target ': [names[i] for i in indexes], 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})

    #df.to_csv('data/dimensions_dependence/Rossenbrockg.csv', index=False)

    df.to_csv('submission/MCHMC/NewTable ' + name_sampler + '.csv', index=False)
    print(df)



def stochastic_volatility():

    target = StochasticVolatility()

    sampler = mchmc.Sampler(target, 1.61 * jnp.sqrt(target.d), 0.63, 'LF', True)


    X, nburnin = sampler.sample(300000)

    thin = 10
    X= X[nburnin::thin, :]
    print('done sampling')

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

    band = posterior_band(np.exp(X[:, :-2]))
    np.save('data/stochastic_volatility/MCHMC_posterior_band.npy', band)


    #np.savez('data/stochastic_volatility/MCHMC_samples.npz', s= X[:, :-2], sigma = X[:, :-2], nu= X[:, :-1], w = W)



def esh_not_converging():

    target = IllConditionedESH()
    bounces= False
    L = np.sqrt(np.average(target.variance)) * np.sqrt(target.d) if bounces else np.inf

    sampler = mchmc.Sampler(target, L, 0.5, 'LF', False)

    X, nburnin = sampler.sample(10000, 500)

    np.save('ESH_not_converging/data/ESHexample_'+('MCHMC' if bounces else 'ESH')+'.npy', X[:, [0, 100, 1000, 10000], :])


def energy_time_chains():

    target = Rosenbrock(d = 100)
    #target = StochasticVolatility()

    sampler = mchmc.Sampler(target, np.sqrt(target.d), 10.0, integrator= 'LF')
    epsilon = np.logspace(np.log10(0.05), np.log10(0.3), 10)
    dE= np.empty((len(epsilon), 4))

    for i in range(len(epsilon)):
        print(epsilon[i])
        sampler.eps = epsilon[i]
        X, E, nburnin = sampler.sample(2000, 300, output = 'energy')
        E = E[:, np.max(nburnin):]
        de1 = np.std(E, axis = 1)**2 / target.d
        de2 = np.average(np.square(E[:, 1:] - E[:, :-1]), axis= 0) / target.d

        dE[i, 0] = np.average(de1)
        dE[i, 1] = np.std(de1)
        dE[i, 2] = np.average(de2)
        dE[i, 3] = np.std(de2)

    df = pd.DataFrame(dE, columns = ['avg time', 'std time', 'avg chains', 'std chains'])
    df['epsilon'] = epsilon
    df.to_csv('energy_fluctuations_time_chains_Rosenbrock.csv')



def plot_energy_time_chains():
    from scipy.stats import linregress
    words = ['time', 'chains']
    fmts= ['s:', 'o:']
    colors = ['tab:blue', 'tab:red']
    target_names = ['STN', "Rosenbrock"]
    DF= [pd.read_csv('energy_fluctuations_time_chains_'+target_name+'.csv') for target_name in target_names]

    for j in range(len(target_names)): #targets
        df = DF[j]
        for i in range(2): #time vs chains
            word = words[i]
            x, y = df['epsilon'], df['avg '+word]
            yerr = df['std '+word]
            label = target_names[j] + '('+word+')'
            plt.errorbar(x, y, yerr = yerr, capsize= 2.0, fmt =fmts[i], label = label, color = colors[j])

            print(label, linregress(np.log(x), np.log(y)))

    plt.legend()
    plt.ylabel('Var[E]/d')
    plt.xlabel('stepsize')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    ill_conditioned()
    #ill_conditioned_tuning_free()
    #table1()

