import numpy as np
from scipy.stats import special_ortho_group

from jax import random
from numpyro.infer import MCMC, NUTS

import benchmark_targets_HMC as targets
import benchmark_targets

import pandas as pd


### Applying NUTS from numpyro to the benchmark problems ###



def bias(X, w, var_true):
    """ Bias = average over dimensions ( variance(samples) - variance(true) )^2 / variance(true)^2 )
        Args:
            X: samples, an array of shape (num_samples, d)
            w: weight of each sample, an array of shape (num_samples, )
            var_true: an exact variance of each dimension, an array of shape (d, )
        Returns:
            variance bias as a function of number of samples (in the way they are ordered)
    """
    var = (np.cumsum((np.square(X.T) * w), axis = 1) / np.cumsum(w)).T

    return np.sqrt(np.average(np.square((var - var_true) / var_true), axis = 1))


def ess_cutoff_crossing(b, step_cost):

    cutoff = 0.1

    n_crossing = 0
    while b[n_crossing] > cutoff:
        n_crossing += 1
        if n_crossing == len(b):
            return 0, n_crossing

    return 200.0 / np.sum(step_cost[:n_crossing + 1]), n_crossing



def sample_nuts(target, target_params, num_samples, key_num = 0, d= 1, names_output = None):
    """ 'default' nuts, tunes the step size with a prerun"""

    # setup
    nuts_setup = NUTS(target, adapt_step_size=True, adapt_mass_matrix=True, dense_mass= False)  # originally: nuts_kernel
    sampler = MCMC(nuts_setup, num_warmup=500, num_samples=num_samples, num_chains=1, progress_bar=False)

    # prior
    if key_num != 0:
        key = random.PRNGKey(key_num)
        key, prior_key = random.split(key)
        x0 = random.normal(prior_key, shape=(d,), dtype='float64')

        # run
        sampler.run(key, *target_params, init_params=x0, extra_fields=['num_steps'])

    else:
        # run
        sampler.run(random.PRNGKey(0), *target_params, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()
    if names_output == None:
        X = np.array(numpyro_samples['x'])

    else:
        X = {name: np.array(numpyro_samples[name]) for name in names_output}

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    return X, steps



def ill_conditioned_scan():

    d = 100
    R = special_ortho_group.rvs(d, random_state=0)
    repeat = 10

    def f(condition_number, num_samples):
        ess_arr = np.zeros(repeat)
        for key_num in range(repeat):
            X, steps = sample_nuts(targets.ill_conditioned_gaussian, [d, condition_number], num_samples, d = d, key_num=key_num)

            variance_true = np.logspace(-0.5*np.log10(condition_number), 0.5*np.log10(condition_number), d)
            ess, n_crossing = ess_cutoff_crossing(bias((R @ X.T).T, np.ones(len(X)), variance_true), steps)
            ess_arr[key_num] = ess

        return np.average(ess_arr), np.std(ess_arr), n_crossing


    kappa_arr = np.logspace(0, 5, 18)

    ess_arr= np.zeros(len(kappa_arr))
    ess_err_arr= np.zeros(len(kappa_arr))

    num_samples= 1000

    for i in range(len(kappa_arr)):
        print(i, num_samples)

        ess_arr[i], ess_err_arr[i], n_crossing = f(kappa_arr[i], num_samples)

        num_samples = n_crossing * 5

    np.save('data/kappa/NUTS.npy', [ess_arr, ess_err_arr, kappa_arr])



def dimension_dependence():


    def gauss(d, num_samples):
        R = special_ortho_group.rvs(d, random_state=0)
        kappa = 100.0
        variance_true = np.logspace(-0.5*np.log10(kappa), 0.5*np.log10(kappa), d)

        X, steps = sample_nuts(targets.ill_conditioned_gaussian, [d, kappa], num_samples)
        X = (R @ X.T).T
        B = bias(X, np.ones(len(X)), variance_true)
        ess, n_crossing = ess_cutoff_crossing(B, steps)

        return ess, n_crossing


    def rosenbrock(d, num_samples):

        Q, var_x, var_y = 0.5, 2.0, 10.498957879911487
        variance_true = np.concatenate((var_x * np.ones(d//2), var_y * np.ones(d//2)))

        # setup
        nuts_setup = NUTS(targets.rosenbrock, adapt_step_size=True, adapt_mass_matrix=True)
        sampler = MCMC(nuts_setup, num_warmup=500, num_samples=num_samples, num_chains=1, progress_bar=False)

        # run
        sampler.run(random.PRNGKey(0), d, Q, extra_fields=['num_steps'])

        # get results
        numpyro_samples = sampler.get_samples()

        steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

        X = np.concatenate((np.array(numpyro_samples['x']).T, np.array(numpyro_samples['y']).T)).T

        B = bias(X, np.ones(len(X)), variance_true)
        ess, n_crossing = ess_cutoff_crossing(B, steps)

        return ess, n_crossing

    dimensions = np.logspace(np.log10(50), 4, 18, dtype= int)

    ess_arr= np.zeros(len(dimensions))
    num_samples= 1000

    for i in range(len(dimensions)):
        print(i, num_samples)

        ess, n_crossing = gauss(dimensions[i], num_samples)
        #ess, n_crossing = rosenbrock(dimensions[i], num_samples)

        ess_arr[i] = ess

        num_samples = n_crossing * 5

    np.save('data/dimensions/Kappa100_NUTS.npy', [ess_arr, dimensions])
    #np.save('data/dimensions/Rosenbrock_NUTS.npy', [ess_arr, dimensions])



def ill_conditioned(key_num, d = 100, condition_number = 100.0):

    R = special_ortho_group.rvs(d, random_state=0)

    nuts_setup = NUTS(targets.ill_conditioned_gaussian, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)
    sampler = MCMC(nuts_setup, num_warmup=500, num_samples=1000, num_chains=1, progress_bar=True)

    # prior
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    x0 = random.normal(prior_key, shape=(d,), dtype='float64')

    # run
    sampler.warmup(key, d, condition_number, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    warmup_calls = np.sum(np.array(sampler.get_extra_fields()['num_steps'], dtype=int))
    sampler.run(key, d, condition_number, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()
    X = np.array(numpyro_samples['x'])

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)
    variance_true = np.logspace(-0.5*np.log10(condition_number), 0.5*np.log10(condition_number), d)
    ess, n_crossing = ess_cutoff_crossing(bias((R @ X.T).T, np.ones(len(X)), variance_true), steps)

    return ess, ess / (1 + ess* warmup_calls / 200.0)



def bimodal(key_num):

    d = 50
    nuts_setup = NUTS(targets.bimodal_hard, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)  # originally: nuts_kernel
    sampler = MCMC(nuts_setup, num_warmup=500, num_samples=10000, num_chains=1, progress_bar=False)

    # prior
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    x0 = random.normal(prior_key, shape=(d,), dtype='float64')

    # run
    sampler.warmup(key, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    warmup_calls = np.sum(np.array(sampler.get_extra_fields()['num_steps'], dtype=int))
    sampler.run(key, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()
    X = np.array(numpyro_samples['x'])

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    variance_true = np.ones(d)
    variance_true[0] += 0.2 * 8.0**2

    ess, n_crossing = ess_cutoff_crossing(bias(X, np.ones(len(X)), variance_true), steps)

    return ess, ess / (1 + ess* warmup_calls / 200.0)



def funnel(key_num):

    d= 20
    #stepsize= 0.01

    # setup
    nuts_setup = NUTS(targets.funnel_noiseless, adapt_step_size=True, adapt_mass_matrix=False)
    sampler = MCMC(nuts_setup, num_warmup=1000, num_samples=10000, num_chains=1, progress_bar=True)


    # prior
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    x0 = random.normal(prior_key, shape = (d, ), dtype = 'float64')

    # run
    sampler.warmup(key, d, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    warmup_calls = np.sum(np.array(sampler.get_extra_fields()['num_steps'], dtype=int))
    sampler.run(key, d, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    z = np.array(numpyro_samples['z'])
    theta = np.array(numpyro_samples['theta'])

    def gaussianize(z, theta):
        return (z.T * np.exp(-0.5 * theta)).T, theta / 3.0

    Gz, Gtheta = gaussianize(z, theta)

    X = np.empty((len(Gtheta), d))
    X[:, :d-1] = Gz
    X[:, -1]= Gtheta

    B = bias(X, np.ones(len(X)), np.ones(d))

    ess, n_crossing = ess_cutoff_crossing(B, steps)

    return ess, ess / (1 + ess* warmup_calls / 200.0)

    #np.savez('data/funnel_HMC', z= np.array(numpyro_samples['z']), theta= np.array(numpyro_samples['theta']), steps= steps)




def rosenbrock(key_num, d = 36):

    #target setup
    #Q, var_x, var_y = 0.1, 2.0, 10.098433122783046
    Q, var_x, var_y = 0.5, 2.0, 10.498957879911487

    variance_true = np.concatenate((var_x * np.ones(d // 2), var_y * np.ones(d // 2)))

    # setup
    nuts_setup = NUTS(targets.rosenbrock, adapt_step_size=True, adapt_mass_matrix=True)#, step_size= stepsize)
    sampler = MCMC(nuts_setup, num_warmup= 1000, num_samples= 10000, num_chains= 1, progress_bar= True)

    # run
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    x0 = random.normal(prior_key, shape = (d, ), dtype = 'float64')

    #run
    sampler.warmup(key, d, Q, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    warmup_calls = np.sum(np.array(sampler.get_extra_fields()['num_steps'], dtype=int))
    sampler.run(key, d, Q, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()

    #X = {name:  for name in ['x', 'y']}

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)


    X = np.concatenate((np.array(numpyro_samples['x']).T, np.array(numpyro_samples['y']).T)).T

    B = bias(X, np.ones(len(X)), variance_true)

    ess, n_crossing = ess_cutoff_crossing(B, steps)

    return ess, ess / (1 + ess* warmup_calls / 200.0)

    #np.savez('data/rosenbrock_HMC', x = np.array(numpyro_samples['x']), y = np.array(numpyro_samples['y']), steps= steps)



def stochastic_volatility(key_num):

    ground_truth = False
    posterior_band = True
    #target setup
    #variance_true = np.concatenate((var_x * np.ones(d // 2), var_y * np.ones(d // 2)))

    # setup
    nuts_setup = NUTS(targets.StochasticVolatility, adapt_step_size=True, adapt_mass_matrix=True)#, step_size= stepsize)

    if ground_truth:
        sampler = MCMC(nuts_setup, num_warmup= 10000, num_samples= 10000 * 50, num_chains= 1, progress_bar= True, thinning=50) #will return num_samples / thinning
    else:
        sampler = MCMC(nuts_setup, num_warmup=500, num_samples=10000, num_chains=1, progress_bar=True)

    # run
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    MCHMC_target = benchmark_targets.StochasticVolatility()

    x0 = MCHMC_target.transform(MCHMC_target.prior_draw(prior_key))


    #run
    sampler.warmup(key, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    warmup_calls = np.sum(np.array(sampler.get_extra_fields()['num_steps'], dtype=int))
    sampler.run(key, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    s= np.array(numpyro_samples['s'])
    sigma= np.array(numpyro_samples['sigma'])
    nu= np.array(numpyro_samples['nu'])

    if posterior_band:
        volatility = np.sort(np.exp(s), axis=0)
        np.save('data/stochastic_volatility/NUTS_posterior_band.npy', [volatility[len(volatility) // 4, :], volatility[len(volatility) // 2, :], volatility[3 * len(volatility) // 4, :]])

    #np.savez('data/stochastic_volatility/NUTS_samples.npz', s= s, sigma = sigma, nu= nu)

    if ground_truth:
        var = np.empty(len(s[0]) + 2)
        var[:-2] = np.average(np.square(s), axis=0)
        var[-2] = np.average(np.square(sigma))
        var[-1] = np.average(np.square(nu))

        np.save('data/stochastic_volatility/ground_truth'+str(key_num)+'.npy', var)

        #np.savez('SVsamples.npz', s= s, sigma = sigma, nu= nu)

    else:
        X = np.empty((len(nu), len(s[0]) + 2))
        X[:, :-2] = s
        X[:, -2] = sigma
        X[:, -1] = nu

        variance_true = np.load('data/stochastic_volatility/ground_truth_moments.npy')

        B = bias(X, np.ones(len(X)), variance_true)

        ess, n_crossing = ess_cutoff_crossing(B, steps)
        print(ess, ess / (1 + ess* warmup_calls / 200.0))

        return ess, ess / (1 + ess* warmup_calls / 200.0)


def bimodal_plot():

    #target setup
    d= 50
    thinning = 100
    # setup
    nuts_setup = NUTS(targets.bimodal_hard, adapt_step_size=True, adapt_mass_matrix=True)#, step_size= stepsize)
    sampler = MCMC(nuts_setup, num_warmup= 1000, num_samples= 2143000, thinning = thinning, num_chains= 1, progress_bar= True)

    # run
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    x0 = random.normal(subkey, shape = (d, ), dtype = 'float64')
    sampler.run(key, init_params= x0, extra_fields=['num_steps'])

    # get results
    X = np.array(sampler.get_samples()['x'])

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)
    print(len(steps))
    np.savez('data/bimodal_marginal/NUTS_hard.npz', x0 = X[:, 0], steps= np.cumsum(steps) * thinning)
    print(np.cumsum(steps))
    import matplotlib.pyplot as plt
    plt.hist(X[:, 0], bins = 30, density=True)
    plt.show()


def table1():
    repeat = 10

    functions = [ill_conditioned, bimodal, rosenbrock, funnel, stochastic_volatility]
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'Stochastic Volatility']
    ess, ess_std = np.zeros(len(names)), np.zeros(len(names))
    ess2, ess_std2 = np.zeros(len(names)), np.zeros(len(names))

    for i in range(len(names)-1, len(names)):
        print(names[i])
        ESS = np.array([functions[i](j) for j in range(repeat)])
        ess[i], ess_std[i] = np.average(ESS[:, 0]), np.std(ESS[:, 0])
        ess2[i], ess_std2[i] = np.average(ESS[:, 1]), np.std(ESS[:, 1])

    data = {'Target ': names, 'ESS': ess, 'ESS std': ess_std, 'ESS (with warmup)': ess2, 'ESS (with warmup) std': ess_std2}
    print(data)
    df = pd.DataFrame(data)
    print(df)
    #df.to_csv('submission/TableNUTS.csv', sep='\t', index=False)


def dimension_scaling():

    target = ill_conditioned
    #target = rosenbrock
    #dimensions = [100, 300, 1000, 3000]
    dimensions = [10000, ]
    repeat = 3

    ess, ess_std = np.zeros(len(dimensions)), np.zeros(len(dimensions))
    ess2, ess_std2 = np.zeros(len(dimensions)), np.zeros(len(dimensions))

    for i in range(len(dimensions)):
        d = dimensions[i]
        print(d)
        for key_num in range(repeat):
            es = target(key_num, d, 100.0)
            print(es)
        #ESS = np.array([target(key_num, d, 100.0) for key_num in range(repeat)])

    #     ess[i], ess_std[i] = np.average(ESS[:, 0]), np.std(ESS[:, 0])
    #     ess2[i], ess_std2[i] = np.average(ESS[:, 1]), np.std(ESS[:, 1])
    #     print(ess[i])
    #
    # np.save('data/dimensions_dependence/HMC_kappa100.npy', [ess, ess2])
    #
    # print(ess)


if __name__ == '__main__':
    X = np.load('data/dimensions_dependence/HMC_kappa100.npy')
    a = np.array([0.003069791714632162, 0.001534978318431252])

    Y = np.concatenate((X, a[:, None]), axis = 1)
    np.save('data/dimensions_dependence/HMC_kappa100.npy', Y)

    # ess, ess2 = np.load('data/dimensions_dependence/HMC_rosenbrock.npy')
    # print(ess)

    #dimension_scaling()
    #stochastic_volatility(0)

    # var = np.array([np.load('ground_truth'+str(i)+'.npy') for i in range(3)])
    # var_avg = np.average(var, axis = 0)
    # np.save('StochasticVolatility_ground_truth_moments.npy', var_avg)
    # bias = [np.sqrt(np.average(np.square((var[i, :] - var_avg) / var_avg))) for i in range(3)]
    # print(bias)

    #SVplot()

    #table1()
    #dimension_dependence()
    #ill_conditioned_scan()
    #bimodal_plot()