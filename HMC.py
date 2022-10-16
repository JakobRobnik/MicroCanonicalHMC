import numpy as np
from scipy.stats import special_ortho_group

from jax import random
from numpyro.infer import MCMC, NUTS

import bias
import targets_HMC as targets

import pandas as pd

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



def ill_conditioned():

    d = 100
    R = special_ortho_group.rvs(d, random_state=0)
    repeat = 10

    def f(condition_number, num_samples):
        ess_arr = np.zeros(repeat)
        for key_num in range(repeat):
            X, steps = sample_nuts(targets.ill_conditioned_gaussian, [d, condition_number], num_samples, d = d, key_num=key_num)

            variance_true = np.logspace(-0.5*np.log10(condition_number), 0.5*np.log10(condition_number), d)
            ess, n_crossing = bias.ess_cutoff_crossing(bias.bias((R @ X.T).T, np.ones(len(X)), variance_true), steps)
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

    np.save('Tests/data/kappa/NUTS.npy', [ess_arr, ess_err_arr, kappa_arr])



def dimension_dependence():


    def gauss(d, num_samples):
        R = special_ortho_group.rvs(d, random_state=0)
        kappa = 100.0
        variance_true = np.logspace(-0.5*np.log10(kappa), 0.5*np.log10(kappa), d)

        X, steps = sample_nuts(targets.ill_conditioned_gaussian, [d, kappa], num_samples)
        X = (R @ X.T).T
        B = bias.bias(X, np.ones(len(X)), variance_true)
        ess, n_crossing = bias.ess_cutoff_crossing(B, steps)

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

        B = bias.bias(X, np.ones(len(X)), variance_true)
        ess, n_crossing = bias.ess_cutoff_crossing(B, steps)

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

    np.save('Tests/data/dimensions/Kappa100_NUTS.npy', [ess_arr, dimensions])
    #np.save('Tests/data/dimensions/Rosenbrock_NUTS.npy', [ess_arr, dimensions])



def bimodal_mixing():

    def avg_mode_mixing_steps(signs, steps):
        L = []
        current_sign = signs[0]
        island_size = steps[0]
        for n in range(1, len(signs)):
            sign = signs[n]
            if sign != current_sign:
                L.append(island_size)
                island_size = 1
                current_sign = sign
            else:
                island_size += steps[n]

            if len(L) == 10:
                return np.average(L)

        print('Maximum number of steps exceeded, num_islands = ' + str(len(L)))
        return len(signs)



    def f(mu, num_samples):
        d = 50
        X, steps = sample_nuts(targets.bimodal, [d, mu], num_samples)

        return avg_mode_mixing_steps(np.sign(X[:, 0] - mu*0.5), steps)


    mu_arr = np.arange(1, 10)

    avg_steps_mode = np.zeros(len(mu_arr))
    num_samples= 10000

    for i in range(len(avg_steps_mode)):
        print(i, num_samples)

        avg_num = f(mu_arr[i], num_samples)

        avg_steps_mode[i] = avg_num

        num_samples = (int)(avg_num * 10 * 30)

    np.save('Tests/data/mode_mixing_NUTS.npy', np.array([avg_steps_mode, mu_arr]))



def kappa100(key_num):

    d, condition_number = 100, 100.0
    R = special_ortho_group.rvs(d, random_state=0)

    nuts_setup = NUTS(targets.ill_conditioned_gaussian, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)  # originally: nuts_kernel
    sampler = MCMC(nuts_setup, num_warmup=500, num_samples=3000, num_chains=1, progress_bar=False)

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
    ess, n_crossing = bias.ess_cutoff_crossing(bias.bias((R @ X.T).T, np.ones(len(X)), variance_true), steps)

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

    ess, n_crossing = bias.ess_cutoff_crossing(bias.bias(X, np.ones(len(X)), variance_true), steps)

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

    B = bias.bias(X, np.ones(len(X)), np.ones(d))

    ess, n_crossing = bias.ess_cutoff_crossing(B, steps)

    return ess, ess / (1 + ess* warmup_calls / 200.0)

    #np.savez('Tests/data/funnel_HMC', z= np.array(numpyro_samples['z']), theta= np.array(numpyro_samples['theta']), steps= steps)




def rosenbrock(key_num):

    #target setup
    d= 36
    Q, var_x, var_y = 0.1, 2.0, 10.098433122783046
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

    B = bias.bias(X, np.ones(len(X)), variance_true)

    ess, n_crossing = bias.ess_cutoff_crossing(B, steps)

    return ess, ess / (1 + ess* warmup_calls / 200.0)

    #np.savez('Tests/data/rosenbrock_HMC', x = np.array(numpyro_samples['x']), y = np.array(numpyro_samples['y']), steps= steps)



def stohastic_volatility(key_num):

    #target setup
    #variance_true = np.concatenate((var_x * np.ones(d // 2), var_y * np.ones(d // 2)))

    # setup
    nuts_setup = NUTS(targets.StohasticVolatility, adapt_step_size=True, adapt_mass_matrix=True)#, step_size= stepsize)
    sampler = MCMC(nuts_setup, num_warmup= 1000, num_samples= 1000, num_chains= 1, progress_bar= True)

    # run
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    x0 = random.normal(prior_key, shape = (2429, ), dtype = 'float64')

    #run
    sampler.warmup(key, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    warmup_calls = np.sum(np.array(sampler.get_extra_fields()['num_steps'], dtype=int))
    sampler.run(key, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()

    #X = {name:  for name in ['x', 'y']}

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    s= np.array(numpyro_samples['s'])
    print(np.shape(s))
    sigma= np.array(numpyro_samples['sigma'])
    nu= np.array(numpyro_samples['nu'])


    np.savez('SVsamples.npz', s= s, sigma = sigma, nu= nu)

    exit()
    X = np.empty((len(nu), len(s[0]) + 2))
    X[:, :-2] = s
    X[:, -2] = sigma
    X[:, -1] = nu

    B = bias.bias(X, np.ones(len(X)), variance_true)

    ess, n_crossing = bias.ess_cutoff_crossing(B, steps)

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
    np.savez('Tests/data/bimodal_marginal/NUTS_hard.npz', x0 = X[:, 0], steps= np.cumsum(steps) * thinning)
    print(np.cumsum(steps))
    import matplotlib.pyplot as plt
    plt.hist(X[:, 0], bins = 30, density=True)
    plt.show()


def SVplot():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from numpyro.examples.datasets import SP500, load_dataset

    _, fetch = load_dataset(SP500, shuffle=False)
    dates, returns = fetch()

    X = np.load('SVsamples.npz')
    s, sigma, nu = X['s'], X['sigma'], X['nu']

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    dates = mdates.num2date(mdates.datestr2num(dates))
    ax.plot(dates, returns, '.', color=  'black', label = 'data')
    # format the ticks
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    volatility = np.sort(np.exp(s), axis = 0)

    ax.plot(dates, volatility[len(volatility)//2, :], color = 'orange')
    ax.fill_between(dates, volatility[len(volatility)//4, :], volatility[3* len(volatility)//4, :], color = 'orange', alpha = 0.5)

    ax.legend()
    ax.set(xlabel="time", ylabel="returns")

    plt.show()


def table1():
    repeat = 10

    functions = [kappa100, bimodal, rosenbrock, funnel]
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel"]
    ess, ess_std = np.zeros(len(names)), np.zeros(len(names))
    ess2, ess_std2 = np.zeros(len(names)), np.zeros(len(names))

    for i in range(len(names)):
        print(names[i])
        ESS = np.array([functions[i](j) for j in range(repeat)])
        ess[i], ess_std[i] = np.average(ESS[:, 0]), np.std(ESS[:, 0])
        ess2[i], ess_std2[i] = np.average(ESS[:, 1]), np.std(ESS[:, 1])

    data = {'Target ': names, 'ESS': ess, 'ESS std': ess_std, 'ESS (with warmup)': ess2, 'ESS (with warmup) std': ess_std2}
    print(data)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv('submission/TableNUTS.csv', sep='\t', index=False)


if __name__ == '__main__':

    #stohastic_volatility(0)
    SVplot()

    #table1()
    #dimension_dependence()
    #ill_conditioned()
    #bimodal_plot()