import numpy as np
from scipy.stats import special_ortho_group

from jax import random
from numpyro.infer import MCMC, NUTS

import bias
import targets_HMC as targets



def sample_nuts(target, target_params, num_samples, names_output = None):
    """ 'default' nuts, tunes the step size with a prerun"""

    # setup
    nuts_setup = NUTS(target, adapt_step_size=True, adapt_mass_matrix=True, dense_mass= False)  # originally: nuts_kernel
    sampler = MCMC(nuts_setup, num_warmup=500, num_samples=num_samples, num_chains=1, progress_bar=False)
    random_seed = random.PRNGKey(0)

    # run
    sampler.run(random_seed, *target_params, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()
    if names_output == None:
        X = np.array(numpyro_samples['x'])

    else:
        X = {name: np.array(numpyro_samples[name]) for name in names_output}

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    return X, steps



def ill_conditioned_gaussian():

    d = 100
    R = special_ortho_group.rvs(d, random_state=0)


    def f(condition_number, num_samples):

        X, steps = sample_nuts(targets.ill_conditioned_gaussian, [d, condition_number], num_samples)

        variance_true = np.logspace(-0.5*np.log10(condition_number), 0.5*np.log10(condition_number), d)
        ess, n_crossing = bias.ess_cutoff_crossing(bias.bias((R @ X.T).T, np.ones(len(X)), variance_true), steps)

        return ess, n_crossing


    kappa_arr = np.logspace(0, 5, 18)

    ess_arr= np.zeros(len(kappa_arr))
    num_samples= 1000

    for i in range(len(kappa_arr)):
        print(i, num_samples)

        ess, n_crossing = f(kappa_arr[i], num_samples)

        ess_arr[i] = ess

        num_samples = n_crossing * 5

    np.save('Tests/data/kappa_NUTS.npy', [ess_arr, kappa_arr])



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



def bimodal():

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


def bimodal_marginal():


    d = 50
    X, steps = sample_nuts(targets.bimodal, [d, mu], num_samples)




def kappa100(key_num):

    d, condition_number = 100, 100.0
    R = special_ortho_group.rvs(d, random_state=0)

    nuts_setup = NUTS(targets.ill_conditioned_gaussian, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)  # originally: nuts_kernel
    sampler = MCMC(nuts_setup, num_warmup=500, num_samples=3000, num_chains=1, progress_bar=False)

    # prior
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    x0 = random.normal(prior_key, shape=(d,), dtype='float64') * np.power(condition_number, 0.25)

    # run
    sampler.run(key, d, condition_number, init_params=x0, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()
    X = np.array(numpyro_samples['x'])

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    variance_true = np.logspace(-0.5*np.log10(condition_number), 0.5*np.log10(condition_number), d)
    ess, n_crossing = bias.ess_cutoff_crossing(bias.bias((R @ X.T).T, np.ones(len(X)), variance_true), steps)

    return ess



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
    sampler.run(key, d, init_params=x0, extra_fields=['num_steps'])

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

    return ess

    #np.savez('Tests/data/funnel_HMC', z= np.array(numpyro_samples['z']), theta= np.array(numpyro_samples['theta']), steps= steps)




def rosenbrock(key_num):

    #target setup
    d= 36
    Q, var_x, var_y = 0.1, 2.0, 10.098433122783046
    variance_true = np.concatenate((var_x * np.ones(d // 2), var_y * np.ones(d // 2)))

    # setup
    nuts_setup = NUTS(targets.rosenbrock, adapt_step_size=True, adapt_mass_matrix=False)#, step_size= stepsize)
    sampler = MCMC(nuts_setup, num_warmup= 1000, num_samples= 10000, num_chains= 1, progress_bar= True)

    # run
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    x0 = random.normal(prior_key, shape = (d, ), dtype = 'float64')
    sampler.run(key, d, Q, init_params= x0, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()

    #X = {name:  for name in ['x', 'y']}

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)


    X = np.concatenate((np.array(numpyro_samples['x']).T, np.array(numpyro_samples['y']).T)).T

    B = bias.bias(X, np.ones(len(X)), variance_true)

    ess, n_crossing = bias.ess_cutoff_crossing(B, steps)

    return ess

    #np.savez('Tests/data/rosenbrock_HMC', x = np.array(numpyro_samples['x']), y = np.array(numpyro_samples['y']), steps= steps)


def table1():
    repeat = 10

    functions = [kappa100, funnel, rosenbrock]
    names = ['Kappa 100', 'Funnel', 'Rosenbrock']
    for i in range(len(names)):
        print(names[i])
        ess = [functions[i](j) for j in range(repeat)]
        print(np.average(ess), np.std(ess))
        print('---------------')

if __name__ == '__main__':
    #bimodal()
    table1()
    #dimension_dependence()
    #ill_conditioned_gaussian()