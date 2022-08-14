import numpy as np
import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist


import bias




def ill_conditioned_gaussian(d, condition_number):
    variance_true = np.logspace(-np.log10(condition_number), np.log10(condition_number), d)
    numpyro.sample('x', dist.Normal(np.zeros(d), np.sqrt(variance_true)) )


def funnel(d, sigma):
    theta = numpyro.sample("theta", dist.Normal(0, 3))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)) )
    numpyro.sample("z", dist.Normal(z, sigma))


def funnel_noiseless(d):
    theta = numpyro.sample("theta", dist.Normal(0, 3))
    numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)) )


def bimodal(d, mu):
    avg= np.zeros(d)
    avg[0]= mu

    mix = dist.Categorical(np.ones(2) / 2.0)

    component_dist = dist.Normal(loc=np.array([avg, -avg]).T)#, scale=np.ones(shape = (d, 2)))

    mixture = dist.MixtureSameFamily(mix, component_dist)

    numpyro.sample('x', mixture)




def sample_nuts(target, target_params, num_samples, names_output = None):

    # setup
    nuts_setup = NUTS(target, adapt_step_size=True, adapt_mass_matrix=False)  # originally: nuts_kernel
    sampler = MCMC(nuts_setup, num_warmup=1000, num_samples=num_samples, num_chains=1, progress_bar=False)
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



def sample_funnel():

    d= 20
    thinning= 5
    num_samples= 1000*thinning
    stepsize= 0.01

    # setup
    nuts_setup = NUTS(funnel_noiseless, adapt_step_size=False, adapt_mass_matrix=False, step_size= stepsize)
    sampler = MCMC(nuts_setup, num_warmup=0, num_samples=num_samples, num_chains=1, progress_bar=True, thinning= thinning)

    random_seed = random.PRNGKey(0)

    # run
    sampler.run(random_seed, d, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()

    X = {name: np.array(numpyro_samples[name]) for name in ['theta', 'z']}

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    print(len(steps), len(X['theta']))
    return X, steps



def ess_kappa():

    def f(condition_number, num_samples):
        d = 100
        X, steps = sample_nuts(ill_conditioned_gaussian, [d, condition_number], num_samples)

        variance_true = np.logspace(-np.log10(condition_number), np.log10(condition_number), d)

        ess, n_crossing = bias.ess_cutoff_crossing(bias.bias(X, np.ones(len(X)), variance_true), steps)

        return ess, n_crossing


    kappa_arr = np.logspace(0, 4, 18)

    ess_arr= np.zeros(len(kappa_arr))
    num_samples= 1000

    for i in range(len(kappa_arr)):
        print(i, num_samples)

        ess, n_crossing = f(kappa_arr[i], num_samples)

        ess_arr[i] = ess

        num_samples = n_crossing * 5

    np.save('Tests/kappa_NUTS.npy', np.concatenate((ess_arr, kappa_arr)).T)

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


def mode_mixing():

    def f(mu, num_samples):
        d = 50
        X, steps = sample_nuts(bimodal, [d, mu], num_samples)

        return avg_mode_mixing_steps(np.sign(X[:, 0]), steps)


    mu_arr = np.arange(1, 6)

    avg_steps_mode = np.zeros(len(mu_arr))
    num_samples= 10000

    for i in range(len(avg_steps_mode)):
        print(i, num_samples)

        avg_num = f(mu_arr[i], num_samples)

        avg_steps_mode[i] = avg_num

        num_samples = (int)(avg_num * 10 * 30)

    np.save('Tests/mode_mixing_NUTS.npy', np.array([avg_steps_mode, mu_arr]))



def funnel_samples():


    samples, steps= sample_funnel()
    print(np.sum(steps))



mode_mixing()

