import numpy as np
import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist

from scipy.stats import norm

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
    z = numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)) )




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




def funnel_samples():


    samples, steps= sample_funnel()
    print(np.sum(steps))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original coordinates')
    plt.plot(samples['z'][:, 0], samples['theta'], '.', color = 'tab:blue')
    plt.xlim(-30, 30)
    plt.xlabel(r'$z_0$')
    plt.ylabel(r'$\theta$')

    plt.subplot(1, 3, 2)
    plt.title('Gaussianized coordinates')
    gaussianized_samples = gaussianize(samples)
    plt.plot(gaussianized_samples['z'][:, 0], gaussianized_samples['theta'], '.', color='tab:blue')

    p_level = np.array([0.6827, 0.9545])
    x_level = np.sqrt(-2 * np.log(1 - p_level))
    phi = np.linspace(0, 2* np.pi, 100)
    for i in range(2):
        plt.plot(x_level[i] * np.cos(phi), x_level[i] * np.sin(phi), color = 'black', alpha= ([0.1, 0.5])[i])

    plt.xlabel(r'$\widetilde{z_0}$')
    plt.ylabel(r'$\widetilde{\theta}$')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.subplot(1, 3, 3)
    plt.title(r'$\theta$-marginal')
    plt.hist(samples['theta'], color='tab:blue', cumulative=True, density=True, bins = 1000)

    t= np.linspace(-10, 10, 100)
    plt.plot(t, norm.cdf(t, scale= 3.0), color= 'black')

    plt.xlabel(r'$\theta$')
    plt.ylabel('CDF')
    plt.savefig('funnel_nuts')

    plt.show()


def gaussianize(samples):
    return {'theta': 0.3 * samples['theta'], 'z': (samples['z'].T * np.exp(-0.5 * samples['theta'])).T }



funnel_samples()




