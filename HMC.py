import numpy as np

from jax import random
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist

import bias



def IllConditionedGaussian(d, condition_number):
    variance_true = np.logspace(-np.log10(condition_number), np.log10(condition_number), d)
    numpyro.sample('x', dist.Normal(np.zeros(d), np.sqrt(variance_true)) )


def ess_kappa(condition_number, num_samples):
    d = 50

    #setup
    nuts_setup = NUTS(IllConditionedGaussian, adapt_step_size = True, adapt_mass_matrix = False) #originally: nuts_kernel
    sampler = MCMC(nuts_setup, num_warmup=1000, num_samples=num_samples, num_chains=1, progress_bar= False)
    random_seed = random.PRNGKey(0)

    #run
    sampler.run(random_seed, d, condition_number, extra_fields= ['num_steps'])

    #get results
    X = np.array(sampler.get_samples()['x'])
    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype = int)

    variance_true = np.logspace(-np.log10(condition_number), np.log10(condition_number), d)

    n_crossing = bias.cutoff_crossing(X, np.ones(len(X)), variance_true)

    return 200.0 / np.sum(steps[:n_crossing+1]), n_crossing


kappa_arr = np.logspace(0, 4, 15)

ess_arr= np.zeros(len(kappa_arr))
num_samples= 1000

for i in range(len(kappa_arr)):
    print(i, num_samples)

    ess, n_crossing = ess_kappa(kappa_arr[i], num_samples)

    ess_arr[i] = ess

    num_samples = n_crossing * 10

print(ess_arr.tolist())
