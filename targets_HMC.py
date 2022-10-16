import numpy as np
from scipy.stats import special_ortho_group

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro.examples.datasets import SP500, load_dataset


_, fetch = load_dataset(SP500, shuffle=False)
SP500_dates, SP500_returns = fetch()


###  targets in the numpyro form  ###



def ill_conditioned_gaussian(d, condition_number):
    variance_true = np.logspace(-0.5*np.log10(condition_number), 0.5*np.log10(condition_number), d)

    # diagonal
    #numpyro.sample('x', dist.Normal(np.zeros(d), np.sqrt(variance_true)))

    #randomly rotated
    Cov = np.diag(variance_true)
    R = special_ortho_group.rvs(d, random_state= 0)
    rotated_Cov = R.T @ Cov @ R
    numpyro.sample('x', dist.MultivariateNormal(jnp.zeros(d), rotated_Cov))


def bimodal(d, mu):
    avg = np.zeros(d)
    avg[0] = mu

    mix = dist.Categorical(np.ones(2) / 2.0)

    component_dist = dist.Normal(loc=np.array([np.zeros(d), avg]).T)  # , scale=np.ones(shape = (d, 2)))

    mixture = dist.MixtureSameFamily(mix, component_dist)

    numpyro.sample('x', mixture)


def bimodal_hard():
    d = 50
    avg = np.zeros(d)
    avg[0] = 8.0

    mix = dist.Categorical(jnp.array([0.8, 0.2]))

    component_dist = dist.Normal(loc=np.array([np.zeros(d), avg]).T)  # , scale=np.ones(shape = (d, 2)))

    mixture = dist.MixtureSameFamily(mix, component_dist)

    numpyro.sample('x', mixture)


def funnel(d, sigma):
    theta = numpyro.sample("theta", dist.Normal(0, 3))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)) )
    numpyro.sample("z", dist.Normal(z, sigma))


def funnel_noiseless(d):
    theta = numpyro.sample("theta", dist.Normal(0, 3))
    numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)) )



def rosenbrock(d, Q):
    """see https://en.wikipedia.org/wiki/Rosenbrock_function (here we set a = 1, b = 1/Q)"""

    x = numpyro.sample("x", dist.Normal(jnp.ones(d // 2), jnp.ones(d // 2)))
    numpyro.sample("y", dist.Normal(jnp.square(x), np.sqrt(Q) * jnp.ones(d // 2)))



def StohasticVolatility():
    step_size = numpyro.sample("sigma", dist.Exponential(50.0))
    nu = numpyro.sample("nu", dist.Exponential(0.1))
    s = numpyro.sample("s", dist.GaussianRandomWalk(scale=step_size, num_steps=jnp.shape(SP500_returns)[0]))
    numpyro.sample("r", dist.StudentT(df=nu, loc=0.0, scale=jnp.exp(s)), obs=SP500_returns)

