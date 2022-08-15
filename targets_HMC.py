import numpy as np
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

###  targets in the numpyro form  ###



def ill_conditioned_gaussian(d, condition_number):
    variance_true = np.logspace(-np.log10(condition_number), np.log10(condition_number), d)
    numpyro.sample('x', dist.Normal(np.zeros(d), np.sqrt(variance_true)))



def bimodal(d, mu):
    avg = np.zeros(d)
    avg[0] = mu

    mix = dist.Categorical(np.ones(2) / 2.0)

    component_dist = dist.Normal(loc=np.array([avg, -avg]).T)  # , scale=np.ones(shape = (d, 2)))

    mixture = dist.MixtureSameFamily(mix, component_dist)

    numpyro.sample('x', mixture)



def funnel(d, sigma):
    theta = numpyro.sample("theta", dist.Normal(0, 3))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)) )
    numpyro.sample("z", dist.Normal(z, sigma))


def funnel_noiseless(d):
    theta = numpyro.sample("theta", dist.Normal(0, 3))
    numpyro.sample("z", dist.Normal(jnp.zeros(d - 1), jnp.exp(0.5 * theta)) )



def rosenbrock(d):
    """see https://en.wikipedia.org/wiki/Rosenbrock_function (here we set a = 1, b = 1/Q)"""
    Q = 0.1

    x = numpyro.sample("x", dist.Normal(jnp.ones(d // 2), jnp.ones(d // 2)))
    y = numpyro.sample("y", dist.Normal(jnp.square(x), Q * jnp.ones(d // 2)))

