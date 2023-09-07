import sys
sys.path.insert(0, './')

from sampling.sampler import Sampler  
from collections import namedtuple
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
import jax
from sampling.dynamics import minimal_norm

MCLMCState = namedtuple("MCLMCState", ["x", "u", "l", "g", "key"])


class StandardGaussian():

  def __init__(self, d):
    self.d = d

  def grad_nlogp(self, x):
    """should return nlogp and gradient of nlogp"""
    return value_grad(x)

  def transform(self, x):
    return x[:2]
    #return x

  def prior_draw(self, key):
    """Args: jax random key
       Returns: one random sample from the prior"""

    return jax.random.normal(key, shape = (self.d, ), dtype = 'float64') * 4

nlogp = lambda x: 0.5*jnp.sum(jnp.square(x))
value_grad = jax.value_and_grad(nlogp)
target = StandardGaussian(d = 2)


class MCLMC(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "x"

    def __init__(self, potential_fn, target, step_size=0.1):
         self.potential_fn = potential_fn
         self.step_size = step_size
         self.eps = 1e-1
         self.L = 10
         self.target = target
         self.sampler = Sampler(target, varEwanted = 5e-4)

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        print("init")
        x, u, l, g, key = (self.sampler.get_initial_conditions('prior', rng_key))
        return MCLMCState(x, u, l, g, key)
     
    # the kernel, in terms of the state space implied by MCLMCState
    def sample(self, state, model_args, model_kwargs):
        (x, u, l, g, key) = state
        print(state, "state")
        xx, uu, ll, gg, kinetic_change, key, time = self.sampler.dynamics_generalized(x, u, g, key, 0, self.L, self.eps, 1.0)
        print("done")
        return MCLMCState(xx, uu, ll, gg, key)


kernel = MCLMC(nlogp, target)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
mcmc.run(random.PRNGKey(0), init_params=jnp.array([1., 2.]))
posterior_samples = mcmc.get_samples()
mcmc.print_summary()  # doctest: +SKIP