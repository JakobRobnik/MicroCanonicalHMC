import sys
sys.path.insert(0, './')

from sampling.sampler import Sampler  
from collections import namedtuple
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
from sampling.dynamics import minimal_norm
from numpyro.infer.util import initialize_model
from numpyro.handlers import condition, seed, substitute, trace


MCLMCState = namedtuple("MCLMCState", ["x", "u", "l", "g", "key"])


class Generic():

  def __init__(self, model, rng_key):
    
    
    model = seed(model, random.PRNGKey(0))
    init_model_trace = trace(model).get_trace()
            
    self.d = len(init_model_trace)
    self.initial_vals = jax.numpy.array([init_model_trace[i]['value'] for i in init_model_trace])


    vars = [init_model_trace[i]['name'] for i in init_model_trace]

    def potential_fn(arr):
        tr = trace(condition(model, dict(zip(vars, arr)) )).get_trace()
        return -sum([tr[x]['fn'].log_prob(arr[i]) for i, x in enumerate(tr)])
     
    self.potential_fn = potential_fn
    self.value_grad = jax.value_and_grad(self.potential_fn)


  def grad_nlogp(self, x):
    """should return nlogp and gradient of nlogp"""
    return self.value_grad(x)

  def transform(self, x):
    return x

  def prior_draw(self, key):
    """Args: jax random key
       Returns: one random sample from the prior"""

    return self.initial_vals



class MCLMC(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "x"

    def __init__(self, model, step_size=0.1):
         
        
        self.step_size = step_size
        self.model = model
        self.eps = 1e-1
        self.L = 10

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):

        target = Generic(rng_key=rng_key,model=self.model)
        self.sampler = Sampler(target, varEwanted = 5e-4)
        x, u, l, g, key = (self.sampler.get_initial_conditions('prior', rng_key))
        return MCLMCState(x, u, l, g, key)
     
    # the kernel, in terms of the state space implied by MCLMCState
    def sample(self, state, model_args, model_kwargs):
        (x, u, l, g, key) = state
        print(state, "state")
        xx, uu, ll, gg, kinetic_change, key, time = self.sampler.dynamics_generalized(x, u, g, key, 0, self.L, self.eps, 1.0)
        print("done")
        return MCLMCState(xx, uu, ll, gg, key)



def m():

    mu = numpyro.sample('mu', dist.Normal(3, 1))
    nu = numpyro.sample('nu', dist.Normal(mu+1, 2))
    

kernel = MCLMC(m)
# kernel = NUTS(m)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
mcmc.run(random.PRNGKey(0))
posterior_samples = mcmc.get_samples()
mcmc.print_summary() 

