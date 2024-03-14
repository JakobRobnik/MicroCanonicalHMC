import sys
sys.path.insert(0, './')

from sampling.sampler import Sampler, Target  
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
import numpyro
import numpyro.distributions as dist
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
import time

# shape of array, but returns one for a float (0 dimensional array)
def compute_length(arr):
  s = arr.shape
  if s==(): 
    return 1
  else: return s[0]

class NumPyroTarget(Target):

  def __init__(self, probabilistic_program):

    # this is just to obtain the shape of the trace
    initial_val, potential_fn_gen, *_ = initialize_model(
        jax.random.PRNGKey(0),
        probabilistic_program,
        model_args=(),
        dynamic_args=True,
    )

    self.initial_trace = initial_val.z
    self.d = sum([compute_length(self.initial_trace[k]) for k in self.initial_trace])
    self.probabilistic_program = probabilistic_program

    Target.__init__(self,self.d,nlogp = lambda x : potential_fn_gen()(self.to_dict(x)))
  
  def to_dict(self, x):
    assert x.shape[0]==self.d, f"The dimensionality of the state, {x.shape[0]}, does not agree with that of the probabilistic program, {self.d}"
    for i,k in enumerate(self.initial_trace):
      s = compute_length(self.initial_trace[k])
      self.initial_trace[k]= x[i:i+s]
    return self.initial_trace
    
  def from_dict(self, tr):
    return jnp.concatenate([tr[k] if tr[k].shape!=() else jnp.array([tr[k]]) for k in tr])

  def prior_draw(self, key):
    """Args: jax random key
       Returns: one random sample from the prior"""

    init_params, *_ = initialize_model(
      key,
      self.probabilistic_program,
      model_args=(),
      dynamic_args=True,
    )
    return self.from_dict(init_params.z)

 
def m():

    mu = numpyro.sample('mu', dist.MultivariateNormal(loc=jnp.array([1,1]), covariance_matrix=jnp.identity(2)))
    nu = numpyro.sample('nu', dist.Normal(1, 2))

target = NumPyroTarget(m)
samples = Sampler(target).sample(100)




# Eight Schools example
J = 8
y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

def eight_schools_noncentered():
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
            theta = numpyro.sample(
                "theta",
                dist.TransformedDistribution(
                    dist.Normal(0.0, 1.0), dist.transforms.AffineTransform(mu, tau)
                ),
            )
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)



target = NumPyroTarget(eight_schools_noncentered)
samples = Sampler(target).sample(100)


## Use in numpyro

MCLMCState = namedtuple("MCLMCState", ["x", "u", "l", "g", "key"])


class MCLMC(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "x"

    def __init__(self, model, step_size=0.1):
         
        
        self.step_size = step_size
        self.model = model
        self.eps = 1e-1
        self.L = 10

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):

        target = NumPyroTarget(probabilistic_program=self.model)
        self.sampler = Sampler(target, varEwanted = 5e-4)
        x, u, l, g, key = (self.sampler.get_initial_conditions(x_initial=None,random_key = rng_key))
        return MCLMCState(x, u, l, g, key)
     
    # the kernel, in terms of the state space implied by MCLMCState
    def sample(self, state, model_args, model_kwargs):
        (x, u, l, g, key) = state
        print(state, "state")
        xx, uu, ll, gg, kinetic_change, key= self.sampler.dynamics(x, u, g, key, self.L, self.eps, 1)
        print("done")
        return MCLMCState(xx, uu, ll, gg, key)



kernel = MCLMC(m)
# kernel = NUTS(m)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)

toc = time.time()
mcmc.run(random.PRNGKey(0))
tic = time.time()
print(f"Time to run was {tic-toc} seconds")
posterior_samples = mcmc.get_samples()
mcmc.print_summary() 