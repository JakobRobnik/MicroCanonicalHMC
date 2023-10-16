# a minimal example of how to run the code

from sampling.sampler import Sampler, Target
import jax.numpy as jnp
import jax.random as jr

# an (unnormalized) distribution you wish to sample from
nlogp = lambda x: 0.5 * jnp.sum(jnp.square(x))

samples = Sampler(Target(d=10, nlogp=nlogp)).sample(
    num_steps=100,
    x_initial=jnp.ones(10,)
    )

print(samples.shape)
