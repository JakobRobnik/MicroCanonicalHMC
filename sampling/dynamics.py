import jax
import jax.numpy as jnp
import numpy as np
import math

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator




def update_momentum(d, sequential):
  """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
  similar to the implementation: https://github.com/gregversteeg/esh_dynamics
  There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
  
  
  def update_sequential(eps, u, g):
      g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
      e = - g / g_norm
      ue = jnp.dot(u, e)
      delta = eps * g_norm / (d-1)
      zeta = jnp.exp(-delta)
      uu = e *(1-zeta)*(1+zeta + ue * (1-zeta)) + 2*zeta* u
      delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1-ue)*zeta**2)
      return uu/jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r

  
  def update_parallel(eps, u, g):
      g_norm = jnp.sqrt(jnp.sum(jnp.square(g), axis=1)).T
      nonzero = g_norm > 1e-13  # if g_norm is zero (we are at the MAP solution) we also want to set e to zero and the function will return u
      inv_g_norm = jnp.nan_to_num(1. / g_norm) * nonzero
      e = - g * inv_g_norm[:, None]
      ue = jnp.sum(u * e, axis=1)
      delta = eps * g_norm / (d - 1)
      zeta = jnp.exp(-delta)
      uu = e * ((1 - zeta) * (1 + zeta + ue * (1 - zeta)))[:, None] + 2 * zeta[:, None] * u
      delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1 - ue) * zeta ** 2)
      return uu / (jnp.sqrt(jnp.sum(jnp.square(uu), axis=1)).T)[:, None], delta_r

  
  return update_sequential if sequential else update_parallel



def update_position(grad_nlogp):
  
  def update(eps, x, u):
    xx = x + eps * u
    ll, gg = grad_nlogp(xx)
    return xx, ll, gg
  
  return update



def minimal_norm(d, T, V):

  def step(x, u, g, eps, sigma):
      """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

      # V T V T V
      uu, r1 = V(eps * lambda_c, u, g * sigma)
      xx, ll, gg = T(eps, x, 0.5*uu*sigma)
      uu, r2 = V(eps * (1 - 2 * lambda_c), uu, gg * sigma)
      xx, ll, gg = T(eps, xx, 0.5*uu*sigma)
      uu, r3 = V(eps * lambda_c, uu, gg * sigma)

      #kinetic energy change
      kinetic_change = (r1 + r2 + r3) * (d-1)

      return xx, uu, ll, gg, kinetic_change
    
  return step, 2



def leapfrog(d, T, V):

  def step(x, u, g, eps, sigma):

    # V T V
    uu, r1 = V(eps * 0.5, u, g * sigma)
    xx, l, gg = T(eps, x, uu*sigma)
    uu, r2 = V(eps * 0.5, uu, gg * sigma)
    
    # kinetic energy change
    kinetic_change = (r1 + r2) * (d-1)

    return xx, uu, l, gg, kinetic_change
  
  return step, 1



def mclmc(hamiltonian_dynamics, partially_refresh_momentum, d):
    
  def step(x, u, g, random_key, L, eps, sigma):
      """One step of the generalized dynamics."""

      # Hamiltonian step
      xx, uu, ll, gg, kinetic_change = hamiltonian_dynamics(x=x, u=u, g=g, eps=eps, sigma = sigma)

      # Langevin-like noise
      nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.) / d)
      uu, key = partially_refresh_momentum(u= uu, random_key= random_key, nu= nu)

      return xx, uu, ll, gg, kinetic_change, key

  return step



def random_unit_vector(d, sequential= True):
  """Generates a random (isotropic) unit vector."""
  
  
  def rng_sequential(random_key):
      key, subkey = jax.random.split(random_key)
      u = jax.random.normal(subkey, shape = (d, ))
      u /= jnp.sqrt(jnp.sum(jnp.square(u)))
      return u, key
    
      
  def rng_parallel(random_key, num_chains):
      key, subkey = jax.random.split(random_key)
      u = jax.random.normal(subkey, shape = (num_chains, d))
      normed_u = u / jnp.sqrt(jnp.sum(jnp.square(u), axis = 1))[:, None]
      return normed_u, key
    
    
  return rng_sequential if sequential else rng_parallel




def partially_refresh_momentum(d, sequential= True):
  """Adds a small noise to u and normalizes."""
    
    
  def rng_sequential(u, random_key, nu):
    key, subkey = jax.random.split(random_key)
    z = nu * jax.random.normal(subkey, shape = (d, ))

    return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key
  

  def rng_parallel(u, random_key, nu):
      key, subkey = jax.random.split(random_key)
      noise = nu * jax.random.normal(subkey, shape= u.shape, dtype=u.dtype)

      return (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise), axis = 1))[:, None], key


  return rng_sequential if sequential else rng_parallel


