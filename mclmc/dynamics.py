from typing import Any, NamedTuple
import jax
import jax.numpy as jnp

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


class State(NamedTuple):
    """Dynamical state"""

    x: jax.Array
    u: jax.Array
    l: float
    g: jax.Array
    key: tuple


# class Info(NamedTuple):

#     transformed_x: jax.Array
#     l: jax.Array
#     de: float


def update_momentum(d):
  """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
  similar to the implementation: https://github.com/gregversteeg/esh_dynamics
  There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
  
  
  def update(eps, u, g):
      g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
      e = - g / g_norm
      ue = jnp.dot(u, e)
      delta = eps * g_norm / (d-1)
      zeta = jnp.exp(-delta)
      uu = e *(1-zeta)*(1+zeta + ue * (1-zeta)) + 2*zeta* u
      delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1-ue)*zeta**2)
      return uu/jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r * (d-1)
  
 
  return update


def update_position(grad_nlogp, boundary):
  
  
  def update(eps, x, u, sigma):
    xx = x + eps * u * sigma
    ll, gg = grad_nlogp(xx)
    return xx, u, ll, gg
  
  def update_with_boundary(eps, x, u, sigma):
    xx, reflect = boundary.map(x + eps * u * sigma)
    ll, gg = grad_nlogp(xx)
    uu = reflect * u
    return xx, uu, ll, gg
  
    
  return update if boundary == None else update_with_boundary



def minimal_norm(T, V):

  def step(x, u, g, eps, sigma):
      """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

      # V T V T V
      uu, r1 = V(eps * lambda_c, u, g * sigma)
      xx, uu, ll, gg = T(0.5 * eps, x, uu, sigma)
      uu, r2 = V(eps * (1 - 2 * lambda_c), uu, gg * sigma)
      xx, uu, ll, gg = T(0.5 * eps, xx, uu, sigma)
      uu, r3 = V(eps * lambda_c, uu, gg * sigma)

      #kinetic energy change
      kinetic_change = (r1 + r2 + r3)

      return xx, uu, ll, gg, kinetic_change
    
  return step, 2



def leapfrog(T, V):

  def step(x, u, g, eps, sigma):

    # V T V
    uu, r1 = V(eps * 0.5, u, g * sigma)
    xx, uu, l, gg = T(eps, x, uu, sigma)
    uu, r2 = V(eps * 0.5, uu, gg * sigma)

    # kinetic energy change
    kinetic_change = (r1 + r2)

    return xx, uu, l, gg, kinetic_change
  
  return step, 1



def mclmc(hamilton, partial, get_nu):
  
  
  def step(dyn, hyp):
      """One step of the generalized dynamics."""
      
      # Hamiltonian step
      x, u, l, g, kinetic_change = hamilton(x=dyn.x, u=dyn.u, g=dyn.g, eps=hyp.eps, sigma = hyp.sigma)

      # Langevin-like noise
      u, key = partial(u= u, random_key= dyn.key, nu= get_nu(hyp.L/hyp.eps))

      energy_change = kinetic_change + l - dyn.l
      
      return State(x, u, l, g, key), energy_change

  return step



def full_refresh(d):
  """Generates a random (isotropic) unit vector."""
  
  
  def rng(random_key):
      key, subkey = jax.random.split(random_key)
      u = jax.random.normal(subkey, shape = (d, ))
      u /= jnp.sqrt(jnp.sum(jnp.square(u)))
      return u, key
  
    
  return rng




def partial_refresh(d):
  """Adds a small noise to u and normalizes."""
    
  def rng(u, random_key, nu):
    key, subkey = jax.random.split(random_key)
    z = nu * jax.random.normal(subkey, shape = (d, ))

    return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key
  
  get_nu = lambda Nd: jnp.sqrt((jnp.exp(2./Nd) - 1.) / d) #MCHMC paper (Nd = L/eps)
  
  return rng, get_nu


