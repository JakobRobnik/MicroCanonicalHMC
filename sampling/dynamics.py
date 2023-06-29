import jax
import jax.numpy as jnp
import numpy as np
import math

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator

grad_evals = {'MN' : 2, 'LF' : 1}

def update_momentum(d, eps):
    
  def update(u, g):
      """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
      similar to the implementation: https://github.com/gregversteeg/esh_dynamics
      There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
      g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
      e = - g / g_norm
      ue = jnp.dot(u, e)
      delta = eps * g_norm / (d-1)
      zeta = jnp.exp(-delta)
      uu = e *(1-zeta)*(1+zeta + ue * (1-zeta)) + 2*zeta* u
      delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1-ue)*zeta**2)
      return uu/jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r
  
  return update

def update_position(eps, u, shift, grad_nlogp):
  def update(x):
    xx = shift(x, eps * u)
    ll, gg = grad_nlogp(xx)
    return xx, ll, gg
  return update

def random_unit_vector(d):
    def given_key(random_key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(subkey, shape = (d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key
    return given_key


def minimal_norm(d, shift, grad_nlogp, sigma):
      def step(u,x,g, eps):
        
          """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

          # V T V T V
          uu, r1 = update_momentum(d, eps * lambda_c)(u, g * sigma)
          xx, ll, gg = update_position(eps, 0.5*uu*sigma, shift, grad_nlogp)(x)
          uu, r2 = update_momentum(d, eps * (1 - 2 * lambda_c))(uu, gg * sigma)
          xx, ll, gg = update_position(eps, 0.5*uu*sigma, shift, grad_nlogp)(xx)
          uu, r3 = update_momentum(d, eps * lambda_c)(uu, gg * sigma)

          #kinetic energy change
          kinetic_change = (r1 + r2 + r3) * (d-1)

          return xx, uu, ll, gg, kinetic_change
      return step

def leapfrog(d, shift, grad_nlogp, sigma):
      def step(u,x,g, eps):
        """leapfrog"""

        # half step in momentum
        uu, delta_r1 = update_momentum(d, eps * 0.5)(u, g * sigma)

        # full step in x
        xx, l, gg = update_position(eps, uu*sigma, shift, grad_nlogp)(x)

        # half step in momentum
        uu, delta_r2 = update_momentum(d, eps * 0.5)(uu, gg * sigma)
        kinetic_change = (delta_r1 + delta_r2) * (d-1)

        return xx, uu, l, gg, kinetic_change
      return step

def hamiltonian_dynamics(integrator, sigma, grad_nlogp, shift, d):
    if integrator == "LF": #leapfrog (first updates the velocity)
        return leapfrog( 
                                                      sigma=sigma, 
                                                      grad_nlogp=grad_nlogp,
                                                      shift=shift,
                                                      d=d
                                                      )

    elif integrator== 'MN': #minimal norm integrator (velocity)


        return minimal_norm(
                                                      sigma=sigma, 
                                                      grad_nlogp=grad_nlogp,
                                                      shift=shift,
                                                      d=d
                                                      )
    else:
        raise Exception("Integrator must be either MN (minimal_norm) or LF (leapfrog)")

def partially_refresh_momentum(d, nu):
    def func(u, random_key):
      """Adds a small noise to u and normalizes."""
      key, subkey = jax.random.split(random_key)
      z = nu * jax.random.normal(subkey, shape = (d, ), dtype = 'float64')

      return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key
    return func 