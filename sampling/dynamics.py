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

# should eventually be merged with the above
def random_unit_vector_broadcast(d, random_key, num_chains):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(subkey, shape = (num_chains, d), dtype = 'float64')
        normed_u = u / jnp.sqrt(jnp.sum(jnp.square(u), axis = 1))[:, None]
        return normed_u, key

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

##################
## resampling code
##################

def systematic_resampling(logw, random_key):
    # Normalize weights
    w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))

    # Compute cumulative sum
    cumsum_w = jnp.cumsum(w)

    # Number of particles
    N = len(logw)

    # Generate N uniform random numbers, then transform them appropriately
    key, subkey = jax.random.split(random_key)
    u = (jnp.arange(N) + jax.random.uniform(subkey)) / N

    # Compute resampled indices
    indices = jnp.searchsorted(cumsum_w, u)

    return indices, key

def resample_particles(logw, x, u, l, g, key, L, eps, T):

            indices, key = systematic_resampling(logw, key)

            x_resampled = jnp.take(x, indices, axis=0)
            u_resampled = jnp.take(u, indices, axis=0)
            l_resampled = jnp.take(l, indices)
            g_resampled = jnp.take(g, indices, axis=0)

            return (x_resampled, u_resampled, l_resampled, g_resampled, key, L, eps, T)

##################
## next temperature choice
##################

def bisection(f, a, b, tol=1e-3, max_iter=100):

    def cond_fn(inputs):
        a, b, _, iter_count = inputs
        return (jnp.abs(a - b) > tol * a) & (iter_count < max_iter)

    def body_fn(inputs):
        a, b, midpoint, iter_count = inputs
        midpoint = (a + b) / 2.0
        a, b = jax.lax.cond(f(midpoint) > 0, lambda _: (a, midpoint), lambda _: (midpoint, b), operand=())

        #jax.debug.print("a: {}, b: {}, midpoint: {}, iter: {}", a, b, midpoint, iter_count)
        return a, b, midpoint, iter_count + 1

    #a, b, midpoint, iter_count = jax.lax.while_loop(cond_fn, body_fn, (a, b, 0.0, 0))
    # Use cond to decide which path to follow, note the condition is now f(b) <= 0
    a, b, midpoint, iter_count = jax.lax.cond(f(b) <= 0, 
                                              lambda _: (b, b, b, 0), 
                                              lambda _: jax.lax.while_loop(cond_fn, body_fn, (a, b, 0.0, 0)), 
                                              operand=())

    return midpoint

def solve_ess(Tprev, ess, l):
      def get_ess(beta):
                logw = -(beta - 1.0/Tprev) * l 
                weights = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
                #jax.debug.print("estimate: {}, ess: {}", 1.0 / jnp.sum(weights**2) / len(weights), ess)
                return ess * len(weights) - 1.0 / jnp.sum(weights**2)
      return get_ess

def update_temp(Tprev, ess, l, target_temp):

            beta = bisection(solve_ess(Tprev, ess, l), 1.0/Tprev, 1.0/target_temp)
            return 1.0 / beta

def initialize(Target, random_key, x_initial, num_chains):


        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains + 1)
                x = jax.vmap(Target.prior_draw)(keys_all[1:])
                key = keys_all[0]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')

        else:  # initial x is given
            x = jnp.copy(x_initial)

        l, g = jax.vmap(Target.grad_nlogp)(x)


        ### initial velocity ###
        u, key = random_unit_vector_broadcast(Target.d, key, num_chains)  # random velocity orientations
        
        ## if you want to use random_unit_vector from dynamics, this is how
        # keys = jax.random.split(key, num=num_chains+1)
        # u, key = jax.vmap(random_unit_vector(self.Target.d))(keys[1:])  # random velocity orientations


        return x, u, l, g, key