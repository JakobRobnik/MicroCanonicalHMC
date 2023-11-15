import jax
import jax.numpy as jnp


lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


def leapfrog(d, T, V):

  def step(x, u, g, eps, sigma):

    # V T V
    uu, r1 = V(eps * 0.5, u, g * sigma)
    xx, l, gg = T(eps, x, uu*sigma)
    uu, r2 = V(eps * 0.5, uu, gg * sigma)
    
    # kinetic energy change
    kinetic_change = (r1 + r2)

    return xx, uu, l, gg, kinetic_change
  
  return step, 1


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
      kinetic_change = (r1 + r2 + r3)

      return xx, uu, ll, gg, kinetic_change
    
  return step, 2



def mclmc(hamilton, partial, d):
    
  def step(x, u, g, random_key, L, eps, sigma):
      """One step of the generalized dynamics."""

      # Hamiltonian step
      xx, uu, ll, gg, kinetic_change = hamilton(x=x, u=u, g=g, eps=eps, sigma = sigma)

      # Langevin-like noise
      nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.) / d)
      uu, key = partial(u= uu, random_key= random_key, nu= nu)

      return xx, uu, ll, gg, kinetic_change, key

  return step



def ma_step(hamilton, full, partial, get_nu, adjust):

  def step(x, u, l, g, random_key, num_steps, num_decoherence, eps, sigma):

      uu, key = full(u, random_key) # do full refreshment (unless full is the identity)
      
      def dynamical_steps(state, useless):
        _x, _u, _l, _g, _kinetic, key = state
        _x, _u, _l, _g, kinetic_change = hamilton(x= _x, u= _u, g= _g, eps= eps, sigma= sigma)
        
        # Langevin-like noise
        nu = get_nu(num_decoherence)
        _u, key = partial(u= _u, random_key= key, nu= nu)

        return (_x, _u, _l, _g, _kinetic + kinetic_change, key), None

      # do num_steps of the Hamiltonian (Langevin) dynamics
      xx, uu, ll, gg, kinetic_change, key = jax.lax.scan(dynamical_steps, init = (x, uu, l, g, 0., key), xs = None, length = num_steps)[0]

      # total energy error
      energy_change = kinetic_change + ll - l

      # accept/reject
      key, key1 = jax.random.split(key)
      acc_prob = jnp.clip(jnp.exp(-energy_change), 0, 1)
      accept = jax.random.bernoulli(key1, acc_prob)
      dyn = jax.tree_util.tree_map(lambda new, old: jax.lax.select(accept, new, old), (xx, uu, ll, gg, key), (x, u, l, g, key))
      
      return dyn, acc_prob
      
  
  def unadjusted_step(x, u, l, g, random_key, num_steps, num_decoherence, eps, sigma):

      uu, key = full(u, random_key) # do full refreshment (unless full is the identity)
      
      def dynamical_steps(state, useless):
        _x, _u, _l, _g, _kinetic, key = state
        _x, _u, _l, _g, kinetic_change = hamilton(x= _x, u= _u, g= _g, eps= eps, sigma= sigma)
        
        # Langevin-like noise
        nu = get_nu(num_decoherence)
        _u, key = partial(u= _u, random_key= key, nu= nu)

        return (_x, _u, _l, _g, _kinetic + kinetic_change, key), _x
      
      # do num_steps of the Hamiltonian (Langevin) dynamics
      state, track = jax.lax.scan(dynamical_steps, init = (x, uu, l, g, 0., key), xs = None, length = num_steps)
      
      xx, uu, ll, gg, kinetic_change, key = state
      
      return (xx, uu, ll, gg, key), track # we also return the entire trajectory

      
  return step if adjust else unadjusted_step





def update_position(grad_nlogp):
  
  def update(eps, x, u):
    xx = x + eps * u
    ll, gg = grad_nlogp(xx)
    return xx, ll, gg
  
  return update



def setup(d, sequential= True, hmc = False):
  
  if hmc:
    N2nu = lambda N2: jnp.exp(-1./N2)
    
    if sequential:

      def full(random_key):
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(subkey, shape = (d, ))
        return u, key

      def partial(u, random_key, nu):
        return u, random_key
        # key, subkey = jax.random.split(random_key)
        # z = jax.random.normal(subkey, shape = (d, ))
        # return u * nu + jnp.sqrt(1- nu**2) * z, key    

      def V(eps, u, g):
          uu = u - eps * g
          return uu, 0.5 * (jnp.dot(uu, uu) - jnp.dot(u, u))

        
    else:
      raise ValueError('Ensemble HMC is not implemented.')
    
  else:

    N2nu = lambda N2: jnp.sqrt((jnp.exp(2./N2) - 1.) / d)

      
    if sequential:
    
      def full(random_key):
          key, subkey = jax.random.split(random_key)
          u = jax.random.normal(subkey, shape = (d, ))
          u /= jnp.sqrt(jnp.sum(jnp.square(u)))
          return u, key
      
        
      def partial(u, random_key, nu):
        key, subkey = jax.random.split(random_key)
        z = nu * jax.random.normal(subkey, shape = (d, ))

        return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key
          
      def V(eps, u, g):
          g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
          e = - g / g_norm
          ue = jnp.dot(u, e)
          delta = eps * g_norm / (d-1)
          zeta = jnp.exp(-delta)
          uu = e *(1-zeta)*(1+zeta + ue * (1-zeta)) + 2*zeta* u
          delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1-ue)*zeta**2)
          return uu/jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r * (d-1)

    else:
      
      def full(random_key, num_chains):
          key, subkey = jax.random.split(random_key)
          u = jax.random.normal(subkey, shape = (num_chains, d))
          normed_u = u / jnp.sqrt(jnp.sum(jnp.square(u), axis = 1))[:, None]
          return normed_u, key

      def partial(u, random_key, nu):
          key, subkey = jax.random.split(random_key)
          noise = nu * jax.random.normal(subkey, shape= u.shape, dtype=u.dtype)

          return (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise), axis = 1))[:, None], key
      
          
      def V(eps, u, g):
          g_norm = jnp.sqrt(jnp.sum(jnp.square(g), axis=1)).T
          nonzero = g_norm > 1e-13  # if g_norm is zero (we are at the MAP solution) we also want to set e to zero and the function will return u
          inv_g_norm = jnp.nan_to_num(1. / g_norm) * nonzero
          e = - g * inv_g_norm[:, None]
          ue = jnp.sum(u * e, axis=1)
          delta = eps * g_norm / (d - 1)
          zeta = jnp.exp(-delta)
          uu = e * ((1 - zeta) * (1 + zeta + ue * (1 - zeta)))[:, None] + 2 * zeta[:, None] * u
          delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1 - ue) * zeta ** 2)
          return uu / (jnp.sqrt(jnp.sum(jnp.square(uu), axis=1)).T)[:, None], delta_r * (d-1)


  return V, full, partial, N2nu
