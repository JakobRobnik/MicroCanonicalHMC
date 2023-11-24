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



def mclmc(hamilton, partial, get_nu):
    
  def step(dyn, hyp):
      """One step of the generalized dynamics."""
      
      # Hamiltonian step
      x, u, l, g, kinetic_change = hamilton(x=dyn['x'], u=dyn['u'], g=dyn['g'], eps=hyp['eps'], sigma = hyp['sigma'])

      # Langevin-like noise
      u, key = partial(u= u, random_key= dyn['key'], nu= get_nu(hyp['L']/hyp['eps']))

      energy_change = kinetic_change + l - dyn['l']
      
      return {'x': x, 'u': u, 'l': l, 'g': g, 'key': key}, energy_change

  return step



def ma_step(hamilton, full, partial, get_nu):

  def step(dyn, hyp):
      
      x, l, g = dyn['x'], dyn['l'], dyn['g']    
      
      u, key = full(dyn['u'], dyn['key']) # do full refreshment (unless full is the identity)
      
      nu = get_nu(0.5 * dyn['L'] / dyn['eps']) # 1/2 because we use two O updates per step
        
      def dynamical_steps(state, useless):
        _x, _u, _l, _g, key, _kinetic = state
        
        # O
        _u, key = partial(u= _u, random_key= key, nu= nu)
        
        # BAB (or BABAB in case of Minimal norm integrator)
        _x, _u, _l, _g, kinetic_change = hamilton(x=_x, u=_u, g=_g, eps= hyp['eps'], sigma= hyp['sigma'])
        
        # O
        _u, key = partial(u= _u, random_key= key, nu= nu)
        
        return (_x, _u, _l, _g, key, _kinetic + kinetic_change), None


      # do num_steps of the Hamiltonian (Langevin) dynamics
      xx, uu, ll, gg, key, kinetic_change = jax.lax.scan(dynamical_steps, init = (x, u, l, g, key, 0.), xs = None, length = hyp['N'])[0]

      # total energy error
      energy_change = kinetic_change + ll - l

      # accept/reject
      key, key1 = jax.random.split(key)
      acc_prob = jnp.clip(jnp.exp(-energy_change), 0, 1)
      accept = jax.random.bernoulli(key1, acc_prob)
      dyn = {'x': xx * accept + x * (1-accept), 
             'u': uu * accept + u * (1-accept), 
             'l': ll * accept + l * (1-accept), 
             'g': gg * accept + g * (1-accept), 
             'key': key}
      
      return dyn, energy_change
      
  
  return step





def update_position(grad_nlogp):
  
  def update(eps, x, u):
    xx = x + eps * u
    ll, gg = grad_nlogp(xx)
    return xx, ll, gg
  
  return update



def setup(d, sequential= True, hmc = False):
  
  if hmc:
    Ndnu = lambda Nd: jnp.exp(-1./Nd) # MEADS paper, Equation 6 (https://proceedings.mlr.press/v151/hoffman22a/hoffman22a.pdf)
    
    if sequential:

      def full(random_key):
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(subkey, shape = (d, ))
        return u, key

      def partial(u, random_key, nu):
        key, subkey = jax.random.split(random_key)
        z = jax.random.normal(subkey, shape = (d, ))
        return u * nu + jnp.sqrt(1- nu**2) * z, key    

      def V(eps, u, g):
          uu = u - eps * g
          return uu, 0.5 * (jnp.dot(uu, uu) - jnp.dot(u, u))

    else:
      raise ValueError('Ensemble HMC is not implemented.')

    
  else:

    Ndnu = lambda Nd: jnp.sqrt((jnp.exp(2./Nd) - 1.) / d) #MCHMC paper (Nd = L/eps)

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


  return V, full, partial, Ndnu
