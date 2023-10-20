import jax 
import jax.numpy as jnp



class Target():

    def __init__(self, nlogp):
        self.d = 2
        self.nlogp = nlogp
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

    def transform(self, x):
        return x

    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ))


def banana(x):
    a, b = 2., 0.2
    y = jnp.array([x[0]/a,  a*x[1] + a*b*(x[0]**2 + a**2) - 4.])
    
    return gauss_nlogp(y, jnp.array([1., 1., 0.5]))


def stn(x):
    return 0.5 * jnp.sum(jnp.square(x))
  
  
def donout(x):
    r0, sigma_sq = 2.6, 0.033,
    r = jnp.sqrt(jnp.sum(jnp.square(x)))
    return jnp.square(r - r0) / sigma_sq
  
  
def invert_cov(Sigma):
    det = Sigma[0] * Sigma[1] - Sigma[2]**2
    H = jnp.array([[Sigma[1], - Sigma[2]], [-Sigma[2], Sigma[0]]]) / det
    return det, H

    
def gauss_p(x, Sigma):
    """sigma = [Sigma[0, 0], Simga[1, 1], Sigma[1, 2]]"""
    det, H = invert_cov(Sigma)
    return jnp.exp(-0.5 * x.T @ H @ x) / (2 * jnp.pi * jnp.sqrt(det))


def gauss_nlogp(x, Sigma):
    """sigma = [Sigma[0, 0], Simga[1, 1], Sigma[1, 2]]"""
    det, H = invert_cov(Sigma)
    return 0.5 * x.T @ H @ x + jnp.log(2 * jnp.pi * jnp.sqrt(det))
    
        
def mixture(x):
    p1 = gauss_p(x + 1.5, jnp.array([0.8, 0.8, 0.]))
    p2 = gauss_p(x - 1.5, jnp.array([0.8, 0.8, 0.]))
    p3 = gauss_p(x - jnp.array([-2, 2]), jnp.array([0.5, 0.5, 0.]))
    return -jnp.log(p1 + p2 + p3)
    

def gauss1d(x, s):
    """-log p"""
    return 0.5 * jnp.log(2*jnp.pi * s) + 0.5 * jnp.square(x / s)
    
    
def funnel(x):
    y = jnp.array([x[1]-2., x[0]])
    return gauss1d(y[0], 3.) + gauss1d(y[1], jnp.exp(0.5 * y[0]))
      
      
def squiggle(x):
    cov= jnp.array([2., 0.5, 0.25])
    y = jnp.array([x[0], x[1] + jnp.sin(5 * x[0])])
    return gauss_nlogp(y, cov)
    
  

targets= {'Banana': Target(banana), 'Donout': Target(donout), 'Standard Normal': Target(stn), 'Gaussian Mixture': Target(mixture), 'Funnel': Target(funnel), 'Squiggle': Target(squiggle)}

