
import sys  
sys.path.insert(0, './')

from sampling.dynamics import update_momentum
import jax 

import jax.numpy as jnp

def update_momentum_unstable(d, eps):

    def update(u, g):
        g_norm = jnp.linalg.norm(g)
        e = - g / g_norm
        delta = eps * g_norm / (d-1)
        uu = (u + e*(jnp.sinh(delta)+jnp.dot(e,u*(jnp.cosh(delta)-1)))) / (jnp.cosh(delta) + jnp.dot(e,u*jnp.sinh(delta)))
        return uu 
    
    return update


# the numerically efficient version of the momentum update used in the code should match the naive implementation according to the equation in the paper
def test_momentum_update():
    d = 3
    eps = 1e-3
    u = jax.random.uniform(key=jax.random.PRNGKey(0),shape=(d,))
    u = u / jnp.linalg.norm(u)
    g = jax.random.uniform(key=jax.random.PRNGKey(1),shape=(d,))
    update_stable = update_momentum(d, eps)
    update_unstable = update_momentum_unstable(d, eps)
    update1 = update_stable(u, g)[0]
    update2 = update_unstable(u, g)
    print(update1, update2)
    assert jnp.allclose(update1,update2)
    

