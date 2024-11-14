import jax
import jax.numpy as jnp



def setup():
            
    # parameter grid
    x = jnp.linspace(0, 1, 5)
    y = jnp.linspace(1, 2, 7)
    z = jnp.linspace(3, 4, 10)
    
    def ess(x, y, z, key):
        """function that we want to evaluate for different values of parameters x, y and different random keys"""
        return jnp.sum(jnp.power(y * jnp.abs(jax.random.normal(key, shape = (10,))), x)) + z
    
    return (x, y, z), ess, 'synthetic'
    

    