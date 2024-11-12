import jax
import jax.numpy as jnp

# parameter grid
x = jnp.linspace(0, 1, 10)
y = jnp.linspace(1, 2, 5)
X, Y = jnp.meshgrid(x, y)

def func(x, y, key):
    """function that we want to evaluate for different values of parameters x, y and different random keys"""
    return jnp.sum(jnp.power(y * jax.random.normal(key, shape = (10,)), x))

func_vmap = jax.vmap(lambda key: jax.vmap(func, ((0, 1), (0, 1), None))(X, Y, key))      

