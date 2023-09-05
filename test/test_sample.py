from sampling.dynamics import update_momentum
import jax 
import jax.numpy as jnp

def update_momentum_unstable(d, eps):
    def update(u, g):
        # delta = eps * jnp.linalg.norm(g)/d
        # e = -g/jnp.linalg.norm(g)

        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        delta = eps * g_norm / (d-1)

        uu = (u + e*(jnp.sinh(delta)+jnp.dot(e,u*(jnp.cosh(delta)-1)))) / (jnp.cosh(delta) + jnp.dot(e,u*jnp.sinh(delta)))
        return uu / jnp.linalg.norm(uu)
    return update


def test_1():
    d = 3
    eps = 1e-3
    u = jax.random.uniform(key=jax.random.PRNGKey(0),shape=(d,))
    g = jax.random.uniform(key=jax.random.PRNGKey(1),shape=(d,))
    update_stable = update_momentum(d, eps)
    update_unstable = update_momentum_unstable(d, eps)
    update1 = update_stable(u, g)[0]
    update2 = update_unstable(u, g)
    print(update1, update2)
    assert jnp.array_equal(update1,update2)
    
    assert 1==1

test_1()