from .phi4 import *


@jax.jit
def ising_action(x: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    """Compute the Euclidean action for the Ising theory after
       Hubbard-Stratonovich transformation.

       0.5 * x @ Kinv @ x - sum logcosh(x)

    Args:
        x: Single field configuration of shape L^d.

    Returns:
        Scalar, the action of the field configuration..
    """
    Kx = jnp.matmul(x, K)
    return 0.5 * jnp.sum(x * Kx) - jnp.sum(jax.nn.softplus(2. * Kx) - Kx - math.log(2.))


@jax.jit
def grad_ising_action(x: jnp.ndarray, Kinv: jnp.ndarray) -> jnp.ndarray:
    """Compute the gradient of Euclidean action for the Ising
       theory after Hubbard-Stratonovich transformation.

       Kinv @ x - tanh(x)

    Args:
        x: Single field configuration of shape L^d.

    Returns:
        jnp array with the same shape as x
    """
    # shape = x.shape
    # x = jnp.ravel(x)
    Kx = jnp.matmul(x, K)
    return Kx - jnp.matmul(jnp.tanh(Kx), K)





@chex.dataclass
class Theory:
    """Ising theory after Hubbard-Stratonovich transformation."""
    K: jnp.ndarray
    d: chex.Scalar

    def nlogp(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the Ising action (negative logp).

        Args:
            x: Either a single field configuration of shape L^d or
               a batch of those field configurations.

        Returns:
            Either a scalar value or a 1d array of actions for the
            field configuration(s).
        """

        # check whether x are a batch or a single sample
        if x.ndim == 1:
            # chex.assert_shape(x, [self.L] * self.dim)
            return ising_action(x, self.K)
        else:
            # chex.assert_shape(x[0], [self.L] * self.dim)
            act = partial(ising_action, K=self.K)
            return jax.vmap(act)(x)

    def grad_nlogp(self, x):
        if x.ndim == 1:
            # chex.assert_shape(x, [self.L] * self.dim)
            return grad_ising_action(x, self.K)
        else:
            # chex.assert_shape(x[0], [self.L] * self.dim)
            act = partial(grad_ising_action, K=self.K)
            return jax.vmap(act)(x)

    def transform(self, x):
        return x

    def prior_draw(self, key):
        """Args: jax random key
           Returns: one random sample from the prior"""

        return jax.random.normal(key, shape=(self.d,), dtype='float64')


    def _energy(self, s, W):
        return -0.5 * np.einsum('ni,ij,nj->n', s, W, s)

    def _magnetization(self, s):
        return np.sum(s, 1)

    def energy(self, s, W, e=None):
        if e is None:
            return np.mean(self._energy(s, W)) / s.shape[1]
        else:
            return np.mean(e) / s.shape[1]

    def specific_heat(self, s, W, T, e=None):
        if e is None:
            e = self._energy(s, W)
        return ((np.mean(e ** 2) - np.mean(e) ** 2) / T ** 2) / s.shape[1]

    def absolute_magnetisation(self, s):
        return np.mean(np.abs(self._magnetization(s))) / s.shape[1]

    def susceptibility(self, s, T):
        M = self._magnetization(s)
        return (np.mean(M ** 2) - np.mean(np.abs(M)) ** 2) / T / s.shape[1]