import jax
import jax.numpy as jnp
import numpy as np



class Lattice:
    def __init__(self, L, d, BC='periodic'):
        self.L = L
        self.d = d
        self.shape = [L]*d
        self.Nsite = L**d
        self.BC = BC

    def move(self, idx, d, shift):
        coord = self.index2coord(idx)
        coord[d] += shift

        if self.BC != 'periodic':
            if (coord[d]>=self.L) or (coord[d]<0):
                return None
        #wrap around because of the PBC
        if (coord[d]>=self.L): coord[d] -= self.L;
        if (coord[d]<0): coord[d] += self.L;

        return self.coord2index(coord)

    def index2coord(self, idx):
        coord = np.zeros(self.d, int)
        for d in range(self.d):
            coord[self.d-d-1] = idx%self.L;
            idx /= self.L
        return coord

    def coord2index(self, coord): #(i, j) -> i * L + j
        idx = coord[0]
        for d in range(1, self.d):
            idx *= self.L;
            idx += coord[d]
        return idx


class Hypercube(Lattice):
    def __init__(self, L, d, BC='periodic'):
        super(Hypercube, self).__init__(L, d, BC)
        self.Adj = np.zeros((self.Nsite,self.Nsite))
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)
                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0


### MCHMC ###

class Theory:
    """Ising theory after Hubbard-Stratonovich transformation."""

    def __init__(self, L, beta):

        self.d = L**2
        self.L = L
        self.beta = beta

        self.K = jnp.asarray(Hypercube(L, 2, 'periodic').Adj) * beta

        self.grad_nlogp = jax.value_and_grad(self.nlogp)



    def nlogp(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the Ising action (negative logp)."""
        Kx = jnp.matmul(x, self.K)
        return 0.5 * jnp.sum(x * Kx) - jnp.sum(jax.nn.softplus(2. * Kx) - Kx - jnp.log(2.))

    def transform(self, x):
        return x

    def prior_draw(self, key):
        """Args: jax random key
           Returns: one random sample from the prior = N(0, 1)"""

        return jax.random.normal(key, shape=(self.d,), dtype='float64')



### observables ###

def _energy(s, W):
    return -0.5 * np.einsum('ni,ij,nj->n', s, W, s)

def _magnetization(s):
    return np.sum(s, 1)

def energy(s, W, e=None):
    if e is None:
        return np.mean(_energy(s, W)) / s.shape[1]
    else:
        return np.mean(e) / s.shape[1]

def specific_heat(s, W, T, e=None):
    if e is None:
        e = _energy(s, W)
    return ((np.mean(e ** 2) - np.mean(e) ** 2) / T ** 2) / s.shape[1]

def absolute_magnetisation(s):
    return np.mean(np.abs(_magnetization(s))) / s.shape[1]

def susceptibility(s, T):
    M = _magnetization(s)
    return (np.mean(M ** 2) - np.mean(np.abs(M)) ** 2) / T / s.shape[1]



### numpyro (NUTS) ###

import numpyro
from numpyro.distributions import constraints


class ising_numpyro(numpyro.distributions.Distribution):
    """Custom defined phi^4 distribution, see https://forum.pyro.ai/t/creating-a-custom-distribution-in-numpyro/3332/3"""

    support = constraints.real_vector

    def __init__(self, L, beta):

        self.d = L**2
        self.L = L
        self.beta = beta

        self.K = jnp.asarray(Hypercube(L, 2, 'periodic').Adj) * beta

        super().__init__(event_shape=(self.d, ))

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, x):
        Kx = jnp.matmul(x, self.K)
        return 0.5 * jnp.sum(x * Kx) - jnp.sum(jax.nn.softplus(2. * Kx) - Kx - jnp.log(2.))


def model(L, beta):
    x = numpyro.sample('x', ising_numpyro(L, beta))