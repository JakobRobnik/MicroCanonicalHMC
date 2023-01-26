import chex
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np


class Lattice:
    def __init__(self,L, d, BC='periodic'):
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

    def coord2index(self, coord):
        idx = coord[0]
        for d in range(1, self.d):
            idx *= self.L;
            idx += coord[d]
        return idx


class Hypercube(Lattice):
    def __init__(self,L, d, BC='periodic'):
        super(Hypercube, self).__init__(L, d, BC)
        self.Adj = np.zeros((self.Nsite,self.Nsite), int)
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)

                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0

#@chex.dataclass
class Theory:
    """Latice Phi^4 theory."""

    def __init__(self, L, m_sq, lam):

        self.d = L**2

        K = jnp.asarray(Hypercube(L, 2, 'periodic').Adj)


        self.nlogp = lambda phi: 0.5 * jnp.dot(phi, jnp.matmul(phi, K)) + jnp.sum(m_sq * jnp.square(phi) + lam * jnp.power(phi, 4))

        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        self.transform = lambda x: x

        self.prior_draw = lambda key: jax.random.normal(key, shape = (self.d, ), dtype = 'float64') #gaussian prior



    def susceptibility1(self, phi):
        """See appendix in https://arxiv.org/pdf/2207.00283.pdf"""
        phibar = jnp.average(phi, axis= 1)

        return self.d * (jnp.average(jnp.square(phibar)) - jnp.square(jnp.average(phibar)))

    def susceptibility2(self, phi):
        """See appendix in https://arxiv.org/pdf/2207.00283.pdf"""
        phibar = jnp.average(phi, axis= 1)

        return self.d * (jnp.average(jnp.square(phibar)) - jnp.square(jnp.average(jnp.abs(phibar))))






