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
    def __init__(self, L, d, BC='periodic', laplace = False):
        super(Hypercube, self).__init__(L, d, BC)
        self.Adj = np.zeros((self.Nsite,self.Nsite))
        for i in range(self.Nsite):
            if laplace:
                self.Adj[i, i] = -2.0 * self.d
            for d in range(self.d):
                j = self.move(i, d, 1)
                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0


class Theory:
    """Latice Phi^4 theory."""

    def __init__(self, L, lam, m_sq = -4):
        """m^2 = -4, such that the diagonal terms cancel out"""

        self.d = L**2
        self.L = L

        if m_sq == -4:
            K = jnp.asarray(Hypercube(L, 2, 'periodic').Adj)
            self.nlogp = lambda phi: - phi @ K @ phi + lam* jnp.sum(jnp.power(phi, 4))

        else:
            laplace = jnp.asarray(Hypercube(L, 2, 'periodic', laplace= True).Adj)
            self.nlogp = lambda phi: - phi @ laplace @ phi + jnp.sum(m_sq * jnp.square(phi) + lam * jnp.power(phi, 4))


        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        self.transform = lambda phi: jnp.ones(1) * jnp.average(phi)

        self.prior_draw = lambda key: jax.random.normal(key, shape = (self.d, ), dtype = 'float64') #gaussian prior


    #Observables

    def susceptibility1(self, phibar):
        """See appendix in https://arxiv.org/pdf/2207.00283.pdf"""
        #phibar = jnp.average(phi, axis= 1)
        return self.d * (jnp.average(jnp.square(phibar)) - jnp.square(jnp.average(phibar))) #= self.d * jnp.std(phibar)**2


    def susceptibility2(self, phibar):
        """See appendix in https://arxiv.org/pdf/2207.00283.pdf"""
        #phibar = jnp.average(phi, axis= 1)

        return self.d * (jnp.average(jnp.square(phibar)) - jnp.square(jnp.average(jnp.abs(phibar))))


    def susceptibility2_full(self, phibar):
        """See appendix in https://arxiv.org/pdf/2207.00283.pdf"""

        steps = jnp.arange(1, phibar.shape[-1]+1)
        phi_1 = jnp.cumsum(jnp.abs(phibar), axis = -1) / steps[None, :]
        phi_2 = jnp.cumsum(jnp.square(phibar), axis = -1) / steps[None, :]

        return self.d * (phi_2 - jnp.square(phi_1))


    def greens_function0(self, phi):
        """zero momentum 2-point Green's function, Equations 22 and 23 in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.034515"""
        phi_lattice = np.reshape(phi, (self.side, self.side))
        phi_t = np.sum(phi_lattice, axis=0)
        phi_omega = np.fft.rfft(phi_t)
        return np.fft.irfft(np.abs(np.square(phi_omega)), self.side)


    def meff(self, phi):
        """effective pole mass estimate for t = 1, 2, 3, ... side-1
            see Equation 28 in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.034515"""
        Gc0 = self.greens_function0(phi)
        return jnp.arccosh((Gc0[0:-2] + Gc0[2:]) / (2*Gc0[1:-1]))






