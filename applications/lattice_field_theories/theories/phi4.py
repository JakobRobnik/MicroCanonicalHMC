import jax
import jax.numpy as jnp
import numpy as np



class Theory:
    """Latice Phi^4 theory (MCHMC target class)"""

    def __init__(self, L, lam):
        """m^2 = -4, such that the diagonal terms cancel out"""

        self.d = L**2
        self.L = L
        self.lam = lam
        
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        self.transform = lambda phi: jnp.ones(1) * jnp.average(phi)

        self.prior_draw = lambda key: jax.random.normal(key, shape = (self.d, ), dtype = 'float64') #gaussian prior
        
    
    def nlogp(self, x):
        """action of the theory"""
        phi = x.reshape(self.L, self.L)
        action_density = self.lam*jnp.power(phi, 4) - phi*(jnp.roll(phi, -1, 0) + jnp.roll(phi, 1, 0) + jnp.roll(phi, -1, 1) + jnp.roll(phi, 1, 1))
        return jnp.sum(action_density)
        
        
    #Observables

    def susceptibility1(self, phibar):
        """See appendix in https://arxiv.org/pdf/2207.00283.pdf"""
        #phibar = jnp.average(phi, axis= 1)
        return self.d * (jnp.average(jnp.square(phibar)) - jnp.square(jnp.average(phibar))) #= self.d * jnp.std(phibar)**2


    def susceptibility2(self, phibar):
        """See appendix in https://arxiv.org/pdf/2207.00283.pdf"""
        #phibar = jnp.average(phi, axis= 1)

        return self.d * (jnp.average(jnp.square(phibar), axis= -1) - jnp.square(jnp.average(jnp.abs(phibar), axis = -1)))


    def susceptibility2_full(self, phibar):
        """See appendix in https://arxiv.org/pdf/2207.00283.pdf"""

        steps = jnp.arange(1, phibar.shape[-1]+1)
        phi_1 = jnp.cumsum(jnp.abs(phibar), axis = -1) / steps[None, :]
        phi_2 = jnp.cumsum(jnp.square(phibar), axis = -1) / steps[None, :]

        return self.d * (phi_2 - jnp.square(phi_1))

    
    def psd(self, phi):
        return jnp.square(jnp.abs(jnp.fft.fft2(phi))) / self.L ** 2

    
#     def greens_function0(self, phi):
#         """zero momentum 2-point Green's function, Equations 22 and 23 in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.034515"""
#         phi_lattice = np.reshape(phi, (self.side, self.side))
#         phi_t = np.sum(phi_lattice, axis=0)
#         phi_omega = np.fft.rfft(phi_t)
#         return np.fft.irfft(np.abs(np.square(phi_omega)), self.side)


#     def meff(self, phi):
#         """effective pole mass estimate for t = 1, 2, 3, ... side-1
#             see Equation 28 in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.034515"""
#         Gc0 = self.greens_function0(phi)
#         return jnp.arccosh((Gc0[0:-2] + Gc0[2:]) / (2*Gc0[1:-1]))



reduced_lam = jnp.linspace(-2.5, 7.5, 16) #lambda range around the critical point (m^2 = -4 is fixed)


def unreduce_lam(reduced_lam, side):
    """see Fig 3 in https://arxiv.org/pdf/2207.00283.pdf"""
    return 4.25 * (reduced_lam * np.power(side, -1.0) + 1.0)


def reduce_chi(chi, side):
    return chi * np.power(side, -7.0/4.0)

