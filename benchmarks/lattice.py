

import pickle
import warnings
import jax
import jax.numpy as jnp
import sys
import time

sys.path.append("./")
import os



class Phi4:
    """Latice Phi^4 theory"""

    def __init__(self, L, lam):
        """m^2 = -4, such that the diagonal terms cancel out"""

        self.name = 'Phi4'
        
        self.ndims = L**2
        self.L = L
        self.lam = lam

        # import os
        # print(os.getcwd(), "foo")
        try:
            print(f'./ground_truth/Phi4/e_x2_{self.L}_{self.lam}.pkl')
            self.E_x2 = pickle.load(open(f'./ground_truth/Phi4/e_x2_{self.L}_{self.lam}.pkl', 'rb'))
            self.E_x4 = pickle.load(open(f'./ground_truth/Phi4/e_x4_{self.L}_{self.lam}.pkl', 'rb'))
            self.Var_x2 = self.E_x4 - self.E_x2**2
        except:
            warnings.warn(f"ground truth for e_x2_{self.L}_{self.lam} not found")
        
        self.transform = self.psd

        self.sample_init = lambda key: jax.random.normal(key, shape = (self.ndims, ))
        
    
    def logdensity_fn(self, x):
        """action of the theory"""
        phi = x.reshape(self.L, self.L)
        action_density = self.lam*jnp.power(phi, 4) - phi*(jnp.roll(phi, -1, 0) + jnp.roll(phi, 1, 0) + jnp.roll(phi, -1, 1) + jnp.roll(phi, 1, 1))
        return -jnp.sum(action_density)

    def psd(self, phi):
        return jnp.square(jnp.abs(jnp.fft.fft2(phi.reshape(self.L, self.L)))) / self.L ** 2

# print(Phi4(3,1).transform(jnp.ones((9,))))


class U1:
    """Lattice U(1) Euclidean gauge theory: lagrangian = -(1/2) sum_{mu < nu} F_{mu nu}^2
        References:
             [1] https://arxiv.org/pdf/2101.08176.pdf
             
        The variables (links) are described as elements of the Lie algebra 'theta', such that the group elements are U = e^{i theta}.
    """

    def __init__(self, Lt, Lx, beta= 1.):
        """Args:
                lattice size = (Lt, Lx)
                beta: inverse temperature
        """
        
        self.name = 'U1'

        # self.E_x2 = jnp.zeros(10,)
        # self.Var_x2 = 2 * jnp.square(self.E_x2)

        self.ndims = 2 * Lt*Lx
        self.Lt, self.Lx, self.beta = Lt, Lx, beta
        self.beta = beta
        self.unflatten = lambda links_flattened: links_flattened.reshape(2, Lt, Lx)
        self.locs = jnp.array([[i//Lx, i%Lx] for i in range(Lt*Lx)]) #the set of all possible lattice sites

        self.sample_init = lambda key: 2 * jnp.pi * jax.random.uniform(key, shape = (self.ndims, ))

        self.transform = self.polyakov_autocorr

        # self.E_x = jnp.zeros(ndims)
        # self.Var_x2 = 2 * jnp.square(self.E_x2)


    def logdensity_fn(self, links):
        """Equation 27 in reference [1]"""
        action_density = jnp.cos(self.plaquete(self.unflatten(links)))
        return self.beta * jnp.sum(action_density)

    
    def plaquete(self, links):
        """Computers theta_{0 1} = Arg(P_{01}(x)) on the lattice. output shape: (L, L)"""

        #       theta_0(x) +    theta_1(x + e0)          - theta_0(x+e1)          - x_1(x)
        return (links[0] + jnp.roll(links[1], -1, 0) - jnp.roll(links[0], -1, 1) - links[1])


    def polyakov_autocorr(self, links_flattened):
        
        links = self.unflatten(links_flattened)
        
        polyakov_angle = jnp.sum(links[0], axis = 0)
        polyakov = jnp.cos(polyakov_angle) + 1j * jnp.sin(polyakov_angle)
        # the result is the same as using [jnp.real(jnp.average(polyakov * jnp.roll(jnp.conjugate(polyakov), -n))) for n in range(self.Lx)], but it is computed faster by the fft
        return jnp.real(jnp.fft.ifft(jnp.square(jnp.abs(jnp.fft.fft(polyakov))))[1:1+self.Lx//2]) / self.Lx # fft based autocorrelation, we only store 1:1+Lx//2 (as the autocorrelation is then periodic)

    
