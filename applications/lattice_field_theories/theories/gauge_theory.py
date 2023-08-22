import jax
import jax.numpy as jnp

from scipy.special import iv as BesselI
import numpy as np



class Theory:
    """Lattice U(1) Euclidean gauge theory: lagrangian = -(1/2) sum_{mu < nu} F_{mu nu}^2
        References:
             [1] https://arxiv.org/pdf/2101.08176.pdf
    """

    def __init__(self, L, beta, observable):
        """Args:
                L:lattice side
                beta: inverse temperature
                observable: which quantity to track. Currently supported:
                            'all': all link variables
                            'topo sin': topological charge with the sinus in the definition (easy to compare with ground truth)
                            'topo int': integer valued topological charge definition
        """
        self.d = 2 * L**2
        self.L = L
        self.beta = beta
        self.link_shape = (2, L, L)

        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        if observable == 'all':
            self.transform = lambda links: links
        elif observable == 'topo sin':
            self.transform = lambda links: self.topo_charge_sin(links) * jnp.ones(1)
        elif observable == 'topo int':
            self.transform = lambda links: self.topo_charge_int(links) * jnp.ones(1)
        else:
            raise ValueError('Observable = ' + observable + ' is not valid parameter.')


    def nlogp(self, links):
        """Equation 27 in reference [1]"""
        action_density = jnp.cos(self.plaquete(links.reshape(self.link_shape)))
        return -self.beta * jnp.sum(action_density)



    def plaquete(self, links):
        """Computers theta_{0 1} = Arg(P_{01}(x)) on the lattice. output shape: (L, L)"""

        #       theta_0(x) +    theta_1(x + e0)          - theta_0(x+e1)          - x_1(x)
        return (links[0] + jnp.roll(links[1], -1, 0) - jnp.roll(links[0], -1, 1) - links[1])


    def topo_charge_int(self, links):
        """Topological charge, an integer. Equation 30 in reference [1]."""

        x = self.plaquete(links.reshape(self.link_shape)) / (2 * jnp.pi)
        x = jnp.remainder(x + 0.5, 1.0) - 0.5

        return jnp.sum(x)


    def topo_charge_sin(self, links):
        """Topological charge, not an integer"""
    
        x = self.plaquete(links.reshape(self.link_shape))
    
        return jnp.sum(jnp.sin(x)) / (2 * np.pi)
    
    
    def Willson_loop(self, links):
        # 
        
        links[]
    

    def prior_draw(self, key):
        """uniform angles [0, 2pi)"""
        return 2 * jnp.pi * jax.random.uniform(key, shape = (self.d, ), dtype = 'float64')


    def chiq(self, Q):
        """topological susceptibility"""
        return jnp.average(jnp.square(Q))


def thermodynamic_ground_truth(beta):
    """topological susceptibility (taking the first term in the expansion"""
    return (BesselI(1, beta)/BesselI(0, beta)) / (beta * 4 * np.pi**2)
