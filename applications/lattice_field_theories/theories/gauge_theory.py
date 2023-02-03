import jax
import jax.numpy as jnp


class Theory:
    """Lattice U(1) Euclidean gauge theory: lagrangian = -(1/2) sum_{mu < nu} F_{mu nu}^2
        References:
             [1] https://arxiv.org/pdf/2101.08176.pdf
    """

    def __init__(self, L, beta):

        self.d = 2 * L**2
        self.L = L
        self.beta = beta

        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        self.transform = lambda links: links


    def nlogp(self, links):
        """Equation 27 in reference [1]"""
        links_lattice = self.reshape(links)
        action_density = jnp.cos(self.plaquete(links_lattice))
        return -self.beta * jnp.sum(action_density)


    def reshape(self, flat_vector):
        """reshape (d,) -> (2, L, L)"""
        return jnp.reshape(flat_vector, (2, self.L, self.L))


    def plaquete(self, links):
        """Arg(P_01(x)). Return shape = (L, L)"""
        #       theta_0(x) +    theta_1(x + e0)          - theta_0(x+e1)          - x_1(x)
        return (links[0] + jnp.roll(links[1], -1, 0) - jnp.roll(links[0], -1, 1) - links[1])


    def topo_charge(self, links):
        """Topological charge, an integer. Equation 30 in reference [1]."""
        links_lattice = self.reshape(links)
        x = self.plaquete(links_lattice) / (2 * jnp.pi)
        x = (x - jnp.floor(x)) - 0.5 # now, -0.5 < x < 0.5
        return jnp.sum(x)
    

    def prior_draw(self, key):
        """uniform angles [0, 2pi)"""
        return 2 * jnp.pi * jax.random.uniform(key, shape = (self.d, ), dtype = 'float64')



