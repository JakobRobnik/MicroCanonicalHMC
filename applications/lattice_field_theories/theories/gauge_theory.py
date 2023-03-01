import jax
import jax.numpy as jnp

from scipy.special import iv as BesselI
import numpy as np



class Theory:
    """Lattice U(1) Euclidean gauge theory: lagrangian = -(1/2) sum_{mu < nu} F_{mu nu}^2
        References:
             [1] https://arxiv.org/pdf/2101.08176.pdf
    """

    def __init__(self, L, beta):
        """Args:
                L:lattice side
                beta: inverse temperature
        """
        self.d = 2 * L**2
        self.L = L
        self.beta = beta
        self.link_shape = (2, L, L)

        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        #self.transform = lambda links: links
        self.transform = lambda links: self.topo_charge(links) * jnp.ones(1)



    def nlogp(self, links):
        """Equation 27 in reference [1]"""
        action_density = jnp.cos(self.plaquete(links.reshape(self.link_shape)))
        return -self.beta * jnp.sum(action_density)



    def plaquete(self, links):
        """Computers theta_{0 1} = Arg(P_{01}(x)) on the lattice. output shape: (L, L)"""

        #       theta_0(x) +    theta_1(x + e0)          - theta_0(x+e1)          - x_1(x)
        return (links[0] + jnp.roll(links[1], -1, 0) - jnp.roll(links[0], -1, 1) - links[1])


    def topo_charge(self, links):
        """Topological charge, an integer. Equation 30 in reference [1]."""

        x = self.plaquete(links.reshape(self.link_shape)) / (2 * jnp.pi)
        x = jnp.remainder(x + 0.5, 1.0) - 0.5

        return jnp.sum(x)



    def prior_draw(self, key):
        """uniform angles [0, 2pi)"""
        return 2 * jnp.pi * jax.random.uniform(key, shape = (self.d, ), dtype = 'float64')


    def chiq(self, Q):
        """topological susceptibility"""
        return jnp.average(jnp.square(Q))


def thermodynamic_ground_truth(beta):
    """topological susceptibility (taking the first term in the expansion"""
    return (BesselI(1, beta)/BesselI(0, beta)) / (beta * 4 * np.pi**2)



def test():
    import torch
    import numpy as np

    torch_device = 'cpu'
    float_dtype = np.float64 # double

    L = 8
    beta = 1.0
    lattice_shape = (L,L)
    link_shape = (2,L,L)
    # some arbitrary configurations
    u1_ex1 = 2*np.pi*np.random.random(size=link_shape).astype(float_dtype)
    u1_ex2 = 2*np.pi*np.random.random(size=link_shape).astype(float_dtype)
    cfgs = torch.from_numpy(np.stack((u1_ex1, u1_ex2), axis=0)).to(torch_device)


    def compute_u1_plaq(links, mu, nu):
        """Compute U(1) plaquettes in the (mu,nu) plane given `links` = arg(U)"""
        return (links[:,mu] + torch.roll(links[:,nu], -1, mu+1) - torch.roll(links[:,mu], -1, nu+1) - links[:,nu])

    class U1GaugeAction:
        def __init__(self, beta):
            self.beta = beta
        def __call__(self, cfgs):
            Nd = cfgs.shape[1]
            action_density = 0
            for mu in range(Nd):
                for nu in range(mu+1,Nd):
                    action_density = action_density + torch.cos(compute_u1_plaq(cfgs, mu, nu))
            return -self.beta * torch.sum(action_density, dim=tuple(range(1,Nd+1)))

    print(U1GaugeAction(beta=beta)(cfgs))

    def torch_mod(x):
        return torch.remainder(x, 2*np.pi)

    def torch_wrap(x):
        return torch_mod(x+np.pi) - np.pi

    def topo_charge(x):
        P01 = torch_wrap(compute_u1_plaq(x, mu=0, nu=1))
        axes = tuple(range(1, len(P01.shape)))
        return torch.sum(P01, dim=axes) / (2*np.pi)

    print(topo_charge(cfgs))

    target = Theory(L, beta)

    u1 = jnp.array(u1_ex1)
    print(target.nlogp(u1), target.topo_charge(u1))
    u2 = jnp.array(u1_ex2)
    print(target.nlogp(u2), target.topo_charge(u2))


