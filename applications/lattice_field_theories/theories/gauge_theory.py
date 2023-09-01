import jax
import jax.numpy as jnp

from scipy.special import iv as BesselI
import numpy as np



class Theory:
    """Lattice U(1) Euclidean gauge theory: lagrangian = -(1/2) sum_{mu < nu} F_{mu nu}^2
        References:
             [1] https://arxiv.org/pdf/2101.08176.pdf
             
        The variables (links) are described as elements of the Lie algebra theta, such that U = e^{i theta}.
    """

    def __init__(self, Lt, Lx, beta, observable, Oparams = None):
        """Args:
                lattice size = (Lt, Lx)
                beta: inverse temperature
                observable: which quantity to track. Currently supported:
                            'all': all link variables
                            'topo sin': topological charge with the sinus in the definition (easy to compare with ground truth)
                            'topo int': integer valued topological charge definition
                            'Wilson loop': Wilson loop (averaged over shifts)
                Oparams: an array of parameters for the observable. Observable will be an array of the same size. 
                            e.g. Wilson loop depends on the parameters nt and nx. Passing [[3, 3], [5, 3]], will produce observable = [W(3, 3), W(5, 3)].
                            can also be 'all' for Wilson which will compute all sizes 0 < nt < Lt , 0 < nx <= Lx/2 
        """
        self.d = 2 * Lt*Lx
        self.Lt, self.Lx, self.beta = Lt, Lx, beta
        self.beta = beta
        self.unflatten = lambda links_flattened: links_flattened.reshape(2, Lt, Lx)
        self.locs = jnp.array([[i//Lx, i%Lx] for i in range(Lt*Lx)]) #the set of all possible lattice sites

        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        if observable == 'all':
            self.transform = lambda links: links
        elif observable == 'topo sin':
            self.transform = lambda links: self.topo_charge_sin(links) * jnp.ones(1)
        elif observable == 'topo int':
            self.transform = lambda links: self.topo_charge_int(links) * jnp.ones(1)
        elif observable == 'Wilson loop':
            
            average_shifts = True
            wl = self.Wilson_loop if average_shifts else self.Wilson_loop_single
            
            if Oparams == None:
                raise ValueError('Wilson loop observable requires parameters nt and nx.')
            elif Oparams == 'all':
                nmax = Lx // 2
                sizes = [[1 + i//nmax, 1 + i%nmax] for i in range((Lt-1)*nmax)]
            else:
                sizes = Oparams
            self.transform = lambda links: jnp.array([wl(links, *ntnx) for ntnx in sizes])
        else:
            raise ValueError('Observable = ' + observable + ' is not valid parameter.')


    def nlogp(self, links):
        """Equation 27 in reference [1]"""
        action_density = jnp.cos(self.plaquete(self.unflatten(links)))
        return -self.beta * jnp.sum(action_density)



    def plaquete(self, links):
        """Computers theta_{0 1} = Arg(P_{01}(x)) on the lattice. output shape: (L, L)"""

        #       theta_0(x) +    theta_1(x + e0)          - theta_0(x+e1)          - x_1(x)
        return (links[0] + jnp.roll(links[1], -1, 0) - jnp.roll(links[0], -1, 1) - links[1])


    def topo_charge_int(self, links):
        """Topological charge, an integer. Equation 30 in reference [1]."""

        x = self.plaquete(self.unflatten(links)) / (2 * jnp.pi)
        x = jnp.remainder(x + 0.5, 1.0) - 0.5

        return jnp.sum(x)


    def topo_charge_sin(self, links):
        """Topological charge, not an integer"""
    
        x = self.plaquete(self.unflatten(links))
    
        return jnp.sum(jnp.sin(x)) / (2 * np.pi)
    
    
    
    def Wilson_loop_single(self, links_flattened, nt, nx):
        """Real part of the Wilson loop, averaged over the shifts"""
        
        links = self.unflatten(links_flattened)
        return jnp.cos(jnp.sum(links[0, :nt, 0]) + jnp.sum(links[1, nt, :nx]) - jnp.sum(links[0, :nt, nx]) - jnp.sum(links[1, 0, :nx]))    
            
        
        
    def Wilson_loop(self, links_flattened, nt, nx):
        """Real part of the Wilson loop, averaged over the shifts"""
        
        
        links0 = self.unflatten(links_flattened)

        def Wilson_loop_single(loc):
            """Real part of the Wilson loop: (0, 0) -> (nt, 0) -> (nt, nx) -> (0, nx) -> (0, 0) shifted to loc = (M, N)"""
            links = jnp.roll(links0, loc, axis = (0, 1))
            return jnp.cos(jnp.sum(links[0, :nt, 0]) + jnp.sum(links[1, nt, :nx]) - jnp.sum(links[0, :nt, nx]) - jnp.sum(links[1, 0, :nx]))    
            
        return jnp.average(jax.lax.map(Wilson_loop_single, self.locs))
    
    
    
    def prior_draw(self, key):
        """uniform angles [0, 2pi)"""
        return 2 * jnp.pi * jax.random.uniform(key, shape = (self.d, ), dtype = 'float64')



    def chiq(self, Q):
        """topological susceptibility"""
        return jnp.average(jnp.square(Q))
    
    


def thermodynamic_ground_truth(beta):
    """topological susceptibility (taking the first term in the expansion"""
    return (BesselI(1, beta)/BesselI(0, beta)) / (beta * 4 * np.pi**2)
