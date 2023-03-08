import jax
import jax.numpy as jnp

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


class Sampler:
    """essentials of the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L = None, eps = None, integrator = 'MN'):
        """Args:
                Target: the target distribution class
                L: momentum decoherence scale
                eps: integration step-size
                integrator: 'LF' (leapfrog) or 'MN' (minimal norm). Typically MN performs better.
        """

        self.Target = Target

        ### integrator ###
        if integrator == "LF":  # leapfrog
            self.hamiltonian_dynamics = self.leapfrog
            self.grad_evals_per_step = 1.0
        elif integrator == 'MN':  # minimal norm integrator
            self.hamiltonian_dynamics = self.minimal_norm
            self.grad_evals_per_step = 2.0
        else:
            print('integrator = ' + integrator + 'is not a valid option.')


        if (not (L is None)) and (not (eps is None)):
            self.set_hyperparameters(L, eps)



    def set_hyperparameters(self, L, eps):
        self.L = L
        self.eps= eps
        self.nu = jnp.sqrt((jnp.exp(2 * self.eps / L) - 1.0) / self.Target.d)



    def random_unit_vector(self, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def partially_refresh_momentum(self, u, key):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(key)
        z = self.nu * jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')

        return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key


    def update_momentum(self, eps, g, u):
        """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""

        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        sh = jnp.sinh(eps * g_norm / (self.Target.d-1))
        ch = jnp.cosh(eps * g_norm / (self.Target.d-1))
        th = jnp.tanh(eps * g_norm / (self.Target.d-1))
        delta_r = jnp.log(ch) + jnp.log1p(ue * th)

        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), delta_r


    def leapfrog(self, x, u, g):
        """leapfrog"""

        #half step in momentum
        uu, r1 = self.update_momentum(self.eps * 0.5, g, u)

        #full step in x
        xx = x + self.eps * uu
        l, gg = self.Target.grad_nlogp(xx)

        #half step in momentum
        uu, r2 = self.update_momentum(self.eps * 0.5, gg, uu)

        kinetic = (r1+r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic



    def minimal_norm(self, x, u, g):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20.
            V T V T V scheme
        """

        #V (momentum update)
        uu, r1 = self.update_momentum(self.eps * lambda_c, g, u)

        #T (postion update)
        xx = x + 0.5 * self.eps * uu
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r2 = self.update_momentum(self.eps * (1 - 2 * lambda_c), gg * self.sigma, uu)

        #T (postion update)
        xx = xx + 0.5 * self.eps * uu
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r3 = self.update_momentum(self.eps * lambda_c, gg, uu)

        #kinetic energy change
        kinetic = (r1 + r2 + r3) * (self.Target.d-1)

        return xx, uu, ll, gg, kinetic


    def dynamics(self, state):
        """One step of the Langevin-like dynamics."""

        x, u, g, key = state

        # Hamiltonian step
        xx, uu, ll, gg, kinetic = self.hamiltonian_dynamics(x, u, g)

        # add noise to the momentum direction
        uu, key = self.partially_refresh_momentum(uu, key)

        return xx, uu, ll, gg, kinetic, key



    def get_initial_conditions(self, x_initial, random_key):

        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        ### initial conditions ###
        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                key, prior_key = jax.random.split(key)
                x = self.Target.prior_draw(prior_key)
            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError(
                    'x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else: #initial x is given
            x = x_initial

        l, g = self.Target.grad_nlogp(x)

        u, key = self.random_unit_vector(key) #random initial direction
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, l, g, key



    def sample(self, num_steps, x_initial = 'prior', random_key= None):
        """Args:
               num_steps: number of integration steps to take.
               x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
               random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).

            Returns:
                samples (shape = (num_steps, self.Target.d))

        """

        def step(state, useless):
            """Tracks transform(x) as a function of number of iterations"""

            x, u, l, g, E, key = state
            xx, uu, ll, gg, kinetic_change, key = self.dynamics(x, u, g, key)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key), (self.Target.transform(xx), ll, EE)

        x, u, l, g, key = self.get_initial_conditions(x_initial, random_key)

        ### do sampling ###
        X, L, E = jax.lax.scan(step, init=(x, u, l, g, 0.0, key), xs=None, length=num_steps)[1]

        #reutrn X, E, burn_in_ending(L)         ### automatically detects the end of burn-in ###
        return X, E



### these two are only needed for the end of burn-in detection:

def find_crossing(array, cutoff):
    """find the smallest M such that array[m] < cutoff for all m > M"""

    def step(carry, element):
        """carry = (, 1 if (array[i] > cutoff for all i < current index) else 0"""
        above_threshold = element > cutoff
        never_been_below = carry[1] * above_threshold  #1 if (array[i] > cutoff for all i < current index) else 0
        return (carry[0] + never_been_below, never_been_below), above_threshold

    state, track = jax.lax.scan(step, init=(0, 1), xs=array, length=len(array))

    return state[0]
    #return jnp.sum(track) #total number of indices for which array[m] < cutoff



def burn_in_ending(loss):
    loss_avg = jnp.median(loss[len(loss) // 2:])
    return 2 * find_crossing(loss - loss_avg, 0.0)  # we add a safety factor of 2
