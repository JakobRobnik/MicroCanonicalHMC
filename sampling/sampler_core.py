import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


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
        sh = jnp.sinh(eps * g_norm / self.Target.d)
        ch = jnp.cosh(eps * g_norm / self.Target.d)

        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh)


    def leapfrog(self, x, g, u):
        """leapfrog"""

        #half step in momentum
        uu = self.update_momentum(self.eps * 0.5, g, u)

        #full step in x
        xx = x + self.eps * uu
        gg = self.Target.grad_nlogp(xx) * self.Target.d / (self.Target.d - 1)

        #half step in momentum
        uu = self.update_momentum(self.eps * 0.5, gg, uu)

        return xx, gg, uu



    def minimal_norm(self, x, g, u):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V

        uu = self.update_momentum(self.eps * lambda_c, g, u)

        xx = x + self.eps * 0.5 * uu
        gg = self.Target.grad_nlogp(xx) * self.Target.d / (self.Target.d - 1)

        uu = self.update_momentum(self.eps * (1 - 2 * lambda_c), gg, uu)

        xx = xx + self.eps * 0.5 * uu
        gg = self.Target.grad_nlogp(xx) * self.Target.d / (self.Target.d - 1)

        uu = self.update_momentum(self.eps * lambda_c, gg, uu)

        return xx, gg, uu



    def dynamics(self, state):
        """One step of the Langevin-like dynamics."""

        x, u, g, key, time = state

        # Hamiltonian step
        xx, gg, uu = self.hamiltonian_dynamics(x, g, u)

        # add noise to the momentum direction
        uu, key = self.partially_refresh_momentum(uu, key)

        return xx, uu, gg, key, 0.0



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

        g = self.Target.grad_nlogp(x) * self.Target.d / (self.Target.d - 1)

        u, key = self.random_unit_vector(key) #random initial direction
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, g, key



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

            x, u, g, key, time = self.dynamics(state)

            return (x, u, g, key, time), self.Target.transform(x)


        x, u, g, key = self.get_initial_conditions(x_initial, random_key)


        ### do sampling ###

        return jax.lax.scan(step, init=(x, u, g, key, 0.0), xs=None, length=num_steps)[1]
