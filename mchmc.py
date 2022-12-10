import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


class Sampler:
    """the mchmc (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L = None, eps = None, integrator = 'MN', generalized= True):
        """Args:
                Target: the target distribution class
                integrator: 'LF' (leapfrog) or 'MN' (minimal norm)
                generalized: True (generalized momentum decoherence) or False (bounces).
        """

        self.Target = Target

        ### integrator ###
        self.hamiltonian_dynamics = self.leapfrog if (integrator=='LF') else self.minimal_norm
        self.grad_evals_per_step = 1.0 if integrator == 'LF' else 2.0
        if integrator != 'LF' and integrator != 'MN':
            print('integrator = ' + integrator + 'is not a valid option.')

        ### decoherence mechanism ###
        self.dynamics = self.dynamics_generalized if generalized else self.dynamics_bounces

        if (not (L is None)) and (not (eps is None)):
            self.set_hyperparameters(L, eps)



    def set_hyperparameters(self, L, eps):
        self.L = L
        self.eps= eps
        self.nu = jnp.sqrt((jnp.exp(2 * self.eps / L) - 1.0) / self.Target.d)


    def energy(self, X, W):
        return 0.5* self.Target.d * jnp.log(self.Target.d * jnp.square(W)) + self.Target.nlogp(X)


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


    def update_momentum(self, eps, g, u, r):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        sh = jnp.sinh(eps * g_norm / self.Target.d)
        ch = jnp.cosh(eps * g_norm / self.Target.d)
        th = jnp.tanh(eps * g_norm / self.Target.d)
        delta_r = jnp.log(ch) + jnp.log1p(ue * th)
        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), r + delta_r


    def leapfrog(self, x, g, u, r):
        """leapfrog"""
        #half step in momentum
        uu, rr = self.update_momentum(self.eps * 0.5, g, u, r)

        #full step in x
        xx = x + self.eps * uu
        gg = self.Target.grad_nlogp(xx)

        #half step in momentum
        uu, rr = self.update_momentum(self.eps * 0.5, gg, uu, rr)
        return xx, gg, uu, rr

    #
    # def leapfrog(self, x, g, u, r):
    #     """adjoint leapfrog"""
    #
    #     #half step in x
    #     xx = x + 0.5 * self.eps * u
    #     gg = self.Target.grad_nlogp(xx)
    #
    #     #full step in momentum
    #     uu, rr = self.update_momentum(self.eps, gg, u, r)
    #
    #     #half step in x
    #     xx = x + 0.5 * self.eps * uu
    #
    #     return xx, gg, uu, rr


    def minimal_norm(self, x, g, u, r):
        """Adjoint integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        xx = x + lambda_c * self.eps * u
        gg = self.Target.grad_nlogp(xx)

        uu, rr = self.update_momentum(0.5*self.eps, gg, u, r)

        xx = xx + (1 - 2*lambda_c) * self.eps* uu
        gg = self.Target.grad_nlogp(xx)

        uu, rr = self.update_momentum(0.5 * self.eps, gg, uu, rr)

        xx = xx + lambda_c * self.eps* uu

        return xx, gg, uu, rr

    # def minimal_norm(self, x, g, u, r):
    #     """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    #     uu, rr = self.update_momentum(self.eps * lambda_c, g, u, r)
    #     xx = x + self.eps * 0.5 * uu
    #     gg = self.Target.grad_nlogp(xx)
    #     uu, rr = self.update_momentum(self.eps * (1 - 2 * lambda_c), gg, uu, rr)
    #     xx = xx + self.eps * 0.5 * uu
    #     gg = self.Target.grad_nlogp(xx)
    #     uu, rr = self.update_momentum(self.eps * lambda_c, gg, uu, rr)
    #
    #     return xx, gg, uu, rr



    def dynamics_bounces(self, state):
        """One step of the dynamics (with bounces)"""
        x, u, g, r, key, time = state

        # Hamiltonian step
        xx, gg, uu, rr = self.hamiltonian_dynamics(x, g, u, r)

        w = jnp.exp(rr) / self.Target.d

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps
        do_bounce = time > self.L
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, gg, rr, key, time, w


    def dynamics_generalized(self, state):
        """One step of the generalized dynamics."""

        x, u, g, r, key, time = state

        # Hamiltonian step
        xx, gg, uu, rr = self.hamiltonian_dynamics(x, g, u, r)
        w = jnp.exp(rr) / self.Target.d

        # bounce
        uu, key = self.partially_refresh_momentum(uu, key)

        return xx, uu, gg, rr, key, 0.0, w


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

        g = self.Target.grad_nlogp(x)
        r = 0.5 + np.log(self.Target.d) - self.Target.nlogp(x) / self.Target.d # initialize r such that all the chains have the same energy = d (1 + log d) / 2. For a standard Gaussian the weights are then around one on the typical set.
        w = jnp.exp(r) / self.Target.d

        u, key = self.random_unit_vector(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, g, r, key, w



    def sample(self, num_steps, x_initial = 'prior', random_key= None, ess=False, monitor_energy= False, prerun= False):


        def step(state, useless):
            """Tracks transform(x) as a function of number of iterations"""

            x, u, g, r, key, time, w = self.dynamics(state)

            return (x, u, g, r, key, time), (self.Target.transform(x), - self.Target.nlogp(x) / self.Target.d)


        def step_with_energy(state, useless):
            """Tracks transform(x) and the energy as a function of number of iterations"""

            x, u, g, r, key, time, w = self.dynamics(state)
            LL = self.Target.nlogp(x)
            energy = LL + 0.5 * self.Target.d * jnp.log(self.Target.d * jnp.square(w))
            return (x, u, g, r, key, time), (self.Target.transform(x),  - LL / self.Target.d, energy)


        def prerun_step(state_track, useless):
            """Only tracks b as a function of number of iterations."""

            x, u, g, r, key, time, w_integrated = self.dynamics(state_track[0])

            W, entropy, F1, F2, E1, E2 = state_track[1]

            #energy and the weight
            l = self.Target.nlogp(x) / self.Target.d
            energy = self.Target.d * (0.5 * jnp.log(self.Target.d * jnp.square(w_integrated)) + l)
            w = jnp.exp(- (l - entropy))

            #expectation values (with the Kalman filter)
            F1 = (F1 * W + (w * x)) / (W + w)  # <x>
            F2 = (F2 * W + (w * jnp.square(x))) / (W + w) # <x^2>
            E1 = (E1 * W + (w * energy)) / (W + w) # <E>
            E2 = (E2 * W + (w * jnp.square(energy))) / (W + w) # <E^2>

            #update the total weights for the next step
            entropy_new = (W * entropy + w * l) / (W + w) # <L(x)> / d
            W = (W + w) * jnp.exp(entropy_new - entropy)

            return ((x, u, g, r, key, time), (W, entropy_new, F1, F2, E1, E2)), None


        def b_step(state_track, useless):
            """Only tracks b as a function of number of iterations."""

            x, u, g, r, key, time, w = self.dynamics(state_track[0])
            W, F2, entropy = state_track[1]
            l = self.Target.nlogp(x) / self.Target.d
            w = jnp.exp(- (l - entropy))

            F2 = (W * F2 + (w * jnp.square(self.Target.transform(x)))) / (W + w)  # Update <f(x)> with a Kalman filter
            entropy_new = (W * entropy + w * l) / (W + w)  # Update entropy = <L(x)/d> with a Kalman filter
            W = (W + w) * jnp.exp(entropy_new - entropy)
            bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))
            #bias = jnp.average((F2 - self.Target.variance) / self.Target.variance)

            return ((x, u, g, r, key, time), (W, F2, entropy_new)), bias


        x, u, g, r, key, w = self.get_initial_conditions(x_initial, random_key)


        ### do sampling ###

        if ess:  # only track the bias

            _, b = jax.lax.scan(b_step, init=((x, u, g, r, key, 0.0), (1.0, jnp.square(x), self.Target.nlogp(x) / self.Target.d)), xs=None, length=num_steps)


            no_nans = 1-jnp.any(jnp.isnan(b))
            cutoff_reached = b[-1] < 0.1

            # plt.plot(bias, '.')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

            return ess_cutoff_crossing(b) * no_nans * cutoff_reached / self.grad_evals_per_step #return 0 if there are nans, or if the bias cutoff was not reached


        elif prerun:  # only track the x moments and the energy moments
            l = self.Target.nlogp(x) / self.Target.d
            en = self.Target.d * (0.5 * jnp.log(self.Target.d * jnp.square(w)) + l)

            W, entropy, F1, F2, E1, E2 = jax.lax.scan(prerun_step, init=((x, u, g, r, key, 0.0), (1.0, l, x, jnp.square(x), en, jnp.square(en))), xs=None, length=num_steps)[0][1]

            sigma = jnp.sqrt(jnp.average(F2 - jnp.square(F1)))
            varE = (E2 - jnp.square(E1)) / self.Target.d

            return sigma, varE


        else: # track the full transform(x)

            if monitor_energy:
                X, logw, E = jax.lax.scan(step_with_energy, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)[1]
                logw -= jnp.median(logw)
                return X, jnp.exp(logw), E

            else:
                final_state, track = jax.lax.scan(step, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)
                X, logw = track[0], track[1]
                logw -= jnp.median(logw)
                return X, jnp.exp(logw)



    def parallel_sample(self, num_chains, num_steps, random_key= None, ess= False, monitor_energy= False):
        """Run multiple chains. The initial conditions for each chain are drawn with self.Target.prior_draw"""

        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key


        def f(key, useless):
            key, key_prior, key_bounces = jax.random.split(key[0], 3)
            x0 = self.Target.prior_draw(key_prior)
            return (key,), self.sample(num_steps, x_initial= x0, random_key = key_bounces, ess= ess, monitor_energy= monitor_energy)


        return jax.lax.scan(f, init= (key, ), xs = None, length = num_chains)[1]


    def tune_hyperparameters(self, x_initial = 'prior', random_key= None):

        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key


        self.set_hyperparameters(np.sqrt(self.Target.d), 0.6)
        varE_wanted = 0.001

        key, subkey = jax.random.split(key)
        x = self.sample(1000, x_initial, subkey)[0][-1, :] #burn-in (exaggerated number of steps)

        def tuning_step(key):

            # get a small number of samples
            key_new, subkey = jax.random.split(key)
            sigma, varE = self.sample(1000, x, subkey, prerun= True)

            #update hyperparameters
            L_new = sigma * np.sqrt(self.Target.d)
            eps_new = self.eps * np.power(varE_wanted / varE, 0.25) #assume varE ~ eps^2
            self.set_hyperparameters(L_new, eps_new)
            print('varE / varE wanted: {}, eps: {}, sigma = L / sqrt(d): {}'.format(varE / varE_wanted, eps_new, sigma))

            return key_new

        for i in range(5):
            key = tuning_step(key)
        print('-------------')


    def weigths_from_energy_conservation(self, X, W):
        w = jnp.copy(W)
        LL = self.Target.nlogp(X)
        LL -= np.median(LL)
        return w.at[:].set(np.exp(- LL / self.Target.d))


    def full_b(self, x_arr, w_arr):

        def step(moments, index):
            F2, W = moments
            x, w = x_arr[index, :], w_arr[index]
            F2 = (F2 * W + (w * jnp.square(self.Target.transform(x)))) / (W + w)  # Update <f(x)> with a Kalman filter
            W += w
            b = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))

            return (F2, W), b

        return jax.lax.scan(step, (jnp.zeros(self.Target.d, ), 0.0), xs= jnp.arange(len(w_arr)))[1]


def ess_cutoff_crossing(bias):

    def find_crossing(carry, b):
        above_threshold = b > 0.1
        never_been_below = carry[1] * above_threshold
        return (carry[0] + never_been_below, never_been_below), above_threshold

    crossing_index = jax.lax.scan(find_crossing, init= (0, 1), xs = bias, length=len(bias))[0][0]

    return 200.0 / np.sum(crossing_index)


def point_reduction(num_points, reduction_factor):
    """reduces the number of points for plotting purposes"""

    indexes = np.concatenate((np.arange(1, 1 + num_points // reduction_factor, dtype=int),
                              np.arange(1 + num_points // reduction_factor, num_points, reduction_factor, dtype=int)))
    return indexes
