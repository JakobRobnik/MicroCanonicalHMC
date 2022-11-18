import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


class Sampler:
    """the ESH sampler (with bounces or the generalzied MCHMC)"""

    def __init__(self, Target, eps):
        self.Target, self.eps = Target, eps


    def energy(self, X, W):
        return 0.5* self.Target.d * jnp.log(self.Target.d * jnp.square(W)) + self.Target.nlogp(X)


    def random_unit_vector(self, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def partially_refresh_momentum(self, u, key, nu):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(key)
        z = nu * jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')

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
        uu, rr = self.update_momentum(self.eps * 0.5, g, u, r)
        xx = x + self.eps * uu
        gg = self.Target.grad_nlogp(xx)
        uu, rr = self.update_momentum(self.eps * 0.5, gg, uu, rr)
        return xx, gg, uu, rr


    def minimal_norm(self, x, g, u, r):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
        uu, rr = self.update_momentum(self.eps * lambda_c, g, u, r)
        xx = x + self.eps * 0.5 * uu
        gg = self.Target.grad_nlogp(xx)
        uu, rr = self.update_momentum(self.eps * (1 - 2 * lambda_c), gg, uu, rr)
        xx = xx + self.eps * 0.5 * uu
        gg = self.Target.grad_nlogp(xx)
        uu, rr = self.update_momentum(self.eps * lambda_c, gg, uu, rr)

        return xx, gg, uu, rr



    def dynamics_bounces_step(self, state, add_time_distance, time_max, integrator_step):
        """One step of the dynamics (with bounces)"""
        x, u, g, r, key, time = state

        # Hamiltonian step
        xx, gg, uu, rr = integrator_step(x, g, u, r)

        w = jnp.exp(rr) / self.Target.d

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += add_time_distance(self.eps, w)
        do_bounce = time > time_max
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, gg, rr, key, time, w


    def generalized_dynamics_step(self, state, nu, integrator_step):
        """One step of the generalized dynamics."""

        x, u, g, r, key, time = state

        # Hamiltonian step
        xx, gg, uu, rr = integrator_step(x, g, u, r)
        w = jnp.exp(rr) / self.Target.d

        # bounce
        uu, key = self.partially_refresh_momentum(uu, key, nu)

        return xx, uu, gg, rr, key, 0.0, w



    def dynamics_bounces_billiard_step(self, state, time_max, xmax, integrator_step):
        """One step of the dynamics with bounces and bounces from the prior walls"""

        x, u, g, r, key, time = state
        # Hamiltonian step
        xx, gg, uu, rr = integrator_step(x, g, u, r)
        w = jnp.exp(rr) / self.Target.d

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps
        do_bounce = time > time_max
        time = time * (1 - do_bounce)  # reset time if the bounce is done

        u_random, key = self.random_unit_vector(key)
        pointing_out = jnp.dot(xx, u_random) > 0.0
        u_wall = u_random - pointing_out * 2 * jnp.dot(u_random, xx) * xx / xmax ** 2
        hit_wall = jnp.sqrt(jnp.sum(jnp.square(xx))) > xmax
        u_return = uu * (1 - hit_wall - do_bounce) + u_wall * hit_wall + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, gg, rr, key, time, w



    def billiard_trajectory(self, num_steps, key, time_bounce, border, xmax, mu):


        def step(state_track, useless):
            """Tracks full x as a function of number of iterations"""

            x, u, g, r, key, time, w = self.dynamics_bounces_billiard_step(*state_track, time_bounce, xmax)
            region = 0 #outside of both modes
            region += (jnp.sqrt(jnp.sum(jnp.square(x))) < border) #in the first mode
            region += 2 * (jnp.sqrt(jnp.sum(jnp.square(x)) + mu**2 - 2 * mu * x[0]) < border) #in the second mode
            return (x, u, g, r, key, time), (x[0], x[1], w, region)

        key, key_prior, key_bounces = jax.random.split(key, 3)
        x0, key = self.random_unit_vector(key)
        x0 *= xmax
        u_random, key = self.random_unit_vector(key)
        pointing_out = jnp.dot(x0, u_random) > 0.0
        u0 = u_random - pointing_out * 2 * jnp.dot(u_random, x0) * x0 / xmax**2
        g = self.Target.grad_nlogp(x0)
        r = 0.0

        track = jax.lax.scan(step, init=(x0, u0, g, r, key, 0.0), xs=None, length=num_steps)[1]

        return track



    def sample(self, x_initial, num_steps, L, random_key, generalized = True, integrator= 'MN', ess=False, monitor_energy= False):

        if isinstance(x_initial, str):
            if x_initial == 'prior':
                key, prior_key = jax.random.split(random_key)
                x0 = self.Target.prior_draw(prior_key)
            else:
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')

        else:
            key = random_key
            x0 = x_initial

        def step(state, useless):
            """Tracks transform(x) as a function of number of iterations"""

            x, u, g, r, key, time, w = dynamics(state)

            return (x, u, g, r, key, time), (self.Target.transform(x), w)


        def step_with_energy(state, useless):
            """Tracks transform(x) and the energy as a function of number of iterations"""

            x, u, g, r, key, time, w = dynamics(state)

            return (x, u, g, r, key, time), (self.Target.transform(x), w, self.energy(x, w))


        def bias_step(state_track, useless):
            """Only tracks bias as a function of number of iterations."""

            x, u, g, r, key, time, w = dynamics(state_track[0])
            W, F2 = state_track[1]

            F2 = (F2 * W + (w * jnp.square(self.Target.transform(x)))) / (W + w)  # Update <f(x)> with a Kalman filter
            W += w
            bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))

            return ((x, u, g, r, key, time), (W, F2)), bias



        # initial conditions
        x = x0
        g = self.Target.grad_nlogp(x0)
        r = 0.5 + np.log(self.Target.d) - self.Target.nlogp(x0) / self.Target.d # initialize r such that all the chains have the same energy = d (1 + log d) / 2. For a standard Gaussian the weights are then around one on the typical set.
        w = jnp.exp(r) / self.Target.d
        # u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p
        u, key = self.random_unit_vector(key)

        #integrator
        integrator_step = self.leapfrog if (integrator=='LF') else self.minimal_norm
        grad_evals_per_step = 1.0 if integrator == 'LF' else 2.0
        if integrator != 'LF' and integrator != 'MN':
            print('integrator = ' + integrator + 'is not a valid option.')

        #bounce mechanism
        dynamics = lambda state: self.dynamics_bounces_step(state, lambda eps, w: eps, L, integrator_step) #bounces equally spaced in distance

        if generalized: #do a continous momentum decoherence (generalized MCHMC)
            nu = jnp.sqrt((jnp.exp(2 * self.eps / L) - 1.0) / self.Target.d)
            dynamics = lambda state: self.generalized_dynamics_step(state, nu, integrator_step)


        #do sampling

        if ess:  # only track the bias

            _, bias = jax.lax.scan(bias_step, init=((x, u, g, r, key, 0.0), (w, jnp.square(x))), xs=None, length=num_steps)

            return bias
            no_nans = 1-jnp.any(jnp.isnan(bias))
            cutoff_reached = bias[-1] < 0.1

            # plt.plot(bias, '.')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

            return ess_cutoff_crossing(bias) * no_nans * cutoff_reached / grad_evals_per_step #return 0 if there are nans, or if the bias cutoff was not reached


        else: # track the full transform(x)

            if monitor_energy:
                X, W, E = jax.lax.scan(step_with_energy, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)[1]
                return X, W, E

            else:
                final_state, track = jax.lax.scan(step, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)
                X, W = track[0], track[1]
                return X, W



    def parallel_bias(self, num_samples, num_chains, L, key, integrator= 'MN'):
        """does not support generalized MCHMC at the moment"""


        def bias_step(state_track, useless):
            """Only tracks bias as a function of a number of iterations."""

            x, u, g, r, key, time, w = dynamics(state_track[0])
            W, F2 = state_track[1]

            F2 = (F2 * W + (w * jnp.square(self.Target.transform(x)))) / (W + w)  # Update <f(x)> with a Kalman filter
            W += w

            return ((x, u, g, r, key, time), (W, F2)), (W, F2)


        def single_chain(track, key):

            #initial conditions
            key_bounces1, key_bounces2, key_prior, key_u = jax.random.split(key, 4)
            x0 = self.Target.prior_draw(key_prior)

            g = self.Target.grad_nlogp(x0)

            #a short burn in
            # state = jax.lax.scan(step, init=(x0, u, g, 0.0, key_bounces2, 0.0), xs=None, length= burn_in_steps)[0]
            # x0, g = state[0], state[2]
            #
            r = 0.5 + np.log(self.Target.d) - self.Target.nlogp(x0) / self.Target.d  # initialize such that all the chains have the same energy
            w = jnp.exp(r) / self.Target.d
            u = self.random_unit_vector(key_u)[0]

            #run the chain
            w, f2 = jax.lax.scan(bias_step, init=((x0, u, g, r, key_bounces1, 0.0), (w, jnp.square(x0))), xs=None, length=num_samples)[1]
            #w is the cumulative weight sum for that chain, f2 is the estimate for the second moment as a function of number of steps
            average_weight = (w[-1] / num_samples)
            w = w / average_weight #we want all chains to contribute equally

            #update the expected moments (as a function of number of steps)
            # F2[i,j] = E[x_j^2] after i steps using the chains that we already computed
            # W[i] = total sum of weights after i steps using all the chains that we already computed
            W, F2 = track
            F2 = (F2.T * W /(W + w)).T + (f2.T * w / (W + w)).T #update F2 with the new chain
            W += w
            return (W, F2), average_weight



        integrator_step = self.leapfrog if (integrator=='LF') else self.minimal_norm
        grad_evals_per_step = 1.0 if integrator == 'LF' else 2.0

        dynamics = lambda state: self.dynamics_bounces_step(state, lambda eps, w: eps, L, integrator_step) #bounces equally spaced in distance


        results, chain_weights = jax.lax.scan(single_chain, init= (jnp.zeros(num_samples), jnp.zeros((num_samples, self.Target.d))), xs = jax.random.split(key, num_chains))
        W, F2 = results


        bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance), axis = 1))
        no_nans = 1-jnp.any(jnp.isnan(bias))
        cutoff_reached = bias[-1] < 0.1

        # print(np.array(bias)[[0, 100, 1000, 10000-1]])
        # plt.plot(bias, '.')
        # plt.plot([0, len(bias)], np.ones(2)*0.1, ':', color = 'black')
        # plt.xlabel('# steps / # chains')
        # plt.ylabel('bias')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        return ess_cutoff_crossing(bias) * no_nans * cutoff_reached / (num_chains * grad_evals_per_step)#return 0 if there are nans, or if the bias cutoff was not reached




    def parallel_sample(self, num_chains, num_steps, L, key, generalized = True, integrator= 'MN', ess=False, monitor_energy= False):
        """Run multiple chains. The initial conditions for each chain are drawn with self.Target.prior_draw(key)"""

        def f(key, useless):
            key, key_prior, key_bounces = jax.random.split(key[0], 3)
            x0 = self.Target.prior_draw(key_prior)
            return (key,), self.sample(x0, num_steps, L, key_bounces, generalized, integrator, ess, monitor_energy)

        return jax.lax.scan(f, init= (key, ), xs = None, length = num_chains)[1]



def ess_cutoff_crossing(bias):

    def find_crossing(carry, b):
        above_threshold = b > 0.1
        never_been_below = carry[1] * above_threshold
        return (carry[0] + never_been_below, never_been_below), above_threshold

    crossing_index = jax.lax.scan(find_crossing, init= (0, 1), xs = bias, length=len(bias))[0][0]

    return 200.0 / np.sum(crossing_index)
