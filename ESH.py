import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


class Sampler:
    """the esh sampler"""

    def __init__(self, Target, eps):
        self.Target, self.eps = Target, eps


    def random_unit_vector(self, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(key, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def f(self, eps, g, u):
        """A convenient function for the leapfrog of the esh dynamics."""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        sh = jnp.sinh(eps * g_norm / self.Target.d)
        ch = jnp.cosh(eps * g_norm / self.Target.d)
        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh)


    def dynamics_bounces_step(self, x, u, g, r, key, time, time_max, add_time_distance):
        """One step of the dynamics (with bounces)"""

        # Hamiltonian step
        uhalf = self.f(self.eps * 0.5, g, u)
        xnew = x + self.eps * uhalf
        gg_new = self.Target.grad_nlogp(xnew)
        unew = self.f(self.eps * 0.5, gg_new, uhalf)
        rnew = r - self.eps * 0.5 * (jnp.dot(u, g) + jnp.dot(unew, gg_new)) / self.Target.d

        w = jnp.exp(rnew) / self.Target.d

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += add_time_distance(self.eps, w)
        do_bounce = time > time_max
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = unew * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xnew, u_return, gg_new, rnew, key, time, w


    def dynamics_bounces_billiard_step(self, x, u, g, r, key, time, time_max, xmax):
        """One step of the dynamics (with bounces)"""

        # Hamiltonian step
        uhalf = self.f(self.eps * 0.5, g, u)
        xnew = x + self.eps * uhalf
        gg_new = self.Target.grad_nlogp(xnew)
        unew = self.f(self.eps * 0.5, gg_new, uhalf)
        rnew = r - self.eps * 0.5 * (jnp.dot(u, g) + jnp.dot(unew, gg_new)) / self.Target.d

        w = jnp.exp(rnew) / self.Target.d

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps
        do_bounce = time > time_max
        time = time * (1 - do_bounce)  # reset time if the bounce is done

        u_random, key = self.random_unit_vector(key)
        pointing_out = jnp.dot(xnew, u_random) > 0.0
        u_wall = u_random - pointing_out * 2 * jnp.dot(u_random, xnew) * xnew / xmax ** 2
        hit_wall = jnp.sqrt(jnp.sum(jnp.square(xnew))) > xmax
        u_return = unew * (1 - hit_wall - do_bounce) + u_wall * hit_wall + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xnew, u_return, gg_new, rnew, key, time, w



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



    def sample(self, x0, num_steps, bounce_length, key, ess=False, update_track=None, initial_track=None, prerun=0, energy_track = False):


        add_distance = lambda eps, w: eps  # a function for updating the rescaled time (i.e. distance)
        add_time = lambda eps, w: eps * w  # a function for updating the Hamiltonian time


        def step(state_track, add_time_distance, time_max):
            """Tracks full x as a function of number of iterations"""

            x, u, g, r, key, time, w = self.dynamics_bounces_step(*state_track, time_max, add_time_distance)

            return (x, u, g, r, key, time), (self.Target.transform(x), w)


        def track_step(state_track, add_time_distance, time_max, update_track):
            """Does not track the full d-dimensional x, but some function (can be a vector function) of x. This can be expected values of some quantities of interest, 1d marginal histogram, etc."""

            x, u, g, r, key, time, w = self.dynamics_bounces_step(*state_track[0], time_max, add_time_distance)

            return ((x, u, g, r, key, time), update_track(state_track[1], self.Target.transform(x), w)), True


        def bias_step(state_track, add_time_distance, time_max):
            """Only tracks bias as a function of number of iterations."""

            x, u, g, r, key, time, w = self.dynamics_bounces_step(*state_track[0], time_max, add_time_distance)
            W, F2 = state_track[1]

            F2 = (F2 * W + (w * jnp.square(self.Target.transform(x)))) / (W + w)  # Update <f(x)> with a Kalman filter

            bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))

            return ((x, u, g, r, key, time), (W+w, F2)), bias


        def energy_step(state_track, add_time_distance, time_max):
            """Only outputs the average energy and it's fluctuations"""

            x, u, g, r, key, time, w = self.dynamics_step(*state_track[0], time_max, add_time_distance)
            W, E1, E2 = state_track[1]

            E = self.energy(x, w)
            E1 = (W * E1 + w * E) / (W + w)  # Update <x> with a Kalman filter
            E2 = (W * E2 + w * jnp.square(E)) / (W + w)  # Update <x^2> with a Kalman filter

            return ((x, u, g, r, key, time), (W + w, E1, E2)), True

        # initial conditions
        x = x0
        g = self.Target.grad_nlogp(x0)
        r = 0.0
        w = jnp.exp(r) / self.Target.d
        # u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p
        #u, key = self.random_unit_vector(key)
        u = jnp.array([1.0, 0.0])
        if ess:  # only track the bias

            if prerun != 0:

                final_state_prerun, bias_trace_prerun = jax.lax.scan(
                    lambda x, _: bias_step(x, add_distance, bounce_length),
                    init=((x, u, g, r, key, 0.0), (w, jnp.square(x))), xs=None, length=prerun)

                w_typical_set = final_state_prerun[1][0] / prerun
                bounce_time = bounce_length * w_typical_set

                final_state, bias_trace = jax.lax.scan(lambda x, _: bias_step(x, add_time, bounce_time),
                                                       init=final_state_prerun, xs=None, length=num_steps)

                bias = jnp.concatenate((bias_trace_prerun, bias_trace))

            else:
                _, bias = jax.lax.scan(lambda x, _: bias_step(x, add_distance, bounce_length),
                                             init=((x, u, g, r, key, 0.0), (w, jnp.square(x))), xs=None,
                                             length=num_steps)

            return ess_cutoff_crossing(bias)

        elif energy_track:  # only track the expected energy and its variance

            energy_tracer = jax.lax.scan(lambda x, _: energy_step(x, add_distance, bounce_length),
                                   init=((x, u, g, r, key, 0.0), (0.0, 0.0, 0.0)), xs=None, length=num_steps)[0][1]

            return [energy_tracer[1], jnp.sqrt(energy_tracer[2] - jnp.square(energy_tracer[1]))]


        elif initial_track != None:  # track some function of x

            track = update_track(initial_track, x, w)

            # init = track_step(init, add_distance, bounce_length, update_track)

            final_state, _ = jax.lax.scan(
                lambda state_track, _: track_step(state_track, add_distance, bounce_length, update_track),
                init=((x, u, g, r, key, 0.0), track), xs=None, length=num_steps)

            return final_state[1]  # return the final values of the quantities that we wanted to track

        else:  # track the full x

            if prerun != 0:

                final_state_prerun, track1 = jax.lax.scan(lambda state, _: step(state, add_distance, bounce_length),
                                                      init=(x, u, g, r, key, 0.0), xs=None, length=prerun)

                w_typical_set = final_state_prerun[1][0] / prerun
                bounce_time = bounce_length * w_typical_set

                final_state, track2 = jax.lax.scan(lambda x, _: step(x, add_time, bounce_time), init=final_state_prerun,
                                               xs=None, length=num_steps)

                X = np.concatenate((track1[0], track2[0]))
                W = np.concatenate((track1[1], track2[1]))


            else:
                final_state, track = jax.lax.scan(lambda state, _: step(state, add_distance, bounce_length),
                                              init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)

                X, W = track[0], track[1]

            return X, W


    def sample_multiple_chains(self, num_chains, num_steps, bounce_length, key, ess=False, update_track=None, initial_track=None, prerun=0, energy_track = False):

        def f(key, useless):
            key, key_prior, key_bounces = jax.random.split(key[0], 3)
            x0 = self.Target.prior_draw(key_prior)
            return (key,), self.sample(x0, num_steps, bounce_length, key_bounces, ess, update_track, initial_track, prerun, energy_track)

        return jax.lax.scan(f, init= (key, ), xs = None, length = num_chains)[1]


    def energy(self, X, W):
        return 0.5* self.Target.d * jnp.log(self.Target.d * jnp.square(W)) + self.Target.nlogp(X)



def update_moments(track, x, w):
    W = track[0]
    F1 = (W * track[1] + w * x) / (W + w)  # Update <x> with a Kalman filter
    F2 = (W * track[2] + w * jnp.square(x)) / (W + w)  # Update <x^2> with a Kalman filter

    return (W + w, F1, F2)


def get_initial_moments(d):
    return (0.0, jnp.zeros(d), jnp.zeros(d))



def update_bins(track, x, w, f, bins):
    fx = f(x)
    mask = (fx > bins[:, 0]) & (fx < bins[:, 1])

    return (track[0] + mask * w, track[1] + w)


def get_uniform_bins(a, b, num):
    d = (b - a) / num
    return jnp.array([[a + d * n, a + d * (n + 1)] for n in range(num)]), (jnp.zeros(num), 0.0)



def ess_cutoff_crossing(bias):

    def find_crossing(carry, b):
        above_threshold = b > 0.1
        never_been_below = carry[1] * above_threshold
        return (carry[0] + never_been_below, never_been_below), above_threshold

    crossing_index = jax.lax.scan(find_crossing, init= (0, 1), xs = bias, length=len(bias))[0][0]

    return 200.0 / np.sum(crossing_index)#, crossing_index != len(bias)

#
#
#
#         if langevin_eta != 0:
#             u = self.langevin_update(u, langevin_eta)
#
#
# def random_unit_vector(self):
#     u = np.random.normal(size=self.Target.d)
#     u /= np.sqrt(np.sum(np.square(u)))
#     return u
#
# def langevin_update(self, u, eta):
#     unew = u + np.random.normal(size=len(u)) * eta
#     return unew / np.sqrt(np.sum(np.square(unew)))




#
# class ModeMixing():
#     """how long does it take to switch between modes (average number of steps per switch after 10 switches)"""
#
#     def __init__(self, x):
#
#         self.L = []
#         self.current_sign = np.sign(x[0])
#         self.island_size = 1
#
#
#     def update(self, x, w):
#
#         sign = np.sign(x[0])
#         if sign != self.current_sign:
#             self.L.append(self.island_size)
#             self.island_size = 1
#             self.current_sign = sign
#
#         else:
#             self.island_size += 1
#
#         return len(self.L) > 9 #it finishes when 10 switches between the modes have been made.
#
#
#     def results(self):
#         return np.average(self.L)
#
#