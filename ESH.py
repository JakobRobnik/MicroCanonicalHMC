import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)


class Sampler:
    """the ESH sampler (with bounces or the generalzied MCHMC)"""

    def __init__(self, Target, eps):
        self.Target, self.eps = Target, eps


    def random_unit_vector(self, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def generalized_step(self, u, key, nu):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(key)
        z = nu * jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')

        return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key


    def momentum_step(self, eps, g, u, r):
        """The momentum updating map for the leapfrog of the esh dynamics."""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        sh = jnp.sinh(eps * g_norm / self.Target.d)
        ch = jnp.cosh(eps * g_norm / self.Target.d)
        th = jnp.tanh(eps * g_norm / self.Target.d)
        delta_r = jnp.log(ch) + jnp.log1p(ue * th)
        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), r + delta_r


    def dynamics_step(self, state):
        """One step of the dynamics (with bounces)"""
        x, u, g, r = state

        # Hamiltonian step
        uhalf, rhalf = self.momentum_step(self.eps * 0.5, g, u, r)
        xnew = x + self.eps * uhalf
        gg_new = self.Target.grad_nlogp(xnew)
        unew, rnew = self.momentum_step(self.eps * 0.5, gg_new, uhalf, rhalf)

        w = jnp.exp(rnew) / self.Target.d

        return xnew, unew, gg_new, rnew, w


    def dynamics_bounces_step(self, state, add_time_distance, time_max):
        """One step of the dynamics (with bounces)"""
        x, u, g, r, key, time = state

        # Hamiltonian step
        uhalf, rhalf = self.momentum_step(self.eps * 0.5, g, u, r)
        xnew = x + self.eps * uhalf
        gg_new = self.Target.grad_nlogp(xnew)
        unew, rnew = self.momentum_step(self.eps * 0.5, gg_new, uhalf, rhalf)

        w = jnp.exp(rnew) / self.Target.d

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += add_time_distance(self.eps, w)
        do_bounce = time > time_max
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = unew * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xnew, u_return, gg_new, rnew, key, time, w


    # def generalized_dynamics_step(self, state, nu):
    #     """One step of the generalized dynamics: Leapfrog integrator"""
    #
    #     x, u, g, r, key, time = state
    #     # Hamiltonian step
    #     uhalf, rhalf = self.momentum_step(self.eps * 0.5, g, u, r)
    #     xnew = x + self.eps * uhalf
    #     gg_new = self.Target.grad_nlogp(xnew)
    #     unew, rnew = self.momentum_step(self.eps * 0.5, gg_new, uhalf, rhalf)
    #
    #     w = jnp.exp(rnew) / self.Target.d
    #
    #     # bounce
    #     unew, key = self.generalized_step(unew, key, nu)
    #
    #     return xnew, unew, gg_new, rnew, key, 0.0, w


    def generalized_dynamics_step(self, state, nu):
        """One step of the generalized dynamics: minimum norm integrator (see Equation 20 in https://arxiv.org/pdf/hep-lat/0505020.pdf)"""

        x, u, g, r, key, time = state

        # Hamiltonian step
        lambda_c = 0.1931833275037836
        uu, rr = self.momentum_step(self.eps * lambda_c, g, u, r)
        xx = x + self.eps * 0.5 * uu
        gg = self.Target.grad_nlogp(xx)
        uu, rr = self.momentum_step(self.eps * (1-2*lambda_c), gg, uu, rr)
        xx = xx + self.eps * 0.5 * uu
        gg = self.Target.grad_nlogp(xx)
        uu, rr = self.momentum_step(self.eps * lambda_c, gg, uu, rr)

        w = jnp.exp(rr) / self.Target.d

        # bounce
        uu, key = self.generalized_step(uu, key, nu)

        return xx, uu, gg, rr, key, 0.0, w



    def dynamics_bounces_billiard_step(self, state, time_max, xmax):
        """One step of the dynamics with bounces and bounces from the prior walls"""

        x, u, g, r, key, time = state
        # Hamiltonian step
        uhalf, rhalf = self.momentum_step(self.eps * 0.5, g, u, r)
        xnew = x + self.eps * uhalf
        gg_new = self.Target.grad_nlogp(xnew)
        unew, rnew = self.momentum_step(self.eps * 0.5, gg_new, uhalf, rhalf)

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


    def tune_epsilon(self, bounce_length):

        dynamics = lambda state: self.dynamics_bounces_step(state, lambda eps, w: eps, bounce_length) #bounces equally spaced in distance

        def step(state, useless):
            """Tracks full x as a function of number of iterations"""

            x, u, g, r, key, time, w = dynamics(state)

            return (x, u, g, r, key, time), self.energy(x, w)

        key, key_prior = jax.random.split(jax.random.PRNGKey(0))
        x0 = self.Target.prior_draw(key_prior)
        g0 = self.Target.grad_nlogp(x0)
        r0 = 0.0
        u0, key = self.random_unit_vector(key)
        trend = []
        eps_arr = [0.5,  5.0]#np.linspace(0.1, 2, 5)
        for i in range(len(eps_arr)):
            self.eps = eps_arr[i]
            print(self.eps)
            E = jax.lax.scan(step, init=(x0, u0, g0, r0, key, 0.0), xs=None, length=1000)[1]
            print(E[-1])
            plt.plot(E, '.')
            #scale = jnp.std(energy[:10])
            #val0 = jnp.average(energy[:10])
            #val1 = jnp.average(energy[-10:])
            #trend.append((val1 - val0) / scale)

        #plt.plot(eps_arr, trend)
        plt.show()




    def sample(self, x0, num_steps, bounce_length, key, generalized = False, ess=False, monitor_energy= False):


        def step(state, useless):
            """Tracks full x as a function of number of iterations"""

            x, u, g, r, key, time, w = dynamics(state)

            return (x, u, g, r, key, time), (self.Target.transform(x), w)


        def step_with_energy(state, useless):
            """Tracks full x as a function of number of iterations"""

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
        r = 0.0
        w = jnp.exp(r) / self.Target.d
        # u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p
        u, key = self.random_unit_vector(key)


        dynamics = lambda state: self.dynamics_bounces_step(state, lambda eps, w: eps, bounce_length) #bounces equally spaced in distance

        if generalized: #do a continous momentum decoherence (generalized MCHMC)
            nu = jnp.sqrt((jnp.exp(2 * self.eps / bounce_length) - 1.0) / self.Target.d)
            dynamics = lambda state: self.generalized_dynamics_step(state, nu)


        if ess:  # only track the bias

            _, bias = jax.lax.scan(bias_step, init=((x, u, g, r, key, 0.0), (w, jnp.square(x))), xs=None, length=num_steps)

            no_nans = 1-jnp.any(jnp.isnan(bias))
            cutoff_reached = bias[-1] < 0.1

            # plt.plot(bias, '.')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

            return 0.5* ess_cutoff_crossing(bias) * no_nans * cutoff_reached #return 0 if there are nans, or if the bias cutoff was not reached


        else:  # track the full x

            if monitor_energy:
                X, W, E = jax.lax.scan(step_with_energy, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)[1]
                return X, W, E

            else:
                final_state, track = jax.lax.scan(step, init=(x, u, g, r, key, 0.0), xs=None, length=num_steps)
                X, W = track[0], track[1]
                return X, W



    def parallel_bias(self, num_samples, num_chains, bounce_length, key):

        dynamics = lambda state: self.dynamics_bounces_step(state, lambda eps, w: eps, bounce_length) #bounces equally spaced in distance


        def bias_step(state_track, useless):
            """Only tracks bias as a function of number of iterations."""

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
            u = -g / jnp.sqrt(jnp.sum(jnp.square(g)))

            #a short burn in
            # state = jax.lax.scan(step, init=(x0, u, g, 0.0, key_bounces2, 0.0), xs=None, length= burn_in_steps)[0]
            # x0, g = state[0], state[2]
            #
            r = 0.5 * (1 + np.log(self.Target.d) - 2 * self.Target.nlogp(x0) / self.Target.d)  # initialize such that all the chains have the same energy
            w = jnp.exp(r) / self.Target.d
            u = self.random_unit_vector(key_u)[0]

            #run the chain
            w, f2 = jax.lax.scan(bias_step, init=((x0, u, g, r, key_bounces1, 0.0), (w, jnp.square(x0))), xs=None, length=num_samples)[1]

            #update the expected moments (as a function of number of steps)
            # F2_ij = E[x_j^2] after i steps using the chains that we already computed
            # W_i = total sum of weights after i steps using all the chains that we already computed
            W, F2 = track
            F2 = (F2.T * W /(W + w)).T + (f2.T * w / (W + w)).T #update F2 with the new chain
            W += w
            return (W, F2), None



        W, F2 = jax.lax.scan(single_chain, init= (jnp.zeros(num_samples), jnp.zeros((num_samples, self.Target.d))), xs = jax.random.split(key, num_chains))[0]

        bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance), axis = 1))
        #bias = jnp.average(F2 / self.Target.variance, axis = 1)
        no_nans = 1-jnp.any(jnp.isnan(bias))
        cutoff_reached = bias[-1] < 0.1

        #plt.plot(W / np.arange(len(W)), '.')
        print(np.array(bias)[[0, 100, 1000, 10000-1]])
        plt.plot(bias, '.')
        plt.plot([0, len(bias)], np.ones(2)*0.1, ':', color = 'black')
        plt.xlabel('# steps / # chains')
        plt.ylabel('bias')
        #plt.ylabel(r'$x^T H x$ / d')
        plt.xscale('log')
        plt.yscale('log')
        #plt.savefig('optimization_eps.png')
        plt.show()

        return ess_cutoff_crossing(bias) * no_nans * cutoff_reached / num_chains #return 0 if there are nans, or if the bias cutoff was not reached




    def sample_multiple_chains(self, num_chains, num_steps, bounce_length, key, generalized= False, ess=False):
        """Run multiple chains. The initial conditions are drawn with self.Target.prior_draw(key)"""

        def f(key, useless):
            key, key_prior, key_bounces = jax.random.split(key[0], 3)
            x0 = self.Target.prior_draw(key_prior)
            return (key,), self.sample(x0, num_steps, bounce_length, key_bounces, generalized, ess)

        return jax.lax.scan(f, init= (key, ), xs = None, length = num_chains)[1]



    def energy(self, X, W):
        return 0.5* self.Target.d * jnp.log(self.Target.d * jnp.square(W)) + self.Target.nlogp(X)



#some functions for tracking quantities (kind of obsolete)

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
#         if generalized_eta != 0:
#             u = self.generalized_update(u, generalized_eta)
#
#
# def random_unit_vector(self):
#     u = np.random.normal(size=self.Target.d)
#     u /= np.sqrt(np.sum(np.square(u)))
#     return u
#
# def generalized_update(self, u, eta):
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