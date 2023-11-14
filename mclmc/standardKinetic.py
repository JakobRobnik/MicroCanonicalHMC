import numpy as np
import jax.numpy as jnp
import jax



class Sampler:
    """Sampler with the standard T+V Hamiltonian"""

    def __init__(self, Target, eps):
        self.Target, self.eps = Target, eps

        cbrt_two = jnp.cbrt(2.0)
        w0, w1 = -cbrt_two / (2.0 - cbrt_two), 1.0 / (2.0 - cbrt_two)
        self.cd = jnp.array([[0.5 * w1, w1], [0.5 * (w0 + w1), w0], [0.5 * (w0 + w1), w1]])


    def random_unit_vector(self, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def V(self, x):
        """potential"""
        return -jnp.exp(-2 * self.Target.nlogp(x) / (self.Target.d - 2))


    def grad_V(self, x):
        v = -jnp.exp(-2 * self.Target.nlogp(x) / (self.Target.d - 2))

        return (-2 * v / (self.Target.d - 2)) * self.Target.grad_nlogp(x)



    def Yoshida_step(self, x0, p0):


        def substep(carry, cd):
            c, d = cd
            x, p = carry
            x += c * p * self.eps
            p -= d * self.grad_V(x) * self.eps
            return (x, p), None

        x, p = jax.lax.scan(substep, init = (x0, p0), xs= self.cd)[0]
        x += self.cd[0, 0] * p * self.eps
        return x, p


    def dynamics_billiard(self, state):
        """One step of the dynamics (with bounces and billiard bounces from the walls)"""
        x, p, key, time = state

        # Hamiltonian step
        xnew, pnew = self.Yoshida_step(x, p)
        speed = jnp.sqrt(jnp.sum(jnp.square(pnew)))
        #speed = jnp.sqrt(-2 * self.V(xnew)) #set total energy = 0


        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps * speed
        do_bounce = time > self.time_max
        time = time * (1 - do_bounce) # reset time if the bounce is done

        #wall bounce
        xmax= 6.0
        u_random, key = self.random_unit_vector(key)
        pointing_out = jnp.dot(xnew, u_random) > 0.0
        u_wall = u_random - pointing_out * 2 * jnp.dot(u_random, xnew) * xnew / xmax ** 2
        hit_wall = jnp.sqrt(jnp.sum(jnp.square(xnew))) > xmax

        p_return= pnew * (1 - do_bounce - hit_wall + hit_wall*do_bounce) + u_wall * jnp.sqrt(-2 * self.V(x)) * hit_wall + u_bounce * speed * do_bounce * (1- hit_wall)  # randomly reorient the momentum if the bounce is done

        return xnew * (1 - hit_wall) + x * hit_wall, p_return, key, time


    def dynamics(self, state):
        """One step of the dynamics (with bounces)"""
        x, p, key, time = state

        # Hamiltonian step
        xnew, pnew = self.Yoshida_step(x, p)
        speed = jnp.sqrt(jnp.sum(jnp.square(pnew)))


        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps * speed
        do_bounce = time > self.time_max
        time = time * (1 - do_bounce) # reset time if the bounce is done

        p_return= pnew * (1 - do_bounce) + u_bounce * speed * do_bounce  # randomly reorient the momentum if the bounce is done

        return xnew, p_return, key, time



    def sample(self, x0, num_steps, bounce_length, key):

        self.time_max = bounce_length

        def bias_step(state_track, useless):
            """Only tracks bias as a function of number of iterations."""

            x, p, key, time = self.dynamics(state_track[0])
            N, F2 = state_track[1]

            F2 = (F2 * N + (jnp.square(self.Target.transform(x)))) / (N+1)  # Update <f(x)> with a Kalman filter

            bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))

            return ((x, p, key, time), (N+1, F2)), bias


        # initial conditions
        p0, key = self.random_unit_vector(key)
        p0 *= jnp.sqrt(-2 * self.V(x0)) #set total energy = 0


        _, bias = jax.lax.scan(bias_step, init=((x0, p0,  key, 0.0), (1, jnp.square(x0))), xs=None, length=num_steps)

        import matplotlib.pyplot as plt
        steps = point_reduction(len(bias), 100)
        plt.plot(steps, bias[steps], '.')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

        no_nans = 1 - jnp.any(jnp.isnan(bias))
        cutoff_reached = bias[-1] < 0.1

        return 0.25*ess_cutoff_crossing(bias) * no_nans * cutoff_reached  # return 0 if there are nans, or if the bias cutoff was not reached




    def sample_multiple_chains(self, num_chains, num_steps, bounce_length, key, generalized = False, ess= True, update_track= None, initial_track= None, prerun= 0, energy_track= False):

        def f(key, useless):
            key, key_prior, key_bounces = jax.random.split(key[0], 3)
            x0 = self.Target.prior_draw(key_prior)
            return (key,), self.sample(x0, num_steps, bounce_length, key_bounces)

        return jax.lax.scan(f, init= (key, ), xs = None, length = num_chains)[1]



def ess_cutoff_crossing(bias):
    cutoff = 0.5

    def find_crossing(carry, b):
        above_threshold = b > cutoff
        never_been_below = carry[1] * above_threshold
        return (carry[0] + never_been_below, never_been_below), above_threshold

    crossing_index = jax.lax.scan(find_crossing, init= (0, 1), xs = bias, length=len(bias))[0][0]

    return (2.0 / cutoff**2) / np.sum(crossing_index)#, crossing_index != len(bias)


def point_reduction(num_points, reduction_factor):
    """reduces the number of points for plotting purposes"""

    indexes = np.concatenate((np.arange(1, 1 + num_points // reduction_factor, dtype=int),
                              np.arange(1 + num_points // reduction_factor, num_points, reduction_factor, dtype=int)))
    return indexes
