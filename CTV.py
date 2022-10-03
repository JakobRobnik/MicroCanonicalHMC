import numpy as np
import bias
import jax.numpy as jnp
import jax


class Sampler:
    """the esh sampler"""

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
        """returns g and it's gradient"""
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
        x += self.cd[0, 1] * p * self.eps
        return x, p


    def dynamics(self, state):
        """One step of the dynamics (with bounces)"""
        x, p, key, time = state

        # Hamiltonian step
        xnew, pnew = self.Yoshida_step(x, p)

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps
        do_bounce = time > self.time_max
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        pneww = pnew * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xnew, pneww, key, time



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

        return ess_cutoff_crossing(bias)



    def sample_multiple_chains(self, num_chains, num_steps, bounce_length, key):

        def f(key, useless):
            key, key_prior, key_bounces = jax.random.split(key[0], 3)
            x0 = self.Target.prior_draw(key_prior)
            return (key,), self.sample(x0, num_steps, bounce_length, key_bounces)

        return jax.lax.scan(f, init= (key, ), xs = None, length = num_chains)[1]



def ess_cutoff_crossing(bias):

    def find_crossing(carry, b):
        above_threshold = b > 0.1
        never_been_below = carry[1] * above_threshold
        return (carry[0] + never_been_below, never_been_below), above_threshold

    crossing_index = jax.lax.scan(find_crossing, init= (0, 1), xs = bias, length=len(bias))[0][0]

    return 200.0 / np.sum(crossing_index)#, crossing_index != len(bias)

#