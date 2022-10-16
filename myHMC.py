import jax.numpy as jnp
import jax



class Sampler:
    """Basic HMC sampler (with Yoshida steps)"""

    def __init__(self, Target, eps):
        self.Target, self.eps = Target, eps

        cbrt_two = jnp.cbrt(2.0)
        w0, w1 = -cbrt_two / (2.0 - cbrt_two), 1.0 / (2.0 - cbrt_two)
        self.cd = jnp.array([[0.5 * w1, w1], [0.5 * (w0 + w1), w0], [0.5 * (w0 + w1), w1]]) #constants for the Yoshida steps


    def resample(self, key):
        """Draws from a Gaussian"""
        key, subkey = jax.random.split(key)
        p = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        return p, key


    def Yoshida_step(self, x0, p0):


        def substep(carry, cd): #Yoshida does 4 of those
            c, d = cd
            x, p = carry
            x += c * p * self.eps
            p -= d * self.Target.grad_nlogp(x) * self.eps
            return (x, p), None

        x, p = jax.lax.scan(substep, init = (x0, p0), xs= self.cd)[0]
        x += self.cd[0, 0] * p * self.eps
        return x, p



    def dynamics(self, state):
        """One step of the dynamics"""
        x, p, key, time = state

        # Hamiltonian step
        xnew, pnew = self.Yoshida_step(x, p)


        # resampling
        p_resampled, key = self.resample(key)
        time += self.eps
        do_resampling = time > self.time_max
        time = time * (1 - do_resampling) # reset time if the bounce is done

        p_return= pnew * (1 - do_resampling) + p_resampled * do_resampling

        return xnew, p_return, key, time



    def sample(self, x0, num_steps, bounce_time, key):

        self.time_max = bounce_time

        def step(state, useless):

            x, p, key, time = self.dynamics(state)

            return (x, p, key, time), x


        # initial conditions
        p0, key = self.resample(key)

        _, X = jax.lax.scan(step, init=(x0, p0,  key, 0.0), xs=None, length=num_steps)

        return X