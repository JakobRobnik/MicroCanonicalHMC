import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt



class Sampler:
    """Unadjusted HMC sampler"""

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


    def leapfrog_step(self, x0, p0, g0):

        p = p0 - self.eps * 0.5 * g0
        x = x0 + self.eps * p
        g = self.Target.grad_nlogp(x)
        p -= self.eps * 0.5 * g

        return x, p, g



    def dynamics(self, state):
        """One step of the dynamics"""
        x, p, g, key, time = state

        # Hamiltonian step
        xnew, pnew, gnew = self.leapfrog_step(x, p, g)


        # resampling
        p_resampled, key = self.resample(key)
        time += self.eps
        do_resampling = time > self.time_max
        time = time * (1 - do_resampling) # reset time if the bounce is done

        p_return = pnew * (1 - do_resampling) + p_resampled * do_resampling

        return xnew, p_return, gnew, key, time



    def sample(self, x_initial, num_steps, L, random_key, generalized= False, integrator= 'LF', ess= False, monitor_energy= False):

        if monitor_energy or integrator != 'LF' or generalized:
            raise KeyError('A parameter was given to the myHMC sampler which is not currently supported. The code will be executed as if this parameter was not given.')


        #set the initial condition
        if isinstance(x_initial, str):
            if x_initial == 'prior': #draw the initial condition from the prior
                key, prior_key = jax.random.split(random_key)
                x0 = self.Target.prior_draw(prior_key)
            else: #if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else:
            key = random_key
            x0 = x_initial


        def step(state, useless):

            x, p, g, key, time = self.dynamics(state)

            return (x, p, g, key, time), (x, p)


        def bias_step(state_track, useless):
            """Only tracks bias as a function of number of iterations."""

            x, p, g, key, time = self.dynamics(state_track[0])
            W, F2 = state_track[1]

            F2 = (F2 * W + jnp.square(self.Target.transform(x))) / (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))
            #bias = jnp.average((F2 - self.Target.variance) / self.Target.variance)

            return ((x, p, g, key, time), (W, F2)), bias


        # initial conditions
        p0, key = self.resample(key)
        g0 = self.Target.grad_nlogp(x0)

        self.time_max = L
        grad_evals_per_step = 1.0


        if ess:  # only track the bias

            _, bias = jax.lax.scan(bias_step, init=((x0, p0, g0, key, 0.0), (1, jnp.square(x0))), xs=None, length=num_steps)

            #steps = point_reduction(len(bias), 100)
            #return bias[steps]
            no_nans = 1 - jnp.any(jnp.isnan(bias))
            cutoff_reached = bias[-1] < 0.1
            #
            # plt.plot(bias, '.')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

            return ess_cutoff_crossing(bias) * no_nans * cutoff_reached / grad_evals_per_step  # return 0 if there are nans, or if the bias cutoff was not reached


        else:  # track the full transform(x)

            X, P = jax.lax.scan(step, init=(x0, p0, g0, key, 0.0), xs=None, length=num_steps)[1]

            return X


    def parallel_sample(self, num_chains, num_steps, L, random_key, generalized = False, integrator= 'LF', ess=False, monitor_energy= False):
        """Run multiple chains. The initial conditions for each chain are drawn with self.Target.prior_draw(key)"""

        def f(key, useless):
            key, key_prior, key_bounces = jax.random.split(key[0], 3)
            x0 = self.Target.prior_draw(key_prior)
            return (key,), self.sample(x0, num_steps, L, key_bounces, generalized, integrator, ess, monitor_energy)

        return jax.lax.scan(f, init= (random_key, ), xs = None, length = num_chains)[1]




def ess_cutoff_crossing(bias):

    def find_crossing(carry, b):
        above_threshold = b > 0.1
        never_been_below = carry[1] * above_threshold
        return (carry[0] + never_been_below, never_been_below), above_threshold

    crossing_index = jax.lax.scan(find_crossing, init= (0, 1), xs = bias, length=len(bias))[0][0]

    return 200.0 / np.sum(crossing_index)
