import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt



class Sampler:
    """Unadjusted HMC sampler"""

    def __init__(self, Target, L, eps, integrator = 'LF'):
        """resamples once every (L / eps) steps"""
        self.Target, self.L, self.eps = Target, L, eps


        ### integrator ###
        if integrator == "LF":  # leapfrog (first updates the velocity)
            self.hamiltonian_dynamics = self.leapfrog
            self.grad_evals_per_step = 1.0

        elif integrator == 'Y':
            self.hamiltonian_dynamics = self.Yoshida_step
            self.grad_evals_per_step = 4.0

            cbrt_two = jnp.cbrt(2.0)
            w0, w1 = -cbrt_two / (2.0 - cbrt_two), 1.0 / (2.0 - cbrt_two)
            self.cd = jnp.array([[0.5 * w1, w1], [0.5 * (w0 + w1), w0], [0.5 * (w0 + w1), w1]])  # constants for the Yoshida steps

        elif integrator == 'RM':
            self.hamiltonian_dynamics = self.randomized_midpoint
            self.grad_evals_per_step = 1.0


        else:
            print('integrator = ' + integrator + 'is not a valid option.')


    def energy(self, x, p):
        return 0.5 * jnp.sum(jnp.square(p)) + self.Target.nlogp(x)


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


    def leapfrog(self, x0, p0, g0, key):

        p = p0 - self.eps * 0.5 * g0
        x = x0 + self.eps * p
        g = self.Target.grad_nlogp(x)
        p -= self.eps * 0.5 * g

        return x, p, g, key


    def randomized_midpoint(self, x0, p0, g0, key0):

        key, key_new = jax.random.split(key0)
        tau = jax.random.uniform(key) * self.eps
        g = self.Target.grad_nlogp(x0 + p0 * tau)
        x = x0 + self.eps * p0 - 0.5 * self.eps**2 * g
        p = p0 - self.eps * g

        return x, p, g, key_new



    def dynamics(self, state):
        """One step of the dynamics"""
        x, p, g, key, time = state

        # Hamiltonian step
        xnew, pnew, gnew, key = self.hamiltonian_dynamics(x, p, g, key)


        # resampling
        p_resampled, key = self.resample(key)
        time += self.eps
        do_resampling = time > self.L
        time = time * (1 - do_resampling) # reset time if the bounce is done

        p_return = pnew * (1 - do_resampling) + p_resampled * do_resampling

        return xnew, p_return, gnew, key, time


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

        p, key = self.resample(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, p, g, key



    def sample(self, num_steps, x_initial = 'prior', random_key= None, ess=False, monitor_energy= False, final_state = False):

        if final_state:
            raise KeyError('A parameter was given to the myHMC sampler which is not currently supported. The code will be executed as if this parameter was not given.')


        def step_energy(state, useless):

            x, p, g, key, time = self.dynamics(state)

            return (x, p, g, key, time), (x, self.energy(x, p))


        def step(state, useless):

            x, p, g, key, time = self.dynamics(state)

            return (x, p, g, key, time), x


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
        x0, p0, g0, key = self.get_initial_conditions(x_initial, random_key)


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

            return ess_cutoff_crossing(bias) * no_nans * cutoff_reached / self.grad_evals_per_step  # return 0 if there are nans, or if the bias cutoff was not reached


        else:  # track the full transform(x)


            if monitor_energy:
                return jax.lax.scan(step_energy, init=(x0, p0, g0, key, 0.0), xs=None, length=num_steps)[1]

            else:
                return jax.lax.scan(step, init=(x0, p0, g0, key, 0.0), xs=None, length=num_steps)[1]



    def parallel_sample(self, num_chains, num_steps, x_initial='prior', random_key=None, ess=False, monitor_energy=False, num_cores=1):
        """Run multiple chains. The initial conditions for each chain are drawn with self.Target.prior_draw"""

        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains * 2)
                x0 = jnp.array([self.Target.prior_draw(keys_all[num_chains + i]) for i in range(num_chains)])
                keys = keys_all[:num_chains]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else:  # initial x is given
            x0 = jnp.copy(x_initial)
            keys = jax.random.split(key, num_chains)

        f = lambda i: self.sample(num_steps, x_initial=x0[i], random_key=keys[i], ess=ess, monitor_energy=monitor_energy)

        if num_cores != 1:  # run the chains on parallel cores
            parallel_function = jax.pmap(jax.vmap(f))
            results = parallel_function(jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores))
            ### reshape results ###
            if type(results) is tuple:  # each chain returned a tuple
                results_reshaped = []
                for i in range(len(results)):
                    res = jnp.array(results[i])
                    results_reshaped.append(res.reshape([num_chains, ] + [res.shape[j] for j in range(2, len(res.shape))]))
                return results_reshaped

            else:
                return results.reshape([num_chains, ] + [results.shape[j] for j in range(2, len(results.shape))])


        else:  # run chains serially on a single core

            return jax.vmap(f)(jnp.arange(num_chains))





def ess_cutoff_crossing(bias):

    def find_crossing(carry, b):
        above_threshold = b > 0.1
        never_been_below = carry[1] * above_threshold
        return (carry[0] + never_been_below, never_been_below), above_threshold

    crossing_index = jax.lax.scan(find_crossing, init= (0, 1), xs = bias, length=len(bias))[0][0]

    return 200.0 / np.sum(crossing_index)
