import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from .jump_identification import remove_jumps
from .correlation_length import ess_corr


jax.config.update('jax_enable_x64', True)


lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L = None, eps = None, integrator = 'MN', generalized= True):
        """Args:
                Target: the target distribution class
                L: momentum decoherence scale (can be tuned automatically)
                eps: integration step-size (can be tuned automatically)
                integrator: 'LF' (leapfrog) or 'MN' (minimal norm). Typically MN performs better.
                generalized: True (Langevin-like momentum decoherence) or False (bounces).
        """

        self.Target = Target

        ### integrator ###
        if integrator == "LF": #leapfrog (first updates the velocity)
            self.hamiltonian_dynamics = self.leapfrog
            self.grad_evals_per_step = 1.0
        elif integrator== 'MN': #minimal norm integrator (velocity)
            self.hamiltonian_dynamics = self.minimal_norm
            self.grad_evals_per_step = 2.0
        elif integrator == 'RM':
            self.hamiltonian_dynamics = self.randomized_midpoint
            self.grad_evals_per_step = 1.0
        else:
            print('integrator = ' + integrator + 'is not a valid option.')


        ### decoherence mechanism ###
        self.dynamics = self.dynamics_generalized if generalized else self.dynamics_bounces

        self.sigma = jnp.ones(self.Target.d)

        if (not (L is None)) and (not (eps is None)):
            self.set_hyperparameters(L, eps)

        else:
            self.set_hyperparameters(jnp.sqrt(Target.d), jnp.sqrt(Target.d) * 0.1)




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
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        sh = jnp.sinh(eps * g_norm / (self.Target.d-1))
        ch = jnp.cosh(eps * g_norm / (self.Target.d-1))
        th = jnp.tanh(eps * g_norm / (self.Target.d-1))
        delta_r = jnp.log(ch) + jnp.log1p(ue * th)

        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), delta_r


    def leapfrog(self, x, u, g, key):
        """leapfrog"""

        z = x / self.sigma # go to the latent space

        # half step in momentum
        uu, delta_r1 = self.update_momentum(self.eps * 0.5, g * self.sigma, u)

        # full step in x
        zz = z + self.eps * uu
        xx = self.sigma * zz # go back to the configuration space
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(self.eps * 0.5, gg * self.sigma, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, key


    def minimal_norm(self, x, u, g, key):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V
        z = x / self.sigma # go to the latent space

        #V (momentum update)
        uu, r1 = self.update_momentum(self.eps * lambda_c, g * self.sigma, u)

        #T (postion update)
        zz = z + 0.5 * self.eps * uu
        xx = self.sigma * zz # go back to the configuration space
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r2 = self.update_momentum(self.eps * (1 - 2 * lambda_c), gg * self.sigma, uu)

        #T (postion update)
        zz = zz + 0.5 * self.eps * uu
        xx = self.sigma * zz  # go back to the configuration space
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r3 = self.update_momentum(self.eps * lambda_c, gg, uu)

        #kinetic energy change
        kinetic_change = (r1 + r2 + r3) * (self.Target.d-1)

        return xx, uu, ll, gg, kinetic_change, key


    #
    # def randomized_midpoint(self, x, u, g, r, key):
    #
    #     key1, key2 = jax.random.split(key)
    #
    #     xx = x + jax.random.uniform(key2) * self.eps * u
    #
    #     gg = self.Target.grad_nlogp(xx)
    #
    #     uu, r1 = self.update_momentum(self.eps, gg, u)
    #
    #     xx = self.update_position_RM(xx, )
    #
    #
    #     return xx, uu, gg, r1 * (self.Target.d-1), key1



    def dynamics_bounces(self, x, u, g, key, time):
        """One step of the dynamics (with bounces)"""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, key)

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += self.eps
        do_bounce = time > self.L
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, ll, gg, kinetic_change, key, time


    def dynamics_generalized(self, x, u, g, key, time):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, key)

        # bounce
        uu, key = self.partially_refresh_momentum(uu, key)

        return xx, uu, ll, gg, kinetic_change, key, time + self.eps



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
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else: #initial x is given
            x = x_initial
        l, g = self.Target.grad_nlogp(x)

        u, key = self.random_unit_vector(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, l, g, key


    def burn_in(self, x0, u0, l0, g0, key0):
        """assuming the prior is wider than the posterior"""

        #adam = Adam(g0)

        maxsteps = 250
        maxsteps_per_level = 50

        Ls = []
        #self.sigma = np.load('simga.npy')
        #self.sigma = jnp.sqrt(self.Target.variance)


        def nan_reject(x, u, l, g, xx, uu, ll, gg):
            """if there are nans, let's reduce the stepsize, and not update the state"""
            no_nans = jnp.all(jnp.isfinite(xx))
            tru = no_nans
            false = (1 - tru)
            new_eps = self.eps * (false * 0.5 + tru * 1.0)
            X = jnp.nan_to_num(xx) * tru + x * false
            U = jnp.nan_to_num(uu) * tru + u * false
            L = jnp.nan_to_num(ll) * tru + l * false
            G = jnp.nan_to_num(gg) * tru + g * false
            return new_eps,X, U, L, G

        def burn_in_step(state):

            index, stationary, x, u, l, g, key, time = state
            #self.sigma = adam.sigma_estimate()  # diagonal conditioner

            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time)
            #energy_change = kinetic_change + ll - l
            #energy_condition = energy_change**2 / self.Target.d < 10000
            new_eps, xx, uu, ll, gg = nan_reject(x, u, l, g, xx, uu, ll, gg)
            self.set_hyperparameters(self.L, new_eps)

            #adam.step(gg)
            Ls.append(ll)

            if len(Ls) > 10:
                stationary = np.std(Ls[-10:]) / np.sqrt(self.Target.d * 0.5) < 1.2
            else:
                stationary = False

            return index + 1, stationary, xx, uu, ll, gg, key, time


        condition = lambda state: (state[0] < maxsteps_per_level) and not state[1] # false if the burn-in should be ended


        x, u, l, g, key = x0, u0, l0, g0, key0
        total_steps = 0
        new_level = True
        l_plateau = np.inf
        while new_level and total_steps < maxsteps:
            steps, stationary, x, u, l, g, key, time = my_while(condition, burn_in_step, (0, False, x, u, l, g, key, 0.0))
            total_steps += steps
            l_plateau_new = np.average(Ls[-10:])
            diff = np.abs(l_plateau_new - l_plateau) / np.sqrt(self.Target.d * 0.5)
            new_level = diff > 1.0
            l_plateau = l_plateau_new
            self.eps = self.eps * 0.5

        # plt.plot(Ls)
        # plt.yscale('log')
        # plt.show()
        # after you are done with developing, replace, my_while with jax.lax.while_loop
        #self.sigma = adam.sigma_estimate()  # diagonal conditioner
        #np.save('simga.npy', self.sigma)
        # plt.plot(self.sigma/np.sqrt(self.Target.variance), 'o')
        # plt.yscale('log')
        # plt.show()

        return total_steps, x, u, l, g, key




    def sample(self, num_steps, num_chains = 1, x_initial = 'prior', random_key= None, output = 'normal', thinning= 1, remove_burn_in= True):
        """Args:
               num_steps: number of integration steps to take.

               num_chains: number of independent chains, defaults to 1. If different than 1, jax will parallelize the computation with the number of available devices (CPU, GPU, TPU),
               as returned by jax.local_device_count().

               x_initial: initial condition for x, shape: (d, ). Defaults to 'prior' in which case the initial condition is drawn from the prior distribution (self.Target.prior_draw).

               random_key: jax radnom seed, defaults to jax.random.PRNGKey(0)

               output: determines the output of the function:
                        'normal': returns Target.transform of the samples (to save memory), shape: (num_samples, len(Target.transform(x)))
                        'full': returns the full samples and the energy at each step, shape: (num_samples, Target.d), (num_samples, )
                        'energy': returns the transformed samples and the energy at each step, shape: (num_samples, len(Target.transform(x))), (num_samples, )
                        'final state': only returns the final state of the chain, shape: (Target.d, )
                        'ess': only ouputs the Effective Sample Size, float. In this case self.Target.variance = <x_i^2>_true should be defined.

               thinning: integer for thinning the chains (every n-th sample is returned), defaults to 1 (no thinning).
                        In unadjusted methods such as MCHMC, all samples contribute to the posterior and thining degrades the quality of the posterior.
                        If thining << # steps needed for one effective sample the loss is not too large.
                        However, in general we recommend no thining, as it can often be avoided by using Target.transform.

               remove_burn_in: removes the samples during the burn-in phase. The end of burn-in is determined by settling of the -log p.
        """

        if num_chains == 1:
            return self.single_chain_sample(num_steps, x_initial, random_key, output, thinning, remove_burn_in) #the function which actually does the sampling

        else:
            num_cores = jax.local_device_count()

            if random_key is None:
                key = jax.random.PRNGKey(0)
            else:
                key = random_key

            if isinstance(x_initial, str):
                if x_initial == 'prior':  # draw the initial x from the prior
                    keys_all = jax.random.split(key, num_chains * 2)
                    x0 = jnp.array([self.Target.prior_draw(keys_all[num_chains+i]) for i in range(num_chains)])
                    keys = keys_all[:num_chains]

                else:  # if not 'prior' the x_initial should specify the initial condition
                    raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
            else: #initial x is given
                x0 = jnp.copy(x_initial)
                keys = jax.random.split(key, num_chains)


            f = lambda i: self.single_chain_sample(num_steps, x0[i], keys[i], output, thinning, remove_burn_in)

            if num_cores != 1: #run the chains on parallel cores
                parallel_function = jax.pmap(jax.vmap(f))
                results = parallel_function(jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores))
                ### reshape results ###
                if type(results) is tuple: #each chain returned a tuple
                    results_reshaped =[]
                    for i in range(len(results)):
                        res = jnp.array(results[i])
                        results_reshaped.append(res.reshape([num_chains, ] + [res.shape[j] for j in range(2, len(res.shape))]))
                    return results_reshaped

                else:
                    return results.reshape([num_chains, ] + [results.shape[j] for j in range(2, len(results.shape))])


            else: #run chains serially on a single core

                return jax.vmap(f)(jnp.arange(num_chains))



    def single_chain_sample(self, num_steps, x_initial = 'prior', random_key= None, output = 'normal', thinning= 1, remove_burn_in= True):


        def step(state, useless):
            """Tracks transform(x) as a function of number of iterations"""

            x, u, l, g, E, key, time = state
            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key, time), (self.Target.transform(xx), ll, EE)


        def step_full_track(state, useless):
            """Tracks transform(x) as a function of number of iterations"""

            x, u, l, g, E, key, time = state
            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key, time), (xx, ll, EE)


        def b_step(state_track, useless):
            """Only tracks b as a function of number of iterations."""

            x, u, l, g, E, key, time = state_track[0]
            x, u, ll, g, kinetic_change, key, time = self.dynamics(x, u, l, g, key, time)
            W, F2 = state_track[1]

            F2 = (W * F2 + jnp.square(self.Target.transform(x)))/ (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            bias = jnp.sqrt(jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance)))
            #bias = jnp.average((F2 - self.Target.variance) / self.Target.variance)

            return ((x, u, ll, g, E + kinetic_change + ll - l, key, time), (W, F2)), bias


        ### initial conditions ###
        x, u, l, g, key = self.get_initial_conditions(x_initial, random_key)

        ### sampling ###

        if output == 'ess':  # only track the bias

            _, b = jax.lax.scan(b_step, init=((x, u, l, g, 0.0, key, 0.0), (1.0, jnp.square(x))), xs=None, length=num_steps)

            no_nans = 1-jnp.any(jnp.isnan(b))
            cutoff_reached = b[-1] < 0.1

            # plt.plot(bias, '.')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

            return ess_cutoff_crossing(b) * no_nans * cutoff_reached / self.grad_evals_per_step #return 0 if there are nans, or if the bias cutoff was not reached

        elif output == 'full': #track everything
            state, track = jax.lax.scan(step_full_track, init=(x, u, l, g, 0.0, key, 0.0), xs=None, length=num_steps)
            x, L, E = track
            if remove_burn_in:
                index_burnin = burn_in_ending(L)
            else:
                index_burnin = 0
            return x[index_burnin::thinning, :], E[index_burnin::thinning]


        else: # track the transform(x) and the energy

            state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, key, 0.0), xs=None, length=num_steps)
            x, L, E = track

            if remove_burn_in:
                index_burnin = burn_in_ending(L)
            else:
                index_burnin = 0

            if output == 'final state': #only return the final x
                return state[0]
            elif output == 'energy': #return the samples X and the energy E
                return x[index_burnin::thinning, :], E[index_burnin::thinning]
            elif output == 'normal': #return the samples X
                return x[index_burnin::thinning]
            else:
                raise ValueError('output = ' + output + 'is not a valid argument for the Sampler.sample')




    def tune_hyperparameters(self, x_initial = 'prior', random_key= None, dialog = False):

        varE_wanted = 0.0005             # targeted energy variance per dimension

        burn_in, samples = 2000, 1000

        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key


        self.set_hyperparameters(np.sqrt(self.Target.d), 0.6)

        key, subkey = jax.random.split(key)
        x0 = self.sample(burn_in, x_initial= x_initial, random_key= subkey, output= 'final state', remove_burn_in = False)
        props = (key, np.inf, 0.0, False)
        if dialog:
            print('Hyperparameter tuning (first stage)')

        def tuning_step(props):

            key, eps_inappropriate, eps_appropriate, success = props

            # get a small number of samples
            key_new, subkey = jax.random.split(key)
            X, E = self.sample(samples, x_initial= x0, random_key= subkey, output= 'full', remove_burn_in = False)

            # remove large jumps in the energy
            E -= jnp.average(E)
            E = remove_jumps(E)


            ### compute quantities of interest ###

            # typical size of the posterior
            x1 = jnp.average(X, axis= 0) #first moments
            x2 = jnp.average(jnp.square(X), axis=0) #second moments
            #sigma = jnp.sqrt(jnp.average(x2 - jnp.square(x1))) #average variance over the dimensions
            sigma_old = self.sigma
            self.sigma = jnp.sqrt(x2 - jnp.square(x1))
            sigma_ratio = jnp.sqrt(jnp.sum(self.sigma**2) / jnp.sum(sigma_old**2))

            # energy fluctuations
            varE = jnp.std(E)**2 / self.Target.d #variance per dimension
            no_divergences = np.isfinite(varE)

            ### update the hyperparameters ###

            if no_divergences:
                #L_new = sigma * jnp.sqrt(self.Target.d)
                eps_new = self.eps * jnp.power(varE_wanted / varE, 0.25) #assume var[E] ~ eps^4
                success = jnp.abs(1.0 - varE / varE_wanted) < 0.2 #we are done

            else:
                L_new = self.L

                if self.eps < eps_inappropriate:
                    eps_inappropriate = self.eps

                eps_new = jnp.inf #will be lowered later


            #update the known region of appropriate eps

            if not no_divergences: # inappropriate epsilon
                if self.eps < eps_inappropriate: #it is the smallest found so far
                    eps_inappropriate = self.eps

            else: # appropriate epsilon
                if self.eps > eps_appropriate: #it is the largest found so far
                    eps_appropriate = self.eps

            # if suggested new eps is inappropriate we switch to bisection
            if eps_new > eps_inappropriate:
                eps_new = 0.5 * (eps_inappropriate + eps_appropriate)

            eps_inappropriate /= sigma_ratio
            eps_appropriate /= sigma_ratio
            eps_new /= sigma_ratio

            self.set_hyperparameters(self.L, eps_new)

            if dialog:
                word = 'bisection' if (not no_divergences) else 'update'
                print('varE / varE wanted: {} ---'.format(np.round(varE / varE_wanted, 4)) + word + '---> eps: {}, sigma = L / sqrt(d): {}'.format(np.round(eps_new, 3), np.round(sigma_ratio, 3)))

            return key_new, eps_inappropriate, eps_appropriate, success


        ### first stage: L = sigma sqrt(d)  ###
        for i in range(10): # = maxiter
            props = tuning_step(props)
            if props[-1]: # success == True
                break

        ### second stage: L = epsilon(best) / ESS(correlations)  ###
        if dialog:
            print('Hyperparameter tuning (second stage)')

        n = np.logspace(2, np.log10(2500), 6).astype(int) # = [100, 190, 362, 689, 1313, 2499]
        n = np.insert(n, [0, ], [1, ])
        X = np.empty((n[-1] + 1, self.Target.d))
        X[0] = x0
        for i in range(1, len(n)):
            key, subkey = jax.random.split(key)
            X[n[i-1]:n[i]] = self.sample(n[i] - n[i-1], x_initial= X[n[i-1]-1], random_key= subkey, output = 'full', remove_burn_in = False)[0]
            ESS = ess_corr(X[:n[i]] / self.sigma[None, :])
            if dialog:
                print('n = {0}, ESS = {1}'.format(n[i], ESS))
            if n[i] > 10.0 / ESS:
                break

        L = 0.4 * self.eps / ESS # = 0.4 * correlation length
        self.set_hyperparameters(L, self.eps)

        if dialog:
            print('L / sqrt(d) = {}, ESS(correlations) = {}'.format(L / np.sqrt(self.Target.d), ESS))
            print('-------------')



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



def burn_in_ending(loss):
    loss_avg = jnp.median(loss[len(loss)//2:])
    above = loss[0] > loss_avg
    i = 0
    while (loss[i] > loss_avg) == above:
         i += 1

    ### plot the removal ###
    # t= np.arange(len(loss))
    # plt.plot(t[:i*2], loss[:i*2], color= 'tab:red')
    # plt.plot(t[i*2:], loss[i*2:], color= 'tab:blue')
    # plt.yscale('log')
    # plt.show()

    return i * 2 #we add a safety factor of 2



def my_while(cond_fun, body_fun, initial_state):
    """see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html"""

    state = initial_state

    while cond_fun(state):
        state = body_fun(state)

    return state


