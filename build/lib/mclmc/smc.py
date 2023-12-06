import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

def bisection(f, a, b, tol=1e-3, max_iter=100):

    def cond_fn(inputs):
        a, b, _, iter_count = inputs
        return (jnp.abs(a - b) > tol * a) & (iter_count < max_iter)

    def body_fn(inputs):
        a, b, midpoint, iter_count = inputs
        midpoint = (a + b) / 2.0
        a, b = jax.lax.cond(f(midpoint) > 0, lambda _: (a, midpoint), lambda _: (midpoint, b), operand=())

        #jax.debug.print("a: {}, b: {}, midpoint: {}, iter: {}", a, b, midpoint, iter_count)
        return a, b, midpoint, iter_count + 1

    #a, b, midpoint, iter_count = jax.lax.while_loop(cond_fn, body_fn, (a, b, 0.0, 0))
    # Use cond to decide which path to follow, note the condition is now f(b) <= 0
    a, b, midpoint, iter_count = jax.lax.cond(f(b) <= 0, 
                                              lambda _: (b, b, b, 0), 
                                              lambda _: jax.lax.while_loop(cond_fn, body_fn, (a, b, 0.0, 0)), 
                                              operand=())

    return midpoint

def systematic_resampling(logw, random_key):
    # Normalize weights
    w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))

    # Compute cumulative sum
    cumsum_w = jnp.cumsum(w)

    # Number of particles
    N = len(logw)

    # Generate N uniform random numbers, then transform them appropriately
    key, subkey = jax.random.split(random_key)
    u = (jnp.arange(N) + jax.random.uniform(subkey)) / N

    # Compute resampled indices
    indices = jnp.searchsorted(cumsum_w, u)

    return indices, key

class vmap_target:
    """A wrapper target class, where jax.vmap has been applied to the functions of a given target"""

    def __init__(self, target):
        """target: a given target to vmap"""

        # obligatory attributes
        self.grad_nlogp = jax.vmap(target.grad_nlogp)
        self.d = target.d

        # optional attributes
        if hasattr(target, 'prior_draw'):
            self.prior_draw = jax.vmap(target.prior_draw)


class Sampler:
    """Ensamble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, alpha = 1.0, varE_wanted = 1e-4):
        """Args:
                Target: the target distribution class.
                alpha: the momentum decoherence scale L = alpha sqrt(d). Optimal alpha is typically around 1, but can also be 10 or so.
                varE_wanted: controls the stepsize after the burn-in. We aim for Var[E] / d = 'varE_wanted'.
        """

        self.Target = vmap_target(Target)

        self.alpha = alpha
        self.L = jnp.sqrt(self.Target.d) * alpha
        self.varEwanted = varE_wanted

        self.grad_evals_per_step = 1.0 # per chain (leapfrog)

        self.eps_initial = jnp.sqrt(self.Target.d)    # this will be changed during the burn-in


    def random_unit_vector(self, random_key, num_chains):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(subkey, shape = (num_chains, self.Target.d), dtype = 'float64')
        normed_u = u / jnp.sqrt(jnp.sum(jnp.square(u), axis = 1))[:, None]
        return normed_u, key


    def partially_refresh_momentum(self, u, random_key, nu):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(random_key)
        noise = nu * jax.random.normal(subkey, shape= u.shape, dtype=u.dtype)

        return (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise), axis = 1))[:, None], key



    def update_momentum(self, eps, g, u):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
        similar to the implementation: https://github.com/gregversteeg/esh_dynamics
        There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g), axis=1)).T
        nonzero = g_norm > 1e-13  # if g_norm is zero (we are at the MAP solution) we also want to set e to zero and the function will return u
        inv_g_norm = jnp.nan_to_num(1.0 / g_norm) * nonzero
        e = - g * inv_g_norm[:, None]
        ue = jnp.sum(u * e, axis=1)
        delta = eps * g_norm / (self.Target.d - 1)
        zeta = jnp.exp(-delta)
        uu = e * ((1 - zeta) * (1 + zeta + ue * (1 - zeta)))[:, None] + 2 * zeta[:, None] * u
        delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1 - ue) * zeta ** 2)
        return uu / (jnp.sqrt(jnp.sum(jnp.square(uu), axis=1)).T)[:, None], delta_r


    def hamiltonian_dynamics(self, x, u, g, key, eps, T):
        """leapfrog"""

        # half step in momentum
        uu, delta_r1 = self.update_momentum(eps * 0.5, g / T, u)

        # full step in x
        xx = x + eps * uu
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(eps * 0.5, gg / T, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, key


    def dynamics(self, x, u, g, random_key, L, eps, T):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, random_key, eps, T)

        # bounce
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = self.partially_refresh_momentum(uu, key, nu)

        return xx, uu, ll, gg, kinetic_change, key


    def initialize(self, random_key, x_initial, num_chains):


        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains + 1)
                x = self.Target.prior_draw(keys_all[1:])
                key = keys_all[0]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')

        else:  # initial x is given
            x = jnp.copy(x_initial)

        l, g = self.Target.grad_nlogp(x)


        ### initial velocity ###
        u, key = self.random_unit_vector(key, num_chains)  # random velocity orientations


        return x, u, l, g, key



    def sample_temp_level(self, num_steps, tune_steps, x0, u0, l0, g0, key0, L0, eps0, T):


        def step(state, tune):

            x, u, l, g, key, L, eps = state 
            x, u, ll, g, kinetic_change, key = self.dynamics(x, u, g, key, L, eps, T)  # update particles by one step


            ### eps tuning ###
            de = jnp.square(kinetic_change + ll - l) / self.Target.d #square energy error per dimension
            varE = jnp.average(de) #averaged over the ensamble

                                #if we are in the tuning phase            #else
            eps *= (tune * jnp.power(varE / self.varEwanted, -1./6.) + (1-tune))


            ### L tuning ###
            #typical width of the posterior
            moment1 = jnp.average(x, axis=0)
            moment2 = jnp.average(jnp.square(x), axis = 0)
            var= moment2 - jnp.square(moment1)
            sig = jnp.sqrt(jnp.average(var)) # average over dimensions (= typical width of the posterior)

            Lnew = self.alpha * sig * jnp.sqrt(self.Target.d)
            L = tune * Lnew + (1-tune) * L #update L if we are in the tuning phase
            

            return (x, u, ll, g, key, L, eps), None


                                                #tuning                     #no tuning
        tune_schedule = jnp.concatenate((jnp.ones(tune_steps), jnp.zeros(num_steps - tune_steps)))

        return jax.lax.scan(step, init= (x0, u0, l0, g0, key0, L0, eps0), xs= tune_schedule, length= num_steps)[0]




    def sample(self, steps_at_each_temp, tune_steps, num_chains, temp_init, temp_final, ess, x_initial= 'prior', random_key= None):

        x0, u0, l0, g0, key0 = self.initialize(random_key, x_initial, num_chains) #initialize the chains

        T0 = temp_init

        def not_terminate(state):
            x, u, l, g, key, L, eps, Tprev = state
            return jnp.abs(Tprev-temp_final) > 1e-2
        
        def update_temp_and_compute_logw(state):
            x, u, l, g, key, L, eps, Tprev = state

            def solve_ess(beta):
                logw = -(beta - 1.0/Tprev) * l 
                weights = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
                #jax.debug.print("estimate: {}, ess: {}", 1.0 / jnp.sum(weights**2) / len(weights), ess)
                return ess * len(weights) - 1.0 / jnp.sum(weights**2)

            beta = bisection(solve_ess, 1.0/Tprev, 1.0/temp_final)
            T = 1.0 / beta

            logw = -(beta - 1.0/Tprev) * l

            return T, logw
        
        def resample_particles(logw, x, u, l, g, key, L, eps, T):

            indices, key = systematic_resampling(logw, key)

            x_resampled = jnp.take(x, indices, axis=0)
            u_resampled = jnp.take(u, indices, axis=0)
            l_resampled = jnp.take(l, indices)
            g_resampled = jnp.take(g, indices, axis=0)

            return (x_resampled, u_resampled, l_resampled, g_resampled, key, L, eps, T)


        def temp_level(state):
            x, u, l, g, key, L, eps, Tprev = state

            T, logw = update_temp_and_compute_logw(state)
            jax.debug.print("T: {}", T)

            L *= jnp.sqrt(T / Tprev)
            eps *= jnp.sqrt(T / Tprev)

            x, u, l, g, key, L, eps, T = resample_particles(logw, x, u, l, g, key, L, eps, T)

            x, u, l, g, key, L, eps = self.sample_temp_level(steps_at_each_temp, tune_steps, x, u, l, g, key, L, eps, T)

            #jax.debug.print("logl_next: {}", l[0])

            return (x, u, l, g, key, L, eps, T)

        
        # do the sampling and return the final x of all the chains
        return jax.lax.while_loop(cond_fun=not_terminate, 
                                  body_fun=temp_level, 
                                  init_val=(x0, u0, l0, g0, key0, self.L, self.eps_initial, T0)
                                 )[0]
        



