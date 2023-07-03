import jaxlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sampling.dynamics import initialize, random_unit_vector, update_temp
from sampling.sampler import hamiltonian_dynamics, grad_evals



class Sampler:
    """Ensamble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, shift_fn = lambda x, y: x + y, alpha = 1.0, varE_wanted = 1e-4, integrator='MN'):
        """Args:
                Target: the target distribution class.
                alpha: the momentum decoherence scale L = alpha sqrt(d). Optimal alpha is typically around 1, but can also be 10 or so.
                varE_wanted: controls the stepsize after the burn-in. We aim for Var[E] / d = 'varE_wanted'.
        """

        self.Target = Target
        self.masses = jnp.ones(self.Target.d)

        self.alpha = alpha
        self.L = jnp.sqrt(self.Target.d) * alpha
        self.varEwanted = varE_wanted
        self.shift_fn = shift_fn

        self.integrator = integrator

        self.grad_evals_per_step = grad_evals[self.integrator]

        self.eps_initial = jnp.sqrt(self.Target.d)    # this will be changed during the burn-in

        # adjust L and eps as a funciton of temperature
        self.temp_func = lambda T, Tprev, L, eps : (L, eps)

        self.resample_particles = lambda logw, x, u, l, g, key, L, eps, T : (x, u, l, g, key, L, eps, T)

    def partially_refresh_momentum(self, u, random_key, nu):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(random_key)
        noise = nu * jax.random.normal(subkey, shape= u.shape, dtype=u.dtype)

        out =  (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise), axis = 1))[:, None]
        return out, key

    def dynamics(self, hamiltonian_dynamics, x, u, g, random_key, L, eps, T):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change = hamiltonian_dynamics(x=x,u=u,g=g/T, eps=jnp.repeat(eps, x.shape[0]))
        ll, gg = ll * T,  gg * T

        # bounce
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = self.partially_refresh_momentum(uu, random_key, nu)

        return xx, uu, ll, gg, kinetic_change, key

    def sample_temp_level(self, num_steps, tune_steps, x0, u0, l0, g0, E0, key0, L0, eps0, T):

        def energy_at_temperature(x):
           l, g = self.Target.grad_nlogp(x)
           return l/T, g/T

        hd = jax.vmap(hamiltonian_dynamics(integrator=self.integrator, sigma=1/jnp.sqrt(self.masses), grad_nlogp=energy_at_temperature, shift=self.shift_fn, d=self.Target.d))
                                   

        def step(state, tune):

            x, u, l, g, E, key, L, eps, T = state 
            x, u, ll, g, kinetic_change, key = self.dynamics(hd, x, u, g, key, L, eps, T)  # update particles by one step
       
            # ## eps tuning ###
            # de = jnp.square(kinetic_change + (ll - l)/T) / self.Target.d
            # varE = jnp.average(de) #averaged over the ensamble

            #                     #if we are in the tuning phase            #else
            # eps *= (tune * jnp.power(varE / self.varEwanted, -1./6.) + (1-tune))


            # ### L tuning ###
            # #typical width of the posterior
            # moment1 = jnp.average(x, axis=0)
            # moment2 = jnp.average(jnp.square(x), axis = 0)
            # var= moment2 - jnp.square(moment1)
            # sig = jnp.sqrt(jnp.average(var)) # average over dimensions (= typical width of the posterior)

            # Lnew = self.alpha * sig * jnp.sqrt(self.Target.d)
            # L = tune * Lnew + (1-tune) * L #update L if we are in the tuning phase


            EE = E + kinetic_change + (ll - l)/T



            return (x, u, ll, g, EE, key, L, eps, T), (x, EE)


                                                #tuning                     #no tuning
        tune_schedule = jnp.concatenate((jnp.ones(tune_steps), jnp.zeros(num_steps - tune_steps)))

        return jax.lax.scan(step, init= (x0, u0, l0, g0, E0, key0, L0, eps0, T), xs= tune_schedule, length= num_steps)


    def sample(self, steps_at_each_temp, tune_steps, num_chains, temp_schedule, x_initial= 'prior', random_key= None, ess=0.9, num_temps=10):

        """
        temp schedule: either list of temps, e.g. [3.0,2.0,1.0] or tuple of inital and target, e.g. (3.0, 1.0)
        """

        x0, u0, l0, g0, key0 = initialize(self.Target, random_key, x_initial, num_chains) #initialize the chains

        if type(temp_schedule) is list:
            self.fixed_schedule = True
            self.initial_temp = temp_schedule[0]
            temp_schedule = jnp.array(temp_schedule)
        elif type(temp_schedule) is tuple and len(temp_schedule)==2:
            self.fixed_schedule = False
            self.initial_temp = temp_schedule[0]
            self.target_temp = temp_schedule[1]
        else:
            print(type(temp_schedule))
            raise Exception("Invalid temp_schedule: must be list of temperatures or pair of initial and target temperatures")

        if self.fixed_schedule:
            temp_schedule_ext = jnp.insert(temp_schedule, 0, temp_schedule[0]) # as if the temp level before the first temp level was the same


        def temp_level(state, iter):
            x, u, l, g, E, key, L, eps, Tprev = state

            if self.fixed_schedule:
                T = temp_schedule_ext[iter]
            else:
                T = update_temp(Tprev, ess, l, self.target_temp)
            # , temp_schedule_ext[iter-1]

            
            # L *= jnp.sqrt(T / Tprev)
            # eps *= jnp.sqrt(T / Tprev)

            L, eps = self.temp_func(T, Tprev, L, eps)
            jax.debug.print("eps: {}, L: {}, T: {}", eps, L, Tprev)


            logw = -(1.0/T - 1.0/Tprev) * l
            x, u, l, g, key, L, eps, T = self.resample_particles(logw, x, u, l, g, key, L, eps, T)



            next_state, (xs, EE) = self.sample_temp_level(steps_at_each_temp, tune_steps, x, u, l, g, E, key, L, eps, T)

            return next_state, (xs, EE)


        if self.fixed_schedule:
            num_temps = len(temp_schedule_ext)

        # jax.debug.print("x {}", x0[0])
        return jax.lax.scan(temp_level, init= (x0, u0, l0, g0, jnp.zeros(x0.shape[0]), key0, self.L, self.eps_initial, self.initial_temp), xs= jnp.arange(1, num_temps))[1]
        
