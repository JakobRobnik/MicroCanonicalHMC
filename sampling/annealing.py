import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sampling import dynamics

from sampling.dynamics import update_momentum


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


class Annealing:
    """Ensamble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Sampler, alpha = 1.0, varE_wanted = 1e-4):
        """Args:
                Target: the target distribution class.
                alpha: the momentum decoherence scale L = alpha sqrt(d). Optimal alpha is typically around 1, but can also be 10 or so.
                varE_wanted: controls the stepsize after the burn-in. We aim for Var[E] / d = 'varE_wanted'.
        """

        self.Target = vmap_target(Sampler.Target)

        self.alpha = alpha
        self.L = jnp.sqrt(self.Target.d) * alpha
        self.varEwanted = varE_wanted

        self.temp_func = lambda T, Tprev, L, eps : (L, eps)

        self.grad_evals_per_step = 1.0 # per chain (leapfrog)

        self.eps_initial = jnp.sqrt(self.Target.d)    # this will be changed during the burn-in

        self.hd, self.grad_evals_per_step = Sampler.integrator(T= dynamics.update_position(self.Target.grad_nlogp), 
                                                                V= dynamics.update_momentum(self.Target.d, sequential=False),
                                                                d= self.Target.d)




    def hamiltonian_dynamics(self, x, u, g, key, eps, T):
        """leapfrog"""

                # sigma = jnp.ones(self.Target.d) 
        # # dynamics.minimal_norm()
        # return dynamics.leapfrog(T= dynamics.update_position(self.Target.grad_nlogp), 
        #                                                         V= dynamics.update_momentum(self.Target.d, sequential=False),
        #                                                         d= self.Target.d)[0](x,u,g/T,eps, sigma)


        # half step in momentum
        uu, delta_r1 = update_momentum(self.Target.d, sequential=False)(eps * 0.5, u, g / T)

        # full step in x
        xx = x + eps * uu
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = update_momentum(self.Target.d, sequential=False)(eps * 0.5, uu, gg / T)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, key


    def dynamics(self, x, u, g, random_key, L, eps, T):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, random_key, eps, T)

        # bounce
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = dynamics.partially_refresh_momentum(d=self.Target.d, sequential=False)(uu, key, nu)

        return xx, uu, ll, gg, kinetic_change, key


    def initialize(self, random_key, x_initial, num_chains):


        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        if x_initial is None:
            keys_all = jax.random.split(key, num_chains + 1)
            x = self.Target.prior_draw(keys_all[1:])
            key = keys_all[0]

        else:  # initial x is given
            x = jnp.copy(x_initial)

        l, g = self.Target.grad_nlogp(x)


        ### initial velocity ###
        u, key = dynamics.random_unit_vector(d=self.Target.d, sequential=False)(key, num_chains)  # random velocity orientations


        return x, u, l, g, key
    

    def sample_temp_level(self, num_steps, tune_steps, x0, u0, l0, g0, E0, key0, L0, eps0, T):

        # def energy_at_temperature(x):
        #    l, g = self.Target.grad_nlogp(x)
        #    return l/T, g/T

        # hd = jax.vmap(hamiltonian_dynamics(integrator=self.integrator, sigma=1/jnp.sqrt(self.masses), grad_nlogp=energy_at_temperature, shift=self.shift_fn, d=self.Target.d))
                                   

        def step(state, tune):

            x, u, l, g, E, key, L, eps = state 
            x, u, ll, g, kinetic_change, key = self.dynamics(x, u, g, key, L, eps, T)
            # x, u, ll, g, kinetic_change, key = self.dynamics(hd, x, u, g, key, L, eps, T)  # update particles by one step
       
            ## eps tuning ###
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

            return (x, u, ll, g, EE, key, L, eps), (x, EE)


                                                #tuning                     #no tuning
        tune_schedule = jnp.concatenate((jnp.ones(tune_steps), jnp.zeros(num_steps - tune_steps)))

        return jax.lax.scan(step, init= (x0, u0, l0, g0, E0, key0, L0, eps0), xs= tune_schedule, length= num_steps)




    def sample(self, steps_at_each_temp, tune_steps, num_chains, temp_schedule, x_initial= None, random_key= None):

        x0, u0, l0, g0, key0 = self.initialize(random_key, x_initial, num_chains) #initialize the chains

        temp_schedule_ext = jnp.insert(temp_schedule, 0, temp_schedule[0]) # as if the temp level before the first temp level was the same


        def temp_level(state, iter):
            x, u, l, g, E, key, L, eps = state
            T, Tprev = temp_schedule_ext[iter], temp_schedule_ext[iter-1]
            
            # L *= jnp.sqrt(T / Tprev)
            # eps *= jnp.sqrt(T / Tprev)

            L, eps = self.temp_func(T, Tprev, L, eps)


            # jax.debug.print("eps: {}, L: {}", eps, L)
            # if self.resample:
            #     logw = -(1.0/T - 1.0/Tprev) * l
            #     x, u, l, g, key, L, eps, T = resample_particles(logw, x, u, l, g, key, L, eps, T)



            next_state, (xs, EE) = self.sample_temp_level(steps_at_each_temp, tune_steps, x, u, l, g, E, key, L, eps, T)

            return next_state, (xs, EE)

        return jax.lax.scan(temp_level, init= (x0, u0, l0, g0, jnp.zeros(x0.shape[0]), key0, self.L, self.eps_initial), xs= jnp.arange(1, len(temp_schedule_ext)))[1]





