import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sampling.dynamics import random_unit_vector
from sampling.sampler import hamiltonian_dynamics, grad_evals


class Sampler:
    """Ensamble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(
        self,
        Target,
        shift_fn=lambda x, y: x + y,
        alpha=1.0,
        varE_wanted=1e-4,
        integrator="MN",
    ):
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

        self.eps_initial = jnp.sqrt(
            self.Target.d
        )  # this will be changed during the burn-in

        # adjust L and eps as a funciton of temperature
        self.temp_func = lambda T, Tprev, L, eps: (L, eps)

    def random_unit_vector(self, random_key, num_chains):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(
            subkey, shape=(num_chains, self.Target.d), dtype="float64"
        )
        normed_u = u / jnp.sqrt(jnp.sum(jnp.square(u), axis=1))[:, None]
        return normed_u, key

    def partially_refresh_momentum(self, u, random_key, nu):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(random_key)
        noise = nu * jax.random.normal(subkey, shape=u.shape, dtype=u.dtype)

        return (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise), axis=1))[
            :, None
        ], key

    def dynamics(self, hamiltonian_dynamics, x, u, g, random_key, L, eps, T):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change = hamiltonian_dynamics(
            x=x, u=u, g=g / T, eps=jnp.repeat(eps, x.shape[0])
        )
        ll, gg = ll * T, gg * T

        # bounce
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = self.partially_refresh_momentum(uu, random_key, nu)

        return xx, uu, ll, gg, kinetic_change, key

    def initialize(self, random_key, x_initial, num_chains):
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        if isinstance(x_initial, str):
            if x_initial == "prior":  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains + 1)
                x = jax.vmap(self.Target.prior_draw)(keys_all[1:])
                key = keys_all[0]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError(
                    'x_initial = "'
                    + x_initial
                    + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array'
                )

        else:  # initial x is given
            x = jnp.copy(x_initial)

        l, g = jax.vmap(self.Target.grad_nlogp)(x)

        ### initial velocity ###
        u, key = self.random_unit_vector(
            key, num_chains
        )  # random velocity orientations

        ## if you want to use random_unit_vector from dynamics, this is how
        # keys = jax.random.split(key, num=num_chains+1)
        # u, key = jax.vmap(random_unit_vector(self.Target.d))(keys[1:])  # random velocity orientations

        return x, u, l, g, key

    def sample_temp_level(
        self, num_steps, tune_steps, x0, u0, l0, g0, E0, key0, L0, eps0, T
    ):
        def energy_at_temperature(x):
            l, g = self.Target.grad_nlogp(x)
            return l / T, g / T

        hd = jax.vmap(
            hamiltonian_dynamics(
                integrator=self.integrator,
                sigma=1 / jnp.sqrt(self.masses),
                grad_nlogp=energy_at_temperature,
                shift=self.shift_fn,
                d=self.Target.d,
            )
        )

        def step(state, tune):
            x, u, l, g, E, key, L, eps = state
            x, u, ll, g, kinetic_change, key = self.dynamics(
                hd, x, u, g, key, L, eps, T
            )  # update particles by one step

            ## eps tuning ###
            de = jnp.square(kinetic_change + (ll - l) / T) / self.Target.d
            varE = jnp.average(de)  # averaged over the ensamble

            # if we are in the tuning phase            #else
            eps *= tune * jnp.power(varE / self.varEwanted, -1.0 / 6.0) + (1 - tune)

            ### L tuning ###
            # typical width of the posterior
            moment1 = jnp.average(x, axis=0)
            moment2 = jnp.average(jnp.square(x), axis=0)
            var = moment2 - jnp.square(moment1)
            sig = jnp.sqrt(
                jnp.average(var)
            )  # average over dimensions (= typical width of the posterior)

            Lnew = self.alpha * sig * jnp.sqrt(self.Target.d)
            L = tune * Lnew + (1 - tune) * L  # update L if we are in the tuning phase

            EE = E + kinetic_change + (ll - l) / T

            return (x, u, ll, g, EE, key, L, eps), (x, EE)

            # tuning                     #no tuning

        tune_schedule = jnp.concatenate(
            (jnp.ones(tune_steps), jnp.zeros(num_steps - tune_steps))
        )

        return jax.lax.scan(
            step,
            init=(x0, u0, l0, g0, E0, key0, L0, eps0),
            xs=tune_schedule,
            length=num_steps,
        )

    def sample(
        self,
        steps_at_each_temp,
        tune_steps,
        num_chains,
        temp_schedule,
        x_initial="prior",
        random_key=None,
    ):
        x0, u0, l0, g0, key0 = self.initialize(
            random_key, x_initial, num_chains
        )  # initialize the chains

        temp_schedule_ext = jnp.insert(
            temp_schedule, 0, temp_schedule[0]
        )  # as if the temp level before the first temp level was the same

        def temp_level(state, iter):
            x, u, l, g, E, key, L, eps = state
            T, Tprev = temp_schedule_ext[iter], temp_schedule_ext[iter - 1]

            # L *= jnp.sqrt(T / Tprev)
            # eps *= jnp.sqrt(T / Tprev)

            L, eps = self.temp_func(T, Tprev, L, eps)

            # jax.debug.print("eps: {}, L: {}", eps, L)
            # if self.resample:
            #     logw = -(1.0/T - 1.0/Tprev) * l
            #     x, u, l, g, key, L, eps, T = resample_particles(logw, x, u, l, g, key, L, eps, T)

            next_state, (xs, EE) = self.sample_temp_level(
                steps_at_each_temp, tune_steps, x, u, l, g, E, key, L, eps, T
            )

            return next_state, (xs, EE)

        return jax.lax.scan(
            temp_level,
            init=(
                x0,
                u0,
                l0,
                g0,
                jnp.zeros(x0.shape[0]),
                key0,
                self.L,
                self.eps_initial,
            ),
            xs=jnp.arange(1, len(temp_schedule_ext)),
        )[1]
