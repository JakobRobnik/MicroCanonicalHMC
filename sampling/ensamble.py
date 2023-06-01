import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sampling.sampler import find_crossing


class vmap_target:
    """A wrapper target class, where jax.vmap has been applied to the functions of a given target"""

    def __init__(self, target):
        """target: a given target to vmap"""

        # obligatory attributes
        self.grad_nlogp = jax.vmap(target.grad_nlogp)
        self.d = target.d


        # optional attributes

        if hasattr(target, 'transform'):
            self.transform = jax.vmap(target.transform)
        else:
            self.transform = lambda x: x #if not given, set it to the identity

        if hasattr(target, 'prior_draw'):
            self.prior_draw = jax.vmap(target.prior_draw)

        if hasattr(target, 'second_moments'):
            self.second_moments = target.second_moments
            self.variance_second_moments = target.variance_second_moments

        if hasattr(target, 'name'):
            self.name = target.name


class pmap_target:
    """A wrapper target class, where jax.pmap(jax.vmap()) has been applied to the functions of a given target."""

    def __init__(self, target, chains):
        """target: a given target to pmap"""
        
        devices= jax.local_device_count()
        self.devices = devices

        # obligatory attributes
        self.d = target.d
        
        pvgrad= jax.pmap(jax.vmap(target.grad_nlogp))
        
        def grad(x):
            l, g = jnp.array(pvgrad(x.reshape(devices, chains // devices, target.d)))
            return l.reshape(chains), g.reshape(chains, target.d)
                
        self.grad_nlogp = grad
        
        # optional attributes

        if hasattr(target, 'transform'):            
            pvtransform= jax.pmap(jax.vmap(target.transform))
            self.transform = lambda x: jnp.array(pvtransform(x.reshape(devices, chains // devices, target.d))).reshape(chains, target.d)

        else:
            self.transform = lambda x: x #if not given, set it to the identity

        if hasattr(target, 'prior_draw'):
            pvdraw= jax.pmap(jax.vmap(target.prior_draw))
            self.transform = lambda x: jnp.array(pvdraw(x.reshape(devices, chains // devices))).reshape(chains)


        if hasattr(target, 'second_moments'):
            self.second_moments = target.second_moments
            self.variance_second_moments = target.variance_second_moments

        if hasattr(target, 'name'):
            self.name = target.name



class Sampler:
    """Ensamble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, chains, alpha = 1.0, diagonal_preconditioning = False):
        """Args:
                Target: the target distribution class.
                alpha: the momentum decoherence scale L = alpha sqrt(d). Optimal alpha is typically around 1, but can also be 10 or so.
                
        """

        self.chains = chains
        
        if jax.local_device_count() == 1:
            self.Target = vmap_target(Target)
        else:
            self.Target = pmap_target(Target, chains)
        
        self.alpha = alpha
        self.program = Program()
        self.diagonal_preconditioning = diagonal_preconditioning
        
        self.grad_evals_per_step = 1.0 # per chain (leapfrog)
                
        self.isotropic_u0 = False
        self.eps_initial = 0.01 * jnp.sqrt(self.Target.d)    # this will be changed during the burn-in

        
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


    def hamiltonian_dynamics(self, x, u, g, key, eps, sigma):
        """leapfrog"""

        z = x / sigma # go to the latent space

        # half step in momentum
        uu, delta_r1 = self.update_momentum(eps * 0.5, g * sigma, u)


        # full step in x
        zz = z + eps * uu
        xx = sigma * zz # go back to the configuration space
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(eps * 0.5, gg * sigma, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, key


    def dynamics(self, x, u, g, random_key, L, eps, sigma):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, random_key, eps, sigma)

        # bounce
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = self.partially_refresh_momentum(uu, key, nu)

        return xx, uu, ll, gg, kinetic_change, key


    def initialize(self, random_key, x_initial):

        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key
            

        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                keys_all = jax.random.split(key, self.chains + 1)
                x = self.Target.prior_draw(keys_all[1:])
                key = keys_all[0]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')

        else:  # initial x is given
            x = jnp.copy(x_initial)
        
        l, g = self.Target.grad_nlogp(x)
        virials = jnp.average(x * g, axis=0)

        ### initial velocity ###
        if self.isotropic_u0:
            u, key = self.random_unit_vector(key, self.chains)  # random velocity orientations

        else: # align the initial velocity with the gradient, with the sign depending on the virial condition
            sgn = -2.0 * (virials < 1.0) + 1.0
            u = - g / jnp.sqrt(jnp.sum(jnp.square(g), axis = 1))[:, None] # initialize momentum in the direction of the gradient of log p
            u = u * sgn[None, :] #if the virial in that direction is smaller than 1, we flip the direction of the momentum in that direction
       
        return x, u, l, g, key



    def sample(self, num_steps, x_initial='prior', random_key= None):
        """Args:
               num_steps: number of integration steps to take during the sampling. There will be some additional steps during the first stage of the burn-in (max 200).
               num_chains: a tuple (number of superchains, number of chains per superchain)
               x_initial: initial condition for x, shape: (num_chains, d). Defaults to 'prior' in which case the initial condition is drawn with self.Target.prior_draw.
               random_key: jax random seed, defaults to jax.random.PRNGKey(0).
        """

        x0, u0, l0, g0, key = self.initialize(random_key, x_initial)
        
        
        def no_nans(x, u, l, g, varE, xx, uu, ll, gg, varE_new):
            """if there are nans we don't want to update the state"""

            no_nans = jnp.all(jnp.isfinite(xx))
            tru = no_nans  # there were no nans
            false = (1 - tru)
            X = jnp.nan_to_num(xx) * tru + x * false
            U = jnp.nan_to_num(uu) * tru + u * false
            L = jnp.nan_to_num(ll) * tru + l * false
            G = jnp.nan_to_num(gg) * tru + g * false
            Var = jnp.nan_to_num(varE_new) * tru + varE * false
            return tru, X, U, L, G, Var


        def step(state, useless):
            
            steps, x, u, l, g, key, L, eps, sigma, varE, t = state

            xx, uu, ll, gg, kinetic_change, key = self.dynamics(x, u, g, key, L, eps, sigma)  # update particles by one step

            de = jnp.square(kinetic_change + ll - l) / self.Target.d
            varE_new = jnp.average(de)

            # if there were nans we don't do the step
            nonans, x, u, l, g, varE = no_nans(x, u, l, g, varE, xx, uu, ll, gg, varE_new)
            
            #diagonal preconditioner
            if self.diagonal_preconditioning:
                update = ((steps+1) % 30 == 0) & steps < num_steps // 3
                sigma_new = jnp.std(x, axis=0) * update + (1 - update) * sigma  # diagonal conditioner
                sigma_ratio = jnp.sqrt(jnp.average(jnp.square(sigma_new) / jnp.average(jnp.square(sigma))))
                u *= sigma / sigma_new
                u /= jnp.sqrt(jnp.sum(jnp.square(u)))
                
            else: 
                sigma_new = sigma
                sigma_ratio = 1.0
                
                ### setting L with sigma ###
                # Lnew = self.alpha * jnp.sqrt(jnp.average(jnp.square(jnp.std(x, axis=0)))) * jnp.sqrt(self.Target.d)
                # update = True#steps < num_steps // 2
                # L = Lnew * update + L *(1-update)
                
                ### L/eps = const ###                
                
                
            #stepsize
            t_nonans = t + (1.-t) / (num_steps - steps)
            t_nans = self.program.inv_vare(self.program.vare(t) / 1.2**6)
            tnew = t_nonans * nonans + t_nans * (1-nonans)
            varEwanted = self.program.vare(tnew)
            eps_factor = jnp.power(varEwanted / varE, 1./6.) * sigma_ratio
            eps_factor = jnp.min(jnp.array([jnp.max(jnp.array([eps_factor, 0.1])), 10.0])) #eps cannot change by more than a factor of 10 in one step
            eps *= eps_factor
            
            # bias
            moments = jnp.average(jnp.square(self.Target.transform(x)), axis = 0)
            bias_d = jnp.square(moments - self.Target.second_moments) / self.Target.variance_second_moments
            bias_avg, bias_max = jnp.average(bias_d), jnp.max(bias_d)

            return (steps + 1, x, u, l, g, key, L, eps, sigma_new, varE, tnew), (eps, varE, varEwanted, bias_avg, bias_max)

        
        
        if self.diagonal_preconditioning:
            sigma0 = jnp.std(x0, axis=0)
            sig = 1.0
            u0 /= sigma0
            u0 /= jnp.sqrt(jnp.sum(jnp.square(u0)))
            
        else:
            sigma0 = jnp.ones(self.Target.d)
            sig = jnp.sqrt(jnp.average(jnp.square(jnp.std(x0, axis=0))))
                           
        #L = self.alpha * sig * jnp.sqrt(self.Target.d)
        #L = num_steps * 0.1 * self.eps_initial
        L = self.alpha * jnp.sqrt(jnp.average(self.Target.second_moments)) * jnp.sqrt(self.Target.d)

        state = (0, x0, u0, l0, g0, key, L, self.eps_initial, sigma0, self.program.vare(0.), 0.)
        state, track = jax.lax.scan(step, init= state, xs= None, length= num_steps)
        
        eps, vare, varew, bias_avg, bias_max = track
        num = 2
        plt.figure(figsize= (6, 3 * num))

        ### stepsize tuning ###
        plt.subplot(num, 1, 1)
        plt.title('stepsize tuning')
        plt.plot(vare, '.-', color='magenta', label='Var[E]/d')
        plt.plot(varew, '.-', color='tab:red', label = 'targeted')
        plt.plot(eps, '.-', color='tab:orange', label = 'eps')
        plt.yscale('log')
        plt.legend()
        
        # plt.subplot(num, 1, 2)
        # plt.title('stepsize tuning')
        # plt.plot(eps_arr, '.-', color='tab:orange', label = 'eps')
        # plt.yscale('log')
        # plt.legend()

        
        ### bias ###
        plt.subplot(num, 1, 2)
        plt.plot(bias_avg, color = 'tab:blue', label= 'average')
        plt.plot(bias_max, color = 'tab:red', label= 'max')
        plt.plot(jnp.ones(len(bias_max)) * 1e-2, '--', color = 'black')
        #num_max, num_avg = find_crossing(bias_max_arr, 0.01), find_crossing(bias_avg_arr, 0.01)
        #plt.title('steps to low bias: {0} (max), {1} (avg)'.format(num_max, num_avg))
        plt.legend()
        plt.xlabel('# gradient evaluations')
        plt.ylabel(r'$\mathrm{bias}^2$')
        plt.yscale('log')
        plt.savefig('plots/tst_ensamble/' + self.Target.name + '.png')
        plt.tight_layout()
        plt.close()

        return state[1]



class Program:
    
    def __init__(self):
        
        #fraction of the total sampling time
        self.t = jnp.array([-1e-10, 4./5., 1. + 1e-10]) 
        
        #log Var[E]/d that we want at the specified times. Other values will be interpolated by the exponential.
        self.m = jnp.log(jnp.array([1.1e-3, 0.9e-3, 1e-5]))

        self.a, self.b = self.get_coeffs(self.t, self.m)
        self.ainv, self.binv = self.get_coeffs(self.m, self.t)


    def get_coeffs(self, x, y):
        """coefficients of the linear interpolation"""
        slope = (y[:-1] - y[1:]) / (x[:-1] - x[1:])
        intercept = (-y[:-1]*x[1:] + y[1:] * x[:-1]) / (x[:-1] - x[1:])
        return slope, intercept


    def vare(self, s):
        """gets the desired Var[E]/d at: number_of_steps = s * total_number_of_steps"""
        select = jnp.searchsorted(self.t, s) - 1
        return jnp.exp(self.a[select] * s + self.b[select])

    def inv_vare(self, m):
        """inverse of the above function"""
        select = jnp.searchsorted(self.m[::-1], jnp.log(m))
        return self.ainv[-select] * jnp.log(m) + self.binv[-select]