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
    """Ensemble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, chains, alpha = 1., diagonal_preconditioning = False):
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
        self.diagonal_preconditioning = diagonal_preconditioning
                
        self.grad_evals_per_step = 1.0 # per chain (leapfrog)
        
        self.isotropic_u0 = False
        self.eps_initial = 0.01 * jnp.sqrt(self.Target.d)    # this will be changed during the burn-in

        self.C = 0.1
        
        
        #self.required_decrease = 0.0 # if log10(change of equipartition loss) / steps is less than required decrease
        self.delay_check = 30

        
        
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
        inv_g_norm = jnp.nan_to_num(1. / g_norm) * nonzero
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
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.) / self.Target.d)
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
        equi = jnp.average(x * g, axis=0)

        ### initial velocity ###
        if self.isotropic_u0:
            u, key = self.random_unit_vector(key, self.chains)  # random velocity orientations

        else: # align the initial velocity with the gradient, with the sign depending on the equipartition condition
            sgn = -2. * (equi < 1.) + 1.
            u = - g / jnp.sqrt(jnp.sum(jnp.square(g), axis = 1))[:, None] # initialize momentum in the direction of the gradient of log p
            u = u * sgn[None, :] #if the equipartition in that direction is smaller than 1, we flip the direction of the momentum in that direction
       
        return x, u, l, g, key


    def computeL(self, x):
        return self.alpha * jnp.sqrt(jnp.sum(jnp.square(x))/x.shape[0])


    def equipartition_loss(self, x, g, random_key):
        """loss = Tr[(1 - E)^T (1 - E)] / d^2
            where Eij = <xi gj> is the equipartition patrix.
            Loss is computed with the Hutchinson's trick."""

        key, key_z = jax.random.split(random_key)
        z = jax.random.rademacher(key_z, (100, self.Target.d)) # <z_i z_j> = delta_ij
        X = z - (g @ z.T).T @ x / x.shape[0]
        return jnp.average(jnp.square(X)) / self.Target.d, key
    
    
    def discretization_bias(self, x, y):
        """Estimate the bias from the difference between the chains.
            x: the precise chain
            y: the sloppy chain
           Bias is computed for the raw x coordinate, without the Target.transform
        """
        
        # moments from the precise chain
        xsq = jnp.square(x)
        moments, var = jnp.average(xsq, 0), jnp.std(xsq, 0)**2
    
        # moments from the sloppy chain
        xsq = jnp.square(y)
        moments_sloppy = jnp.average(xsq, 0) 
        
        # compute the bias of the sloppy chain as if the precise chain was the ground truth
        bias_d = jnp.square(moments_sloppy - moments) / var
        bias = jnp.average(bias_d)
        
        return bias / 16.


    def ground_truth_bias(self, x):

        moments = jnp.average(jnp.square(self.Target.transform(x)), axis = 0)
        bias_d = jnp.square(moments - self.Target.second_moments) / self.Target.variance_second_moments
        bias_avg, bias_max = jnp.average(bias_d), jnp.max(bias_d)

        return bias_avg, bias_max


        
    def sample(self, num_steps, x_initial='prior', random_key= None):
        """Args:
               num_steps: number of integration steps to take during the sampling. There will be some additional steps during the first stage of the burn-in (max 200).
               num_chains: a tuple (number of superchains, number of chains per superchain)
               x_initial: initial condition for x, shape: (num_chains, d). Defaults to 'prior' in which case the initial condition is drawn with self.Target.prior_draw.
               random_key: jax random seed, defaults to jax.random.PRNGKey(0).
        """

        x0, u0, l0, g0, key = self.initialize(random_key, x_initial)
        
        
        def step(state, useless):
            
            steps, history, phase1, x, u, l, g, x2, u2, l2, g2, vare, key, L, eps, sigma = state
            
            ### two steps of the precise dynamics ###
            xx, uu, ll, gg, dK, key = self.dynamics(x, u, g, key, L, eps, sigma)
            de = jnp.square(dK + ll - l) / self.Target.d
            varee = jnp.average(de)
            xx, uu, ll, gg, _, key = self.dynamics(xx, uu, gg, key, L, eps, sigma)
        
            ### one step of the sloppy dynamics ###
            xx2, uu2, ll2, gg2, _, key = self.dynamics(x2, u2, g2, key, L, 2 * eps, sigma)
                        
            # if there were nans we don't do the step
            nonans = jnp.all(jnp.isfinite(xx)) & jnp.all(jnp.isfinite(xx2))
            x, u, l, g = switch(nonans, x, u, l, g, xx, uu, ll, gg)
            x2, u2, l2, g2 = switch(nonans, x2, u2, l2, g2, xx2, uu2, ll2, gg2)
            vare = jnp.nan_to_num(varee) * nonans + vare * (1-nonans)
            
            # bias
            initialization_bias, key = self.equipartition_loss(x, g, key) #estimate the bias from the equipartition loss
            disrcretization_bias = self.discretization_bias(x, x2) #estimate the bias from the chains with larger stepsize
            bias_avg, bias_max = self.ground_truth_bias(x) #actual bias

            # hyperparameters for the next step
            
            history = jnp.concatenate((jnp.ones(1) * initialization_bias, history[:-1]))
            phase1 = steps < 0.5 * num_steps#(jnp.log10(history[-1] / history[0]) / self.delay_check > 0.0)

            bias = initialization_bias * phase1 + disrcretization_bias * (1-phase1)
            varew = self.C * jnp.power(bias, 3./8.)
            varew = 1e-3 * (1-phase1) + varew * phase1
            eps_factor = jnp.power(varew / vare, 1./6.) * nonans + (1-nonans) * 0.5
            eps_factor = jnp.min(jnp.array([jnp.max(jnp.array([eps_factor, 0.3])), 3.])) # eps cannot change by too much
            eps *= eps_factor
            
            L = self.computeL(x)
            
            
            return (steps + 2, history, phase1, x, u, l, g, x2, u2, l2, g2, vare, key, L, eps, sigma), (eps, vare, varew, L, initialization_bias, disrcretization_bias, bias_avg, bias_max)


        # initialize the hyperparameters            
        sigma0 = jnp.ones(self.Target.d)
        L = self.computeL(x0)
        history = jnp.concatenate((jnp.ones(1) * 1e50, jnp.ones(self.delay_check-1) * jnp.inf))
        state = (0, history, True, x0, u0, l0, g0, x0, u0, l0, g0, 1e-3, key, L, self.eps_initial, sigma0)
        
        # run the chains
        state, track = jax.lax.scan(step, init= state, xs= None, length= num_steps//2)
        
        self.analyze_results(track)
        
        #return state[0]
    
    

    def analyze_results(self, track):
    
        eps, vare, varew, Ls, bias1, bias2, bias_avg, bias_max = track
        
        print(find_crossing(bias_max, 0.01) * 2)
        
        num = 4
        plt.figure(figsize= (6, 3 * num))

        steps = jnp.arange(0, 2*len(eps), 2)
        
        ### bias ###
        plt.subplot(num, 1, 1)
        plt.title('bias')
        plt.plot(steps, bias_avg, color = 'tab:red', alpha = 0.5, label= 'average')
        plt.plot(steps, bias_max, color = 'tab:red', label= 'max')
        plt.plot(steps, bias1, color = 'tab:blue', label = 'equipartition loss')
        plt.plot(steps, bias2, color = 'tab:green', label = 'from pairs')
        plt.plot(steps, jnp.ones(steps.shape) * 1e-2, '-', color = 'black')
        plt.legend()
        plt.ylabel(r'$\mathrm{bias}^2$')
        plt.ylim(1e-4, 1e2)
        plt.yscale('log')
        
        ### stepsize tuning ###
        plt.subplot(num, 1, 2)
        plt.title('energy error')
        plt.plot(steps, vare, '.', color='tab:blue', alpha = 0.5, label='measured')
        plt.plot(steps, varew, '.-', color='purple', label = 'targeted')
        plt.ylabel("Var[E]/d")
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(num, 1, 3)
        plt.title('stepsize')
        plt.plot(steps, eps, '.-', color='royalblue')
        plt.ylabel(r"$\epsilon$")
        plt.yscale('log')
        
        ### L tuning ###
        plt.subplot(num, 1, 4)
        plt.title('L')
        L0 = self.alpha * jnp.sqrt(jnp.sum(self.Target.second_moments))
        plt.plot(steps, Ls, '.-', color='tab:orange')
        plt.plot(steps, L0 * jnp.ones(steps.shape), '-', color='black')

        plt.ylabel("L")
        #plt.yscale('log')
        plt.xlabel('# gradient evaluations')
        plt.tight_layout()
        plt.savefig('plots/tst_ensemble/' + self.Target.name + '.png')
        plt.close()

        

def switch(tru, x, u, l, g, xx, uu, ll, gg):
    """if there are nans we don't want to update the state"""

    false = (1 - tru)
    X = jnp.nan_to_num(xx) * tru + x * false
    U = jnp.nan_to_num(uu) * tru + u * false
    L = jnp.nan_to_num(ll) * tru + l * false
    G = jnp.nan_to_num(gg) * tru + g * false
    
    return X, U, L, G

