import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd


from .sampler import find_crossing
from .sampler import my_while



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
    """A wrapper target class, where jax.pmap has been applied to the functions of a given target"""

    def __init__(self, target):
        """target: a given target to pmap"""

        # obligatory attributes
        self.grad_nlogp = jax.pmap(target.grad_nlogp)
        self.d = target.d

        # optional attributes

        if hasattr(target, 'transform'):
            self.transform = jax.pmap(jax.vmap(target.transform))
        else:
            self.transform = lambda x: x  # if not given, set it to the identity

        if hasattr(target, 'prior_draw'):
            self.prior_draw = jax.pmap(target.prior_draw)

        if hasattr(target, 'variance'):
            self.variance = target.variance



class Sampler:
    """Ensamble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, alpha = 1.0, varE_wanted = 1e-3, diagonal_preconditioning = False, pmap = False):
        """Args:
                Target: the target distribution class.
                alpha: the momentum decoherence scale L = alpha sqrt(d). Optimal alpha is typically around 1, but can also be 10 or so.
                varE_wanted: controls the stepsize after the burn-in. We aim for Var[E] / d = 'varE_wanted'.
                pmap: if True, jax.pmap will be applied to the Target functions.
                      In this case, the number of available devices (as returned by jax.local_device_count())
                      should be equal or larger than the number of ensamble chains that we are running.
                      if False, jax.vmap will be applied to the Target functions. The operation will be run on a single device.
        """

        if pmap:
            self.Target = pmap_target(Target)
        else:
            self.Target = vmap_target(Target)

        self.alpha = alpha
        self.varEwanted = varE_wanted
        self.diagonal_preconditioning = diagonal_preconditioning
        
        self.grad_evals_per_step = 1.0 # per chain (leapfrog)

        ### Hyperparameters of the burn in. The value of those parameters typically will not have large impact on the performance ###
        
        self.isotropic_u0 = False
        self.hutchinson_repeat = 100 #how many realizations to take in hutchinson's trick to compute the virial loss. 100 typically ensures 1% accuracy of the equipartition loss.

        self.required_decrease = 0.01 # if log10(change of virial loss) / steps if less than required decrease we start collecting samples
        self.delay_check = 20

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


    # def virial_loss(self, x, g, key):
    #     """loss^2 = (1/d) sum_i (virial_i - 1)^2"""
    #
    #     virials = jnp.average(x*g, axis=0) #should be all close to 1 if we have reached the typical set
    #     return jnp.sqrt(jnp.average(jnp.square(virials - 1.0))), key


    def vloss(self, x, g, random_key):
        """loss^2 = Tr[(1 - V)^T (1 - V)] / d
            where Vij = <xi gj> is the matrix of virials.
            Loss is computed with the Hutchinson's trick."""

        z = jax.random.rademacher(random_key, (self.hutchinson_repeat, self.Target.d)) # <z_i z_j> = delta_ij
        X = z - (g @ z.T).T @ x / x.shape[0]
        return jnp.sqrt(jnp.average(jnp.square(X)))

    
    def virial_loss(self, x, g, random_key, num_chains):
        
        key1, key2 = jax.random.split(random_key)
        
        if num_chains is tuple:
            shape = (num_chains[0], num_chains[1], self.Target.d)
            virials = jax.vmap(self.vloss)(x.reshape(shape), g.reshape(shape), jax.random.split(key2, num_chains[0]))
            return jnp.median(virials), virials, key1
        else: 
            return self.vloss(x, g, key2), key1
                 

    def splitR(self, x, num_chains):
        """See: https://arxiv.org/pdf/2110.13017.pdf, Definition 8."""

        X = x.reshape(num_chains[0], num_chains[1], x.shape[1])
        avg = jnp.average(X, axis=1)  # average for each superchain
        avg2 = jnp.average(jnp.square(X), axis=1)
        var = avg2 - jnp.square(avg)  # variance for each superchain

        #identify outlier superchains by the different average
        discrepancy = jnp.max(jnp.abs(avg - jnp.median(avg)), axis = 1) #discrepancy of the average
        perm = jnp.argsort(discrepancy) # sort the chains by discrepancy

        Avg = avg[perm][:(len(avg) * self.keep_frac).astype(int)] #remove some percentage of the worst chains
        Var = var[perm][:(len(avg) * self.keep_frac).astype(int)]

        avg_all = jnp.average(Avg, axis=0)  # average of the good chains
        var_between = jnp.sum(jnp.square(Avg - avg_all), axis=0) / (len(Avg) - 1) # in between chains variance for the good chains

        var_within = jnp.average(Var, axis=0) # average within chain vairance

        R = jnp.sqrt(1. + var_between / var_within) #Gelman-Rubin R

        return jnp.max(R), perm


    def initialize(self, random_key, x_initial, chain_shape):

        if chain_shape is tuple:
            num_chains = chain_shape[0]
        else:
            num_chains = chain_shape

            
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
        virials = jnp.average(x * g, axis=0)
        xg = jnp.average(x * g, axis = -1)

        ### initial velocity ###
        if self.isotropic_u0:
            u, key = self.random_unit_vector(key, num_chains)  # random velocity orientations

        else: # align the initial velocity with the gradient, with the sign depending on the virial condition
            sgn = -2.0 * (virials < 1.0) + 1.0
            u = - g / jnp.sqrt(jnp.sum(jnp.square(g), axis = 1))[:, None] # initialize momentum in the direction of the gradient of log p
            u = u * sgn[None, :] #if the virial in that direction is smaller than 1, we flip the direction of the momentum in that direction


        
        if chain_shape is tuple:
            loss = jnp.median(self.Target.d  * jnp.square(xg) - 2 * xg + 1)

            return loss, jnp.repeat(x, num_chains[1], axis= 0), jnp.repeat(u, num_chains[1], axis= 0), jnp.repeat(l, num_chains[1], axis= 0), jnp.repeat(g, num_chains[1], axis= 0), key
        else:
            loss, key = self.virial_loss(x, g, key, chain_shape)
            return loss, x, u, l, g, key
    


    def sample(self, num_chains, neff= 200, x_initial='prior', random_key= None, maxiter= 1000):
        """Args:
               num_steps: number of integration steps to take during the sampling. There will be some additional steps during the first stage of the burn-in (max 200).
               num_chains: a tuple (number of superchains, number of chains per superchain)
               x_initial: initial condition for x, shape: (num_chains, d). Defaults to 'prior' in which case the initial condition is drawn with self.Target.prior_draw.
               random_key: jax random seed, defaults to jax.random.PRNGKey(0).
        """

        
        loss_wanted = jnp.sqrt((self.Target.d + 1) / neff)
        loss_wanted5 = jnp.sqrt((self.Target.d + 1) / 5.0)
        loss_wanted100 = jnp.sqrt((self.Target.d + 1) / 100.0)
        
        # chains are grouped in superchains. chains within the same group have the same intial conditions
        loss, x0, u0, l0, g0, key = self.initialize(random_key, x_initial, num_chains)
        

        def accept_reject_step(x, u, l, g, varE, xx, uu, ll, gg, varE_new):
            """if there are nans we don't want to update the state"""

            no_nans = jnp.all(jnp.isfinite(xx))
            tru = no_nans  # loss went down and there were no nans
            false = (1 - tru)
            X = jnp.nan_to_num(xx) * tru + x * false
            U = jnp.nan_to_num(uu) * tru + u * false
            L = jnp.nan_to_num(ll) * tru + l * false
            G = jnp.nan_to_num(gg) * tru + g * false
            Var = jnp.nan_to_num(varE_new) * tru + varE * false
            return tru, X, U, L, G, Var


        def burn_in_step(state):
            """one step of the burn-in"""

            steps, loss_history, first_stage, x, u, l, g, key, L, eps, sigma, varE = state

            xx, uu, ll, gg, kinetic_change, key = self.dynamics(x, u, g, key, L, eps, sigma)  # update particles by one step

            de = jnp.square(kinetic_change + ll - l) / self.Target.d
            varE_new = jnp.average(de)

            #will we accept the step?
            accept, x, u, l, g, varE = accept_reject_step(x, u, l, g, varE, xx, uu, ll, gg, varE_new)

            
            # good superchains based on the virial condition inside the chain
            if num_chains is tuple:
                loss, loss_sc, key = self.virial_loss(x, g, key, num_chains)
                goodsc = num_chains[0] #// 2
                order = jnp.argsort(loss_sc)
                X = ((x.reshape(num_chains[0], num_chains[1], self.Target.d)[order])[:goodsc]).reshape(goodsc * num_chains[1], self.Target.d)
            else:
                loss, key = self.virial_loss(x, g, key, num_chains)
                X = x
            
                        #loss
            loss_history_new = jnp.concatenate((jnp.ones(1) * loss, loss_history[:-1]))
            first_stage *= (jnp.log10(loss_history_new[-1] / loss_history_new[0]) / self.delay_check > self.required_decrease)

            #diagonal preconditioner
            if self.diagonal_preconditioning:
                update = first_stage# & ((steps+1) % 30 == 0)
                sigma_new = jnp.std(x, axis=0) * update + (1 - update) * sigma  # diagonal conditioner
                sigma_ratio = jnp.sqrt(jnp.average(jnp.square(sigma_new) / jnp.average(jnp.square(sigma))))
                u *= sigma / sigma_new
                u /= jnp.sqrt(jnp.sum(jnp.square(u)))
                
            else: 
                sigma_new = sigma
                sigma_ratio = 1.0
                L = self.alpha * jnp.sqrt(jnp.average(jnp.square(jnp.std(x, axis=0)))) * jnp.sqrt(self.Target.d)
                
            #stepsize
            bias_factor = jnp.power(0.5 * loss / 0.1, 0.25) * first_stage + (1 - first_stage) * 1.0
            eps *= (bias_factor * jnp.power(varE / self.varEwanted, -1./6.) * sigma_ratio) * accept + (1-accept) * 0.5
            eps_arr.append(eps)

            # tracking
            loss_arr.append(loss)
            entropy_arr.append(jnp.median(l))
            vare_arr.append(varE)
            
            # bias
            moments = jnp.average(jnp.square(self.Target.transform(X)), axis = 0)
            bias_d = jnp.square(moments - self.Target.second_moments) / self.Target.variance_second_moments
            bias_avg, bias_max = jnp.average(bias_d), jnp.max(bias_d)
            bias_avg_arr.append(bias_avg)
            bias_max_arr.append(bias_max)

            return steps + 1, loss_history_new, first_stage, x, u, l, g, key, L, eps, sigma_new, varE

        
        loss_arr = []
        entropy_arr = []
        
        eps_arr = []
        vare_arr = []

        bias_avg_arr = []
        bias_max_arr = []
        
        condition = lambda state: (loss_wanted < state[1][0]) * (state[0] < maxiter)# * (jnp.log10(state[1][-1]/state[1][0]) / self.delay_check > self.required_decrease) # true during the burn-in
        
        if self.diagonal_preconditioning:
            sigma0 = jnp.std(x0, axis=0)
            sig = 1.0
            u0 /= sigma0
            u0 /= jnp.sqrt(jnp.sum(jnp.square(u0)))
            
        else:
            sigma0 = jnp.ones(self.Target.d)
            sig = jnp.sqrt(jnp.average(jnp.square(jnp.std(x0, axis=0))))
                           
        loss_history = jnp.concatenate((jnp.ones(1) * 1e50, jnp.ones(self.delay_check-1) * jnp.inf))
                            
        state = (0, loss_history, True, x0, u0, l0, g0, key, self.alpha * sig * jnp.sqrt(self.Target.d), self.eps_initial, sigma0 , 1e4)
        steps, loss_history, first_stage, x, u, l, g, key, L, eps, sigma, varE = my_while(condition, burn_in_step, state)
        #steps, loss, fail_count, never_rejected, x, u, l, g, key, L, eps, sigma, varE = jax.lax.while_loop(condition, burn_in_step, state)
        eps = eps * jnp.power(self.varEwanted / varE, 1./6.)
        n1 = np.arange(len(loss_arr))

        
        num = 3
        plt.figure(figsize= (10, 5 * num))

        ### convergence diagnostics ###
        plt.subplot(num, 1, 1)
        plt.title('convergence diagnostics')
        plt.plot(n1, loss_arr, '.-', color = 'tab:blue', label = 'virial')
        plt.plot(n1, np.ones(len(n1)) * loss_wanted, '--', color = 'black', alpha = 0.7)
        plt.plot(n1, np.ones(len(n1)) * loss_wanted5, '--', color = 'black', alpha = 0.2)

        plt.plot(n1, entropy_arr - np.min(entropy_arr) + 1, '.-', color = 'tab:orange', label = 'entropy')
        plt.legend()
        plt.yscale('log')

        
        ### stepsize tuning ###
        plt.subplot(num, 1, 2)
        plt.title('stepsize tuning')
        plt.plot(n1, vare_arr, '.-', color='magenta', label='Var[E]/d')
        plt.plot(n1, eps_arr, '.-', color='tab:orange', label = 'eps')
        plt.plot(np.ones(1) * len(loss_arr), jnp.ones(1) * eps, 'o', color = 'tab:red')
        plt.yscale('log')
        plt.legend()

        ### bias ###
        plt.subplot(num, 1, 3)
        plt.plot(bias_avg_arr, color = 'tab:blue', label= 'average')
        plt.plot(bias_max_arr, color = 'tab:red', label= 'max')
        plt.plot(jnp.ones(len(bias_max_arr)) * 1e-2, '--', color = 'black')
        #num_max, num_avg = find_crossing(bias_max_arr, 0.01), find_crossing(bias_avg_arr, 0.01)
        #plt.title('steps to low bias: {0} (max), {1} (avg)'.format(num_max, num_avg))
        plt.legend()
        plt.xlabel('# gradient evaluations')
        plt.ylabel(r'$\mathrm{bias}^2$')
        plt.yscale('log')
        #plt.savefig('tst_ensamble/' + self.Target.name + '.png')
        plt.show()

        return x
        #return steps, x, u, l, g, key, L, eps, sigma



    def further_sample(self, num_steps, num_chains, x_initial='prior', output= 'normal', random_key= None):

        ### sampling ###

        def step(state, useless):
            x, u, g, key = state
            x, u, l, g, kinetic_change, key = self.dynamics(x, u, g, key, L, eps, sigma) # update particles by one step
            return (x, u, g, key), x

        def step_ess(state, useless):
            x, u, g, key = state
            x, u, l, g, kinetic_change, key = self.dynamics(x, u, g, key, L, eps, sigma) # update particles by one step
            return (x, u, g, key), jnp.average(jnp.square(self.Target.transform(x)), axis = 0)

        if output == 'normal':
            X = jax.lax.scan(step, init= (x, u, g, key), xs= None, length= num_steps)[1] #do the sampling
            X = jnp.swapaxes(X, 0, 1)

            f = jnp.average(jnp.average(jnp.square(X), axis= 0), axis= -1)
            burnin2_steps = self.remove_burn_in(f, self.relative_accuracy)

            return X[:, burnin2_steps:, :]

        elif output == 'ess':
            Xsq = jax.lax.scan(step_ess, init=(x, u, g, key), xs=None, length=num_steps)[1] #do the sampling


            # instantaneous moments bias
            bias_d_t = jnp.square(Xsq - self.Target.second_moments[None, :]) / self.Target.variance_second_moments[None, :]
            bias_avg_t, bias_max_t = jnp.average(bias_d_t, axis=-1), jnp.max(bias_d_t, axis=-1)

            # time-averaged moments bias
            burnin2_steps = self.remove_burn_in(jnp.average(Xsq, axis = -1), self.relative_accuracy)
            second_moments = jnp.cumsum(Xsq[burnin2_steps:], axis=0) / jnp.arange(1, num_steps - burnin2_steps+1)[:, None]
            bias_d = jnp.square(second_moments - self.Target.second_moments[None, :]) / self.Target.variance_second_moments[None, :]
            bias_avg, bias_max = jnp.average(bias_d, axis=-1), jnp.max(bias_d, axis=-1)

            plt.plot(bias_avg_t, '--', color='tab:orange', alpha = 0.5, label = 'average')
            plt.plot(bias_max_t, '--', color='tab:red', alpha = 0.5, label = 'max')

            plt.plot(bias_avg, '.-', color='tab:orange', label = 'average')
            plt.plot(bias_max, '.-', color='tab:red', label = 'max')
            plt.yscale('log')
            plt.savefig('tst_ensamble/' + self.Target.name + '/bias2.png')
            plt.show()

            num_t = find_crossing(bias_max_t, 0.01)
            num = find_crossing(bias_max, 0.01)

            return num_t + burnin_steps, num + burnin_steps + burnin2_steps


    def remove_burn_in(self, f, relative_accuracy):


        f_avg = jnp.average(f[-len(f) // 10])
        burnin2_steps = len(f) - find_crossing(-jnp.abs(f[::-1] - f_avg) / f_avg, -relative_accuracy)

        plt.plot(f, '.-')
        plt.plot([0, len(f) - 1], jnp.ones(2) * f_avg, color='black')
        plt.plot([0, len(f) -1], jnp.ones(2) * f_avg*(1+relative_accuracy), color = 'black', alpha= 0.3)
        plt.plot([0, len(f) - 1], jnp.ones(2) * f_avg*(1-relative_accuracy), color= 'black', alpha= 0.3)

        plt.xlabel('steps')
        plt.ylabel('f')
        plt.savefig('tst_ensamble/' + self.Target.name + '/secondary_burn_in2.png')
        plt.close()

        return burnin2_steps

