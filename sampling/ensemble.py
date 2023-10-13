import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from sampling.sampler import find_crossing
from sampling import dynamics


lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


#TODO: parallelize paired chains



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
            
        
        self.map_to_worst = target.map_to_worst
        self.maxmin = target.maxmin


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
            l, g = pvgrad(x.reshape(devices, chains // devices, target.d))
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
            self.prior_draw = lambda keys: jnp.array(pvdraw(keys.reshape(devices, chains // devices, 2))).reshape(chains, target.d)


        if hasattr(target, 'second_moments'):
            self.second_moments = target.second_moments
            self.variance_second_moments = target.variance_second_moments

        if hasattr(target, 'name'):
            self.name = target.name


        # self.map_to_worst = target.map_to_worst
        # self.maxmin = target.maxmin
        
        
class Sampler:
    """Ensemble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, chains, 
                 alpha = 1., integrator = 'LF', isotropic_u0 = False, C= 0.1, equipartition_definition = 'full',
                 debug= True, plotdir = None):
        """Args:
                Target: The target distribution class.
                chains: The number of chains to run in parallel. 
                        If multiple devices are available (as seen by jax.local_device_count()),
                        pmap will be used to distribute the computation over the devices.
                        
                alpha: The momentum decoherence scale L = alpha sqrt(d). 
                       Optimal alpha is typically around 1, but can also be 10 or so.
                integrator: in the first stage the leapfrog integrator is used, because it is cheaper to evaluate and high accuracy is not required.
                            In the second stage we use either 'LF' (leapfrog) or 'MN' (minimal norm). MN is expected to perform better, because of the higher required accuracy.
                isotropic_u0: If True, initial velocity will be randomly oriented for each particle. If False, it will be aligned or anti-aligned with the gradient, depending on the initial equipartition.
                C: Proportionality constant for the stepsize.
                equipartition_definition: 'full' or 'diagonal'. See the paper.
                delay_frac: TODO
                neff: effective number of steps used to determine the stepsize in the adaptive step
                n: proportionality factor in converting from bias ratio in successive steps to utility
                debug: If True, the non-jax while loop will be run in sample. Diagnostics like the energy error, bias and stepsize will be saved at each step.
                plotdir: If debug, diagnostics plots will be produced and saved in plotdir.
        """

        self.chains = chains
        
        if jax.local_device_count() == 1: # vmap if only one device is available
            self.Target = vmap_target(Target)
        else: #pmap(vmap()) if multiple devices are available
            self.Target = pmap_target(Target, chains)
        
        
        ### integrator ###
        self.random_unit_vector = dynamics.random_unit_vector(self.Target.d, False)
        self.partially_refresh_momentum = dynamics.partially_refresh_momentum(self.Target.d, False)

        self.dynamics = dynamics.mclmc(dynamics.hamiltonian(integrator= 'LF', grad_nlogp= self.Target.grad_nlogp, d= self.Target.d),
                                       self.partially_refresh_momentum)
        self.grad_evals_per_step = 1
        
        self.integrator = integrator

        # initialization
        self.isotropic_u0 = isotropic_u0 # isotropic direction of the initial velocity (if false, aligned with the gradient)
        self.eps_initial = 0.01 * jnp.sqrt(self.Target.d) # stepsize of the first step

        ### hyperparameters ###
        self.alpha = alpha # momentum decoherence scale L = alpha sqrt(Tr[Sigma]), where Sigma is the covariance matrix
        
        #first stage
        self.C = C # proportionality constant in determining the stepsize (varew \propto C)
        self.delay_frac = 0.05
        
        #second stage
        self.n = 2 # n eps-steps with the precise chain and n-1 eps'-steps with the sloopy chains (eps' = eps n/(n-1) )
        
        self.vmap_twogroup_dynamics = jax.vmap(self.twogroup_dynamics, (None, 0))
        
        
        self.debug = debug 
        if debug:
            if plotdir == None:
                raise ValueError('if debug == True, plotdir must be given.')
            self.plot_dir = plotdir + self.Target.name + '_debug'

        if equipartition_definition == 'full':
            self.equipartition = self.equipartition_fullrank
        elif equipartition_definition == 'diagonal':
            self.equipartition = self.equipartition_diagonal
        else:
            raise ValueError('equipartition_definition = ' + equipartition_definition + "is not a valid option, should be either 'full' or 'diagonal'.")



    def twogroup_dynamics(self, dyn, hyp):
        """do two steps with hyp['eps'] for one group of chains and one step with hyp['eps'] for the other group"""
        
        x, u, l, g, x2, u2, l2, g2, vare, key = dyn['x'], dyn['u'], dyn['l'], dyn['g'], dyn['x2'], dyn['u2'], dyn['l2'], dyn['g2'], dyn['vare'], dyn['key']
        L, eps, sigma = hyp['L'], hyp['eps'], hyp['sigma']
   
        ### one extra step of the precise dynamics ###
        xx, uu, ll, gg, dK, key = self.dynamics(x, u, g, key, L, eps, sigma)
        de = jnp.square(dK + ll - l) / self.Target.d
        varee = jnp.average(de)
        
        ### n-1 steps for both groups
        def step(state, useless):
            x, u, l, g, _, key = self.dynamics(state['x'], state['u'], state['g'], state['key'], L, eps, sigma)
            x2, u2, l2, g2, _, key = self.dynamics(state['x2'], state['u2'], state['g2'], key, L, eps * self.n / (self.n - 1), sigma)
            
            return {'x': x, 'u': u, 'l': l, 'g': g, 'x2': x2, 'u2': u2, 'l2': l2, 'g2': g2, 'key': key}, None

        s = jax.lax.scan(step, init= {'x': xx, 'u': uu, 'l': ll, 'g': gg, 'x2': x2, 'u2': u2, 'l2': l2, 'g2': g2, 'key': key}, length = self.n - 1, xs = None)[0]
                
                    
        ### if there were nans we don't do the step ###
        nonans = jnp.all(jnp.isfinite(s['x'])) & jnp.all(jnp.isfinite(s['x2']))

        x, u, l, g, x2, u2, l2, g2, vare = jax.tree_map(lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old), 
                                                        (s['x'], s['u'], s['l'], s['g'], s['x2'], s['u2'], s['l2'], s['g2'], varee), #if no nans
                                                        (x, u, l, g, x2, u2, l2, g2, vare)) #if nans
        
        return {'x': x, 'u': u, 'l': l, 'g': g, 'x2': x2, 'u2': u2, 'l2': l2, 'g2': g2, 'vare': vare, 'key': key}, nonans
        

    def initialize(self, random_key, x_initial):
        """initialize the ensemble"""

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
    
    

    def initialize_paired(self, num_steps, random_key, x_initial):
            
        delay_num = jnp.rint(self.delay_frac * num_steps / self.grads_per_step).astype(int)
        
        x0, u0, l0, g0, key = self.initialize(random_key, x_initial)

        sigma0 = jnp.ones(self.Target.d)
        L = self.computeL(x0)
        history = jnp.concatenate((jnp.ones(1) * 1e50, jnp.ones(delay_num-1) * jnp.inf)) # loss history
        dyn0 = {'x': x0, 'u': u0, 'l': l0, 'g': g0, 'x2': x0, 'u2': u0, 'l2': l0, 'g2': g0, 'vare': 1e-3, 'key': key} #dynamical variables
        hyp0 = {'L': L, 'eps': self.eps_initial, 'sigma': sigma0} #hyperparameters
        state = (0, history, True, dyn0, hyp0)
        
        return state



    def computeL(self, x):
        return self.alpha * jnp.sqrt(jnp.sum(jnp.square(x))/x.shape[0]) #average over the ensemble, sum over th


    # def equipartition_fullrank(self, x, g, random_key):
    #     """loss = Tr[(1 - E)^T (1 - E)] / d^2
    #         where Eij = <xi gj> is the equipartition patrix.
    #         Loss is computed with the Hutchinson's trick."""

    #     key, key_z = jax.random.split(random_key)
    #     z = jax.random.rademacher(key_z, (100, self.Target.d)) # <z_i z_j> = delta_ij
    #     X = z - (g @ z.T).T @ x / x.shape[0]
    #     return jnp.average(jnp.square(X)) / self.Target.d, key
    
    
    def equipartition_fullrank(self, x, g, random_key):
        """loss = Tr[(1 - E)^T (1 - E)] / d^2
            where Eij = <xi gj> is the equipartition patrix.
            Loss is computed with the Hutchinson's trick."""

        key, key_z = jax.random.split(random_key)
        z = jax.random.rademacher(key_z, (100, self.Target.d)) # <z_i z_j> = delta_ij
        X = z - (g @ z.T).T @ x / x.shape[0]
        Y = z - (x @ z.T).T @ g / x.shape[0]
        return jnp.average(X*Y), key
    
    
    def equipartition_diagonal(self, x, g, random_key):
        return jnp.average(jnp.square(1. - jnp.average(x*g, axis = 0))), random_key
    
        
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
        bias, bias_max = jnp.average(bias_d), jnp.max(bias_d)
        
        adjust = ((self.n - 1.)/self.n) ** 4
        return bias * adjust, bias_max * adjust


    def ground_truth_bias(self, x):

        moments = jnp.average(jnp.square(self.Target.transform(x)), axis = 0)
        bias_d = jnp.square(moments - self.Target.second_moments) / self.Target.variance_second_moments
        bias_avg, bias_max = jnp.average(bias_d), jnp.max(bias_d)

        return bias_avg, bias_max

    
    def compute_diagnostics(self, dyn, hyp):
        x, g, x2, key = dyn['x'], dyn['g'], dyn['x2'], dyn['key']
        
        ### diagnostics ###
        equi_diag, key = self.equipartition_diagonal(x, g, key) # estimate the bias from the equipartition loss
        equi_full, key = self.equipartition_fullrank(x, g, key)
        
        bpair = self.discretization_bias(x, x2) # estimate the bias from the chains with larger stepsize
        btrue = self.ground_truth_bias(x) #actual bias
        
        varew = self.C * jnp.power(equi_full, 3./8.)
        
        dyn['key'] = key
        
        return jnp.array([hyp['eps'], hyp['L'], dyn['vare'], varew, bpair[0], bpair[1], btrue[0], btrue[1], equi_diag, equi_full]), dyn
    


    def sample(self, num_steps, x_initial='prior', random_key= None):
        """Args:
               num_steps: number of integration steps to take during the sampling
               num_chains: number of chains
               x_initial: initial condition for x, shape: (num_chains, d). Defaults to 'prior' in which case the initial condition is drawn with self.Target.prior_draw.
               random_key: jax random seed, defaults to jax.random.PRNGKey(0).
           Returns:
               final x: shape = (num_chains, d)
               If debug, additional diagnostics plots will be produced.
        """

        state = self.initialize_paired(num_steps, random_key, x_initial)
        
        
        def step1(state):
            steps, history, decreassing, dyn, hyp = state
            dyn, nonans = self.twogroup_dynamics(dyn, hyp)
            
            ### hyperparameters for the next step ###
            equi, dyn['key'] = self.equipartition(dyn['x'], dyn['g'], dyn['key']) #estimate the bias from the equipartition loss

            history = jnp.concatenate((jnp.ones(1) * equi, history[:-1]))
            decreassing *= (history[-1] > history[0])

            varew = self.C * jnp.power(equi, 3./8.)
            #stage1 = steps < 500
            #varew = varew * stage1 + (1-stage1) * 1e-4
            eps_factor = jnp.power(varew / dyn['vare'], 1./6.) * nonans + (1-nonans) * 0.5
            eps_factor = jnp.min(jnp.array([jnp.max(jnp.array([eps_factor, 0.3])), 3.])) # eps cannot change by too much
            hyp['eps'] = eps_factor * hyp['eps']
            
            hyp['L'] = self.computeL(dyn['x'])
            
            return (steps + self.n, history, decreassing, dyn, hyp)
        
        
        def step2(state):
            dyn, hyp = state
            dyn, nonans = self.twogroup_dynamics(dyn, hyp)
            
            ### hyperparameters for the next step ###
            varew = 1e-4
            eps_factor = jnp.power(varew / dyn['vare'], 1./6.) * nonans + (1-nonans) * 0.5
            eps_factor = jnp.min(jnp.array([jnp.max(jnp.array([eps_factor, 0.3])), 3.])) # eps cannot change by too much
            hyp['eps'] = eps_factor * hyp['eps']
            
            hyp['L'] = self.computeL(dyn['x'])
            
            return (dyn, hyp)
        
        
        def step1_debug(state):
            state = step1(state)
            steps, history, decreassing, dyn, hyp = state
            # if steps % 100 == 0:
            #     plott(self.Target, steps, dyn)
            _diagnostics, dyn = self.compute_diagnostics(dyn, hyp)
            diagnostics1.append(np.array(_diagnostics))
            return (steps, history, decreassing, dyn, hyp)
            
        
        def step2_debug(state, useless):
            dyn, hyp = step2(state)
            _diagnostics, dyn = self.compute_diagnostics(dyn, hyp)
            return (dyn, hyp), _diagnostics
            

        def cond(state):
            steps, history, decreassing, dyn, hyp = state
            return steps < 500
            #return decreassing & (steps < num_steps * 0.8)
        
        
        if self.debug:
            diagnostics1 = []
            state = mywhile(cond, step1_debug, state)
        else:
            state = jax.lax.while_loop(cond, step1, state)
        
        steps_used, history, decreassing, dyn, hyp = state
        steps_left = num_steps - steps_used
        #loss0 = self.discretization_bias(dyn['x'], dyn['x2'])[1]
        
        
        # def step2(state):
        #     dyn, hyp = state
        #     dyn, nonans = self.twogroup_dynamics(dyn, hyp)
            
        #     varew = 1e-4
            
        #     eps_factor = jnp.power(varew / dyn['vare'], 1./6.) * nonans + (1-nonans) * 0.5
        #     eps_factor = jnp.min(jnp.array([jnp.max(jnp.array([eps_factor, 0.3])), 3.])) # eps cannot change by too much
        #     hyp['eps'] = eps_factor * hyp['eps']
        #     hyp['L'] = self.computeL(dyn['x'])
            
        #     _diagnostics, dyn = self.compute_diagnostics(dyn, hyp)
            
        #     return (dyn, hyp), _diagnostics
        
        
        # if self.integrator == 'MN':            
        
        # self.dynamics = dynamics.mclmc(dynamics.hamiltonian(integrator= 'MN', grad_nlogp= self.Target.grad_nlogp, d= self.Target.d),
        #                                self.partially_refresh_momentum)
        #     self.grads_per_step = 2
        #     self.grads_per_step = 2
        #     #hyp['eps'] *= jnp.sqrt(10.)
        
        
        if self.debug:
            
            state, diagnostics2 = jax.lax.scan(step2_debug, init = (dyn, hyp), length = steps_left // (self.n * self.grads_per_step), xs = None)
        
        
            # state, diagnostics2 = jax.lax.scan(step2, init = (dyn, hyp), length = steps_left // (self.n * self.grads_per_step), xs = None)
            diagnostics = np.concatenate((np.array(diagnostics1), np.array(diagnostics2)))
            self.debug_plots(diagnostics, steps_used)
            
            #plott(self.Target, 800, state[1])
            
            return state[-2]['x']

        
        else:
            raise ValueError('debug = False option is not implemented yet.')     

    

    def debug_plots(self, diagnostics, steps1):
    
        eps, L, vare, varew, bpair_avg, bpair_max, btrue_avg, btrue_max, equi_diag, equi_full = diagnostics.T
        
        print(find_crossing(btrue_max, 0.01) * self.n)
        
        end_stage1 = lambda: plt.plot(steps1 * np.ones(2), plt.gca().get_ylim(), color = 'grey', alpha = 0.2)
        
        num = 4
        plt.figure(figsize= (6, 3 * num))

        steps = jnp.arange(0, self.n*len(eps), self.n)
        
        ### bias ###
        plt.subplot(num, 1, 1)
        plt.title('bias')
        
        # true
        plt.plot(steps, btrue_avg, color = 'tab:blue', label= 'average')
        plt.plot(steps, btrue_max, color = 'tab:red', label = 'max')
        
        # pair
        plt.plot(steps, bpair_avg, '--', color = 'tab:blue')
        plt.plot(steps, bpair_max, '--', color = 'tab:red')
        plt.plot([], [], '--', color = 'grey', label= 'pair')

        # equipartition
        plt.plot(steps, equi_diag, color = 'tab:olive', label = 'diagonal equipartition')
        plt.plot(steps, equi_full, '--', color = 'tab:green', label = 'full rank equipartition')
        
        plt.plot(steps, jnp.ones(steps.shape) * 1e-2, '-', color = 'black')
        plt.legend()
        plt.ylabel(r'$\mathrm{bias}^2$')
        plt.ylim(1e-4, 1e2)
        plt.yscale('log')
        end_stage1()
        
        ### stepsize tuning ###
        plt.subplot(num, 1, 2)
        plt.title('energy error')
        plt.plot(steps, vare, '.', color='tab:blue', alpha = 0.5, label='measured')
        plt.plot(steps, varew, '.-', color='purple', label = 'targeted')
        plt.ylabel("Var[E]/d")
        plt.yscale('log')
        end_stage1()
        plt.legend()
        
        plt.subplot(num, 1, 3)
        plt.title('stepsize')
        plt.plot(steps, eps, '.-', color='royalblue')
        plt.ylabel(r"$\epsilon$")
        plt.yscale('log')
        end_stage1()
        
        ### L tuning ###
        plt.subplot(num, 1, 4)
        plt.title('L')
        L0 = self.alpha * jnp.sqrt(jnp.sum(self.Target.second_moments))
        plt.plot(steps, L, '.-', color='tab:orange')
        plt.plot(steps, L0 * jnp.ones(steps.shape), '-', color='black')
        end_stage1()
        plt.ylabel("L")
        #plt.yscale('log')
        plt.xlabel('# gradient evaluations')
        plt.tight_layout()
        plt.savefig(self.plot_dir)
        plt.close()



def mywhile(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def plott(target, steps, dyn):
    
    plt.title('steps = ' + str(steps))
    
    for i in range(2):
        w = ['', '2'][i]
        y = dyn['x' + w] @ target.map_to_worst.T
        plt.plot(y[:, 0], y[:, 1], '.', color= ['teal', 'tab:red'][i], label = 'eps'+w)
    
    
    # ground truth
    from matplotlib.patches import Ellipse
    a, b = target.maxmin
    ax = plt.gca()
    for i in range(2):
        factor, alpha, word = [(1.52, 1., '68'), (2.48, 0.5, '95')][i]
        ax.add_patch(Ellipse(xy=np.zeros(2), width=a*factor*2, height=b*factor*2, facecolor = "None",  edgecolor = 'grey', alpha = alpha, label = 'exact posterior\n('+word+'% confidence)'))
        
    plt.legend()
    plt.xlabel('widest dimension')
    plt.ylabel('narrowest dimension')
    
    plt.tight_layout()
    plt.savefig('img/different_stepsizes_'+str(steps) + '.png')
    plt.close()