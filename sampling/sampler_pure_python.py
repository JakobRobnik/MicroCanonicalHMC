import numpy as np

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator


class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, L = None, eps = None, integrator = 'MN'):
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
        else:
            print('integrator = ' + integrator + 'is not a valid option.')


        self.sigma = np.ones(self.Target.d)

        if (not (L is None)) and (not (eps is None)):
            self.set_hyperparameters(L, eps)

        else:
            self.set_hyperparameters(np.sqrt(Target.d), np.sqrt(Target.d) * 0.1)




    def set_hyperparameters(self, L, eps):
        self.L = L
        self.eps= eps
        self.nu = np.sqrt((np.exp(2 * self.eps / L) - 1.0) / self.Target.d)


    def random_unit_vector(self):
        """Generates a random (isotropic) unit vector."""
        u = self.rng.randn(shape = (self.Target.d, ))
        u /= np.sqrt(np.sum(np.square(u)))
        return u


    def partially_refresh_momentum(self, u):
        """Adds a small noise to u and normalizes."""
        z = self.nu * self.rng.randn(shape = (self.Target.d, ))

        return (u + z) / np.sqrt(np.sum(np.square(u + z)))


    def update_momentum(self, eps, g, u):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
        g_norm = np.sqrt(np.sum(np.square(g)))
        e = - g / g_norm
        ue = np.dot(u, e)
        sh = np.sinh(eps * g_norm / (self.Target.d-1))
        ch = np.cosh(eps * g_norm / (self.Target.d-1))
        th = np.tanh(eps * g_norm / (self.Target.d-1))
        delta_r = np.log(ch) + np.log1p(ue * th)

        return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), delta_r


    def leapfrog(self, x, u, g):
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

        return xx, uu, l, gg, kinetic_change


    def minimal_norm(self, x, u, g):
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

        return xx, uu, ll, gg, kinetic_change


    def dynamics(self, x, u, g):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change = self.hamiltonian_dynamics(x, u, g)

        # bounce
        uu = self.partially_refresh_momentum(uu)

        return xx, uu, ll, gg, kinetic_change



    def get_initial_conditions(self, x_initial, random_number_generator):

        ### random number generator ###
        if random_number_generator is None:
            self.rng = np.random.RandomState(seed= 0)

        else:
            self.rng = random_number_generator

        ### initial conditions ###
        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                x = self.Target.prior_draw()
            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else: #initial x is given
            x = x_initial
        l, g = self.Target.grad_nlogp(x)

        u = self.random_unit_vector()
        #u = - g / np.sqrt(np.sum(np.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, l, g


    def sample(self, num_steps, x_initial = 'prior', random_number_generator= None, output = 'normal', thinning= 1, remove_burn_in= True):
        """Args:
               num_steps: number of integration steps to take.

               x_initial: initial condition for x, shape: (d, ). Defaults to 'prior' in which case the initial condition is drawn from the prior distribution (self.Target.prior_draw).

               random_number_generator: numpy random number generator, defaults to np.random.RandomState(seed= 0)

               output: determines the output of the function:
                        'normal': returns Target.transform of the samples (to save memory), shape: (num_samples, len(Target.transform(x)))
                        'full': returns the full samples and the energy at each step, shape: (num_samples, Target.d), (num_samples, )
                        'energy': returns the transformed samples and the energy at each step, shape: (num_samples, len(Target.transform(x))), (num_samples, )
                        'final state': only returns the final state of the chain, shape: (Target.d, )
                        'ess': only ouputs the Effective Sample Size, float. In this case, self.Target.variance = <x_i^2>_true should be defined.

               thinning: integer for thinning the chains (every n-th sample is returned), defaults to 1 (no thinning).
                        In unadjusted methods such as MCHMC, all samples contribute to the posterior and thining degrades the quality of the posterior.
                        If thining << # steps needed for one effective sample the loss is not too large.
                        However, in general we recommend no thining, as it can often be avoided by using Target.transform.

               remove_burn_in: removes the samples during the burn-in phase. The end of burn-in is determined by settling of the -log p.
        """


        def step(state):
            """Tracks transform(x) as a function of number of iterations"""

            x, u, l, g, E = state
            xx, uu, ll, gg, kinetic_change = self.dynamics(x, u, g)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE), (self.Target.transform(xx), ll, EE)



        ### initial conditions ###
        x, u, l, g = self.get_initial_conditions(x_initial, random_number_generator)

        ### sampling ###

        if output == 'ess':  # only track the bias

            b = np.empty(num_steps)
            E, W, F2 = 0.0, 1.0, np.square(x)

            for i in range(num_steps):

                x, u, l, g, kinetic_change = self.dynamics(x, u, l, g)
                F2 = (W * F2 + np.square(self.Target.transform(x))) / (W + 1)  # Update <f(x)> with a Kalman filter
                W += 1
                bias = np.sqrt(np.average(np.square((F2 - self.Target.variance) / self.Target.variance)))  # bias = np.average((F2 - self.Target.variance) / self.Target.variance)
                b[i] = bias

            no_nans = 1-np.any(np.isnan(b))
            cutoff_reached = b[-1] < 0.1

            # plt.plot(bias, '.')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

            return ess_cutoff_crossing(b) * no_nans * cutoff_reached / self.grad_evals_per_step #return 0 if there are nans, or if the bias cutoff was not reached

        elif output == 'full': #track everything
            E = 0.0
            xarr, larr, Earr = np.empty((num_steps, self.Target.d)), np.empty(num_steps), np.empty(num_steps)
            for i in range(num_steps):
                x, u, ll, g, kinetic_change = self.dynamics(x, u, g)
                E += kinetic_change + ll - l
                l = ll
                xarr[i], larr[i], Earr[i] = x, l, E


            if remove_burn_in:
                index_burnin = burn_in_ending(larr)
            else:
                index_burnin = 0
            return xarr[index_burnin::thinning, :], Earr[index_burnin::thinning]


        else: # track the transform(x) and the energy
            E = 0.0
            xarr, larr, Earr = np.empty((num_steps, len(self.Target.transform(x)))), np.empty(num_steps), np.empty(num_steps)
            for i in range(num_steps):
                x, u, ll, g, kinetic_change = self.dynamics(x, u, g)
                E += kinetic_change + ll - l
                l = ll
                xarr[i], larr[i], Earr[i] = self.Target.transform(x), l, E

            if remove_burn_in:
                index_burnin = burn_in_ending(larr)
            else:
                index_burnin = 0

            if output == 'final state': #only return the final x
                return state[0]
            elif output == 'energy': #return the samples X and the energy E
                return xarr[index_burnin::thinning, :], Earr[index_burnin::thinning]
            elif output == 'normal': #return the samples X
                return xarr[index_burnin::thinning]
            else:
                raise ValueError('output = ' + output + 'is not a valid argument for the Sampler.sample')




    def tune_hyperparameters(self, x_initial = 'prior', random_number_generator= None, varE_wanted = 0.0005, dialog = False, initial_eps= 0.6):

        varE_wanted = 0.0005             # targeted energy variance per dimension
        burn_in, samples = 2000, 1000

        ### random number generator ###
        if random_number_generator is None:
            self.rng = np.random.RandomState(seed=0)

        else:
            self.rng = random_number_generator


        self.set_hyperparameters(np.sqrt(self.Target.d), initial_eps)

        x0 = self.sample(burn_in, x_initial= x_initial, random_number_generator = self.rng, output = 'final state', remove_burn_in= False)
        props = (np.inf, 0.0, False)
        if dialog:
            print('Hyperparameter tuning (first stage)')

        def tuning_step(props):

            eps_inappropriate, eps_appropriate, success = props

            # get a small number of samples
            X, E = self.sample(samples, x_initial= x0, random_number_generator = self.rng, output= 'full', remove_burn_in= False)

            # remove large jumps in the energy
            E -= np.average(E)
            #E = remove_jumps(E)

            ### compute quantities of interest ###

            # typical size of the posterior
            x1 = np.average(X, axis= 0) #first moments
            x2 = np.average(np.square(X), axis=0) #second moments
            sigma = np.sqrt(np.average(x2 - np.square(x1))) #average variance over the dimensions

            # energy fluctuations
            varE = np.std(E)**2 / self.Target.d #variance per dimension
            no_divergences = np.isfinite(varE)

            ### update the hyperparameters ###

            if no_divergences: #appropriate eps

                L_new = sigma * np.sqrt(self.Target.d)
                eps_new = self.eps * np.power(varE_wanted / varE, 0.25) #assume var[E] ~ eps^4
                success = np.abs(1.0 - varE / varE_wanted) < 0.2 #we are done

                if self.eps > eps_appropriate: #it is the largest appropriate eps found so far
                    eps_appropriate = self.eps

            else: #inappropriate eps

                L_new = self.L

                if self.eps < eps_inappropriate: #it is the smallest inappropriatre eps found so far
                    eps_inappropriate = self.eps

                eps_new = np.inf #will be lowered later


            # if suggested new eps is inappropriate we switch to bisection
            if eps_new > eps_inappropriate:
                eps_new = 0.5 * (eps_inappropriate + eps_appropriate)

            self.set_hyperparameters(L_new, eps_new)


            if dialog:
                word = 'bisection' if (not no_divergences) else 'update'
                print('varE / varE wanted: {} ---'.format(np.round(varE / varE_wanted, 4)) + word + '---> eps: {}, sigma = L / sqrt(d): {}'.format(np.round(eps_new, 3), np.round(L_new / np.sqrt(self.Target.d), 3)))

            return eps_inappropriate, eps_appropriate, success


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
            X[n[i-1]:n[i]] = self.sample(n[i] - n[i-1], x_initial= X[n[i-1]-1], random_number_generator = self.rng, output= 'full', remove_burn_in = False)[0]
            ESS = ess_corr(X[:n[i]])
            if dialog:
                print('n = {0}, ESS = {1}'.format(n[i], ESS))
            if n[i] > 10.0 / ESS:
                break

        L = 0.4 * self.eps / ESS # = 0.4 * correlation length
        self.set_hyperparameters(L, self.eps)

        if dialog:
            print('L / sqrt(d) = {}, ESS(correlations) = {}'.format(L / np.sqrt(self.Target.d), ESS))
            print('-------------')


def find_crossing(array, cutoff):

    for m in range(len(array)):
        if array[m] < cutoff:
            break

    return m #the smallest M such that array[m] > cutoff for all m < M.



def ess_cutoff_crossing(bias):

    return 200.0 / find_crossing(bias, 0.1)



def point_reduction(num_points, reduction_factor):
    """reduces the number of points for plotting purposes"""

    indexes = np.concatenate((np.arange(1, 1 + num_points // reduction_factor, dtype=int),
                              np.arange(1 + num_points // reduction_factor, num_points, reduction_factor, dtype=int)))
    return indexes



def burn_in_ending(loss):
    loss_avg = np.median(loss[len(loss)//2:])
    return 2 * find_crossing(loss - loss_avg, 0.0) #we add a safety factor of 2

    ### plot the removal ###
    # t= np.arange(len(loss))
    # plt.plot(t[:i*2], loss[:i*2], color= 'tab:red')
    # plt.plot(t[i*2:], loss[i*2:], color= 'tab:blue')
    # plt.yscale('log')
    # plt.show()


