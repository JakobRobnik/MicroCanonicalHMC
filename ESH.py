import matplotlib.pyplot as plt
import numpy as np
import bias




class Sampler:
    """the esh sampler"""

    def __init__(self, Target, eps):
        self.Target, self.eps = Target, eps


    def step(self, x, u, gg, r):
        """Equation 7 in https://proceedings.neurips.cc/paper/2021/file/5b970a1d9be0fd100063fd6cd688b73e-Paper.pdf"""

        def f(eps, g, u):
            g_norm = np.sqrt(np.sum(np.square(g)))
            e = - g / g_norm
            ue = np.dot(u, e)
            sh = np.sinh(eps * g_norm / self.Target.d)
            ch = np.cosh(eps * g_norm / self.Target.d)
            return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh)

        uhalf = f(self.eps * 0.5, gg, u)
        xnew = x + self.eps * uhalf
        gg_new = self.Target.grad_nlogp(xnew)
        unew = f(self.eps * 0.5, gg_new, uhalf)
        rnew = r - self.eps * 0.5 *(np.dot(u, gg) + np.dot(unew, gg_new)) / self.Target.d

        return xnew, unew, gg_new, rnew


    def trajectory(self, x0, total_steps):
        """Computes the trajectory in the phase space and it's energy as a function of time.
            Args:
                x0: initial condition for x (initial p will be of unit length with random direction)
                total_steps: how many steps in total

            Returns:
                t: rescaled time, an array of shape (total_steps, )
                x: position, an array of shape (total_steps, d)
                p: momentum, an array of shape (total_steps, d)
                E: energy, an array of shape (total_steps, )
        """

        #initial conditions
        x = x0
        g = self.Target.grad_nlogp(x0)
        X = [x0, ]
        r = 0.0

        u = np.random.normal(size = self.Target.d)
        u /= np.sqrt(np.sum(np.square(u)))
        P = [u, ]

        E = [0.0* self.Target.d + self.Target.nlogp(x0), ]

        #evolve
        for i in range(total_steps):
            x, u, g, r = self.step(x, u, g, r)

            X.append(x)
            P.append(np.exp(r) * u)

            E.append(self.Target.d * r + self.Target.nlogp(x))

        return np.arange(total_steps) * self.eps, X, P, E


    # def ess_with_averaging(self, x0_arr, free_steps):
    #     """x0_arr must have shape (num_averaging, d)"""
    #
    #     num_averaging = len(x0_arr)
    #
    #     required_number_estimate = (int)(200 / self.ess(x0_arr[0], free_steps)) #run ess once to get an estimate of the required number of steps
    #
    #     u = self.ess(x0_arr[0], free_steps, max_steps=required_number_estimate, terminate=False)
    #
    #     B = np.array([self.ess(x0_arr[i], free_steps, max_steps= required_number_estimate, terminate= False) for i in range(num_averaging)]) #array of biases for different realizations
    #
    #     print(np.shape(B))
    #     b_median = np.median(B, axis=0)
    #     b_upper_quarter = [np.median((B[:, i])[B[:, i] > b_median[i]]) for i in range(len(b_median))]
    #     b_lower_quarter = [np.median((B[:, i])[B[:, i] < b_median[i]]) for i in range(len(b_median))]
    #
    #     return bias.ess_cutoff_crossing(b_median, np.ones(len(B)))[0],\
    #            bias.ess_cutoff_crossing(b_upper_quarter, np.ones(len(B)))[0] / np.sqrt(num_averaging),\
    #            bias.ess_cutoff_crossing(b_lower_quarter, np.ones(len(B)))[0] / np.sqrt(num_averaging)


    def sample(self, x0, bounce_length, max_steps= 1000000, prerun_steps= 0, track= 'ESS', langevin_eta= 0):

        """Determines the effective sample size by monitoring the bias in the estimated variance.
            Args:
                x0: initial condition for x (initial p will be of unit length with random direction)
                free_steps: how many steps are performed before a bounce occurs

            Returns:
                ess: effective sample size
        """

        # initial conditions
        x = x0
        g = self.Target.grad_nlogp(x0)
        r = 0.0
        w = np.exp(r) / self.Target.d
        u = self.random_unit_vector()#- g / np.sqrt(np.sum(np.square(g))) #initialize momentum in the direction of the gradient of log p


        ### bounce program ###

        if prerun_steps != 0:
            bounce_tracker= Distance_equally_spaced(self.eps, bounce_length)
            W = 0.0
            num_steps = 0

            while num_steps < prerun_steps:

                # do a step
                x, u, g, r = self.step(x, u, g, r)
                w = np.exp(r) / self.Target.d
                W += w
                num_steps += 1

                if bounce_tracker.update(w):
                    u = self.random_unit_vector()

            w_typical_set = W / prerun_steps
            bounce_time = bounce_length * w_typical_set
            bounce_tracker = Time_equally_spaced(self.eps, bounce_time)

            #initialize again
            x = x0
            g = self.Target.grad_nlogp(x0)
            r = 0.0
            w = np.exp(r) / self.Target.d
            u = self.random_unit_vector()#- g / np.sqrt(np.sum(np.square(g)))  # initialize momentum in the direction of the gradient of log p


        else:
            bounce_tracker= Distance_equally_spaced(self.eps, bounce_length)

            #the target must be preconditioned and normalized and even then this is only an approximation
            # w_typical_set = np.exp(-0.5 + self.Target.nlogp(x0) / self.Target.d) / self.Target.d
            # bounce_time = bounce_length * w_typical_set
            # bounce_tracker = Time_equally_spaced(self.eps, bounce_time)



        ### quantities to track (to save memory we do not store full x(t) ) ###
        if track == 'ESS':
            tracker= Ess(x, w, self.Target.variance, self.Target.gaussianize if self.Target.gaussianization_available else (lambda x: x))
        elif track == 'FullBias':
            tracker = FullBias(x, w, self.Target.variance, max_steps, self.Target.gaussianize if self.Target.gaussianization_available else (lambda x: x))
        elif track == 'FullTrajectory':
            tracker= Full_trajectory(x, w, 100000)
        elif track == 'ModeMixing':
            tracker= ModeMixing(x)
        elif track == 'Marginal1d':
            tracker= Marginal1d(bins, num_steps, lambda x: x[0])
        else:
            print(str(track) + ' is not a valid track option.')
            exit()

        num_steps = 0
        #time = []
        while num_steps < max_steps:

            #do a step
            x, u, g, r = self.step(x, u, g, r)
            w = np.exp(r) / self.Target.d
            num_steps += 1
            
            if tracker.update(x, w): #update tracker
                return tracker.results()

            if langevin_eta != 0:
                u = self.langevin_update(u, langevin_eta)

            else:
                if bounce_tracker.update(w):
                    u = self.random_unit_vector()


        print('Maximum number of steps exceeded')
        return tracker.results()




    def fine_tune(self, show):
        np.random.seed(0)
        L = np.array([3, 5, 7 ])
        epsilon = np.array([0.5, 0.7, 1.0])
        max_steps = 50000

        x0 = self.Target.draw(1)[0]  # we draw an initial condition from the target
        ess = np.empty((len(L), len(epsilon)))


        for i in range(len(L)):
            for j in range(len(epsilon)):
                self.eps = epsilon[j]
                length = L[i] * np.sqrt(self.Target.d)
                print(L[i], self.eps)
                ess[i, j] = 2.0 / (self.sample(x0, length, max_steps=max_steps, prerun_steps=500, track= 'FullBias')[-1]**2 * max_steps)


        I = np.argmax(ess)
        eps_best = epsilon[I % (len(epsilon))]
        L_best = L[I // len(epsilon)]
        ess_best = np.max(ess)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(ess)
            fig.colorbar(cax)
            plt.title(r'ESS = {0} (with optimal L = {1}, $\epsilon$ = {2})'.format(ess_best, *np.round([L_best, eps_best], 2)))

            ax.set_xticklabels([''] + [str(e) for e in epsilon])
            ax.set_yticklabels([''] + [str(l) for l in L])
            ax.xaxis.set_label_position('top')
            ax.set_xlabel(r"$\epsilon$")
            ax.set_ylabel("L")
            ax.invert_yaxis()

            plt.show()

        return ess_best, L_best, eps_best



    def random_unit_vector(self):
        u = np.random.normal(size=self.Target.d)
        u /= np.sqrt(np.sum(np.square(u)))
        return u

    def langevin_update(self, u, eta):
        unew = u + np.random.normal(size=len(u)) * eta
        return unew / np.sqrt(np.sum(np.square(unew)))


    def half_sphere_bounce(self, u):
        v= self.random_unit_vector()

        projection = np.dot(u, v)

        if projection < 0.0: #force the component of the new velocity along the old velocity to be positive
            return v - 2 * projection * u
        else:
            return v

    def perpendicular_bounce(self, u):
        v = self.random_unit_vector()

        projection = np.dot(u, v)

        return v - projection * u



### bounce trackers ###
# The bounce program is implemented as a class which must at a minimum have defined the function update(w).
# This function returns True if the bounce should occur and False otherwise.

class Distance_equally_spaced():
    """the bounces occur equally spaced in the distance travelled"""

    def __init__(self, eps, tau_max):
        self.num_steps = 0
        self.eps = eps
        self.max_steps = (int)(tau_max / eps)

    def update(self, w):
        self.num_steps += 1

        if self.num_steps < self.max_steps:
            return False #don't do a bounce

        else:
            self.num_steps = 0 #reset the bounce condition
            return True #do a bounce


class Time_equally_spaced():
    """the bounces occur equally spaced in the time passed"""

    def __init__(self, eps, time_max):
        self.eps = eps
        self.time = 0.0
        self.time_max = time_max


    def update(self, w):
        self.time += self.eps * w

        if self.time < self.time_max:
            return False

        else:
            self.time = 0.0
            return True



### quantities to keep track of during sampling ###
# The tracked qunatity is implemented as a class which must at a minimum have defined functions:
# - update(self, x w): updates the quantity being tracked after each step. It returns True if the sampling can be finished and False otherwise (e.g. bias is below 0.1)
# - results(self): returns the tracked quantity. This function is called after update returns True.


class Full_trajectory():
    """Stores x(t). Contains all the information but can be memory intensive."""

    def __init__(self, x, w, max_steps):
        self.max_steps = max_steps
        self.X = np.empty((max_steps, len(x)))
        self.W = np.empty(max_steps)
        self.X[0], self.W[0] = x, w
        self.num_steps = 0


    def update(self, x, w):
        self.num_steps += 1
        self.X[self.num_steps] = x
        self.W[self.num_steps] = w

        return self.num_steps > self.max_steps-2


    def results(self):
        return self.X, self.W



class Ess():
    """effective sample size computed from the bias"""

    def __init__(self, x, w, variance, gaussianization):
        self.gaussianization = gaussianization
        self.variance = variance
        self.F = np.square(gaussianization(x))  # <f(x)> estimate after one step, in this case f(x) = x^2
        self.W = w  # sum of weights
        self.bias = np.sqrt(np.average(np.square((self.F - self.variance) / self.variance)))
        self.num_steps = 0


    def update(self, x, w):
        self.F = (self.F + (w * np.square(self.gaussianization(x)) / self.W)) / (1 + (w / self.W))  # Update <f(x)> with a Kalman filter
        self.W += w
        self.num_steps += 1

        self.bias = np.sqrt(np.average(np.square((self.F - self.variance) / self.variance)))

        return self.bias < 0.1


    def results(self):
        return 200.0 / self.num_steps #ess



class FullBias():
    """Variance bias"""

    def __init__(self, x, w, variance, num_max, gaussianization):
        self.variance = variance #defined in the gaussianized coordinates
        self.gaussianization = gaussianization
        self.F = np.square(gaussianization(x))  # <f(x)> estimate after one step, in this case f(x) = x^2
        self.W = w  # sum of weights
        self.bias = np.empty(num_max)
        self.bias[0] = np.sqrt(np.average(np.square((self.F - self.variance) / self.variance)))
        self.num_steps = 0
        self.num_max = num_max

    def update(self, x, w):
        self.F = (self.F + (w * np.square(self.gaussianization(x)) / self.W)) / (1 + (w / self.W))  # Update <f(x)> with a Kalman filter
        self.W += w
        self.num_steps += 1

        self.bias[self.num_steps] = np.sqrt(np.average(np.square((self.F - self.variance) / self.variance)))

        return self.num_steps == self.num_max - 1


    def results(self):
        return self.bias #ess




class ExpectedValue():
    """expected value of some quantities f(x)"""

    def __init__(self, x, w, f, max_num_steps):
        self.F = f(x)  # <f(x)> estimate after one step, can be a vector
        self.W = w  # sum of weights
        self.max_num_steps = max_num_steps
        self.num_steps = 0


    def update(self, x, w):
        self.F = (self.F + (w * self.f(x) / self.W)) / (1 + (w / self.W))  # Update <f(x)> with a Kalman filter
        self.W += w
        self.num_steps += 1

        return self.num_steps < self.max_num_steps


    def results(self):
        return self.F



class ModeMixing():
    """how long does it take to switch between modes (average number of steps per switch after 10 switches)"""

    def __init__(self, x):

        self.L = []
        self.current_sign = np.sign(x[0])
        self.island_size = 1


    def update(self, x, w):

        sign = np.sign(x[0])
        if sign != self.current_sign:
            self.L.append(self.island_size)
            self.island_size = 1
            self.current_sign = sign

        else:
            self.island_size += 1

        return len(self.L) > 9 #it finishes when 10 switches between the modes have been made.


    def results(self):
        return np.average(self.L)




class Marginal1d():
    """pdf of some marginal quantity f(x)"""

    def __init__(self, bins, total_steps, f):
        self.bins, self.total_steps = bins, total_steps
        self.count = np.zeros(len(bins) + 1)
        self.f = f
        self.step_count = 0

    def which_bin(self, x):

        for i in range(len(self.bins)):
            if x > self.bins[i][0] and x < self.bins[i][1]:
                return i

        return len(self.bins)  # if it is not in any of the bins

    def update(self, x, w):
        self.count[self.which_bin(self.f(x))] += w
        self.step_count += 1
        return self.step_count < self.total_steps


    def results(self):
        return self.count