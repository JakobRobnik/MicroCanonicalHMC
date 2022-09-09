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


    def sample(self, x0, time_bounce, max_steps = 1000000):

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
        u = - g / np.sqrt(np.sum(np.square(g))) #initialize momentum in the direction of the gradient of log p

        ### bounce program ###
        bounce_tracker= Rescaled_time_bounces(self.eps, time_bounce)
        #bounce_tracker = rescaled_time_bounces(self.eps, time_bounce, self.Target)

        ### quantities to track (to save memory we do not store full x(t) ###
        tracker= Ess(x, w, self.Target.variance)
        #tracker= Full_trajectory(x, w, max_steps?)
        #tracker= Mode_mixing(x)
        #tracker= Marginal_1d(bins, num_steps, lambda x: x[0])

        num_steps = 0

        while num_steps < max_steps:

            #do a step
            x, u, g, r = self.step(x, u, g, r)
            w = np.exp(r) / self.Target.d
            num_steps += 1
            
            if tracker.update(x, w): #update tracker
                return tracker.results()

            if bounce_tracker.update(x): #perhaps do a bounce
                u = self.random_unit_vector()


        print('Maximum number of steps exceeded')
        return tracker.results()



    def random_unit_vector(self):
        u = np.random.normal(size=self.Target.d)
        u /= np.sqrt(np.sum(np.square(u)))
        return u

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

class Rescaled_time_bounces():

    def __init__(self, eps, tau_max):
        self.num_steps = 0
        self.max_steps = (int)(tau_max / eps)

    def update(self, x):
        self.num_steps += 1

        if self.num_steps < self.max_steps:
            return False

        else:
            self.num_steps = 0
            return True


class Hamiltonian_time_bounces():

    def __init__(self, x, time_max, Target):
        self.x = x
        self.Target = Target
        self.time = 0.0
        self.max_steps = time_max

    def update(self, x):
        self.time += np.sqrt(np.sum(np.square(x - self.x))) * np.exp(- 2 * self.Target.nlogp / self.Target.d)
        self.x = x

        if self.num_steps < self.max_steps:
            return False

        else:
            self.time = 0.0
            return True



### quantities to keep during sampling ###

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
        self.X[self.num_steps, :] = x
        self.W[self.num_steps] = w

        return self.num_steps < self.max_steps


    def results(self):
        return self.X, self.W



class Ess():
    """effective sample size computed from the bias"""

    def __init__(self, x, w, variance):
        self.variance = variance
        self.F = np.square(x)  # <f(x)> estimate after one step, in this case f(x) = x^2
        self.W = w  # sum of weights
        self.bias = np.sqrt(np.average(np.square((self.F - self.variance) / self.variance)))
        self.num_steps = 0


    def update(self, x, w):
        self.F = (self.F + (w * np.square(x) / self.W)) / (1 + (w / self.W))  # Update <f(x)> with a Kalman filter
        self.W += w
        self.num_steps += 1

        self.bias = np.sqrt(np.average(np.square((self.F - self.variance) / self.variance)))

        return self.bias < 0.1


    def results(self):
        return 200.0 / self.num_steps #ess




class Expected_value():
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



class Mode_mixing():
    """how long does it take to switch between modes"""

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

        return len(self.L) > 9


    def results(self):
        return np.average(self.L)




class Marginal_1d():
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