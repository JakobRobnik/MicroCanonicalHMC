import numpy as np
import bias
from pdb import set_trace as bp

class Sampler:
    """the esh sampler"""

    def __init__(self, Target, eps):
        self.Target, self.eps = Target, eps
        self.stop_bouncing_threshold = .8 #value of target^{2/d} below which stop bouncing


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

        #Boundary condition: a reflective box
        if np.isinf(xnew).any(): ## gives an array with the same shape as xnew and True at the position of infinite component
            unew = u-2*u*np.isinf(xnew)  ## bounce back the momentum in the infinite direction. Is it better to use uhalf here?
            xnew = x
            gg_new = gg
            rnew = r
        else:
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

        E = [self.Target.nlogp(x0), ]

        #evolve
        for i in range(total_steps):
            x, u, g, r = self.step(x, u, g, r)

            X.append(x)
            P.append(np.exp(r) * u)

            E.append(self.Target.d * r + self.Target.nlogp(x))

        return np.arange(total_steps) * self.eps, X, P, E


    def sample(self, x0, free_steps, total_steps):
        """Samples from the target distribution.
            Args:
                x0: initial condition for x (initial p will be of unit length with random direction)
                free_steps: how many steps are performed before a bounce occurs
                total_steps: how many steps in total

            Returns:
                X: array of samples of shape (total_steps, d)
                w: weights of samples, an array of shape (total_steps, )
        """

        #initial conditions
        x = x0
        g = self.Target.grad_nlogp(x0)
        X = [x0, ]
        r = 0.0
        w = [np.exp(r) / self.Target.d, ]
        
        u = np.random.normal(size = self.Target.d)
        #u = self.random_unit_vector()

        for k in range(total_steps//free_steps): #number of bounces
            # bounce
            #u = self.half_sphere_bounce(u)
            #u = self.perpendicular_bounce(u)
            
            
            #This is target^{2/d}

            if -self.Target.nlogp(x) > (self.Target.d/2.0)*np.log(self.stop_bouncing_threshold):
                u = self.random_unit_vector()

            #evolve
            for i in range(free_steps):
                x, u, g, r = self.step(x, u, g, r)

                X.append(x)
                w.append(np.exp(r) / self.Target.d)


        return np.array(X), np.array(w)


    def ess_with_averaging(self, x0_arr, free_steps):
        """x0_arr must have shape (num_averaging, d)"""

        num_averaging = len(x0_arr)

        required_number_estimate = (int)(200 / self.ess(x0_arr[0], free_steps)) #run ess once to get an estimate of the required number of steps

        u = self.ess(x0_arr[0], free_steps, max_steps=required_number_estimate, terminate=False)

        B = np.array([self.ess(x0_arr[i], free_steps, max_steps= required_number_estimate, terminate= False) for i in range(num_averaging)]) #array of biases for different realizations

        print(np.shape(B))
        b_median = np.median(B, axis=0)
        b_upper_quarter = [np.median((B[:, i])[B[:, i] > b_median[i]]) for i in range(len(b_median))]
        b_lower_quarter = [np.median((B[:, i])[B[:, i] < b_median[i]]) for i in range(len(b_median))]

        return bias.ess_cutoff_crossing(b_median, np.ones(len(B)))[0],\
               bias.ess_cutoff_crossing(b_upper_quarter, np.ones(len(B)))[0] / np.sqrt(num_averaging),\
               bias.ess_cutoff_crossing(b_lower_quarter, np.ones(len(B)))[0] / np.sqrt(num_averaging)


    def ess(self, x0, free_steps, max_steps = 1000000, terminate = True):

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
        u = np.random.normal(size = self.Target.d)

        F = np.square(x) #<f(x)> estimate after one step
        W = w #sum of weights

        if not terminate:
            bias_arr = []

        for k in range(max_steps // free_steps):  # number of bounces
            # bounce
            
            #This is target^{2/d}
            if -self.Target.nlogp(x) > (self.Target.d/2.0)*np.log(self.stop_bouncing_threshold):
                u = self.random_unit_vector()

            # evolve
            for i in range(free_steps):
                x, u, g, r = self.step(x, u, g, r)
                w= np.exp(r) / self.Target.d

                F = (F + (w * np.square(x) / W)) / (1 + (w / W))
                W += w

                bias = np.sqrt(np.average(np.square((F - self.Target.variance) / self.Target.variance)))

                if terminate:
                    if bias < 0.1:
                        return 200.0 / (k*free_steps + i)

                else:
                    bias_arr.append(bias)


        if terminate:
            print('Maximum number of steps exceeded')
            return 0.0

        else:
            return bias_arr



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



    def mode_mixing(self, free_steps, max_steps = 10000000):

        """Determines the mode mixing property by monitoring the number of steps the sampler spends in one mode.
            Args:
                x0: initial condition for x (initial p will be of unit length with random direction)
                free_steps: how many steps are performed before a bounce occurs

            Returns:
                ess: average number of steps spent in a mode (time of computation is such that the mode is switched 10 times
        """



        # initial conditions

        x = np.random.normal(size= self.Target.d)
        x[0] = self.Target.mu
        g = self.Target.grad_nlogp(x)
        r = 0.0
        w = np.exp(r) / self.Target.d
        u = np.random.normal(size = self.Target.d)


        L = []
        current_sign = 1
        island_size = 1

        for k in range(max_steps // free_steps):  # number of bounces
            # bounce
            
            #This is target^{2/d}. 

            #print(np.exp(-(2.0*self.Target.nlogp(x))/self.Target.d))
            #if np.exp(-self.Target.nlogp(x))>.8:
                #print("check: ", np.exp(-(2.0*self.Target.nlogp(x))/self.Target.d))
            #    print("check: ", -self.Target.nlogp(x))
            #print(np.exp(-self.Target.nlogp(x)))
            
            #if np.exp(-(2.0*self.Target.nlogp(x))/self.Target.d) > self.stop_bouncing_threshold:
            if -self.Target.nlogp(x) > (self.Target.d/2.0)*np.log(self.stop_bouncing_threshold):
                u = self.random_unit_vector()


            # evolve
            for i in range(free_steps):
                x, u, g, r = self.step(x, u, g, r)
                w= np.exp(r) / self.Target.d

                sign = np.sign(x[0])
                if sign != current_sign:
                    L.append(island_size)
                    island_size = 1
                    current_sign = sign

                else:
                    island_size += 1

                if len(L) == 10:
                    return np.average(L)


        print('Maximum number of steps exceeded, num_islands = ' + str(len(L)))
        return max_steps


