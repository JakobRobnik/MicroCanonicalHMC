import numpy as np


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
        u = self.random_unit_vector()

        for k in range(total_steps//free_steps): #number of bounces
            # bounce
            #u = self.half_sphere_bounce(u)
            #u = self.perpendicular_bounce(u)
            u = self.random_unit_vector()

            #evolve
            for i in range(free_steps):
                x, u, g, r = self.step(x, u, g, r)

                X.append(x)
                w.append(np.exp(r) / self.Target.d)


        return np.array(X), np.array(w)


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
