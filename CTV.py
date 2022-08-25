import numpy as np
import bias


class Sampler:
    """the Canonical T+V Hamiltonian sampler"""

    def __init__(self, Target, eps):
        self.Target, self.eps = Target, eps
        self.step = self.Yoshida_step


    def V(self, x):
        """potential"""
        return -np.exp(-2 * self.Target.nlogp(x) / (self.Target.d - 2))


    def grad_V(self, x):
        """returns g and it's gradient"""
        v = -np.exp(-2 * self.Target.nlogp(x) / (self.Target.d - 2))

        return v, (-2 * v / (self.Target.d - 2)) * self.Target.grad_nlogp(x)


    def initialize(self):
        """randomly draws the initial x from the unit gaussian and the initial momentum with random direction and the magnitude such that total energy is zero"""

        x0, p0 = np.random.normal(size=self.Target.d), np.random.normal(size=self.Target.d)
        p0 *= np.sqrt(-2 * self.V(x0)) / np.sqrt(np.sum(np.square(p0)))
        return x0, p0

    def initialize_momentum(self, x0):
        """randomly draws the initial momentum with random direction and the magnitude such that total energy is zero"""

        p0 = np.random.normal(size=self.Target.d)
        p0 *= np.sqrt(-2 * self.V(x0)) / np.sqrt(np.sum(np.square(p0)))
        return p0


    def hamiltonian(self, x, p):
        """"H = T + V"""
        return 0.5 * np.sum(np.square(p)) + self.V(x)


    def symplectic_Euler_step(self, x, p):
        v, Dv = self.grad_V(x)
        pnew = p - self.eps * Dv
        xnew = x + self.eps * pnew

        return xnew, pnew


    def Yoshida_step(self, x0, p0):
        x = np.copy(x0)
        p = np.copy(p0)
        cbrt_two = np.cbrt(2.0)
        w0, w1 = -cbrt_two / (2.0 - cbrt_two), 1.0 / (2.0 - cbrt_two)
        c = [0.5 * w1, 0.5 * (w0 + w1), 0.5 * (w0 + w1)]
        d = [w1, w0, w1]
        for i in range(3):
            x += c[i] * p * self.eps
            p -= d[i] * self.grad_V(x)[1] * self.eps

        x += c[0] * p * self.eps

        return x, p



    def trajectory(self, time):
        steps = (int)(time / self.eps)
        x0, p0 = self.initialize()
        X = np.zeros((steps + 1, len(x0)))
        P = np.zeros((steps + 1, len(p0)))
        X[0], P[0] = x0, p0

        for i in range(steps):
            X[i + 1], P[i + 1] = self.step(X[i], P[i])

        return X, P

  
    
    # def sample(self, free_time, num_bounces):
    #     x, p = self.initialize()
    #     free_steps = (int)(free_time / self.eps)
    #     X = np.empty((num_bounces * free_steps, self.Target.d))
    #
    #     for k in range(num_bounces):  # number of bounces
    #         # bounce
    #         p = self.isotropic_bounce(p)
    #
    #         # evolve
    #         for i in range(free_steps):
    #             x, p = self.step(x, p)
    #
    #             X[k*free_steps+i, :] = x
    #
    #     return X


    def ess(self, x0, free_length, max_steps= 1000000):
        x = np.copy(x0)
        p = self.initialize_momentum(x)

        F = np.square(x)
        N = 1

        while N < max_steps:  # number of bounces
            # bounce
            p = self.isotropic_bounce(p)
            length_transversed = 0.0
            # evolve
            while length_transversed < free_length:
                xnew, p = self.step(x, p)
                length_transversed += np.sqrt(np.sum(np.square(xnew - x)))
                x= xnew

                F = (N * F + np.square(x)) / (N + 1)
                N += 1

                bias = np.sqrt(np.average(np.square((F - self.Target.variance) / self.Target.variance)))
                #B.append(bias)

                if bias < 0.1:
                    return 200.0 / N

        print('Desired effective sample size not achieved.')
        return 200.0 / max_steps


    def isotropic_bounce(self, p):
        p_size = np.sqrt(np.sum(np.square(p)))
        p_new = np.random.normal(size=self.Target.d)
        p_new *= p_size / np.sqrt(np.sum(np.square(p_new)))
    
        return p_new