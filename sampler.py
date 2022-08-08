import numpy as np
from numba import njit  #I use numba, which makes the python evaluations significantly faster (as if it was a c code). If you do not want to download numba comment out the @njit decorators.

from scipy.stats import special_ortho_group

#from numba import jitclass          # import the decorator
#from numba import int32, float32    # import the types


#@jitclass
class Sampler():

    #@njit
    def trajectory(self, x0, p0, steps):
        X = np.zeros((steps+1, len(x0)))
        P = np.zeros((steps+1, len(p0)))
        X[0], P[0] = x0, p0

        for i in range(steps):
            X[i+1], P[i+1] = self.step(X[i], P[i])

        return X, P

    def trajectory_fixed_time(self, x0, p0, total_time):
        X = [x0, ]
        P = [p0, ]
        x = np.copy(x0)
        p = np.copy(p0)
        self.time = 0.0
        while self.time < total_time:
            x, p = self.step(x, p)
            X.append(x)
            P.append(p)

        return np.array(X), np.array(P)


    #@njit
    # def time_evolve(self, x0, p0, E, dt, steps):
    #     x, p = x0, p0
    #     for i in range(steps):
    #         x, p = self.step(x, p, E, dt)
    #     return x, p

    #@njit
    def billiard_evolve(self, x0, p0, free_time, rot):
        x, p = x0, p0
        #X = np.ones((len(rot)*num_free_steps, len(x0)))
        X = []
        time = []
        t0 = 0.0
        self.time = 0.0
        #X = np.empty((len(rot), len(x0)))
        for k in range(len(rot)): #number of bounces
            # bounce
            p = self.transform_initial_condition(x, np.dot(rot[k], self.transform_to_synchronized(x, p)))

            #evolve
            while self.time < free_time:
                x, p = self.step(x, p)

                #X[k*num_free_steps+i, :] = x
                X.append(x)
                time.append(t0 + self.time)

            t0 += self.time
            self.time = 0.0

        return np.array(time), np.array(X)




    def sample(self, free_time, num_bounces):
        x0, p0 = self.initialize()
        self.E = self.hamiltonian(x0, p0)

        rot = np.array([special_ortho_group.rvs(len(x0)) for k in range(num_bounces)])  #generate random rotations

        t, x = self.billiard_evolve(x0, self.transform_initial_condition(x0, p0), free_time, rot)

        return t, x


#
# spec = [
#     ('d', int32),               # a simple scalar field
#     ('array', float32[:]),          # an array field
# ]



class Seperable(Sampler):

    def __init__(self, negative_log_p, grad_negative_log_p, d, step):
        """Args:
             negative_log_p: - log p of the target distribution
             d: dimension of x"""

        self.nlogp = negative_log_p
        self.grad_nlogp = grad_negative_log_p
        self.d = d #dimensionality of x
        self.E = 0.0
        self.step_name = step
        if step == 'Euler' or step == 'Leapfrog':
            self.step = self.symplectic_Euler_step

        elif step == 'Yoshida':
            self.step = self.Yoshida_step

        else:
            print("step name '" + step + "' is not valid.")
            raise (ValueError)

    #@njit
    def V(self, x):
        """potential"""
        return -np.exp(-2 * self.nlogp(x) / (self.d-2))

    #@njit
    def grad_V(self, x):
        """returns g and it's gradient"""
        v = -np.exp(-2 * self.nlogp(x) / (self.d-2))

        return v, (-2 * v / (self.d-2)) * self.grad_nlogp(x)


    def initialize(self):
        """randomly draws the initial x from the unit gaussian and the initial momentum with random direction and the magnitude such that total energy is zero"""

        x0, p0 = np.random.normal(size=self.d), np.random.normal(size=self.d)
        p0 *= np.sqrt(-2 * self.V(x0)) / np.sqrt(np.sum(np.square(p0)))
        return x0, p0

    def transform_initial_condition(self, x, p):
        """shifts the given p_i to p_{i-1/2} as needed for the leapfrog"""

        return p + 0.5 * self.dt * self.grad_V(x)[1] if self.step_name == 'Leapfrog' else p


    def transform_to_synchronized(self, x, p):
        """shifts the given p_{i-1/2} to p_{i} as needed for computing the energy"""
        return p - 0.5 * self.dt * self.grad_V(x)[1] if self.step_name == 'Leapfrog' else p

    #@njit
    def hamiltonian(self, x, p):
        """"H = g(x) p^2"""
        return 0.5* np.sum(np.square(p)) + self.V(x)


    #@njit
    def symplectic_Euler_step(self, x, p):
        v, Dv = self.grad_V(x)
        pnew = p - self.dt * Dv
        xnew = x + self.dt * pnew
        self.time += self.dt

        return xnew, pnew


    def Yoshida_step(self, x0, p0):
        x = np.copy(x0)
        p = np.copy(p0)
        cbrt_two = np.cbrt(2.0)
        w0, w1 = -cbrt_two / (2.0 - cbrt_two), 1.0 / (2.0 - cbrt_two)
        c = [0.5*w1, 0.5*(w0+w1), 0.5*(w0+w1)]
        d = [w1, w0, w1]
        for i in range(3):
            x += c[i] * p * self.dt
            p -= d[i] * self.grad_V(x)[1] * self.dt

        x += c[0] * p * self.dt
        self.time += self.dt

        return x, p




class Ruthless(Sampler):

    def __init__(self, negative_log_p, grad_negative_log_p, d, step):
        """Args:
             negative_log_p: - log p of the target distribution
             d: dimension of x"""

        self.nlogp = negative_log_p
        self.grad_nlogp = grad_negative_log_p
        self.d = d
        self.E = 0.0
        self.step_name = step
        self.time = 0.0

        if step == 'Euler':
            self.step = self.symplectic_Euler_step

        elif step == 'Leapfrog':
            self.step = self.leapfrog_step

        elif step == 'Adaptive Leapfrog':
            self.step = self.adaptive_leapfrog_step

        elif step == 'EPSSP':
            self.step = self.EPSSP_step

        elif step == 'RK4':
            self.step = self.RK4_step

        else:
            print("step name '" + step + "' is not valid.")
            raise (ValueError)

    # @njit
    def g(self, x):
        """inverse mass"""
        return np.exp(2 * self.nlogp(x) / self.d)

    # @njit
    def grad_g(self, x):
        """returns g and it's gradient"""
        gg = np.exp(2 * self.nlogp(x) / self.d)
        return gg, (2 * gg / self.d) * self.grad_nlogp(x)

    # @njit
    def grad_g(self, x):
        """returns g and it's gradient"""
        gg = np.exp(2 * self.nlogp(x) / self.d)
        return gg, (2 * gg / self.d) * self.grad_nlogp(x)

    # @njit
    def hamiltonian(self, x, p):
        """"H = g(x) p^2"""
        return self.g(x) * np.sum(np.square(p))

    # @njit
    def symplectic_Euler_step(self, x, p):
        gg, Dg = self.grad_g(x)
        pnew = p - self.dt * Dg * self.E / gg
        xnew = x + self.dt * 2 * gg * pnew
        self.time += self.dt
        return xnew, pnew

    # iterative (momentum) leapfrog functions

    def transform_initial_condition(self, x, p):
        """for Leapfrog: shifts the given p_i to v_{i-1/2} as needed for the leapfrog
           for anything else: returns p"""
        if self.step_name == 'Leapfrog' or self.step_name == 'Adaptive Leapfrog':
            gg, Dg = self.grad_g(x)
            v = 2 * gg * p
            return v - 0.5 * self.dt * (-2 * self.E * Dg + np.dot(v, Dg) * v / gg)

        else:
            return p


    def transform_to_synchronized(self, x, v):
        """for Leapfrog: shifts the given v_{i-1/2} to p_{i} as needed for computing the energy
            for anything else: returns v"""

        if self.step_name == 'Leapfrog' or self.step_name == 'Adaptive Leapfrog':
            gg, Dg = self.grad_g(x)
            return self.midpoint_velocity(v, gg, Dg) / (2 * gg)
        else:
            return v


    def midpoint_velocity(self, v, gg, Dg, error_flag = False):
        max_iter, tol = 10, 1e-12  # terminates iteration when one of the criterion is met
        vmid = np.copy(v)
        for i in range(max_iter):
            # vmid_previous = np.copy(vmid)
            vmid = v + 0.5 * self.dt * (-2 * self.E * Dg + np.dot(vmid, Dg) * vmid / gg)  # if np.sum(np.abs(vmid_previous - vmid)) < tol:  #     #print('tolerance reached')  #     print(np.sum(np.abs(vmid_previous - vmid)) , i)  #     return vmid
            if error_flag:
                print(vmid)

        if error_flag:
            print('------')
        # residual = np.sqrt(np.sum(np.square(vmid - (v + 0.5 * self.dt * (-2 * self.E * Dg + np.dot(vmid, Dg) * vmid / gg)))))
        # if residual > tol:
        #     print('max iterations reached, tol = {0}'.format(residual))

        return vmid


    # @njit
    def leapfrog_step(self, x, v):
        gg, Dg = self.grad_g(x)
        # iterate to find vn

        midpoint_v = self.midpoint_velocity(v, gg, Dg)#, x[0] < -2.8)
        vnew = 2 * midpoint_v - v
        xnew = x + vnew * self.dt
        self.time += self.dt

        return xnew, vnew


    def adaptive_leapfrog_step(self, x, v):
        gg, Dg = self.grad_g(x)

        midpoint_v = self.midpoint_velocity(v, gg, Dg)

        dt_previous = self.dt
        dt = self.dx / np.sqrt(np.sum(np.square(midpoint_v)))
        self.dt = dt
        self.time += dt

        vnew = (1+ dt/dt_previous) * midpoint_v - (dt / dt_previous) * v
        xnew = x + vnew * dt

        return xnew, vnew


    def EPSSP_step(self, x, p):
        """Implementation of the second order symplectic integrator proposed in https://arxiv.org/pdf/2111.10915.pdf
            EPSSP = Extended Phase Space with Symmetric Projection"""

        def extended_step(x, xtilde, p, ptilde):
            """Equation 6"""

            gg, Dg = self.grad_g(xtilde)
            x1 = x + 2 * gg * p * 0.5 * self.dt
            p1tilde = ptilde - Dg * np.sum(np.square(p)) * 0.5 * self.dt

            gg, Dg = self.grad_g(x1)
            Xtilde = xtilde + 2 * gg * p1tilde * self.dt
            P = p - Dg * np.sum(np.square(p1tilde)) * self.dt

            gg, Dg = self.grad_g(Xtilde)
            X = x1 + 2 * gg * P * 0.5 * self.dt
            Ptilde = p1tilde - Dg * np.sum(np.square(P)) * 0.5 * self.dt

            return X, Xtilde, P, Ptilde

        def iteration(mu, nu):
            """Equation 23 (and 22)"""
            X, Xtilde, P, Ptilde = extended_step(x + mu, x - mu, p + nu, p - nu)

            f_mu, f_nu = X - Xtilde + 2 * mu, P - Ptilde + 2 * nu

            return mu - 0.25 * f_mu, nu - 0.25 * f_nu, X + mu, P + nu

        # initialize
        mu, nu = np.zeros(len(x)), np.zeros(len(p))
        X, P = np.copy(x), np.copy(p)
        tol = 1e-10
        # iterate
        for i in range(2):
            X0, P0 = np.copy(X), np.copy(P)
            mu, nu, X, P = iteration(mu, nu)

            tolerance = 0.5 * (np.average(np.square(X - X0)) + np.average(np.square(P - P0)))
            # print(tolerance)
            if tolerance < tol:  # we have converged
                break
        self.time += self.dt
        return X, P


    def RK4_step(self, x, p):

        def Hamilton_eqs(x, p):
            """returns [dx/dt, dp/dt]"""
            gg, Dg = self.grad_g(x)

            return [2 * gg * p, - Dg * self.E / gg]

        k1 = Hamilton_eqs(x, p)
        k2 = Hamilton_eqs(x + 0.5 * self.dt * k1[0], p + 0.5 * self.dt * k1[1])
        k3 = Hamilton_eqs(x + 0.5 * self.dt * k2[0], p + 0.5 * self.dt * k2[1])
        k4 = Hamilton_eqs(x + self.dt * k3[0], p + self.dt * k3[1])
        self.time += self.dt

        return x + self.dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0, p + self.dt * (
                    k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0



# class BI(Sampler):
#
#     def __init__(self, negative_log_p, grad_negative_log_p, d):
#         """Args:
#              negative_log_p: - log p of the target distribution
#              d: dimension of x"""
#
#         self.nlogp = negative_log_p
#         self.grad_nlogp = grad_negative_log_p
#         self.d = d
#
#     @njit
#     def c_sq(self, x):
#         """squared speed of light"""
#         return np.exp(2 * self.nlogp(x) / self.d)
#
#     @njit
#     def grad_c_sq(self, x):
#         """returns c^2 and it's gradient"""
#         gg = np.exp(2 * self.nlogp(x) / self.d)
#         return gg, (2 * gg / self.d) * self.grad_nlogp(x)
#
#     @njit
#     def hamiltonian(self, x, p):
#         """"H = g(x) p^2"""
#         cc = self.c_sq(x)
#         return np.sqrt(cc(x) ** 2 + cc * np.sum(np.square(p)))
#
#
#     @njit
#     def symplectic_Euler_step(self, x, p):
#         cc, Dcc = self.grad_g(x)
#         pnew = p - self.dt * 0.5 * Dcc * (self.E / (cc + 1e-8) + cc / self.E)
#         xnew = x + self.dt * cc * pnew / self.E
#         return xnew, pnew
#


