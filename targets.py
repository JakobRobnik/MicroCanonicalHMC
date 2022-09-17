import numpy as np
from scipy.stats import norm

### targets that we want to sample from ###


class StandardNormal():
    """an example of a target class: Standard Normal distribution in d dimensions"""

    def __init__(self, d):
        self.d = d
        self.variance = np.ones(d)
        self.gaussianization_available = False

    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * np.sum(np.square(x))

    def grad_nlogp(self, x):
        return x

    def draw(self, num_samples):
        """direct sampler from a target"""
        return np.random.normal(size = (num_samples, self.d))



class IllConditionedGaussian():
    """Gaussian distribution. Covariance matrix has eigenvalues equally spaced in log-space, going from 1/condition_bnumber to condition_number."""

    def __init__(self, d, condition_number):
        self.d = d
        self.variance = np.logspace(-0.5*np.log10(condition_number), 0.5*np.log10(condition_number), d)
        self.gaussianization_available = False


    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * np.sum(np.square(x) / self.variance)

    def grad_nlogp(self, x):
        return x / self.variance

    def draw(self, num_samples):
        """direct sampler from a target"""
        return np.random.normal(size = (num_samples, self.d)) * np.sqrt(self.variance)


class BiModal():
    """A Gaussian mixture p(x) = f N(x | mu1, sigma1) + (1-f) N(x | mu2, sigma2)."""

    def __init__(self, d, mu1, mu2, sigma1, sigma2, f):

        self.d = d

        self.mu1 = np.insert(np.zeros(d-1), 0, mu1)
        self.mu2 = np.insert(np.zeros(d - 1), 0, mu2)
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.f = f
        self.gaussianization_available = False


    def nlogp(self, x):
        """- log p of the target distribution"""

        N1 = (1.0 - self.f) * np.exp(-0.5 * np.sum(np.square(x - self.mu1)) / self.sigma1 ** 2) / np.power(2 * np.pi * self.sigma1 ** 2, self.d * 0.5)
        N2 = self.f * np.exp(-0.5 * np.sum(np.square(x - self.mu2)) / self.sigma2 ** 2) / np.power(2 * np.pi * self.sigma2 ** 2, self.d * 0.5)

        return -np.log(N1 + N2)


    def grad_nlogp(self, x):

        N1 = (1.0 - self.f) * np.exp(-0.5 * np.sum(np.square(x - self.mu1)) / self.sigma1 ** 2) / np.power(2 * np.pi * self.sigma1 ** 2, self.d * 0.5)
        N2 = self.f * np.exp(-0.5 * np.sum(np.square(x - self.mu2)) / self.sigma2 ** 2) / np.power(2 * np.pi * self.sigma2 ** 2, self.d * 0.5)

        return (N1 * (x - self.mu1) / self.sigma1**2 + N2 * (x - self.mu2) / self.sigma2**2) / (N1 + N2)


    def draw(self, num_samples):
        """direct sampler from a target"""
        X = np.random.normal(size = (num_samples, self.d))
        mask = np.random.uniform(0, 1, num_samples) < self.f
        X[mask] = (X[mask] * self.sigma) + self.mu

        return X




class BiModalEqual():
    """Mixture of two Gaussians, one centered at x = [mu/2, 0, 0, ...], the other at x = [-mu/2, 0, 0, ...].
        Both have equal probability mass."""

    def __init__(self, d, mu):

        self.d = d
        self.mu = mu
        self.gaussianization_available = False

    def nlogp(self, x):
        """- log p of the target distribution"""

        return 0.5 * np.sum(np.square(x)) - np.log(np.cosh(0.5*self.mu*x[0]))


    def grad_nlogp(self, x):
        grad = np.copy(x)
        grad[0] -= 0.5*self.mu * np.tanh(0.5 * self.mu * x[0])

        return grad

    def draw(self, num_samples):
        """direct sampler from a target"""
        X = np.random.normal(size = (num_samples, self.d))
        mask = np.random.uniform(0, 1, num_samples) < 0.5
        X[mask, 0] += 0.5*self.mu
        X[~mask, 0] -= 0.5 * self.mu

        return X




class Funnel():
    """Noise-less funnel"""

    def __init__(self, d):

        self.d = d
        self.sigma_theta= 3.0
        self.variance = np.ones(d)
        self.gaussianization_available = True


    def nlogp(self, x):
        """ - log p of the target distribution
                x = [z_0, z_1, ... z_{d-1}, theta] """
        theta = x[-1]
        return 0.5* np.square(theta / self.sigma_theta) + 0.5 * (self.d - 1) * x[-1] + 0.5 * np.exp(-theta) * np.sum(np.square(x[:-1]))


    def grad_nlogp(self, x):
        theta = x[-1]
        return np.append(np.exp(-theta) * x[:self.d - 1], (theta / self.sigma_theta**2) + 0.5 * (self.d - 1) - 0.5 * np.sum(np.square(x[:-1])) * np.exp(-theta))


    def draw(self, num_samples):
        """direct sampler from a target"""
        return self.inverse_gaussianize(np.random.normal(size = (num_samples, self.d)))


    def inverse_gaussianize(self, xtilde):
        x= np.empty(np.shape(xtilde))
        x[:, -1] = 3 * xtilde[:, -1]
        x[:, :-1] = xtilde[:, -1] * np.exp(1.5*xtilde[:, -1])
        return x


    def gaussianize(self, x):
        xtilde = np.empty(np.shape(x))
        xtilde[-1] =  x.T[-1] / 3.0
        xtilde[:-1] = x.T[:-1] * np.exp(-0.5*x.T[-1])
        return xtilde.T



class Rosenbrock():
    """Mixture of two Gaussians, one centered at x = [mu/2, 0, 0, ...], the other at x = [-mu/2, 0, 0, ...]"""

    def __init__(self, d):

        self.d = d
        self.Q, var_x, var_y = 0.1, 2.0, 10.098433122783046 #var_y is computed numerically (see compute_variance below)
        #self.Q, var_x, var_y = 0.5, 2.0, 10.498957879911487
        self.variance = np.concatenate((var_x * np.ones(d//2), var_y * np.ones(d//2)))
        self.gaussianization_available = False


    def nlogp(self, x):
        """- log p of the target distribution"""
        X, Y = x[:self.d//2], x[self.d//2:]
        return 0.5 * np.sum(np.square(X - 1.0) + np.square(np.square(X) - Y) / self.Q)


    def grad_nlogp(self, x):
        X, Y = x[:self.d//2], x[self.d//2:]

        return np.concatenate((X - 1.0 + 2*(np.square(X) - Y) * X / self.Q, (Y - np.square(X)) / self.Q))

    def draw(self, num):
        n = self.d // 2
        X= np.empty((num, self.d))
        X[:, :n] = np.random.normal(loc= 1.0, scale= 1.0, size= (num, n))
        X[:, n:] = np.random.normal(loc= np.square(X[:, :n]), scale= np.sqrt(self.Q), size= (num, n))

        return X


    def compute_variance(self):
        num = 100000000
        x = np.random.normal(loc=1.0, scale=1.0, size=num)
        y = np.random.normal(loc=np.square(x), scale=np.sqrt(self.Q), size=num)

        var_x = np.sum(np.square(x)) / (num - 1)
        var_y = np.sum(np.square(y)) / (num - 1)
        print(var_x, var_y)


class DiagonalPreconditioned():
    """A target instance which takes some other target and preconditions it"""

    def __init__(self, Target, a):
        """ Target: original target
            a: scaling vector (sqrt(variance of x)), such that x' = x / a"""

        self.Target = Target
        self.d= Target.d
        self.a = a
        self.variance = Target.variance / np.square(a)
        self.gaussianization_available = False


    def nlogp(self, x):
        return self.Target.nlogp(self.a * x)

    def grad_nlogp(self, x):
        return self.Target.grad_nlogp(self.a * x) * self.a

    def draw(self, num):
        return self.Target.draw(num) / self.a




def check_gradient(target, x):
    """check the analytical gradient of the target at point x"""

    from scipy import optimize

    approx_grad= optimize.approx_fprime(x, target.nlogp, 1e-8)

    grad= target.grad_nlogp(x)

    print('numerical grad: ', approx_grad)
    print('analytical grad: ', grad)
    print('ratio: ', grad / approx_grad)




if __name__ == '__main__':

    d = 10

    #target.compute_variance()
    # d = 36

