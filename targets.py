import numpy as np

### targets that we want to sample from ###

class StandardNormal():
    """an example of a target class: Standard Normal distribution in d dimensions"""

    def __init__(self, d):
        self.d = d
        self.variance = np.ones(d)

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
        self.variance = np.logspace(-np.log10(condition_number), np.log10(condition_number), d)

    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * np.sum(np.square(x) / self.variance)

    def grad_nlogp(self, x):
        return x / self.variance

    def draw(self, num_samples):
        """direct sampler from a target"""
        return np.random.normal(size = (num_samples, self.d)) * np.sqrt(self.variance)



class Funnel():
    """Noise-less funnel"""

    def __init__(self, d):

        self.d = d
        self.sigma_theta= 3.0


    def nlogp(self, x):
        """ - log p of the target distribution
                x = [z_0, z_1, ... z_{d-1}, theta] """

        return 0.5* np.square(x[-1] / self.sigma_theta) + 0.25 * (self.d - 1) * x[-1] + 0.5 * np.exp(-x[-1]) * np.sum(np.square(x[:-1]))


    def grad_nlogp(self, x):
        theta = x[-1]
        return np.append(np.exp(-theta) * x[:self.d - 1], (theta / self.sigma_theta**2) + 0.25 * (self.d - 1) - 0.5 * np.sum(np.square(x[:-1])) * np.exp(-theta))


    def draw(self, num_samples):
        """direct sampler from a target"""
        return self.inverse_gaussianize(np.random.normal(size = (num_samples, self.d)))


    def inverse_gaussianize(self, xtilde):
        x= np.empty(np.shape(xtilde))
        x[:, -1] = 3 * xtilde[:, -1]
        x[:, -1] = xtilde[:, -1] * np.exp(0.5*x[:, -1])



class BiModal():
    """Mixture of two Gaussians, one centered at x = [mu/2, 0, 0, ...], the other at x = [-mu/2, 0, 0, ...]"""

    def __init__(self, d, mu):

        self.d = d
        self.mu = mu

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
        mask = np.random.uniform(num_samples) < 0.5
        X[mask, 0] += 0.5*self.mu
        X[~mask, 0] -= 0.5 * self.mu

        return X


def check_gradient(target, x):
    """check the analytical gradient of the target at point x"""

    from scipy import optimize

    approx_grad= optimize.approx_fprime(x, target.nlogp, 1e-8)

    grad= target.grad_nlogp(x)

    print('numerical grad: ', approx_grad)
    print('analytical grad: ', grad)
    print('ratio: ', grad / approx_grad)



if __name__ == '__main__':
    #target = Funnel(d= 20)
    #check_gradient(target, np.random.normal(size = 20))

    target = Funnel(d = 2)

    z0, theta = np.linspace(-20, 20, 1000), np.linspace(-8, 5, 1000)

    Z0, Theta = np.meshgrid(z0, theta)
    logp = np.array([[-target.nlogp(np.array([z0[i], theta[j]])) for i in range(len(z0))] for j in range(len(Theta))])
    cutoff = -20.0
    logp[logp< cutoff] = cutoff
    import matplotlib.pyplot as plt
    plt.contourf(Z0, Theta, logp, levels= 20)#, levels = [0, 1, 10, 100, 1000])
    plt.colorbar()
    plt.savefig('funnel_2d_logp.png')
    plt.show()
