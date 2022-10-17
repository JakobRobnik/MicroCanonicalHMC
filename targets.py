import numpy as np
from scipy.stats import norm
import jax
import jax.numpy as jnp

from numpyro.examples.datasets import SP500, load_dataset
from numpyro.distributions import StudentT


### targets that we want to sample from ###


class StandardNormal():
    """an example of a target class: Standard Normal distribution in d dimensions"""

    def __init__(self, d):
        self.d = d
        self.variance = jnp.ones(d)

    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * jnp.sum(jnp.square(x), axis= -1)

    def grad_nlogp(self, x):
        return x

    def transform(self, x):
        return x

    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64')



class IllConditionedGaussian():
    """Gaussian distribution. Covariance matrix has eigenvalues equally spaced in log-space, going from 1/condition_bnumber^1/2 to condition_number^1/2."""

    def __init__(self, d, condition_number):
        self.d = d
        self.variance = jnp.logspace(-0.5*jnp.log10(condition_number), 0.5*jnp.log10(condition_number), d)


    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * jnp.sum(jnp.square(x) / self.variance, axis= -1)

    def grad_nlogp(self, x):
        return x / self.variance

    def transform(self, x):
        return x

    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64')



class IllConditionedGaussianGamma():
    """Gaussian distribution. Covariance matrix has eigenvalues drawn from the Gamma distribution."""

    def __init__(self):
        self.d = 1000

        seed = 1234
        np.random.seed(seed)

        rng = np.random.RandomState(seed=10)
        eigs = np.sort(rng.gamma(shape=1.0, scale=1., size=self.d)) #get the variance
        R, _ = np.linalg.qr(rng.randn(self.d, self.d)) #random rotation
        self.Hessian = (R * eigs).dot(R.T)

        self.variance = jnp.diagonal((R * eigs ** -1).dot(R.T))


    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * x.T @ self.Hessian @ x

    def grad_nlogp(self, x):
        return self.Hessian @ x

    def transform(self, x):
        return x

    def prior_draw(self, key):

        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64') #* 100


class BiModal():
    """A Gaussian mixture p(x) = f N(x | mu1, sigma1) + (1-f) N(x | mu2, sigma2)."""

    def __init__(self, d, mu1, mu2, sigma1, sigma2, f):

        self.d = d

        self.mu1 = jnp.insert(jnp.zeros(d-1), 0, mu1)
        self.mu2 = jnp.insert(jnp.zeros(d - 1), 0, mu2)
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.f = f
        self.variance = jnp.insert(jnp.ones(d-1) * ((1 - f) * sigma1**2 + f * sigma2**2), 0, (1-f)*(sigma1**2 + mu1**2) + f*(sigma2**2 + mu2**2))


    def nlogp(self, x):
        """- log p of the target distribution"""

        N1 = (1.0 - self.f) * jnp.exp(-0.5 * jnp.sum(jnp.square(x - self.mu1), axis= -1) / self.sigma1 ** 2) / jnp.power(2 * jnp.pi * self.sigma1 ** 2, self.d * 0.5)
        N2 = self.f * jnp.exp(-0.5 * jnp.sum(jnp.square(x - self.mu2), axis= -1) / self.sigma2 ** 2) / jnp.power(2 * jnp.pi * self.sigma2 ** 2, self.d * 0.5)

        return -jnp.log(N1 + N2)


    def grad_nlogp(self, x):

        N1 = (1.0 - self.f) * jnp.exp(-0.5 * jnp.sum(jnp.square(x - self.mu1), axis= -1) / self.sigma1 ** 2) / jnp.power(2 * jnp.pi * self.sigma1 ** 2, self.d * 0.5)
        N2 = self.f * jnp.exp(-0.5 * jnp.sum(jnp.square(x - self.mu2), axis= -1) / self.sigma2 ** 2) / jnp.power(2 * jnp.pi * self.sigma2 ** 2, self.d * 0.5)

        return (N1 * (x - self.mu1) / self.sigma1**2 + N2 * (x - self.mu2) / self.sigma2**2) / (N1 + N2)


    def draw(self, num_samples):
        """direct sampler from a target"""
        X = np.random.normal(size = (num_samples, self.d))
        mask = np.random.uniform(0, 1, num_samples) < self.f
        X[mask, :] = (X[mask, :] * self.sigma2) + self.mu2
        X[~mask] = (X[~mask] * self.sigma1) + self.mu1

        return X


    def transform(self, x):
        return x

    def prior_draw(self, key):
        z = jax.random.normal(key, shape = (self.d, ), dtype = 'float64') *self.sigma1
        #z= z.at[0].set(self.mu1 + z[0])
        return z


class BiModalEqual():
    """Mixture of two Gaussians, one centered at x = [mu/2, 0, 0, ...], the other at x = [-mu/2, 0, 0, ...].
        Both have equal probability mass."""

    def __init__(self, d, mu):

        self.d = d
        self.mu = mu

    def nlogp(self, x):
        """- log p of the target distribution"""

        return 0.5 * jnp.sum(jnp.square(x), axis= -1) - jnp.log(jnp.cosh(0.5*self.mu*x[0])) + 0.5* self.d * jnp.log(2 * jnp.pi) + self.mu**2 / 8.0


    def grad_nlogp(self, x):
        grad = jnp.copy(x)
        grad[0] -= 0.5*self.mu * jnp.tanh(0.5 * self.mu * x[0])

        return grad

    def draw(self, num_samples):
        """direct sampler from a target"""
        X = np.random.normal(size = (num_samples, self.d))
        mask = np.random.uniform(0, 1, num_samples) < 0.5
        X[mask, 0] += 0.5*self.mu
        X[~mask, 0] -= 0.5 * self.mu

        return X

    def transform(self, x):
        return x


class Funnel():
    """Noise-less funnel"""

    def __init__(self, d):

        self.d = d
        self.sigma_theta= 3.0
        self.variance = jnp.ones(d)


    def nlogp(self, x):
        """ - log p of the target distribution
                x = [z_0, z_1, ... z_{d-1}, theta] """
        theta = x[-1]
        X = x[..., :- 1]

        return 0.5* jnp.square(theta / self.sigma_theta) + 0.5 * (self.d - 1) * theta + 0.5 * jnp.exp(-theta) * jnp.sum(jnp.square(X), axis = -1)


    def grad_nlogp(self, x):
        theta = x[..., -1]
        X = x[..., :- 1]
        return jnp.append(jnp.exp(-theta) * X, (theta / self.sigma_theta**2) + 0.5 * (self.d - 1) - 0.5 * jnp.sum(jnp.square(X), axis =-1) * jnp.exp(-theta))


    def draw(self, num_samples):
        """direct sampler from a target"""
        return self.inverse_transform(np.random.normal(size = (num_samples, self.d)))


    def inverse_transform(self, xtilde):
        x= jnp.empty(jnp.shape(xtilde))
        x[:, -1] = 3 * xtilde[:, -1]
        x[:, :-1] = xtilde[:, -1] * jnp.exp(1.5*xtilde[:, -1])
        return x


    def transform(self, x):
        """gaussianization"""
        xtilde = jnp.empty(x.shape)
        xtilde = xtilde.at[-1].set(x.T[-1] / 3.0)
        xtilde = xtilde.at[:-1].set(x.T[:-1] * jnp.exp(-0.5*x.T[-1]))
        return xtilde.T


    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64')



class Rosenbrock():
    """Mixture of two Gaussians, one centered at x = [mu/2, 0, 0, ...], the other at x = [-mu/2, 0, 0, ...]"""

    def __init__(self, d):

        self.d = d
        self.Q, var_x, var_y = 0.1, 2.0, 10.098433122783046 #var_y is computed numerically (see compute_variance below)
        #self.Q, var_x, var_y = 0.5, 2.0, 10.498957879911487
        self.variance = jnp.concatenate((var_x * jnp.ones(d//2), var_y * jnp.ones(d//2)))


    def nlogp(self, x):
        """- log p of the target distribution"""
        X, Y = x[..., :self.d//2], x[..., self.d//2:]
        return 0.5 * jnp.sum(jnp.square(X - 1.0) + jnp.square(jnp.square(X) - Y) / self.Q, axis= -1)


    def grad_nlogp(self, x):
        X, Y = x[..., :self.d//2], x[..., self.d//2:]

        return jnp.concatenate((X - 1.0 + 2*(jnp.square(X) - Y) * X / self.Q, (Y - jnp.square(X)) / self.Q))


    def draw(self, num):
        n = self.d // 2
        X= np.empty((num, self.d))
        X[:, :n] = np.random.normal(loc= 1.0, scale= 1.0, size= (num, n))
        X[:, n:] = np.random.normal(loc= jnp.square(X[:, :n]), scale= jnp.sqrt(self.Q), size= (num, n))

        return X


    def transform(self, x):
        return x

    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64')


    def compute_variance(self):
        num = 100000000
        x = np.random.normal(loc=1.0, scale=1.0, size=num)
        y = np.random.normal(loc=np.square(x), scale=jnp.sqrt(self.Q), size=num)

        var_x = jnp.sum(jnp.square(x)) / (num - 1)
        var_y = jnp.sum(jnp.square(y)) / (num - 1)
        print(var_x, var_y)


class StohasticVolatility():

    def __init__(self):

        _, fetch = load_dataset(SP500, shuffle=False)
        self.dates, self.returns = fetch()
        self.d = 2429

        self.typical_sigma, self.typical_nu = 0.02, 10.0 # := 1 / lambda

        self.variance = np.load('Tests/data/stohastic_volatility/ground_truth_moments.npy')
        self.grad_nlogp = jax.grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""

        sigma = jnp.exp(x[-2]) * self.typical_sigma #we used this transformation to make x unconstrained
        nu = jnp.exp(x[-1]) * self.typical_nu

        l1= jnp.sum(jnp.exp(x[-2:]) - x[-2:])
        l2 = (self.d - 2) * jnp.log(sigma) + 0.5 * (jnp.square(x[0]) + jnp.sum(jnp.square(x[1:] - x[:-1]))) / sigma * 2
        l3 = - jnp.sum(StudentT(df=nu).log_prob(self.returns / jnp.exp(x[:-2])))

        return l1 + l2 + l3


    def transform(self, x):
        """sigma and nu were transformed to make them unconstrained"""

        z = jnp.empty(x.shape)
        z = z.at[:-2].set(x.T[:-2])
        z = z.at[-2].set(jnp.exp(x.T[-2]) * self.typical_sigma)
        z = z.at[-1].set(jnp.exp(x.T[-1]) * self.typical_nu)

        return z.T



    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64')



class DiagonalPreconditioned():
    """A target instance which takes some other target and preconditions it"""

    def __init__(self, Target, a):
        """ Target: original target
            a: scaling vector (sqrt(variance of x)), such that x' = x / a"""

        self.Target = Target
        self.d= Target.d
        self.a = a
        self.variance = Target.variance / jnp.square(a)


    def nlogp(self, x):
        return self.Target.nlogp(self.a * x)

    def grad_nlogp(self, x):
        return self.Target.grad_nlogp(self.a * x) * self.a

    def draw(self, num):
        return self.Target.draw(num) / self.a

    def transform(self, x):
        return x



def check_gradient(target, x):
    """check the analytical gradient of the target at point x"""

    from scipy import optimize

    approx_grad= optimize.approx_fprime(x, target.nlogp, 1e-8)

    grad= target.grad_nlogp(x)

    print('numerical grad: ', approx_grad)
    print('analytical grad: ', grad)
    print('ratio: ', grad / approx_grad)




if __name__ == '__main__':


    target = StohasticVolatility()
    target.nlogp(jnp.zeros(target.d))
    #target.compute_variance()
    # d = 36

