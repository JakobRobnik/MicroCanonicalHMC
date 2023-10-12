import numpy as np
from scipy.stats import norm
import jax
import jax.numpy as jnp
import os
dirr = os.path.dirname(os.path.realpath(__file__))



class StandardNormal():
    """Standard Normal distribution in d dimensions"""

    def __init__(self, d):
        self.d = d
        self.variance = jnp.ones(d)
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        self.second_moments = jnp.ones(d)
        self.variance_second_moments = 2 * self.second_moments
        

    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * jnp.sum(jnp.square(x), axis= -1)


    def transform(self, x):
        return x

    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ))




class IllConditionedGaussian():
    """Gaussian distribution. Covariance matrix has eigenvalues equally spaced in log-space, going from 1/condition_bnumber^1/2 to condition_number^1/2."""


    def __init__(self, d, condition_number, numpy_seed=None, prior= 'prior'):
        """numpy_seed is used to generate a random rotation for the covariance matrix.
            If None, the covariance matrix is diagonal."""

        self.name = 'ICG_easy'
        self.d = d
        self.condition_number = condition_number
        eigs = jnp.logspace(-0.5 * jnp.log10(condition_number), 0.5 * jnp.log10(condition_number), d)

        if numpy_seed == None:  # diagonal
            self.second_moments = eigs
            self.R = jnp.eye(d)
            self.Hessian = jnp.diag(1 / eigs)
            self.Cov = jnp.diag(eigs)

        else:  # randomly rotate
            rng = np.random.RandomState(seed=numpy_seed)
            D = jnp.diag(eigs)
            inv_D = jnp.diag(1 / eigs)
            R, _ = jnp.array(np.linalg.qr(rng.randn(self.d, self.d)))  # random rotation
            self.R = R
            self.Hessian = R @ inv_D @ R.T
            self.Cov = R @ D @ R.T
            self.second_moments = jnp.diagonal(R @ D @ R.T)

            #Cov_precond = jnp.diag(1 / jnp.sqrt(self.second_moments)) @ self.Cov @ jnp.diag(1 / jnp.sqrt(self.second_moments))

            #print(jnp.linalg.cond(Cov_precond) / jnp.linalg.cond(self.Cov))

        self.variance_second_moments = 2 * jnp.square(self.second_moments)


        self.nlogp = lambda x: 0.5 * x.T @ self.Hessian @ x
        self.transform = lambda x: x
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


        if prior == 'map':
            self.prior_draw = lambda key: jnp.zeros(self.d)

        elif prior == 'posterior':
            self.prior_draw = lambda key: self.R @ (jax.random.normal(key, shape=(self.d,)) * jnp.sqrt(eigs))

        else: # N(0, sigma_true_max)
            self.prior_draw = lambda key: jax.random.normal(key, shape=(self.d,)) * jnp.max(jnp.sqrt(eigs))



class IllConditionedESH():
    """ICG from the ESH paper."""

    def __init__(self):
        self.d = 50
        self.variance = jnp.linspace(0.01, 1, self.d)

        self.grad_nlogp = jax.value_and_grad(self.nlogp)

    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * jnp.sum(jnp.square(x) / self.variance, axis= -1)


    def transform(self, x):
        return x

    def draw(self, key):
        return jax.random.normal(key, shape = (self.d, )) * jnp.sqrt(self.variance)

    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ))




class IllConditionedGaussianGamma():
    """Inference gym's Ill conditioned Gaussian"""

    def __init__(self, prior = 'prior'):
        self.name = 'ICG'
        self.d = 100

        # define the Hessian
        rng = np.random.RandomState(seed=10 & (2 ** 32 - 1))
        eigs = np.sort(rng.gamma(shape=0.5, scale=1., size=self.d)) #eigenvalues of the Hessian
        eigs *= jnp.average(1.0/eigs)
        self.entropy = 0.5 * self.d
        self.maxmin = (1./jnp.sqrt(eigs[0]), 1./jnp.sqrt(eigs[-1])) 
        R, _ = np.linalg.qr(rng.randn(self.d, self.d)) #random rotation
        self.map_to_worst = (R.T)[[0, -1], :]
        self.Hessian = R @ np.diag(eigs) @ R.T

        # analytic ground truth moments
        self.second_moments = jnp.diagonal(R @ np.diag(1.0/eigs) @ R.T)
        self.variance_second_moments = 2 * jnp.square(self.second_moments)

        # norm = jnp.diag(1/jnp.sqrt(self.second_moments))
        # Sigma = R @ np.diag(1/eigs) @ R.T
        # reduced = norm @ Sigma @ norm
        # print(np.linalg.cond(reduced), np.linalg.cond(Sigma))
        
        # gradient
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        if prior == 'map':
            self.prior_draw = lambda key: jnp.zeros(self.d)

        elif prior == 'posterior':
            self.prior_draw = lambda key: R @ (jax.random.normal(key, shape=(self.d,)) / jnp.sqrt(eigs))

        else: # N(0, sigma_true_max)
            self.prior_draw = lambda key: jax.random.normal(key, shape=(self.d,)) * jnp.max(1.0/jnp.sqrt(eigs))
            
    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * x.T @ self.Hessian @ x

    def transform(self, x):
        return x
    
    


class Banana():
    """Banana target fromm the Inference Gym"""

    def __init__(self, prior = 'map'):
        self.name = 'Banana'
        self.curvature = 0.03
        self.d = 2
        self.grad_nlogp = jax.value_and_grad(self.nlogp)
        self.transform = lambda x: x
        self.second_moments = jnp.array([100.0, 19.0]) #the first is analytic the second is by drawing 10^8 samples from the generative model. Relative accuracy is around 10^-5.
        self.variance_second_moments = jnp.array([20000.0, 4600.898])

        if prior == 'map':
            self.prior_draw = lambda key: jnp.array([0, -100.0 * self.curvature])
        elif prior == 'posterior':
            self.prior_draw = lambda key: self.posterior_draw(key)
        elif prior == 'prior':
            self.prior_draw = lambda key: jax.random.normal(key, shape=(self.d,)) * jnp.array([10.0, 5.0]) * 2
        else:
            raise ValueError('prior = '+prior +' is not defined.')

    def nlogp(self, x):
        mu2 = self.curvature * (x[0] ** 2 - 100)
        return 0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))

    def posterior_draw(self, key):
        z = jax.random.normal(key, shape = (2, ))
        x0 = 10.0 * z[0]
        x1 = self.curvature * (x0 ** 2 - 100) + z[1]
        return jnp.array([x0, x1])

    def ground_truth(self):
        x = jax.vmap(self.posterior_draw)(jax.random.split(jax.random.PRNGKey(0), 100000000))
        print(jnp.average(x, axis=0))
        print(jnp.average(jnp.square(x), axis=0))
        print(jnp.std(jnp.square(x[:, 0])) ** 2, jnp.std(jnp.square(x[:, 1])) ** 2)

    def plott(self):
        xmin, xmax = -20.0, 20.0
        ymin, ymax = -10.0, 10.0
        X, Y, Z = get_contour_plot(self, jnp.linspace(xmin, xmax, 100), jnp.linspace(ymin, ymax, 100))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.contourf(X, Y, jnp.exp(-Z))

        x = np.linspace(xmin, xmax, 100)
        plt.plot(x, 0.03 * (x ** 2 - 100), color='tab:red')
        plt.savefig('../tst_ensamble/Banana/banana.png')
        plt.show()



class Cauchy():
    """d indpendent copies of the standard Cauchy distribution"""

    def __init__(self, d):
        self.d = d

        self.nlogp = lambda x: jnp.sum(jnp.log(1. + jnp.square(x)))
        self.grad_nlogp = jax.value_and_grad(self.nlogp)
        self.transform = lambda x: x        
        self.prior_draw = lambda key: jax.random.normal(key, shape=(self.d,))




class HardConvex():

    def __init__(self, d, kappa, theta = 0.1):
        """d is the dimension, kappa = condition number, 0 < theta < 1/4"""
        self.d = d
        self.theta, self.kappa = theta, kappa
        C = jnp.power(d-1, 0.25 - theta)
        self.nlogp = lambda x: 0.5 * jnp.sum(jnp.square(x[:-1])) + (0.75 / kappa)* x[-1]**2 - 0.5 * jnp.sum(jnp.cos(C * x[:-1])) / C**2
        self.grad_nlogp = jax.value_and_grad(self.nlogp)
        self.transform = lambda x: x

        # numerically precomputed variances
        num_integration = [0.93295, 0.968802, 0.990595, 0.998002, 0.999819]
        if d == 100:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[0], jnp.ones(1) * 2.0*kappa/3.0))
        elif d == 300:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[1], jnp.ones(1) * 2.0*kappa/3.0))
        elif d == 1000:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[2], jnp.ones(1) * 2.0*kappa/3.0))
        elif d == 3000:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[3], jnp.ones(1) * 2.0*kappa/3.0))
        elif d == 10000:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[4], jnp.ones(1) * 2.0*kappa/3.0))
        else:
            None


    def prior_draw(self, key):
        """Gaussian prior with approximately estimating the variance along each dimension"""
        scale = jnp.concatenate((jnp.ones(self.d-1), jnp.ones(1) * jnp.sqrt(2.0 * self.kappa / 3.0)))
        return jax.random.normal(key, shape=(self.d,)) * scale




class BiModal():
    """A Gaussian mixture p(x) = f N(x | mu1, sigma1) + (1-f) N(x | mu2, sigma2)."""

    def __init__(self, d = 50, mu1 = 0.0, mu2 = 8.0, sigma1 = 1.0, sigma2 = 1.0, f = 0.2):

        self.d = d

        self.mu1 = jnp.insert(jnp.zeros(d-1), 0, mu1)
        self.mu2 = jnp.insert(jnp.zeros(d - 1), 0, mu2)
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.f = f
        self.variance = jnp.insert(jnp.ones(d-1) * ((1 - f) * sigma1**2 + f * sigma2**2), 0, (1-f)*(sigma1**2 + mu1**2) + f*(sigma2**2 + mu2**2))
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""

        N1 = (1.0 - self.f) * jnp.exp(-0.5 * jnp.sum(jnp.square(x - self.mu1), axis= -1) / self.sigma1 ** 2) / jnp.power(2 * jnp.pi * self.sigma1 ** 2, self.d * 0.5)
        N2 = self.f * jnp.exp(-0.5 * jnp.sum(jnp.square(x - self.mu2), axis= -1) / self.sigma2 ** 2) / jnp.power(2 * jnp.pi * self.sigma2 ** 2, self.d * 0.5)

        return -jnp.log(N1 + N2)


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
        z = jax.random.normal(key, shape = (self.d, )) *self.sigma1
        #z= z.at[0].set(self.mu1 + z[0])
        return z


class BiModalEqual():
    """Mixture of two Gaussians, one centered at x = [mu/2, 0, 0, ...], the other at x = [-mu/2, 0, 0, ...].
        Both have equal probability mass."""

    def __init__(self, d, mu):

        self.d = d
        self.mu = mu
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""

        return 0.5 * jnp.sum(jnp.square(x), axis= -1) - jnp.log(jnp.cosh(0.5*self.mu*x[0])) + 0.5* self.d * jnp.log(2 * jnp.pi) + self.mu**2 / 8.0


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

    def __init__(self, d = 20):

        self.d = d
        self.sigma_theta= 3.0
        self.variance = jnp.ones(d)
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """ - log p of the target distribution
                x = [z_0, z_1, ... z_{d-1}, theta] """
        theta = x[-1]
        X = x[..., :- 1]

        return 0.5* jnp.square(theta / self.sigma_theta) + 0.5 * (self.d - 1) * theta + 0.5 * jnp.exp(-theta) * jnp.sum(jnp.square(X), axis = -1)

    def inverse_transform(self, xtilde):
        theta = 3 * xtilde[-1]
        return jnp.concatenate((xtilde[:-1] * jnp.exp(0.5 * theta), jnp.ones(1)*theta))


    def transform(self, x):
        """gaussianization"""
        xtilde = jnp.empty(x.shape)
        xtilde = xtilde.at[-1].set(x.T[-1] / 3.0)
        xtilde = xtilde.at[:-1].set(x.T[:-1] * jnp.exp(-0.5*x.T[-1]))
        return xtilde.T


    def prior_draw(self, key):
        return self.inverse_transform(jax.random.normal(key, shape = (self.d, )))




class Funnel_with_Data():

    def __init__(self, d, sigma, minibatch_size, key):

        self.d = d
        self.sigma_theta= 3.0
        self.theta_true = 0.0
        self.sigma_data = sigma
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        self.data = self.simulate_data()

        self.batch = minibatch_size

    def simulate_data(self):

        norm = jax.random.normal(jax.random.PRNGKey(123), shape = (2*(self.d-1), ))
        z_true = norm[:self.d-1] * jnp.exp(self.theta_true * 0.5)
        self.data = z_true + norm[self.d-1:] * self.sigma_data


    def nlogp(self, x, subset):
        """ - log p of the target distribution
                x = [z_0, z_1, ... z_{d-1}, theta] """
        theta = x[-1]
        z = x[:- 1][subset]

        prior_theta = jnp.square(theta / self.sigma_theta)
        prior_z = np.sum(subset) * theta + jnp.exp(-theta) * jnp.sum(jnp.square(z*subset))
        likelihood = jnp.sum(jnp.square((z - self.data)*subset / self.sigma_data))

        return 0.5 * (prior_theta + prior_z + likelihood)


    def transform(self, x):
        """gaussianization"""
        return x

    def prior_draw(self, key):
        key1, key2 = jax.random.split(key)
        theta = jax.random.normal(key1) * self.sigma_theta
        z = jax.random.normal(key2, shape = (self.d-1, )) * jnp.exp(theta * 0.5)
        return jnp.concatenate((z, theta))




class Rosenbrock():

    def __init__(self, d = 36, Q = 0.1):

        self.d = d
        self.Q = Q

        #ground truth moments
        var_x = 2.0

        #these two options were precomputed:
        if Q == 0.1:
            var_y = 10.098433122783046 # var_y is computed numerically (see class function compute_variance)
        elif Q == 0.5:
            var_y = 10.498957879911487
        else:
            raise ValueError('Ground truth moments for Q = ' + str(Q) + ' were not precomputed. Use Q = 0.1 or 0.5.')

        self.variance = jnp.concatenate((var_x * jnp.ones(d//2), var_y * jnp.ones(d//2)))

        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""
        X, Y = x[..., :self.d//2], x[..., self.d//2:]
        return 0.5 * jnp.sum(jnp.square(X - 1.0) + jnp.square(jnp.square(X) - Y) / self.Q, axis= -1)



    def draw(self, num):
        n = self.d // 2
        X= np.empty((num, self.d))
        X[:, :n] = np.random.normal(loc= 1.0, scale= 1.0, size= (num, n))
        X[:, n:] = np.random.normal(loc= jnp.square(X[:, :n]), scale= jnp.sqrt(self.Q), size= (num, n))

        return X


    def transform(self, x):
        return x


    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ))


    def compute_moments(self):
        num = 100000000
        x = np.random.normal(loc=1.0, scale=1.0, size=num)
        y = np.random.normal(loc=np.square(x), scale=jnp.sqrt(self.Q), size=num)

        x2 = jnp.sum(jnp.square(x)) / (num - 1)
        y2 = jnp.sum(jnp.square(y)) / (num - 1)

        x1 = np.average(x)
        y1 = np.average(y)

        print(np.sqrt(0.5*(np.square(np.std(x)) + np.square(np.std(y)))))

        print(x2, y2)



class StochasticVolatility():
    """Example from https://num.pyro.ai/en/latest/examples/stochastic_volatility.html"""

    def __init__(self):
        self.SP500_returns = np.load(dirr + '/SP500.npy')

        self.name = 'SV'
        self.d = 2429

        self.typical_sigma, self.typical_nu = 0.02, 10.0 # := 1 / lambda

        data = np.load(dirr + '/ground_truth/stochastic_volatility/ground_truth_0.npy')
        self.second_moments = data[0]
        self.variance_second_moments = data[1]
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution
            x=  [s1, s2, ... s2427, log sigma / typical_sigma, log nu / typical_nu]"""

        sigma = jnp.exp(x[-2]) * self.typical_sigma #we used this transformation to make x unconstrained
        nu = jnp.exp(x[-1]) * self.typical_nu

        l1= (jnp.exp(x[-2]) - x[-2]) + (jnp.exp(x[-1]) - x[-1])
        l2 = (self.d - 2) * jnp.log(sigma) + 0.5 * (jnp.square(x[0]) + jnp.sum(jnp.square(x[1:-2] - x[:-3]))) / jnp.square(sigma)
        l3 = jnp.sum(nlogp_StudentT(self.SP500_returns, nu, jnp.exp(x[:-2])))

        return l1 + l2 + l3


    def transform(self, x):
        """transforms to the variables which are used by numpyro (and in which we have the ground truth moments)"""

        z = jnp.empty(x.shape)
        z = z.at[:-2].set(x[:-2]) # = s = log R
        z = z.at[-2].set(jnp.exp(x[-2]) * self.typical_sigma) # = sigma
        z = z.at[-1].set(jnp.exp(x[-1]) * self.typical_nu) # = nu

        return z


    def prior_draw(self, key):
        """draws x from the prior"""

        key_walk, key_exp = jax.random.split(key)

        scales = jnp.array([self.typical_sigma, self.typical_nu])
        #params = jax.random.exponential(key_exp, shape = (2, )) * scales
        params= scales
        walk = random_walk(key_walk, self.d - 2) * params[0]
        return jnp.concatenate((walk, jnp.log(params/scales)))
    
    
def nlogp_StudentT(x, df, scale):
    y = x / scale
    z = (
        jnp.log(scale)
        + 0.5 * jnp.log(df)
        + 0.5 * jnp.log(jnp.pi)
        + jax.scipy.special.gammaln(0.5 * df)
        - jax.scipy.special.gammaln(0.5 * (df + 1.0))
    )
    return 0.5 * (df + 1.0) * jnp.log1p(y**2.0 / df) + z


def random_walk(key, num):
    """ Genereting process for the standard normal walk:
        x[0] ~ N(0, 1)
        x[n+1] ~ N(x[n], 1)

        Args:
            key: jax random key
            num: number of points in the walk
        Returns:
            1 realization of the random walk (array of length num)
    """

    def step(track, useless):
        x, key = track
        randkey, subkey = jax.random.split(key)
        x += jax.random.normal(subkey)
        return (x, randkey), x

    return jax.lax.scan(step, init=(0.0, key), xs=None, length=num)[1]


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



def get_contour_plot(target, x, y):
    """Args:
            target with target.nlogp defined (configuration space must be two dimensional)
            x = linspace over the x axis
            y = linspace over the y axis
       Returns:
            X, Y, Z = nlogp(X, Y), so that you can for example do a contour plot with
            plt.contourf(X, Y, Z)
    """

    nx, ny = len(x), len(y)
    X, Y = np.meshgrid(x, y)
    R = jnp.array([X, Y])
    R = jnp.concatenate(jnp.moveaxis(R, [0, 1, 2], [2, 0, 1]))
    Z = jax.vmap(target.nlogp)(R).reshape(ny, nx)

    return X, Y, Z



def check_gradient(target, x):
    """check the analytical gradient of the target at point x"""

    from scipy import optimize

    approx_grad= optimize.approx_fprime(x, target.nlogp, 1e-3)

    grad= target.grad_nlogp(x)

    print('numerical grad: ', approx_grad)
    print('analytical grad: ', grad)
    print('ratio: ', grad / approx_grad)



if __name__ == '__main__':


    rng = np.random.RandomState(seed=10 & (2 ** 32 - 1))
    eigs = rng.gamma(shape=0.5, scale=1., size=100) #eigenvalues of the Hessian
    print(np.max(eigs)/np.min(eigs))