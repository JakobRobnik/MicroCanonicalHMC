import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt





class MultiGauss:

    def __init__(self, d, condition_number, num_components):
        key = jax.random.PRNGKey(42)


        self.kappa = condition_number
        self.d = d
        self.num_components = num_components

        keys= jax.random.split(key, num_components+2)
        key = keys[0]
        Hessians, log_det_cov = [], []
        for i in range(num_components):
            H, det = self.draw_Hessian(keys[2 + i])
            Hessians.append(H)
            log_det_cov.append(det)

        self.Hessians = jnp.array(Hessians)
        self.log_det_cov = jnp.array(log_det_cov)

        locations = [self.prior_draw(keys[1]), ]

        for i in range(1, num_components):

            separated = False
            while not separated:
                key, subkey = jax.random.split(key)
                mu = self.prior_draw(subkey)

                separated = True
                for j in range(i):
                    x= locations[j] - mu
                    distance = min(x @ Hessians[j] @ x, x @ Hessians[i] @ x)
                    if distance < 10.0**2:
                        separated = False
                        break

            locations.append(mu)

        key, subkey = jax.random.split(key)
        relative_contributions = jax.random.uniform(subkey, shape = (num_components, ))
        self.relative_contributions = relative_contributions / jnp.sum(relative_contributions)
        self.locations = jnp.array(locations)

        self.grad_nlogp = jax.value_and_grad(self.nlogp)



    def draw_Hessian(self, key):
        key1, key2 = jax.random.split(key)
        eigs= jnp.exp(jax.random.uniform(key1, shape= (self.d, ), minval= -0.5 * np.log(self.kappa), maxval= 0.5 * np.log(self.kappa)))
        inv_D = jnp.diag(1 / eigs)
        R, _ = jnp.linalg.qr(jax.random.normal(key2, shape = (self.d, self.d), dtype = 'float64'))  # random rotation
        return R @ inv_D @ R.T, jnp.sum(jnp.log(eigs))


    def nlogp(self, x):
        """- log p of the target distribution"""

        def add_component(logp, i):
            y = x - self.locations[i]
            return jnp.logaddexp(logp, -0.5 * y @ self.Hessians[i] @ y + jnp.log(self.relative_contributions[i]) - 0.5 * self.log_det_cov[i]), None

        logp = jax.lax.scan(add_component, -jnp.inf, xs = jnp.arange(self.num_components))[0]

        return -logp


    def transform(self, x):
        return x

    def prior_draw(self, key):
        return jax.random.normal(key, shape=(self.d,), dtype='float64') * 10.0




def estimate_covariance_matrix(X):
    Y = X - jnp.average(X, axis = 0)[None, :]
    S = jnp.einsum('ki, kj', Y, Y) / len(X)
    return S

def evidence(X, L):

    entropy = jnp.average(L)

    d = len(X[0])

    return 0.5 * d * (1 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.log(jnp.linalg.det(estimate_covariance_matrix(X))) - entropy




target = MultiGauss(2, 10.0, 3)

num = 2

x = np.linspace(-1, 1, 10) * 5 + target.locations[num][0]
y = np.linspace(-1, 1, 10) * 5 + target.locations[num][1]
X, Y = np.meshgrid(x, y)



Z = np.exp(-np.array([[target.nlogp(jnp.array([X[i, j], Y[i, j]])) for j in range(len(X[0]))] for i in range(len(X))]))

plt.contourf(X, Y, Z)
plt.show()
