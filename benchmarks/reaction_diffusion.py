import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit, hessian, lax
from jax.scipy.stats import norm, multivariate_normal

import itertools
from functools import partial
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

from scipy.linalg import eigh
from scipy.interpolate import griddata
     
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

def distance_matrix(x, y):
    return jnp.sqrt(jnp.sum(jnp.abs(x[:, jnp.newaxis, :] - y[jnp.newaxis, :, :]) ** 2, axis=-1))


# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


def kernel(X, Z, var, length):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    return k


class SquaredExponential:
    def __init__(self, coords, mkl, lamb):
        """
        This class sets up a random process
        on a grid and generates
        a realisation of the process, given
        parameters or a random vector.
        """

        # Internalise the grid and set number of vertices.
        self.coords = coords
        self.n_points = self.coords.shape[0]
        self.eigenvalues = None
        self.eigenvectors = None
        self.parameters = None
        self.random_field = None

        # Set some random field parameters.
        self.mkl = mkl
        self.lamb = lamb

        self.assemble_covariance_matrix()

    def assemble_covariance_matrix(self):
        """
        Create a snazzy distance-matrix for rapid
        computation of the covariance matrix.
        """
        #dist = distance_matrix(self.coords, self.coords)
        diffs = jnp.expand_dims(self.coords / self.lamb, 1) - \
                jnp.expand_dims(self.coords / self.lamb, 0)
        r2 = jnp.sum(diffs**2, axis=2)
        self.cov = jnp.exp(-0.5 * r2)

    def plot_covariance_matrix(self):
        """
        Plot the covariance matrix.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.cov, cmap="binary")
        plt.colorbar()
        plt.show()

    def compute_eigenpairs(self):
        """
        Find eigenvalues and eigenvectors using Arnoldi iteration.
        """
        eigvals, eigvecs = eigh(self.cov, eigvals=(self.n_points - self.mkl, self.n_points - 1))

        order = jnp.flip(jnp.argsort(eigvals))
        self.eigenvalues = jnp.asarray(eigvals[order])
        self.eigenvectors = jnp.asarray(eigvecs[:, order])

    def generate(self, parameters=None, key=None):
        """
        Generate a random field, see
        Scarth, C., Adhikari, S., Cabral, P. H.,
        Silva, G. H. C., & Prado, A. P. do. (2019).
        Random field simulation over curved surfaces:
        Applications to computational structural mechanics.
        Computer Methods in Applied Mechanics and Engineering,
        345, 283–301. https://doi.org/10.1016/j.cma.2018.10.026
        """

        if parameters is None:
            if key is None:
                key = random.PRNGKey(0)
            self.parameters = random.normal(key, shape=(self.mkl,))
        else:
            self.parameters = jnp.array(parameters).flatten()

        self.random_field = jnp.linalg.multi_dot(
            (self.eigenvectors, jnp.sqrt(jnp.diag(self.eigenvalues)), self.parameters)
        )

    def plot(self, lognormal=True):
        """
        Plot the random field.
        """

        if lognormal:
            random_field = self.random_field
            contour_levels = jnp.linspace(min(random_field), max(random_field), 20)
        else:
            random_field = jnp.exp(self.random_field)
            contour_levels = jnp.linspace(min(random_field), max(random_field), 20)

        plt.figure(figsize=(12, 10))
        plt.tricontourf(
            self.coords[:, 0],
            self.coords[:, 1],
            random_field,
            levels=contour_levels,
            cmap="plasma",
        )
        plt.colorbar()
        plt.show()


class Matern52(SquaredExponential):
    def assemble_covariance_matrix(self):
        """
        This class inherits from RandomProcess and creates a Matern 5/2 covariance matrix.
        """

        # Compute scaled distances.
        dist = jnp.sqrt(5.0) * distance_matrix(self.coords, self.coords) / self.lamb

        # Set up Matern 5/2 covariance matrix.
        self.cov = (1 + dist + dist**2 / 3) * jnp.exp(-dist)