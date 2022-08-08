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



class IllConcitionedGaussian():
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

