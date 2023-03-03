import numpyro
from numpyro.distributions import constraints


def mchmc_target_to_numpyro(target):
    """given mchmc target outputs the corresponding numpyto model"""

    class distribution(numpyro.distributions.Distribution):
        """Custom defined distribution, see https://forum.pyro.ai/t/creating-a-custom-distribution-in-numpyro/3332/3"""

        support = constraints.real_vector

        def __init__(self, *args):
            self.target = target(*args)
            self.log_prob = lambda x: -self.target.nlogp(x)
            super().__init__(event_shape=(self.target.d,))

        def sample(self, key, sample_shape=()):
            raise NotImplementedError

    def model(*args):
        x = numpyro.sample('x', distribution(*args))

    return model