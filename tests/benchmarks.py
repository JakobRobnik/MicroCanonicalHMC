from benchmarks.benchmarks_mchmc import *
from sampling.sampler import Sampler

# from benchmarks import german_credit
import os
import jax

num_cores = 6  # specific to my PC
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(num_cores)

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)


def test():
    target = IllConditionedGaussian(100, 100.0)

    def ESS(target, num_samples, diagonal_preconditioning):
        sampler = Sampler(
            target, integrator="MN", diagonal_preconditioning=diagonal_preconditioning
        )
        ess = jnp.average(sampler.sample(num_samples, num_chains=12, output="ess"))
        return ess, sampler.L / np.sqrt(target.d), sampler.eps

    print(ESS(target, 30000, False))
    print(ESS(target, 30000, True))


test()
