from benchmarks.benchmarks_mchmc import *
from sampling.sampler import Sampler
#from benchmarks import german_credit
import pandas as pd
import os
import jax

num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)



def test():

    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0), BiModal(), Rosenbrock(), Funnel(), StochasticVolatility()]

    def ESS(target, num_samples):
        sampler = Sampler(target, integrator= 'MN')
        ess= jnp.average(sampler.sample(num_samples, 12, output= 'ess'))
        return ess, sampler.L / np.sqrt(target.d), sampler.eps

    num_samples= [30000, 100000, 300000, 100000, 10000]

    results = np.array([ESS(targets[i], num_samples[i]) for i in range(len(names))])
    df = pd.DataFrame({'Target ': names, 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})
    print(df)
    

test()
