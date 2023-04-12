import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

from sampling.ensamble1 import Sampler as EnsambleSampler

from benchmarks.benchmarks_mchmc import *
from benchmarks.german_credit import Target as GermanCredit
from benchmarks.brownian import Target as Brownian
from benchmarks.IRT import Target as IRT


num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)


def problems():
    import time


    def problem(num):
        t0 = time.time()

        num_samples = [500, 1000, 1000, 1000, 1000, 2000][num]
        target = [Banana(prior = 'prior'),
                  IllConditionedGaussianGamma(prior = 'prior'),
                  GermanCredit(),
                  Brownian(),
                  IRT(),
                  StochasticVolatility()][num]


        sampler = EnsambleSampler(target, alpha = 1.0, varE_wanted= 1e-3)
        n1, n2 = sampler.sample(num_samples, 4096, output = 'ess')

        t1 = time.time()
        print(time.strftime('%H:%M:%S', time.gmtime(t1 - t0)))

        dat= np.array([n1, n2])
        print(dat)
        return dat

    data = [problem(num) for num in [3 ]]


problems()