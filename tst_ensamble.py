import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

from sampling.ensamble import Sampler as EnsambleSampler

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

        target = [Banana(prior = 'prior'),
                  IllConditionedGaussianGamma(prior = 'prior'),
                  GermanCredit(),
                  Brownian(),
                  IRT(),
                  StochasticVolatility()][num]


        sampler = EnsambleSampler(target, varE_wanted= 1e-3)
        n1, n2 = sampler.sample((256, 16))

        t1 = time.time()
        print(time.strftime('%H:%M:%S', time.gmtime(t1 - t0)))

        dat= np.array([n1, n2])
        print(dat)
        return dat

    data = [problem(num) for num in [3, ]]


problems()

# target = Brownian()
# sampler = EnsambleSampler(target, alpha=1.0, varE_wanted=1e-3)
# x = sampler.sample(1000, 200, output='normal')
# np.save('brown.npy', x)
# print(x.shape)

#
# x = np.load('brown.npy')
#
# for i in range(100):
#     plt.plot(x[i, :, 0], x[i, :, 1])
#
# plt.plot(np.log([0.1, ]), np.log([0.15, ]), '*', color = 'tab:red', markersize = 5)
#
# plt.show()