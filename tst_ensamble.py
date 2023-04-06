import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

from sampling.benchmark_targets import *
from sampling.ensamble import Sampler as EnsambleSampler

from sampling.german_credit import Target as GermanCredit
from sampling.brownian import Target as Brownian
from sampling.IRT import Target as IRT


num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

def virial_loss_calibration():
    """calibration with a Gaussian"""

    d = 36

    def viral_particles(chains):
        repeat = 30
        X = np.random.normal(size = (repeat, chains, d))
        virials = np.average(np.square(X), axis = 1)
        virial = np.sqrt(np.average(np.square(virials - 1.0), axis = 1))
        return [np.average(virial), np.std(virial)]


    chains = [5, 10, 30, 50, 100, 200, 300, 500, 700, 1000]
    data = np.array([viral_particles(c) for c in chains])

    plt.errorbar(chains, data[:, 0], yerr= data[:, 1], capsize= 2, fmt = 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('particles')
    plt.ylabel('virial loss (for exact samples)')
    plt.savefig('virial_loss_calibration')
    plt.show()




def benchmarks():
    """For generating Table 1 in the paper"""

    particles = 300

    # targets
    names = ['Ill-Conditioned', 'Bi-Modal', 'Rosenbrock', "Neal's Funnel", 'German Credit', 'Stochastic Volatility']
    targets = [IllConditionedGaussian(100, 100.0, numpy_seed= 42), BiModal(), Rosenbrock(), Funnel(), german_credit.Target(), StochasticVolatility()]
    indexes = [i for i in range(len(names))]

    # dimensions = [100, 300, 1000, 3000, 10000]
    # names= [str(d) for d in dimensions]
    # #targets= [StandardNormal(d) for d in dimensions]
    # #targets = [IllConditionedGaussian(d, 100.0) for d in dimensions]
    # targets= [Rosenbrock(d) for d in dimensions]
    #
    key = jax.random.PRNGKey(0)

    def ESS(alpha, varE, target, num_samples):
        sampler = EnsambleSampler(target, alpha * np.sqrt(target.d), np.sqrt(target.d / 100) * 5, varE)
        n = np.sum(sampler.sample(particles, num_samples, ess=True))
        return n

    ESS(10.0, 1e-3, targets[2], 500)
    exit()

    borders_alpha = np.array([[0.3, 3], [0.3, 3], [10, 40], [0.3, 10], [0.3, 3], [0.3, 3]])

    num_samples = [100, 100, 300, 300, 300, 200]

    results = np.array([grid_search.search_wrapper(lambda a, e: ESS(a, e, targets[i], num_samples[i]), borders_alpha[i][0], borders_alpha[i][1], 1e-4, 1e-2) for i in indexes])
    print(results)
    df = pd.DataFrame({'Target ': [names[i] for i in indexes], 'ESS': results[:, 0], 'alpha': results[:, 1], 'eps': results[:, 2]})


    df.to_csv('ensamble/first_results.csv', index=False)
    print(df)


def problems():
    import time


    def problem(num):
        t0 = time.time()

        num_samples = [500, 1000, 2000, 1000, 1000, 2000][num]
        target = [Banana(prior = 'prior'),
                  IllConditionedGaussianGamma(prior = 'prior'),
                  GermanCredit(),
                  Brownian(),
                  IRT(),
                  StochasticVolatility()][num]


        sampler = EnsambleSampler(target, alpha = 1.0, varE_wanted= 1e-3)
        n1, n2, nburn = sampler.sample(num_samples, 4096, output = 'ess')

        t1 = time.time()
        print(time.strftime('%H:%M:%S', time.gmtime(t1 - t0)))

        return [n1 + nburn, n2 + nburn]

    data = [problem(num) for num in range(6)]
    print(data)

problems()