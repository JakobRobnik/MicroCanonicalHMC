import numpy as np
import matplotlib.pyplot as plt

from HMC.myHMC import Sampler as HMCsampler
from sampling.benchmark_targets import *
from graphs import plot_power_lines

factorial = lambda n: 1 if n == 0 else n * factorial(n-1)


def compute_accuracy(names):
    """"returns steps and diff
        diff[i, j, 0] = trajectory error for i-th integrator with steps[j] number of steps
        diff[i, j, 1] = STD[energy]
    """

    #properties of the trajectory
    total_time = 5 #total time of the trajectory
    key = jax.random.PRNGKey(0)


    #integrators
    target = StandardNormal(d = 100)
    #target = Rosenbrock()
    Integrators = [HMCsampler(target, np.inf, 1.0, name) for name in names]
    x0 = target.prior_draw(jax.random.PRNGKey(43))

    #base solution: very small stepsize with the best method (the last one in the Samplers array)
    num_points, base = 10, 2
    Integrators[-1].eps = total_time/base**(num_points-1)
    x_base = Integrators[-1].sample(base**(num_points-1), random_key= key, x_initial=x0)
    x_base = np.concatenate(([x0, ], x_base))

    #main computation
    diff = np.empty((len(names), num_points, 2))
    for num_sampler in range(len(names)): # integration method
        print('sampler ' + str(names[num_sampler]))
        for n in range(num_points): #stepsize
            Sampler = Integrators[num_sampler]
            eps = total_time / base ** n
            Sampler.eps = eps

            #get the trajectory
            x, E = Sampler.sample(base ** n, random_key= key, x_initial= x0, monitor_energy= True)
            x = np.concatenate(([x0, ], x))

            #results
            diff[num_sampler, n, 0] = np.max(np.sqrt(np.sum(np.square(x_base[::base**(num_points -1 - n), :] - x), axis = 1)))
            diff[num_sampler, n, 1] = np.std(E)

    #stepsize = [total_time / base**n for n in range(1, nmax+1)]
    steps = np.power(base, np.arange(num_points))

    return steps, diff



def accuracy_plot():

    names = ['RM', 'LF']

    steps, diff = compute_accuracy(names)
    base, num_points = steps[1], len(steps)
    plt.figure(figsize=(15, 10))
    plt.suptitle('Integrator comparisson')

    plt.subplot(1, 2, 1)
    for num_sampler in range(len(names)):
        plt.plot(steps[:-1], diff[num_sampler, :-1, 0], '.:', label = names[num_sampler])

    plt.xscale('log')
    plt.yscale('log')
    plot_power_lines(-2.0)
    plt.xlabel('# gradient evaluations')
    plt.ylabel(r'$\mathrm{max}_t |x(t) - x_{\mathrm{true}}(t) |$')
    plt.legend(loc = 3)

    plt.xlim(1, base**(num_points-1))
    #plt.ylim(1e-16, 1)

    plt.subplot(1, 2, 2)
    for num_sampler in range(len(names)):
        plt.plot(steps, diff[num_sampler, :, 1], '.:', label=names[num_sampler])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    plt.ylabel('STD[E]')
    plt.legend(loc = 3)

    plt.xlim(1, base**(num_points-1))

    plt.savefig("integrators_gaussian.png")
    plt.show()


accuracy_plot()