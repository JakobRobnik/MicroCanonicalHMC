import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm

import ESH
import CTV
import parallel
from targets import *
import bias


#
# def compute_free_time(n):
#     #free_steps_arr = (np.linspace(50, 250, 18)).astype(int)
#     free_steps_arr = [1, 10, 100, 1000]
#
#     d = 100
#     free_steps = free_steps_arr[n]
#     eps= 0.5
#     sampler = ESH.Sampler(StandardNormal(d=d), eps= eps)
#
#
#     x0 = sampler.Target.draw(1)[0] #we draw an initial condition from the target
#
#     ess = sampler.ess(x0, free_steps*eps)
#
#     return [ess, free_steps, eps, d]


def compute_free_time(n, d):

    # free_steps_arr = (np.linspace(50, 250, 18)).astype(int)
    length = (3.6 * np.sqrt(d) * np.logspace(-0.4, 0.4, 12))[n]

    #sampler = CTV.Sampler(Target = IllConditionedGaussian(d= d, condition_number=100), eps= 3)
    sampler = ESH.Sampler(Target= Rosenbrock(d= d), eps= 0.5)

    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target

    ess = sampler.sample(x0, length)

    return [ess, length, sampler.eps, d]



def compute_eps(n):
    eps_arr = np.logspace(np.log10(0.1), np.log10(16), 6)
    eps = eps_arr[n]
    d = 100
    free_time = 16.0
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps=eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    ess = esh.ess(x0, free_time)

    return [ess, free_steps, eps, d]



def compute_kappa(n):
    kappa = np.logspace(0, 3, 18)[n]#([1, 10, 100, 1000])[n]
    d = 100
    eps, free_time = 1, 16
    esh = ESH.Sampler(Target=IllConditionedGaussian(d=d, condition_number=kappa), eps=eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    ess = esh.ess(x0, free_time)

    return [ess, free_steps, eps, d, kappa]


def compute_dimension(n):
    d = ([2, 3, 5, 10])[n]  # ([1, 10, 100, 1000])[n]

    eps, free_steps = 1.0, 16
    esh = ESH.Sampler(Target= StandardNormal(d=d), eps=eps)
    np.random.seed(0)
    num_averaging = 10
    x0 = esh.Target.draw(num_averaging)

    ess, ess_upper, ess_lower = esh.ess_with_averaging(x0, free_steps)

    return [ess, ess_upper, ess_lower, free_steps, eps, d]



def compute_energy(n):
    eps_arr = [0.05, 0.1, 0.5, 1, 2]
    eps = eps_arr[n]
    d = 50
    total_num = 1000000
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps= eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    t, X, P, E = esh.trajectory(x0, total_num)
    np.save('Tests/energy/E'+str(n)+'.npy', E)



def compute_mode_mixing(n):
    mu = np.arange(1, 9)[n]

    eps, free_steps = 1.0, 16
    d = 2
    esh = ESH.Sampler(Target= BiModal(d=d, mu= mu), eps=eps)
    np.random.seed(0)

    avg_island_size = esh.mode_mixing(free_steps)
    print(avg_island_size)
    sys.stdout.flush()

    return [avg_island_size, free_steps, eps, d, mu]



def funnel():

    eps = 0.1
    free_time = 6
    free_steps = (int)(free_time / eps)
    d = 20
    esh = ESH.Sampler(Target= Funnel(d=d), eps=eps)
    np.random.seed(0)
    x0 = np.zeros(d)
    samples, w = esh.sample(x0, free_steps, 5000000)
    np.savez('Tests/data/funnel_free'+str(free_time) + '_eps'+str(eps), z = samples[:, :-1], theta= samples[:, -1], w = w)



def rosenbrock():

    eps = 0.001
    free_steps = (int)(1 / eps)
    d = 36
    esh = ESH.Sampler(Target= Rosenbrock(d=d), eps=eps)
    np.random.seed(0)
    x0 = np.zeros(d)
    samples, w = esh.sample(x0, free_steps, 10000000)
    np.savez('Tests/data/rosenbrock3', samples = samples[::10, :], w = w[::10])


def my_hist(bins, count):
    probability = count / np.sum(count)
    print(probability[-1])

    for i in range(len(bins)):
        density = probability[i] / (bins[i][1] - bins[i][0])
        plt.fill_between(bins[i], np.zeros(2), density * np.ones(2), alpha = 0.5, color = 'tab:blue')



def bimodal():

    def get_bins(mu, sigma, num_bins_per_mode):
        xmax =3.0
        bins_mode = np.array([[- xmax + i * 2 * xmax / num_bins_per_mode, - xmax + (i+1) * 2 * xmax / num_bins_per_mode] for i in range(num_bins_per_mode)])

        bins = np.concatenate((bins_mode, (bins_mode * sigma) + mu))

        return bins

    eps = 1.0
    #free_steps = (int)(1 / eps)
    d, mu, sigma, f = 50, 9, 0.5, 0.1
    esh = ESH.Sampler(Target= BiModal(d=d, mu = mu, sigma= sigma, f= f), eps=eps)
    bins_per_mode = 10
    X = esh.Target.draw(1000)[:, 0]

    bins = get_bins(mu, sigma, bins_per_mode)

    def which_bin(x):
        for i in range(len(bins)):
            if x > bins[i][0] and x < bins[i][1]:
                return i

        return len(bins)  # if it is not in any of the bins

    P = np.zeros(len(bins) + 1)
    for i in range(len(X)):
        P[which_bin(X[i])] += 0.1

    my_hist(bins, P)
    f1, f2 = np.sum(P[:bins_per_mode]), np.sum(P[bins_per_mode : 2 * bins_per_mode])
    print(f2 / (f1 + f2))

    t = np.linspace(-5, 12, 1000)

    plt.plot(t, (1- f)*norm.pdf(t) + f * norm.pdf(t, loc = mu, scale = sigma), ':', color = 'gold')
    plt.show()

    #np.random.seed(0)
    #x0 = np.zeros(d)
    #samples, w = esh.sample(x0, free_steps, 10000000)


def inference_gym(n):

    d = 50
    #time_bounce = 2 * np.pi * 0.26 * np.sqrt(d)
    time_bounce = (2 * np.pi * 0.26 * np.sqrt(d) * np.logspace(-0.4, 0.4, 12))[n]
    eps = 0.5

    # standard Gaussian
    sampler = ESH.Sampler(StandardNormal(d=d), eps=eps)
    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    ess1 = sampler.sample(x0, time_bounce)

    # ill-conditioned Gaussian
    sampler = ESH.Sampler(IllConditionedGaussian(d=d, condition_number= 100), eps=eps)
    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    ess2 = sampler.sample(x0, time_bounce)

    # Rosenbrock
    sampler = ESH.Sampler(Rosenbrock(d=d), eps=eps)
    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    ess3 = sampler.sample(x0, time_bounce)


    return [ess1, ess2, ess3, time_bounce, eps, d]


if __name__ == '__main__':

    #funnel()
    #parallel run:
    #parallel.run_collect(inference_gym, runs=2, working_folder='working/', name_results='Tests/data/inference_gym')

    dimensions = [50, 100, 200, 500, 1000]#, 3000, 10000]

    for d in dimensions:
        parallel.run_collect(lambda n: compute_free_time(n, d), runs= 2, working_folder= 'working/', name_results= 'Tests/data/dimensions/'+str(d) + 'Rosenbrock')

