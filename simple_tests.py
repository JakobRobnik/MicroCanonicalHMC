import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
import os

import ESH
import CTV
import parallel
from targets import *
import bias



def bounce_frequency(n, d, kappa = 100.0):

    # free_steps_arr = (np.linspace(50, 250, 18)).astype(int)
    length = (1.5 * np.sqrt(d) * np.logspace(-0.8, 0.8, 24))[n]
    #length = 1.5 * np.sqrt(d)
    #sampler = CTV.Sampler(Target = IllConditionedGaussian(d= d, condition_number=100), eps= 3)
    #sampler = ESH.Sampler(Target= StandardNormal(d= d), eps=1.0)
    sampler = ESH.Sampler(Target= IllConditionedGaussian(d= d, condition_number= kappa), eps=1.5)
    #sampler = ESH.Sampler(Target= Rosenbrock(d= d), eps= 0.5)
    #a= np.sqrt(np.concatenate((np.ones(d//2) * 2.0, np.ones(d//2) * 10.498957879911487)))
    #sampler = ESH.Sampler(Target= DiagonalPreconditioned(Rosenbrock(d= d), a), eps= 0.5)

    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    ess = sampler.sample(x0, length, prerun_steps= 500, track= 'ESS')

    return [ess, length, sampler.eps, d]



def bounce_frequency_full_bias(n, d):
    length = [2, 5, 10, 30, 50, 75, 80, 90, 100, 1000, 10000, 10000000]
    L = length[n]

    sampler = ESH.Sampler(Target= StandardNormal(d= d), eps=1)
    np.random.seed(1)
    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    bias = sampler.sample(x0, L, track= 'FullBias')

    return bias



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



def bimodal_mixing(n):
    mu = np.arange(3.0, 5.0, 7.0, 10.0, 15.0)[n]

    d = 50
    eps, L = 1.0, 1.5 * np.sqrt(d)
    esh = ESH.Sampler(Target= BiModal(d=d, mu= mu), eps=eps)
    np.random.seed(0)
    x0 = np.random.normal(size= d)

    avg_island_size = esh.sample(x0, L, prerun_steps=500, track= 'ModeMixing')
    print(avg_island_size)
    sys.stdout.flush()

    return [avg_island_size, mu, L, eps, d]


def ill_conditioned(n):
    kappa = np.logspace(0, 5, 18)[n]
    eps = np.array(10 * [2.0, ] + 2 * [1.5, ] + 3*[1.0, ] + [0.7,  ] + 2 * [0.5])[n]
    d = 100
    L = 1.5 * np.sqrt(d)
    esh = ESH.Sampler(Target=IllConditionedGaussian(d=d, condition_number=kappa), eps=eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    ess = esh.sample(x0, L, prerun_steps= 500, track= 'ESS')

    return [ess, L, eps, d, kappa]



def funnel_plot():

    eps = 0.1
    free_time = 6
    free_steps = (int)(free_time / eps)
    d = 20
    esh = ESH.Sampler(Target= Funnel(d=d), eps=eps)
    np.random.seed(0)
    x0 = np.zeros(d)
    samples, w = esh.sample(x0, free_steps, 5000000, track= 'FullTrajectory')
    np.savez('Tests/data/funnel_free'+str(free_time) + '_eps'+str(eps), z = samples[:, :-1], theta= samples[:, -1], w = w)


def funnel_ess(n):
    length = (1 * np.sqrt(d) * np.logspace(-0.4, 0.4, 12))[n]
    sampler = ESH.Sampler(Target=Funnel(d =20), eps= 0.1)

    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    ess = sampler.sample(x0, length, prerun_steps=500, track= 'ESS')

    return [ess, length, sampler.eps, d]



def rosenbrock():

    eps = 0.001
    free_steps = (int)(1 / eps)
    d = 36
    esh = ESH.Sampler(Target= Rosenbrock(d=d), eps=eps)
    np.random.seed(0)
    x0 = np.zeros(d)
    samples, w = esh.sample(x0, free_steps, max_steps= 10000000, prerun_steps= 500, track= 'FullTrajectory')
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



def dimension_dependence():


    dimensions = [50, 100, 200, 500, 1000]#, 3000, 10000]
    #dimensions = [3000, 10000]

    #condition_numbers = [1, 10, 100, 1000, 10000]
    condition_numbers = np.logspace(0, 5, 18)


    #name_folder= 'Tests/data/dimensions/Rosenbrock_t'
    name_folder = 'Tests/data/kappa/'

    # if not os.path.exists(name_folder):
    #     os.mkdir(name_folder)

    # for d in dimensions:
    #     parallel.run_collect(lambda n: bounce_frequency(n, d), runs= 2, working_folder= 'working/', name_results= name_folder+'/'+str(d))
    #
    # for i in range(len(condition_numbers)):
    #     parallel.run_collect(lambda n: bounce_frequency(n, 100, condition_numbers[i]), runs= 4, working_folder='working/', name_results=name_folder + '/' + str(i)+ 'eps0.7')

    parallel.run_collect(lambda n: bounce_frequency(n, 100, condition_numbers[11]), runs=4, working_folder='working/', name_results=name_folder + '/' + str(11) + 'eps1.5')


if __name__ == '__main__':

    #parallel.run_collect(lambda n: bounce_frequency_full_bias(n, 250), runs=2, working_folder='working/', name_results= 'Tests/data/bounces_eps1')

    parallel.run_collect(bimodal_mixing, runs=1, working_folder='working/', name_results= 'Tests/data/mode_mixing')


    #parallel.run_collect(lambda n: bounce_frequency(n, 32), runs=2, working_folder='working/', name_results= 'Tests/data/rosenbrock')
    #dimension_dependence()

    #parallel.run_collect(ill_conditioned, runs=3, working_folder='working/', name_results= 'Tests/data/kappa/L1.5')