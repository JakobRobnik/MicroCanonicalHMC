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
    eta = (2.93 * np.power(d, -0.78) * np.logspace(-0.8, 0.8, 24))[n]

    #length = 1.5 * np.sqrt(d)
    #sampler = CTV.Sampler(Target = IllConditionedGaussian(d= d, condition_number=100), eps= 3)
    #sampler = ESH.Sampler(Target= StandardNormal(d= d), eps=1.0)
    sampler = ESH.Sampler(Target= IllConditionedGaussian(d= d, condition_number= kappa), eps=1.0)
    #sampler = ESH.Sampler(Target= Rosenbrock(d= d), eps= 0.5)
    #a= np.sqrt(np.concatenate((np.ones(d//2) * 2.0, np.ones(d//2) * 10.498957879911487)))
    #sampler = ESH.Sampler(Target= DiagonalPreconditioned(Rosenbrock(d= d), a), eps= 0.5)

    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    # ess = sampler.sample(x0, length, track= 'ESS', langevin_eta= eta)
    # return [ess, eta, sampler.eps, d]

    ess = sampler.sample(x0, length, track= 'ESS')
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
    mu = np.arange(1, 9)[n]

    d = 50
    eps, L = 1.0, 1.5 * np.sqrt(d)
    esh = ESH.Sampler(Target= BiModalEqual(d, mu), eps=eps)
    np.random.seed(1)
    x0 = np.random.normal(size= d)
    x0[0] += mu * 0.5

    avg_island_size = esh.sample(x0, L, prerun_steps= 500, track= 'ModeMixing')

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
    print('probability outside of bins', probability[-1])

    for i in range(len(bins)):
        density = probability[i] / (bins[i][1] - bins[i][0])
        plt.fill_between(bins[i], np.zeros(2), density * np.ones(2), alpha = 0.5, color = 'tab:blue')



def bimodal():

    xmax = 3.5 #how many sigma away from the mean of the gaussians do we want to have bins

    def get_bins(mu, sigma, num_bins_per_mode):
        bins_mode = np.array([[- xmax + i * 2 * xmax / num_bins_per_mode, - xmax + (i+1) * 2 * xmax / num_bins_per_mode] for i in range(num_bins_per_mode)])

        bins = np.concatenate(( (bins_mode * sigma[0]) + mu[0], (bins_mode * sigma[1]) + mu[1]  ))

        return bins

    d = 50
    mu1, sigma1= 0.0, 1.0 # the first Gaussian
    mu2, sigma2, f = 7.0, 0.5, 0.2 #the second Gaussian

    name = 'sep'+str(mu2)+'_f'+str(f)+'_sigma'+str(sigma2)
    load = False

    eps, L = 1.0, 1.5 * np.sqrt(d)
    bins_per_mode = 20
    bins = get_bins([mu1, mu2], [sigma1, sigma2], bins_per_mode)

    sampler = ESH.Sampler(Target= BiModal(d=d, mu1= mu1, mu2 = mu2, sigma1= sigma1, sigma2 = sigma2, f= f), eps=eps)

    #X = esh.Target.draw(1000)[:, 0]
    np.random.seed(0)
    x0 = sampler.Target.draw(1)[0]  # we draw an initial condition from the target
    if load:
        P = np.load('Tests/data/bimodal_marginal'+name+'.npy')
    else:
        P = sampler.sample(x0, L, prerun_steps=5000, max_steps= 50000000, track='Marginal1d', bins= bins)

    my_hist(bins, P)
    f1, f2 = np.sum(P[:bins_per_mode]), np.sum(P[bins_per_mode : 2 * bins_per_mode])
    print('f = ' + str(f2 / (f1 + f2)) + '  (true f = ' + str(f) + ')')

    t = np.linspace(-xmax*sigma1+mu1-0.5, xmax*sigma2 + mu2 + 0.5, 1000)

    plt.plot(t, (1- f)*norm.pdf(t, loc = mu1, scale = sigma1) + f * norm.pdf(t, loc = mu2, scale = sigma2), color = 'black')
    plt.xlim(t[0], t[-1])
    plt.ylim(0, 0.4)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$p(x_1)$')

    plt.savefig('Tests/'+name+'.png')
    np.save('Tests/data/bimodal_marginal'+name+'.npy', P)

    plt.show()



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


def bimodal_explore():
    mu = 7.0

    d = 10
    eps, L = 1.0, 20 * np.sqrt(d)
    esh = ESH.Sampler(Target=BiModalEqual(d=d, mu=mu), eps=eps)
    np.random.seed(1)
    x0 = np.random.normal(size=d)
    x0[0] += mu * 0.5

    X, W = esh.sample(x0, L, prerun_steps=500, track='FullTrajectory')

    plt.subplot(2, 1, 1)
    plt.plot(X[:, 0])
    plt.ylabel('x1')
    plt.xlim(22000, 25000)

    plt.subplot(2, 1, 2)
    plt.plot([2 * esh.Target.nlogp(x) for x in X])
    plt.ylabel('- 2 log p')
    plt.xlabel('steps')
    plt.xlim(22000, 25000)
    plt.savefig('no_jump_in_logp')
    plt.show()



if __name__ == '__main__':


    #parallel.run_collect(lambda n: bounce_frequency_full_bias(n, 250), runs=2, working_folder='working/', name_results= 'Tests/data/bounces_eps1')

    bimodal()
    #parallel.run_collect(bimodal_mixing, runs=2, working_folder='working/', name_results= 'Tests/data/mode_mixing_d50_L1.5')

    #parallel.run_collect(lambda n: bounce_frequency(n, 32), runs=2, working_folder='working/', name_results= 'Tests/data/rosenbrock')
    #parallel.run_collect(lambda n: bounce_frequency(n, 100, 10000.0), runs=4, working_folder='working/', name_results= 'Tests/data/no_langevin_kappa10000')

    #parallel.run_collect(ill_conditioned, runs=3, working_folder='working/', name_results= 'Tests/data/kappa/L1.5')
