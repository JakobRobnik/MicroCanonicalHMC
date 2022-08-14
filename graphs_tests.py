import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import ESH
import targets
from bias import *

tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def bias_bounce_frequency():
    nn = [1, 10, 100, 1000, 10000, 100000]
    d = 50
    fig, ax = plt.subplots()

    esh = ESH.Sampler(Target= targets.StandardNormal(d=d), eps=0.1)

    for n in range(len(nn)):
        num_bounces = nn[n]

        X = np.load('Tests/bounce_frequency/X'+str(num_bounces)+'.npy')
        w = np.load('Tests/bounce_frequency/w'+str(num_bounces)+'.npy')

        variance_bias = bias(X, w, esh.Target.variance)
        plt.plot(variance_bias, label = '# bounces = ' + str(num_bounces), color = tab_colors[n])

        X = np.load('Tests/bounce_frequency/X_half_sphere_'+str(num_bounces)+'.npy')
        w = np.load('Tests/bounce_frequency/w_half_sphere_'+str(num_bounces)+'.npy')

        variance_bias = bias(X, w, esh.Target.variance)
        plt.plot(variance_bias, ls = ':', color = tab_colors[n])


    plt.plot([0, len(variance_bias)], [0.1, 0.1], ':', color='black', alpha = 0.5) #threshold for effective sample size 200
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel('bias')
    plt.xlim(1, 1e6)
    ess_axis(ax)
    plt.savefig('Tests/bounce_frequency.png')
    plt.show()



def ess_free_time():

    X = np.load('Tests/free_time_ill.npy')
    plt.plot(X[:, 1] * 0.1, X[:, 0], '.', color = 'black')


    # X = np.load('Tests/free_time.npy')
    # plt.plot(X[:, 1] * 0.1, X[:, 0], '.', color = 'black')

    #
    # X = np.load('Tests/free_time2.npy')
    # plt.plot(X[:, 1] * 0.1, X[:, 0], '.', color = 'black')


    plt.ylabel('ESS')
    plt.yscale('log')
    plt.xlabel("rescaled time between bounces")
    #plt.savefig('Tests/free_time_fine_tuning.png')
    plt.show()


def ess_epsilon():

    X = np.load('Tests/eps3.npy')
    plt.plot(X[:, 2], X[:, 0] , '.', color = 'black')

    plt.ylabel('ESS')
    plt.xlabel(r"$\epsilon$")
    #plt.savefig('Tests/eps_fine_tuning.png')
    plt.show()


def plot_kappa():
    kappa = [1, 10, 100, 1000]
    d = 50
    #import matplotlib.colors
    colors= plt.cm.viridis(np.linspace(1, 0, len(kappa)))
    fig, ax = plt.subplots()

    for n in range(len(kappa)):
        target = targets.IllConditionedGaussian(d=d, condition_number=kappa[n])

        X = np.load('Tests/kappa/X'+str(kappa[n])+'.npy')
        w = np.load('Tests/kappa/w'+str(kappa[n])+'.npy')

        variance_bias = bias(X, w, target.variance)
        steps = np.arange(1, 1+len(variance_bias))
        plt.plot(steps, variance_bias, label = r'$\kappa$ = {0}'.format(kappa[n]), color =colors[n])

        # X = np.load('Tests/kappa/X_perp_'+str(kappa[n])+'.npy')
        # w = np.load('Tests/kappa/w_perp_'+str(kappa[n])+'.npy')
        # variance_bias = bias(X, w, target.variance)
        #
        # plt.plot(steps, variance_bias, ':', color =tab_colors[n])
        #
        # X = np.load('Tests/kappa/X_half_sphere_'+str(kappa[n])+'.npy')
        # w = np.load('Tests/kappa/w_half_sphere_'+str(kappa[n])+'.npy')
        # variance_bias = bias(X, w, target.variance)
        #
        # plt.plot(steps, variance_bias, '--', color =tab_colors[n])
        #


    #plt.plot([1, len(variance_bias)+1], [0.1, 0.1], ':', color='black', alpha = 0.5) #threshold for effective sample size 200
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel('bias')
    plt.xlim(1, 1e6)
    ess_axis(ax)
    plt.savefig('Tests/kappa.png')
    plt.show()

def ess_axis(ax):

    steps_to_ess = lambda n: 200/n
    ess_to_steps = lambda e: 200/e

    ymin, ymax = plt.ylim()
    ax2 = ax.secondary_xaxis(np.log10(0.1 / ymin) / np.log10(ymax / ymin), functions=(steps_to_ess, ess_to_steps))
    ax2.set_xlabel('ESS')


def plot_eps():
    eps_arr = np.logspace(np.log10(0.05), np.log10(0.5), 6)
    #[0.05, 0.1, 0.5, 1, 2]
    d= 50

    target = targets.StandardNormal(d=d)
    fig, ax = plt.subplots()

    for n in range(len(eps_arr)):

        X = np.load('Tests/eps/X'+str(n)+'.npy')
        w = np.load('Tests/eps/w'+str(n)+'.npy')

        variance_bias = bias(X, w, target.variance)
        plt.plot(np.arange(1, 1+len(variance_bias)), variance_bias, label = r'$\epsilon$ = {0}'.format(eps_arr[n]))


    plt.plot([1, len(variance_bias)+1], [0.1, 0.1], ':', color='black', alpha = 0.5) #threshold for effective sample size 200
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel('bias')
    plt.xlim(1, 1e6)
    ess_axis(ax)
    #plt.savefig('Tests/eps.png')
    plt.show()



def plot_energy():
    eps_arr = [0.05, 0.1, 0.5, 1, 2]

    var = []
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for n in range(len(eps_arr)):

        E = np.load('Tests/energy/E'+str(n)+'.npy')
        var.append(np.std(E))
        plt.plot(np.arange(1, 1+len(E))[::1000], E[::1000], label = r'$\epsilon$ = {0}'.format(eps_arr[n]), zorder = -n, color = tab_colors[n])


    plt.legend(loc= 4)
    plt.xlabel('rescaled time')
    plt.ylabel('energy')
    #plt.xscale('log')
    plt.xlim(1, 1e6)

    plt.subplot(1, 2, 2)
    for n in range(len(eps_arr)):
        plt.plot([eps_arr[n], ], [var[n], ], 'o', color = tab_colors[n])
    plt.yscale('log')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('STD[energy]')

    plt.savefig('Tests/energy.png')
    plt.show()


def point_reduction(points, reduction_factor):
    """reduces the number of points for plotting purposes"""

    indexes=  np.concatenate((np.arange(1, 1 + points//reduction_factor, dtype = int), np.arange(1 + points//reduction_factor, points, reduction_factor, dtype = int)))
    return indexes


def plot_bounce_condition():
    d = 50
    fig, ax = plt.subplots()

    esh = ESH.Sampler(Target= targets.StandardNormal(d=d), eps=0.1)


    num_bounces = 1

    X = np.load('Tests/bounce_frequency/X'+str(num_bounces)+'.npy')
    w = np.load('Tests/bounce_frequency/w'+str(num_bounces)+'.npy')

    variance_bias = bias(X, w, esh.Target.variance)
    plt.plot(variance_bias, label = '# bounces = ' + str(num_bounces))



    #plt.plot([0, len(variance_bias)], [0.1, 0.1], ':', color='black', alpha = 0.5) #threshold for effective sample size 200
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel('bias')
    #plt.xlim(1, 1e3)
    #ess_axis(ax)
    #plt.savefig('Tests/bounce_frequency.png')
    plt.show()



def kappa_comparisson():
    #kappa_mchmc = [1, 10, 100, 1000]
    #ess_mchmc = [0.00796591, 0.00345943, 0.00129648, 0.00030617]
    kappa = np.logspace(0, 3, 12)
    kappa_nuts = np.logspace(0, 4, 18)


    ess_mchmc = np.load('Tests/kappa.npy')[:, 0]
    ess_nuts = np.load('Tests/kappa_NUTS.npy')[:18]


    #ess_hmc = [0.045207956600361664, 0.06207324643078833, 0.020695364238410598, 0.015683814303638646, 0.00991866693116445, 0.0061502506227128755, 0.0023066188427693264, 0.0014471465887137037, 0.0009537116071471148, 0.00030081611411760103, 0.00033305800538221736, 0.000291579192908794, 0.00012951547612803123, 9.597615184578936e-05, 2.8260107156674315e-05]
        #[0.04527960153950645, 0.05140066820868671, 0.02563116749967961, 0.00956892014736137, 0.0056228738508251564, 0.002301469488268259, 0.0014498216719343521, 0.0006908057212529834, 0.0003906494155884743, 0.0004268642763690071]

    plt.plot(kappa_nuts, ess_nuts, 'o:', color = 'gold', label = 'NUTS')
    plt.plot(kappa, ess_mchmc, 'o:', color = 'black', label = 'MCHMC')


    plt.xlabel('condition number')
    plt.ylabel('ESS')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Tests/kappa_comparisson.png')
    plt.show()


def mode_mixing():

    num_mixing = np.load('Tests/mode_mixing.npy')[:, 0]
    mu = np.arange(1, 9)

    plt.plot(mu, num_mixing, 'o:', label = 'MCHMC')

    num_mixing = np.load('Tests/mode_mixing_NUTS.npy')[0]
    mu = np.load('Tests/mode_mixing_NUTS.npy')[1]

    plt.plot(mu, num_mixing, 'o:', label= 'NUTS')


    plt.yscale('log')
    plt.xlabel(r'$\mu$')
    plt.ylabel('average steps spent in a mode')
    plt.legend()
    plt.savefig('mode_mixing.png')

    plt.show()


def plot_funnel():

    samples = np.load('funnel_samples.npy')
    w = np.load('funnel_w.npy')

    d = 20
    theta, z = samples[:, -1], samples[:, :d-1]

    #z0, theta = samples['z'][:, 0], samples['theta']

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    plt.title('Original coordinates')
    plt.hist2d(z[:, 0], theta, weights= w, bins = 30, density=True )
    plt.xlim(-30, 30)
    plt.xlabel(r'$z_0$')
    plt.ylabel(r'$\theta$')

    plt.subplot(2, 2, 2)
    plt.title('Gaussianized coordinates')
    Gz, Gtheta = gaussianize(z, theta)
    plt.hist2d(Gz[:, 0], Gtheta, weights= w, bins = 20, density=True )


    p_level = np.array([0.6827, 0.9545])
    x_level = np.sqrt(-2 * np.log(1 - p_level))
    phi = np.linspace(0, 2* np.pi, 100)
    for i in range(2):
        plt.plot(x_level[i] * np.cos(phi), x_level[i] * np.sin(phi), color = 'black', alpha= ([0.1, 0.5])[i])

    plt.xlabel(r'$\widetilde{z_0}$')
    plt.ylabel(r'$\widetilde{\theta}$')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.subplot(2, 2, 3)
    plt.title(r'$\theta$-marginal')
    plt.hist(theta, weights= w, color='tab:blue', cumulative=True, density=True, bins = 1000)

    t= np.linspace(-10, 10, 100)
    plt.plot(t, norm.cdf(t, scale= 3.0), color= 'black')

    plt.xlabel(r'$\theta$')
    plt.ylabel('CDF')
    plt.savefig('funnel_nuts')

    plt.show()


def gaussianize(z, theta):
    return (z.T * np.exp(-0.5 * theta)).T, 0.3 * theta



plot_funnel()

#kappa_comparisson()
#plot_energy()
#plot_kappa()