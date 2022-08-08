import numpy as np
import matplotlib.pyplot as plt
import ESH
import targets
from bias import *

tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plot_bounce_frequency():
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



def plot_kappa():
    kappa = [1, 10, 100, 1000]
    d = 50
    #import matplotlib.colors
    colors= plt.cm.viridis(np.linspace(1, 0, len(kappa)))
    fig, ax = plt.subplots()

    for n in range(len(kappa)):
        target = targets.IllConcitionedGaussian(d=d, condition_number=kappa[n])

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
    eps_arr = [0.05, 0.1, 0.5, 1, 2]
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
    plt.savefig('Tests/eps.png')
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



#plot_bounce_f+requency()
#plot_eps()
#plot_energy()
plot_kappa()