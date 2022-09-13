import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def inference_gym():
    data = np.load('Tests/data/inference_gym.npy')
    names= [r'Gaussian ($\kappa = 1$)', r'Gaussian ($\kappa = 10^4$)', r'Rosenbrock ($Q = 0.5$)']
    for i in range(3):
        plt.plot(data[:, 3], data[:, i], 'o', color= tab_colors[i], label= names[i])

    plt.xlabel(r'$\Delta x$')
    plt.ylabel('ESS')
    plt.legend()
    plt.yscale('log')
    plt.show()


def ess_free_time():

    dimensions = [50, 100, 200, 500, 1000, 3000, 10000]

    E, L = [], []
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    folder_name = 'StandardNormal_t'
    #folder_name = 'Rosenbrock_precondition_t'
    factor = 1.0 #Leapfrog
    #factor = 0.25 #Yoshida
    for i in range(len(dimensions)):
        d = dimensions[i]
        X = np.load('Tests/data/dimensions/'+folder_name+'/'+str(d)+'.npy')
        print(X)
        #peak
        plt.plot(X[:, 1], factor*X[:, 0], color = tab_colors[i], alpha= 0.5)

        #highest point
        imax= np.argmax(factor*X[:, 0])
        L.append(X[imax, 1])
        E.append(X[imax, 0])
        plt.plot(X[imax, 1], factor*X[imax, 0], '.', color = tab_colors[i])
        plt.text(X[imax, 1] * 1.05, factor*X[imax, 0]*1.03, 'd= '+str(d), color = tab_colors[i], alpha = 0.5) #dimension tag


    plt.ylabel('ESS')
    plt.xscale('log')
    plt.xlabel("orbit length between bounces")

    ###  L ~ sqrt(d)  ###
    plt.subplot(1, 3, 2)
    for i in range(len(dimensions)):
        plt.plot(dimensions[i], L[i], 'o', color = tab_colors[i])

    slope= np.dot(np.sqrt(dimensions[1:]), L[1:]) / np.sum(dimensions[1:])
    print(slope)
    plt.title(r'$L \approx$' +'{0:.4}'.format(slope) + r' $\sqrt{d}$')
    plt.plot(dimensions, slope * np.sqrt(dimensions), ':', color = 'black')
    plt.xlabel('d')
    plt.ylabel('optimal orbit length between bounces')
    plt.xscale('log')
    plt.yscale('log')


    #ESS(d)
    plt.subplot(1, 3, 3)

    for i in range(len(dimensions)):
        plt.plot(dimensions[i], E[i], 'o', color= tab_colors[i])

    from scipy.stats import linregress

    res = linregress(np.log(dimensions), np.log(E))


    plt.title(r'$L \propto d^{-\alpha}, \quad \alpha = $' + '{0}'.format(np.round(-res.slope, 2)))
    plt.plot(dimensions, np.exp(res.intercept) * np.power(dimensions, res.slope), ':', color='black')
    plt.xlabel('d')
    plt.ylabel('ESS')
    plt.xscale('log')
    plt.yscale('log')

    #plt.savefig('Tests/bounce_dimension_dependence/'+folder_name+'.png')
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

#
# def plot_bounce_condition():
#     d = 50
#     fig, ax = plt.subplots()
#
#     esh = ESH.Sampler(Target= targets.StandardNormal(d=d), eps=0.1)
#
#
#     num_bounces = 1
#
#     X = np.load('Tests/bounce_frequency/X'+str(num_bounces)+'.npy')
#     w = np.load('Tests/bounce_frequency/w'+str(num_bounces)+'.npy')
#
#     variance_bias = bias(X, w, esh.Target.variance)
#     plt.plot(variance_bias, label = '# bounces = ' + str(num_bounces))
#
#
#
#     #plt.plot([0, len(variance_bias)], [0.1, 0.1], ':', color='black', alpha = 0.5) #threshold for effective sample size 200
#     plt.legend()
#     plt.yscale('log')
#     plt.xscale('log')
#     plt.xlabel('# evaluations')
#     plt.ylabel('bias')
#     #plt.xlim(1, 1e3)
#     #ess_axis(ax)
#     #plt.savefig('Tests/bounce_frequency.png')
#     plt.show()



def kappa_comparisson():
    #kappa_mchmc = [1, 10, 100, 1000]
    #ess_mchmc = [0.00796591, 0.00345943, 0.00129648, 0.00030617]
    kappa = np.logspace(0, 3, 12)
    kappa_nuts = np.logspace(0, 4, 18)


    ess_mchmc = np.load('Tests/data/kappa.npy')[:, 0]
    ess_nuts = np.load('Tests/data/kappa_NUTS_adapt_mass.npy')#[:18]


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

    num_mixing = np.load('Tests/data/mode_mixing.npy')[:, 0]
    mu = np.arange(1, 9)

    plt.plot(mu, num_mixing, 'o:', label = 'MCHMC')

    num_mixing = np.load('Tests/data/mode_mixing_NUTS.npy')[0]
    mu = np.load('Tests/mode_mixing_NUTS.npy')[1]

    plt.plot(mu, num_mixing, 'o:', label= 'NUTS')


    plt.yscale('log')
    plt.xlabel(r'$\mu$')
    plt.ylabel('average steps spent in a mode')
    plt.legend()
    plt.savefig('Tests/mode_mixing.png')

    plt.show()


def plot_funnel():

    def gaussianize(z, theta):
        return (z.T * np.exp(-0.5 * theta)).T, 0.3 * theta

    eps, free_time = 0.1, 6
    data = np.load('Tests/data/funnel_free'+str(free_time) + '_eps'+str(eps)+'.npz')
    z, theta, w = data['z'], data['theta'], data['w']


    data = np.load('Tests/data/funnel_HMC.npz')
    zHMC, thetaHMC = data['z'], data['theta']


    ff, ff_title, ff_ticks = 18, 20, 14
    plt.rcParams['xtick.labelsize'] = ff_ticks
    plt.rcParams['ytick.labelsize'] = ff_ticks
    plt.figure(figsize=(24, 8))


    ####   2d marginal in the original coordinates ####
    plt.subplot(1, 3, 1)
    plt.title('Original coordinates', fontsize = ff_title)
    plt.plot(zHMC[:, 0], thetaHMC, '.', ms= 1, color = 'tab:orange', label = 'NUTS')

    #plt.hist2d(z[:, 0], theta, cmap = 'Blues', weights= w, bins = 70, density=True, range= [[-30, 30], [-8, 8]], label ='MCHMC')
    plt.plot(z[::5000, 0], theta[::5000], '.', ms= 1, color = 'tab:blue', label = 'MCHMC')
    #plt.hist2d(z[:, 0], theta, weights= w, bins = 100, density=True, label = 'MCHMC')
    plt.xlim(-30, 30)
    plt.ylim(-8, 8)
    plt.xlabel(r'$z_0$', fontsize = ff)
    plt.ylabel(r'$\theta$', fontsize = ff)



    #### 2d marginal in the gaussianized coordinates ####
    plt.subplot(1, 3, 2)
    plt.title('Gaussianized coordinates', fontsize = ff_title)
    Gz, Gtheta = gaussianize(z, theta)
    plt.hist2d(Gz[:, 0], Gtheta, cmap = 'Blues', weights= w, bins = 70, density=True, range= [[-3, 3], [-3, 3]], label ='MCHMC')
    GzHMC, GthetaHMC = gaussianize(zHMC, thetaHMC)
    plt.plot(GzHMC[:, 0], GthetaHMC, '.', ms= 4, color = 'tab:orange', alpha = 0.5, label ='NUTS')

    #level sets
    p_level = np.array([0.6827, 0.9545])
    x_level = np.sqrt(-2 * np.log(1 - p_level))
    phi = np.linspace(0, 2* np.pi, 100)
    for i in range(2):
        plt.plot(x_level[i] * np.cos(phi), x_level[i] * np.sin(phi), color = 'black', alpha= ([1, 0.5])[i])

    plt.xlabel(r'$\widetilde{z_0}$', fontsize = ff)
    plt.ylabel(r'$\widetilde{\theta}$', fontsize = ff)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)



    #### 1d theta marginal####
    plt.subplot(1, 3, 3)
    plt.title(r'$\theta$-marginal', fontsize = ff_title)
    plt.hist(thetaHMC, color='tab:orange', density=True, bins = 20, alpha = 0.5, label = 'NUTS')
    plt.hist(theta, weights= w, color='tab:blue', density=True, bins = 20, alpha = 0.5,  label = 'MCHMC')

    t= np.linspace(-10, 10, 100)
    plt.plot(t, norm.pdf(t, scale= 3.0), ':', color= 'black', alpha = 0.5, label = 'exact')

    #xmax = np.min([np.max(thetaHMC), np.max(theta)])
    #plt.xlim(-xmax, xmax)
    #plt.ylim(0, 1)

    plt.legend(fontsize = ff)
    plt.xlabel(r'$\theta$', fontsize = ff)
    plt.ylabel(r'$p(\theta)$', fontsize = ff)
    plt.savefig('submission/funnel.pdf')

    plt.show()





def plot_rosenbrock():


    xmin, xmax, ymin, ymax = -2.3, 3.9, -2, 16

    #ff_ticks =
    #plt.rcParams['xtick.labelsize'] = ff_ticks
    #plt.rcParams['ytick.labelsize'] = ff_ticks
    plot = sns.JointGrid(height=10, xlim=(xmin, xmax), ylim=(ymin, ymax))
    ff = 20

    # MCHMC
    d = 36
    X = np.load('Tests/data/rosenbrock.npz')
    x, y = X['samples'][:, 0], X['samples'][:, d // 2]
    w = X['w']
    sns.histplot(x=x, y=y, weights=w, bins=200, ax=plot.ax_joint)

    #sns.scatterplot(x=[], y=[], ax= plot.ax_joint, color = 'tab:blue')

    # # marginals
    sns.histplot(x= x, weights= w, bins= 40, fill= True, alpha = 0.5, linewidth= 0, ax= plot.ax_marg_x, stat= 'density', color= 'tab:blue', zorder = 2)
    sns.histplot(y= y, weights= w, bins= 40, fill= True, alpha = 0.5, linewidth= 0, ax= plot.ax_marg_y, stat= 'density', color= 'tab:blue', label= 'MCHMC', zorder = 2)


    # NUTS
    X= np.load('Tests/data/rosenbrock_HMC.npz')
    x, y = X['x'][:, 0], X['y'][:, 0]

    sns.scatterplot(x, y, s= 6, linewidth= 0, ax= plot.ax_joint, alpha = 0.7, color= 'tab:orange')

    # marginals
    sns.histplot(x=x, bins= 40, fill= True, alpha = 0.5, linewidth= 0, ax= plot.ax_marg_x, stat= 'density', color= 'tab:orange', zorder = 1)
    sns.histplot(y=y, bins= 40, fill= True, alpha = 0.5, linewidth= 0, ax= plot.ax_marg_y, stat= 'density', color= 'tab:orange', label= 'NUTS', zorder = 1)


    #exact
    import targets
    ros = targets.Rosenbrock(d = 2)
    X = ros.draw(1000)
    x, y = X[:, 0], X[:, 1]

    sns.scatterplot(x, y, s= 6, linewidth= 0, ax= plot.ax_joint, color= 'black', alpha = 0.5)

    # marginals
    sns.lineplot(x, np.exp(-0.5 * np.square(x - 1)) / np.sqrt(2 * np.pi), linewidth= 1, ax= plot.ax_marg_x, color= 'black', alpha = 0.5)
    ros = targets.Rosenbrock(d=2)
    X = ros.draw(5000000)
    x, y = X[:, 0], X[:, 1]
    sns.histplot(y=y, bins= 2000, fill= False, element= 'step', linewidth= 1, ax= plot.ax_marg_y, stat= 'density', color= 'black', alpha = 0.5, label= 'exact\nsamples')


    plot.ax_marg_y.legend(fontsize = ff)

    plot.set_axis_labels(r'$x_0$', r'$y_0$', fontsize= ff)
    plt.tight_layout()
    plt.savefig('submission/rosenbrock.pdf')
    plt.show()


def funnel_debug():

    data = np.load('Tests/data/funnel.npz')
    z, theta, w = data['z'], data['theta'], data['w']

    data = np.load('Tests/data/funnel_HMC.npz')
    zHMC, thetaHMC = data['z'], data['theta']


    num = 100000
    plt.plot(z[:num, 0], theta[:num])
    plt.plot(zHMC[:, 0], thetaHMC, '.', ms=1, color='tab:orange', label='NUTS')
    # plt.xlim(-30, 30)
    # plt.ylim(-8, 8)
    plt.xlabel(r'$z_0$')
    plt.ylabel(r'$\theta$')
    plt.show()


#ess_free_time()
plot_rosenbrock()
#plot_funnel()
# d = 36
# X = np.load('Tests/data/rosenbrock.npz')
# x, y = X['samples'][:, 0], X['samples'][:, d // 2]
# w = X['w']
# plt.plot(x[:10000], y[:10000])
# plt.show()
#
