import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import linregress
import arviz as az


import ESH
import targets
from bias import *

tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']





def bounce_frequency_full_bias():
    """ Figure 1 """

    def point_reduction(points, reduction_factor):
        """reduces the number of points for plotting purposes"""

        indexes = np.concatenate((np.arange(1, 1 + points // reduction_factor, dtype=int),
                                  np.arange(1 + points // reduction_factor, points, reduction_factor, dtype=int)))
        return indexes

    def ess_axis(ax, ff):

        steps_to_ess = lambda n: 200 / n
        ess_to_steps = lambda e: 200 / e

        ymin, ymax = plt.ylim()
        ax2 = ax.secondary_xaxis(np.log10(0.1 / ymin) / np.log10(ymax / ymin), functions=(steps_to_ess, ess_to_steps))
        plt.text(8.5e2, 0.07, 'ESS', fontsize = ff)


    length = [    2,     5,     10,   30,   50,    75,    80,   90,   100,  1000, 10000, 10000000]
    mask_plot = [True, True, False, True, False, False, True, True, True, False, False, False]
    #mask_plot = len(length) * [True, ]

    X = np.load('Tests/data/bounces_eps1.npy')
    indexes = point_reduction(len(X[0]), 100)


    ff, ff_title, ff_ticks = 18, 20, 14
    plt.rcParams['xtick.labelsize'] = ff_ticks
    plt.rcParams['ytick.labelsize'] = ff_ticks
    plt.figure(figsize= (20, 8))
    ax = plt.gca()
    ff, ff_title, ff_ticks = 18, 20, 16
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    colors = plt.cm.gist_heat(np.linspace(0.1, 0.9, np.sum(mask_plot)))[::-1]
    plotted = 0
    dash_size = [1, 5, 15, 25, 40, 80]
    for n in range(len(length)):
        if mask_plot[n]:
            plt.plot(indexes, X[n, indexes], linestyle='--', dashes=(dash_size[plotted], 2),  color = colors[plotted], label = r'$\Delta x = $' + str(length[n]))
            plt.plot(indexes, X[n, indexes], alpha= 0.1, color=colors[plotted])

            plotted += 1
    # plt.plot([0, len(variance_bias)], [0.1, 0.1], ':', color='black', alpha = 0.5) #threshold for effective sample size 200
    plt.legend(fontsize = ff)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('# gradient evaluations', fontsize = ff)
    plt.ylabel('bias', fontsize = ff)
    plt.xlim(1, 1e6)
    ess_axis(ax, ff)
    plt.savefig('submission/FullBias.pdf')
    plt.show()



def dimension_dependence_appendix():
    generalized = True

    if generalized:
        dimensions = [100, 200, 500, 1000, 3000, 10000]
        df = pd.read_csv('Tests/data/dimensions/StandardNormal_t.csv', sep='\t')
    else:
        dimensions = [100, 200, 1000, 3000, 10000]
        df = pd.read_csv('Tests/data/dimensions/StandardNormal.csv', sep='\t')

    skip_large = -7
    alpha = np.array(df['alpha'])[:skip_large]
    E, L = [], []

    ff, ff_title, ff_ticks = 18, 20, 14
    plt.rcParams['xtick.labelsize'] = ff_ticks
    plt.rcParams['ytick.labelsize'] = ff_ticks
    plt.figure(figsize= (20, 7))

    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #folder_name = 'Rosenbrock_precondition_t'
    factor = 1.0 #Leapfrog
    #factor = 0.25 #Yoshida
    for i in range(len(dimensions)):
        d = dimensions[i]
        #peak
        ess = factor * np.array(df['ess (d='+str(d)+')'])[:skip_large]
        ess_err = factor * np.array(df['err ess (d=' + str(d) + ')'])[:skip_large]
        plt.plot(alpha * np.sqrt(d), ess, '.:', color = tab_colors[i], alpha = 0.5)
        plt.fill_between(alpha * np.sqrt(d), ess - ess_err, ess + ess_err, color = tab_colors[i], alpha = 0.1)

        #highest point
        imax= np.argmax(ess)
        L.append(alpha[imax] * np.sqrt(d))
        E.append(ess[imax])
        plt.plot(L[-1], E[-1], 'o', markersize =10, color = tab_colors[i])
        plt.text(L[-1] * 1.05, E[-1]*1.03, 'd= '+str(d), color = tab_colors[i], alpha = 0.5) #dimension tag


    plt.xscale('log')
    plt.ylabel('ESS', fontsize = ff)
    if generalized:
        plt.xlabel(r'$L(\nu) = \epsilon / \log \sqrt{1 + \nu^2 d}$', fontsize = ff)
    else:
        plt.xlabel(r'$L$', fontsize=ff)

    ###  L ~ sqrt(d)  ###
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(len(dimensions)):
        plt.plot(dimensions[i], L[i], 'o', markersize =10, color = tab_colors[i])

    skip = 0
    slope= np.dot(np.sqrt(dimensions[skip:]), L[skip:]) / np.sum(dimensions[skip:])
    print(slope)
    plt.title(r'$L \approx$' +'{0:.4}'.format(slope) + r' $\sqrt{d}$', fontsize = ff)
    plt.plot(dimensions, slope * np.sqrt(dimensions), ':', color = 'black')
    plt.xlabel('d', fontsize = ff)
    plt.ylabel(r'optimal $L$', fontsize = ff)
    if generalized:
        plt.ylabel(r'optimal $L(\nu)$', fontsize=ff)

    plt.xscale('log')
    plt.yscale('log')


    # #ESS(d)
    # plt.subplot(1, 3, 3)
    #
    # for i in range(len(dimensions)):
    #     plt.plot(dimensions[i], E[i], 'o', color= tab_colors[i])
    #
    # from scipy.stats import linregress
    #
    # res = linregress(np.log(dimensions[skip:]), np.log(E[skip:]))
    #
    #
    # plt.title(r'ESS $\propto d^{-\alpha}, \quad \alpha = $' + '{0}'.format(np.round(-res.slope, 2)))
    # plt.plot(dimensions, np.exp(res.intercept) * np.power(dimensions, res.slope), ':', color='black')
    # plt.xlabel('d')
    # plt.ylabel('ESS')
    # plt.xscale('log')
    # plt.yscale('log')

    if generalized:
        plt.savefig('submission/GeneralizedTuning.pdf')
    else:
        plt.savefig('submission/BounceTuning.pdf')
    plt.show()


def dimension_dependence():

    dimensions = [50, 100, 200, 500, 1000]#, 3000, 10000]
    name= 'StandardNormal'
    df = pd.read_csv('Tests/data/dimensions/'+name+'.csv', sep='\t')
    df_t = pd.read_csv('Tests/data/dimensions/'+name+'_t.csv', sep='\t')
    alpha = np.array(df['alpha'])
    E, L = [], []
    Et, Lt = [], []
    plt.figure(figsize=(10, 5))
    factor = 1.0 #Leapfrog
    #factor = 0.25 #Yoshida
    for i in range(len(dimensions)):
        d = dimensions[i]
        ess = factor * np.array(df['ess (d='+str(d)+')'])

        #highest point
        imax= np.argmax(ess)
        L.append(alpha[imax] * np.sqrt(d))
        E.append(ess[imax])

        ess = factor * np.array(df_t['ess (d=' + str(d) + ')'])

        # highest point
        imax = np.argmax(ess)
        Lt.append(alpha[imax] * np.sqrt(d))
        Et.append(ess[imax])

    skip=1

    from scipy.stats import linregress
    plt.plot(dimensions, E, '.', color='tab:blue', label='equally spaced in distance')
    plt.plot(dimensions, Et, '.', color='tab:orange', label='equally spaced in time')

    res = linregress(np.log(dimensions[skip:]), np.log(E[skip:]))
    plt.plot(dimensions, np.exp(res.intercept) * np.power(dimensions, res.slope), color = 'tab:blue', alpha = 0.5)
    res = linregress(np.log(dimensions[skip:]), np.log(Et[skip:]))
    plt.plot(dimensions, np.exp(res.intercept) * np.power(dimensions, res.slope), color = 'tab:orange', alpha = 0.5)
    plt.legend()
    plt.xlabel('d')
    plt.ylabel('ESS')
    plt.xscale('log')
    plt.yscale('log')

    #plt.savefig('Tests/kappa100corrected.png')
    plt.show()


def kappa_dependence_prelim():

    kappa = [1, 10, 100, 10000, 10000]
    factor = 1.0  # Leapfrog
    # factor = 0.25 #Yoshida
    E, L = [], []


    plt.figure(figsize=(15, 5))

    ###  peaks ESS(L) for different kappa  ###
    plt.subplot(1, 3, 1)

    for i in range(len(kappa)):
        K = kappa[i]
        X = np.load('Tests/data/kappa/'+str(K)+'.npy')
        #peak
        plt.plot(X[:, 1], factor*X[:, 0], color = tab_colors[i], alpha= 0.5)

        #highest point
        imax= np.argmax(factor*X[:, 0])
        L.append(X[imax, 1])
        E.append(X[imax, 0])
        plt.plot(X[imax, 1], factor*X[imax, 0], '.', color = tab_colors[i])
        plt.text(X[imax, 1] * 1.05, factor*X[imax, 0]*1.03, 'kappa = '+str(K), color = tab_colors[i], alpha = 0.5) #dimension tag


    plt.ylabel('ESS')
    plt.xlabel("orbit length between bounces")


    ###  optimal L as a function of kappa  ###
    plt.subplot(1, 3, 2)
    for i in range(len(kappa)):
        plt.plot(kappa[i], L[i], 'o', color = tab_colors[i])

    # skip = 1
    # slope= np.dot(np.sqrt(dimensions[skip:]), L[skip:]) / np.sum(dimensions[1:])
    # print(slope)
    # plt.title(r'$L \approx$' +'{0:.4}'.format(slope) + r' $\sqrt{d}$')
    # plt.plot(dimensions, slope * np.sqrt(dimensions), ':', color = 'black')
    plt.xlabel('condition number')
    plt.ylabel('optimal orbit length between bounces')
    plt.xscale('log')
    #plt.yscale('log')


    ###  optimal ESS as a function of kappa  ###
    plt.subplot(1, 3, 3)

    for i in range(len(kappa)):
        plt.plot(kappa[i], E[i], 'o', color= tab_colors[i])

    #from scipy.stats import linregress
    #res = linregress(np.log(dimensions[skip:]), np.log(E[skip:]))


    #plt.title(r'ESS $\propto d^{-\alpha}, \quad \alpha = $' + '{0}'.format(np.round(-res.slope, 2)))
    #plt.plot(dimensions, np.exp(res.intercept) * np.power(dimensions, res.slope), ':', color='black')
    plt.xlabel('condition number')
    plt.ylabel('ESS')
    plt.xscale('log')
    plt.yscale('log')

    #plt.savefig('Tests/bounce_dimension_dependence/'+folder_name+'.png')
    plt.show()




def ill_conditioned():
    """Figure 2"""


    ff, ff_title, ff_ticks = 18, 20, 17

    plt.rcParams['xtick.labelsize'] = ff_ticks
    plt.rcParams['ytick.labelsize'] = ff_ticks
    plt.figure(figsize= (20, 8))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    kappa = np.logspace(0, 5, 18)


    # ess= [np.max(np.load('Tests/data/kappa/' + str(i) + '.npy')[:, 0]) for i in range(18)]
    # plt.plot(kappa, ess, 'o:', color = 'tab:purple',  label = 'MCHMC (fine tuned)')

    ess_mchmc = np.load('Tests/data/kappa/fine_tuned_l.npy')
    color = 'indigo'
    plt.plot(kappa, ess_mchmc[:, 0], 'o:', color=color, label='generalized MCHMC')
    plt.fill_between(kappa, ess_mchmc[:, 0] - ess_mchmc[:, 1], ess_mchmc[:, 0] + ess_mchmc[:, 1], color=color, alpha=0.07)

    ess_mchmc = np.load('Tests/data/kappa/fine_tuned_t.npy')
    color = 'cornflowerblue'
    plt.plot(kappa, ess_mchmc[:, 0], 'o:', color=color, label='MCHMC (fine tuning)')
    plt.fill_between(kappa, ess_mchmc[:, 0] - ess_mchmc[:, 1], ess_mchmc[:, 0] + ess_mchmc[:, 1], color=color, alpha=0.07)

    # ess_mchmc = np.load('Tests/data/kappa/fine_tuned.npy')
    # plt.plot(kappa, ess_mchmc[:, 0], 'o:', color='tab:blue', label='MCHMC (bounces in distance)')
    # plt.fill_between(kappa, ess_mchmc[:, 0] - ess_mchmc[:, 1], ess_mchmc[:, 0] + ess_mchmc[:, 1], color='tab:blue', alpha=0.07)

    # ess_mchmc = np.load('Tests/data/kappa/L1.5.npy')[:, 0]
    # plt.plot(kappa, ess_mchmc, 'o:', color = 'tab:blue', alpha = 0.5, label = 'MCHMC (tuning free)')
    ess_mchmc = np.load('Tests/data/kappa/no_tuning_t.npy')
    color = 'teal'
    plt.plot(kappa, ess_mchmc[:, 0], 'o:', color = color, label = 'MCHMC (tuning free)')
    plt.fill_between(kappa, ess_mchmc[:, 0] - ess_mchmc[:, 1], ess_mchmc[:, 0] + ess_mchmc[:, 1], color = color, alpha = 0.07)


    ess_nuts = np.load('Tests/data/kappa/NUTS.npy').T
    color = 'tab:orange'
    plt.plot(kappa, ess_nuts[:, 0], 'o:', color=color, label='NUTS')
    plt.fill_between(kappa, ess_nuts[:, 0] - ess_nuts[:, 1], ess_nuts[:, 0] + ess_nuts[:, 1], color=color, alpha=0.07)

    plt.ylabel('ESS', fontsize= ff)
    plt.xlabel('condition number', fontsize= ff)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize= ff)
    plt.savefig('submission/kappa.pdf')
    plt.show()



def BimodalMixing():
    """Figure 3"""

    ff, ff_title, ff_ticks = 19, 20, 17
    plt.rcParams['xtick.labelsize'] = ff_ticks
    plt.rcParams['ytick.labelsize'] = ff_ticks
    plt.figure(figsize=(20, 8))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    mu = np.arange(1, 9)

    plt.plot(mu, np.load('Tests/data/bimodal_marginal/mode_mixing_d50_L1.5.npy')[:, 0], 'o:', label='MCHMC')

    nuts_results = np.load('Tests/data/mode_mixing_NUTS.npy')

    plt.plot(nuts_results[1], nuts_results[0], 'o:', label= 'NUTS')


    plt.yscale('log')
    plt.xlabel(r'$\mu$', fontsize = ff)
    plt.ylabel('average steps spent in a mode', fontsize = ff)
    plt.legend(fontsize = ff)
    #plt.savefig('submission/mode_mixing.pdf')

    plt.show()



def BimodalMarginal():

    #the problem parameters:
    d = 50
    mu1, sigma1= 0.0, 1.0 # the first Gaussian
    mu2, sigma2, f = 8.0, 1.0, 0.2 #the second Gaussian

    #plot parameters
    ff, ff_title, ff_ticks = 19, 20, 17
    plt.rcParams['xtick.labelsize'] = ff_ticks
    plt.rcParams['ytick.labelsize'] = ff_ticks
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    #NUTS
    X = np.load('Tests/data/bimodal_marginal/NUTS_hard.npz')
    x0, steps = np.array(X['x0']), np.array(X['steps'])

    plt.hist(x0, density=True, bins = 30, alpha = 0.5, color = 'tab:orange', label = 'NUTS', zorder = 0)

    #MCHMC

    def my_hist(bins, count):
        probability = count / np.sum(count)
        print('probability outside of bins', probability[-1])

        for i in range(len(bins)):
            density = probability[i] / (bins[i][1] - bins[i][0])
            plt.fill_between(bins[i], np.zeros(2), density * np.ones(2), alpha = 0.5, color='tab:blue', zorder = 1)

        plt.fill_between([], [], [], alpha = 0.5, color = 'tab:blue', label = 'MCHMC')

    xmax = 3.5 #how many sigma away from the mean of the gaussians do we want to have bins

    def get_bins(mu, sigma, num_bins_per_mode):
        bins_mode = np.array([[- xmax + i * 2 * xmax / num_bins_per_mode, - xmax + (i+1) * 2 * xmax / num_bins_per_mode] for i in range(num_bins_per_mode)])

        bins = np.concatenate(( (bins_mode * sigma[0]) + mu[0], (bins_mode * sigma[1]) + mu[1]  ))

        return bins

    bins_per_mode = 20
    bins = get_bins([mu1, mu2], [sigma1, sigma2], bins_per_mode)
    P = np.load('Tests/data/bimodal_marginal/sep'+str(mu2)+'_f'+str(f)+'_sigma'+str(sigma2)+'.npy')

    my_hist(bins, P)
    f1, f2 = np.sum(P[:bins_per_mode]), np.sum(P[bins_per_mode : 2 * bins_per_mode])
    print('f = ' + str(f2 / (f1 + f2)) + '  (true f = ' + str(f) + ')')


    #exact
    t = np.linspace(-xmax*sigma1+mu1-0.5, xmax*sigma2 + mu2 + 0.5, 1000)
    plt.plot(t, (1- f)*norm.pdf(t, loc = mu1, scale = sigma1) + f * norm.pdf(t, loc = mu2, scale = sigma2), color = 'black', label = 'exact', zorder = 2)

    plt.legend(fontsize = ff)
    plt.xlim(t[0], t[-1])
    plt.ylim(0, 0.4)
    plt.xlabel(r'$x_1$', fontsize = ff)
    plt.ylabel(r'$p(x_1)$', fontsize = ff)
    plt.savefig('submission/BimodalMarginal.pdf')
    plt.show()


def Funnel():
    """Figure 4"""

    def gaussianize(z, theta):
        return (z.T * np.exp(-0.5 * theta)).T, theta / 3.0

    eps, free_time = 0.1, 6
    data = np.load('Tests/data/funnel_free'+str(free_time) + '_eps'+str(eps)+'.npz')
    z, theta, w = data['z'], data['theta'], data['w']


    data = np.load('Tests/data/funnel_HMC.npz')
    zHMC, thetaHMC = data['z'], data['theta']


    ff, ff_title, ff_ticks = 18, 20, 14
    plt.rcParams['xtick.labelsize'] = ff_ticks
    plt.rcParams['ytick.labelsize'] = ff_ticks
    plt.figure(figsize=(24, 8))
    ff, ff_title, ff_ticks = 18, 20, 16



    ####   2d marginal in the original coordinates ####
    plt.subplot(1, 3, 1)
    plt.title('Original coordinates', fontsize = ff_title)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(zHMC[:, 0], thetaHMC, '.', ms= 1, color = 'tab:orange', label = 'NUTS')

    #plt.hist2d(z[:, 0], theta, cmap = 'Blues', weights= w, bins = 70, density=True, range= [[-30, 30], [-8, 8]], label ='MCHMC')
    plt.plot(z[::5000, 0], theta[::5000], '.', ms= 1, color = 'tab:blue', label = 'MCHMC')

    #plt.hist2d(z[:, 0], theta, weights= w, bins = 100, density=True, label = 'MCHMC')
    plt.xlim(-30, 30)
    plt.ylim(-8, 8)
    plt.xlabel(r'$z_1$', fontsize = ff)
    plt.ylabel(r'$\theta$', fontsize = ff)

    #### 2d marginal in the gaussianized coordinates ####
    plt.subplot(1, 3, 2)
    plt.title('Gaussianized coordinates', fontsize = ff_title)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    Gz, Gtheta = gaussianize(z, theta)
    plt.hexbin(Gz[:, 0], Gtheta, C= w, cmap='Blues', gridsize=50, label='MCHMC', reduce_C_function=np.sum)

    GzHMC, GthetaHMC = gaussianize(zHMC, thetaHMC)
    plt.plot(GzHMC[:, 0], GthetaHMC, '.', ms= 4, color = 'tab:orange', alpha = 0.5, label ='NUTS')

    #level sets
    p_level = np.array([0.6827, 0.9545])
    x_level = np.sqrt(-2 * np.log(1 - p_level))
    phi = np.linspace(0, 2* np.pi, 100)
    for i in range(2):
        plt.plot(x_level[i] * np.cos(phi), x_level[i] * np.sin(phi), color = 'black', alpha= ([1, 0.5])[i])

    plt.xlabel(r'$\widetilde{z_1}$', fontsize = ff)
    plt.ylabel(r'$\widetilde{\theta}$', fontsize = ff)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)


    #### 1d theta marginal####
    plt.subplot(1, 3, 3)
    plt.title(r'$\theta$-marginal', fontsize = ff_title)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.hist(thetaHMC, color='tab:orange', density=True, bins = 20, alpha = 0.5, label = 'NUTS')
    plt.hist(theta, weights= w, color='tab:blue', density=True, bins = 20, alpha = 0.5,  label = 'MCHMC')

    t= np.linspace(-10, 10, 100)
    plt.plot(t, norm.pdf(t, scale= 3.0), color= 'black', label = 'exact')

    #xmax = np.min([np.max(thetaHMC), np.max(theta)])
    #plt.xlim(-xmax, xmax)
    #plt.ylim(0, 1)


    plt.legend(fontsize = ff)
    plt.xlabel(r'$\theta$', fontsize = ff)
    plt.ylabel(r'$p(\theta)$', fontsize = ff)
    plt.savefig('submission/funnel.pdf')

    plt.show()





def Rosenbrock():
    """Figure 5"""

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
    sns.histplot(y=y, bins= 2000, fill= False, element= 'step', linewidth= 1, ax= plot.ax_marg_y, stat= 'density', color= 'black', alpha = 0.5, label= 'exact')


    plot.ax_marg_y.legend(fontsize = ff)

    plot.set_axis_labels(r'$x_1$', r'$y_1$', fontsize= ff)
    plt.tight_layout()
    plt.savefig('submission/rosenbrock.pdf')
    plt.show()



### old functions ###

def ess_epsilon():

    X = np.load('Tests/eps3.npy')
    plt.plot(X[:, 2], X[:, 0] , '.', color = 'black')

    plt.ylabel('ESS')
    plt.xlabel(r"$\epsilon$")
    #plt.savefig('Tests/eps_fine_tuning.png')
    plt.show()



def energy():
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

#
# def dimension_dependence():
#     ###  dimension dependence ###
#     plt.subplot(1, 2, 1)
#     dimensions = [50, 100, 200, 500, 1000, 3000, 10000]
#     markers = ['o:', 's:', 'v:']
#
#     # tuned MCHMC
#     target_names = ['StandardNormal_t', 'Kappa100_t', 'Rosenbrock_t']
#     for i in range(len(target_names)):
#         if i == 2:
#             dim = [50, 100, 200, 500, 1000, 3000]
#         else:
#             dim = dimensions
#         ess = [np.max(np.load('Tests/data/dimensions/' + target_names[i] + '/' + str(d) + '.npy')[:, 0]) for d in dim]
#         plt.plot(dim, ess, markers[i], color='tab:blue')
#
#     # NUTS
#     target_names = ['StandardNormal', 'Kappa100', 'Rosenbrock']
#     for i in range(len(target_names)):
#         ess = np.load('Tests/data/dimensions/' + target_names[i] + '_NUTS.npy')[0]
#         plt.plot(dimensions, ess, markers[i], color='tab:orange')
#
#     plt.xlabel('d', fontsize=ff)
#     plt.ylabel('ESS', fontsize=ff)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xticks([100, 1000, 10000], [r'$10^2$', r'$10^3$', r'$10^4$'])
#
#

def langevin():

    ymax = 0.015
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title('Langevin')
    X = np.load('Tests/data/langevin_kappa10000.npy')
    plt.plot(X[:, 1], X[:, 0], 'o:')
    plt.ylim(0, ymax)
    plt.ylabel('ESS')
    plt.xlabel(r'$\eta$')
    plt.xscale('log')

    plt.subplot(1, 2, 2)
    X = np.load('Tests/data/no_langevin_kappa10000.npy')
    plt.title('Bounces')
    plt.plot(X[:, 1] / np.sqrt(100), X[:, 0], 'o:')
    plt.ylim(0, ymax)
    plt.xscale('log')
    plt.xlabel(r'$\alpha$')

    plt.savefig('Langevin_kappa10000.png')

    plt.show()


def german_credit():
    folder = 'Tests/data/german_credit/'

    mchmc_data = np.load(folder + 'mchmc.npz')
    X, W = mchmc_data['x'], mchmc_data['w']

    hmc_data = az.from_netcdf(folder + 'inference_data_german_credit_mcmc.nc')
    tuning_steps = np.loadtxt(folder + 'german_credit_warmup_n_steps.txt')

    # hmc_steps = np.array(hmc_data['sample_stats']['n_steps'])
    # print(np.shape(hmc_steps))


    ff, ff_title, ff_ticks = 19, 20, 17
    plt.rcParams['xtick.labelsize'] = ff_ticks
    plt.rcParams['ytick.labelsize'] = ff_ticks
    plt.figure(figsize=(20, 8))

    hmc_bins = 100
    mchmc_bins = 100

    plt.subplot(1, 3, 1)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    tau = np.concatenate(np.array(hmc_data['posterior']['tau']))
    print(np.average(tau))
    plt.hist(np.log(tau), bins = hmc_bins, density=True, alpha = 0.5, color= 'tab:orange', label = 'NUTS')

    tau = X[:, 0]
    plt.hist(np.log(tau), bins=mchmc_bins, weights= W, density=True, alpha=0.5, color='tab:blue', label='MCHMC')

    plt.xlabel(r'$\log \tau$', fontsize= ff)
    plt.ylabel('Density', fontsize= ff)


    plt.subplot(1, 3, 2)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    lambda1 = np.concatenate(np.array(hmc_data['posterior']['lam'])[:, :, 0])
    plt.hist(np.log(lambda1), bins = hmc_bins, density=True, alpha = 0.5, color= 'tab:orange', label = 'NUTS')

    lambda1 = X[:, 1]
    plt.hist(np.log(lambda1), bins=mchmc_bins, weights= W, density=True, alpha=0.5, color='tab:blue', label='MCHMC')

    plt.xlabel(r'$\log \lambda_1$', fontsize= ff)


    plt.subplot(1, 3, 3)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    beta1 = np.concatenate(np.array(hmc_data['posterior']['beta'])[:, :, 0])
    plt.hist(beta1, bins = hmc_bins, density=True, alpha = 0.5, color= 'tab:orange', label = 'NUTS')

    beta1 = X[:, 2]
    plt.hist(beta1, bins=mchmc_bins, weights= W, density=True, alpha=0.5, color='tab:blue', label='MCHMC')

    plt.xlabel(r'$\beta_1$', fontsize= ff)

    plt.savefig('submission/german_credit_posterior.pdf')
    plt.show()


#bounce_frequency_full_bias()
#ill_conditioned()
#BimodalMarginal()

#Funnel()
#Rosenbrock()
dimension_dependence_appendix()
#german_credit()