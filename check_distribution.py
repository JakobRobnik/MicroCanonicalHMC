import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit  #I use numba, which makes the python evaluations significantly faster (as if it was a c code). If you do not want to download numba comment out the @njit decorators.
import sampler


def virial_theorem(samples, grad_samples):
    """Checks if the samples satisfy the virial theorem.
        Args:
            samples[i, :] = x of i-th sample
            grad_nlogp[i, :] = - grad log p_target(x of i-th sample)

        Returns:
            an array with entries <x_i  grad_i -logp(x_i) > / d, where expectation is taken over the samples. The virial theorem predicts all values should be 1.
    """

    return [np.average(samples[:, i] * grad_samples[:, i]) for i in range(len(samples[0]))]



def cross_entropy(samples, Sampler):
    return np.sum([V(x) for x in samples]) / len(samples)



def check_posterior(Sampler):

    t, samples = Sampler.sample(free_time=300.0, num_bounces=50)
    samples = samples[::10]

    print('done sampling')

    #samples = np.random.normal(size = (1000, 2))
    grad_samples = np.array([Sampler.grad_nlogp(x) for x in samples])
    virial = np.round(virial_theorem(samples, grad_samples), 3)

    x, y = samples[:, 0], samples[:, 1]

    xmin, xmax, ymin, ymax = -4, 4, -4, 4
    X = np.linspace(xmin, xmax, 100)
    Y = np.linspace(ymin, ymax, 100)
    px = np.exp(-0.5*np.square(X)) / np.sqrt(2 * np.pi)
    py = np.exp(-0.5*np.square(Y)) / np.sqrt(2 * np.pi)
    Xmesh, Ymesh = np.meshgrid(X, Y)

    Z = (np.square(Xmesh) + np.square(Ymesh)) * 0.5

    plot = sns.JointGrid(xlim = (xmin, xmax), ylim = (ymin, ymax), height=10)
    ff = 16
    plot.fig.suptitle(r"$ -\langle x \, \partial_x \log p \rangle = $" + "{0}".format(virial[0]) +
                      '\n' + r"$ -\langle y \, \partial_y \log p \rangle = $" + "{0}".format(virial[1]) +
                      '\n' + r"$ -\langle z \, \partial_z \log p \rangle = $" + "{0}".format(virial[2]) , x = 0.85, y = 0.9, fontsize = ff)

    plot.ax_joint.contourf(Xmesh, Ymesh, -Z, cmap = 'Blues', levels = np.linspace(-4.7, 0, 20))
    sns.scatterplot(x, y, s=10, linewidth=1.5, ax=plot.ax_joint, color= 'black')
    sns.kdeplot(x, y, bw_adjust=2, ax=plot.ax_joint, levels= np.array([0.2, 0.4, 0.6, 0.8]), alpha = 0.5, color= 'black')

    #marginals
    sns.histplot(x=x, fill=False, linewidth=2, ax=plot.ax_marg_x, stat = 'density', color = 'black')
    sns.histplot(y=y, fill=False, linewidth=2, ax=plot.ax_marg_y, stat='density', color = 'black')
    #sns.kdeplot(x=x, linewidth=2, ax=plot.ax_marg_x)
    #sns.kdeplot(y=y, linewidth=2, ax=plot.ax_marg_y)

    sns.scatterplot(X, px, ax = plot.ax_marg_x, s = 5, color = 'tab:blue')
    sns.scatterplot(py, Y, ax=plot.ax_marg_y, s = 5, color = 'tab:blue')

    plot.set_axis_labels('x', 'y', fontsize =ff)
    plt.tight_layout()
    plt.savefig('gaussian_3d_canonical_2_2.png')
    plt.show()
    # x1 = X[:, -1, 0]
    # plt.hist(x1, bins=30, density=True, label='ruthless sampler')
    # t = np.linspace(-6, 6, 200)
    #
    # # t = np.linspace(min(x1), max(x1), 100)
    # p = np.exp(-np.array([V(np.array([tt, 0.0])) for tt in t]))
    # p /= np.sum(p) * (t[1] - t[0])
    # plt.plot(t, p, label='analytical')
    #
    # plt.xlabel('$x_1$')
    # plt.ylabel('pdf')
    # plt.legend()
    # plt.savefig('bimodal_ruthless.png')
    # plt.show()


def variance(Sampler):

    t, X = Sampler.sample(300, 100)#[:, 0]
    print('done sampling')
    #np.save('gauss5.npy', X)
    #X = np.load('gauss4.npy')
    #x = X[:, 0]
    var_true = 1.0

    x = X[:, 0]
    variance_bias = np.square(np.cumsum(np.square(x)) / (1 + np.arange(len(x))) - var_true)
    plt.plot(variance_bias, '.')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("steps")
    plt.ylabel(r"$Bias^2$")
    #plt.savefig('bias5.png')
    plt.show()



def eta200(X, var_true):
    """estimates effecitve sample size / sample size"""
    variance_bias = np.square((np.cumsum(np.square(X), axis = 0).T / (1 + np.arange(len(X)))).T - var_true)
    flag = np.all(variance_bias < 0.1, axis = 1)
    fraction = np.cumsum(flag) / (np.arange(len(flag))+1) #fraction of time points which have bias^2 < 0.01 as a function of time

    plt.plot(fraction, '.')
    plt.show()

    i = len(fraction) -1
    while fraction[i] > 0.5:
        i-=1
        if i < 0:
            return 0

    #the first time for which the fraction is smaller than 0.5 (starting from the above)
    return 200.0 / i



def eta200_with_error(X, var_true):
    """reduces the eta estimate """
    num = 100
    eta = [eta200(X[np.random.permutation(len(X)), :], var_true) for i in range(num)]

    return np.median(eta), np.std(eta)/ np.sqrt(num)


def eta_fit(x, var_true):
    variance_bias = np.square(np.cumsum(np.square(x)) / (1 + np.arange(len(x))) - var_true)
    eta = 2.0 / (variance_bias * np.arange(1, 1+len(variance_bias)))

    return np.median(eta)



def variance_tst():

    #x = Sampler.sample(1000, 1000)[:, 0]

    #eta = [eta200(np.random.normal(size = 1000), 1.0) for i in range(1000)] #ess / ss
    rand = np.random.normal(size = 1000)
    #eta_bootstrap = [eta200(rand[np.random.permutation(len(rand))], 1.0)for i in range(1000)] #ess / ss
    #eta2 = [eta200_with_error(np.random.normal(size = 1000), 1.0)[0] for i in range(1000)] #ess / ss
    eta2 = [eta200_with_error(np.random.normal(size = 1000), 1.0)[0] for i in range(1000)] #ess / ss


    #plt.hist(eta, bins = 1000, density=True, cumulative=True, histtype = 'step', label = 'simulation')
    #plt.hist(eta_bootstrap, bins = 1000, density=True, cumulative=True, histtype='step', label = 'bootstrap')
    plt.hist(eta2, bins=1000, density=True, cumulative=True, histtype='step', label='simulation averaged over permutations')

    plt.legend(loc = 4)
    plt.plot([1, 1], [0, 1], ':', color= 'black')
    plt.xlim(0, 5)
    plt.ylim(0, 1)
    plt.xlabel('ESS/SS')
    #plt.savefig('tst_bootstrap.png')
    plt.show()



def projection_hist(Sampler):
    samples= Sampler.sample(1000, 60)
    x1 = samples[:, 0]
    plt.hist(x1, bins=30, density=True, label='ruthless sampler')
    t = np.linspace(-6, 6, 200)

    #t = np.linspace(min(x1), max(x1), 100)
    p = np.exp(-np.array([V(np.array([tt, 0.0])) for tt in t]))
    p /= np.sum(p) * (t[1] - t[0])
    plt.plot(t, p, label = 'analytical')

    plt.xlabel('$x_1$')
    plt.ylabel('pdf')
    plt.legend()
    plt.savefig('bimodal_ruthless.png')
    plt.show()



def plot_trajectory(Sampler):
    #background
    # num = 100
    # x= np.linspace(-4, 4, num)
    # y = np.linspace(-3, 3, num)
    # X, Y = np.meshgrid(x, y)
    # Z = 0.5* (X**2 + Y**2)#b * X**4 - X**2 - a*X**3 - c + Y**2
    # plt.contourf(X, Y, Z)
    # plt.colorbar()

    total_time = 300
    steps = (int)(total_time / Sampler.dt)
    x0, p0 = Sampler.initialize()
    Sampler.time = 0.0
    x, p = Sampler.trajectory(x0, p0, steps)
    skip_num = 5

    plt.plot(x[::skip_num, 0], x[::skip_num, 1], '.', color = 'red')
    plt.plot(x[:, 0], x[:, 1], ':', color='red', alpha = 0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # plt.plot(np.arange(steps) / parts_per_sec, [Sampler.nlogp(x[i, :]) for i in range(len(x))])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()


def movie():
    # importing movie py libraries
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage

    # numpy array
    x = np.linspace(-2, 2, 200)

    # duration of the video
    duration = 5

    # matplot subplot
    fig, ax = plt.subplots()
    # background
    num = 100
    x = np.linspace(-4, 4, num)
    y = np.linspace(-3, 3, num)
    X, Y = np.meshgrid(x, y)
    Z = 0.5* (X**2 + Y**2)#b * X**4 - X**2 - a*X**3 - c + Y**2
    #ax.contourf(X, Y, Z)

    #ax.colorbar()

    parts_per_sec = 10000
    steps = duration * parts_per_sec
    x, p = trajectory(np.array([0, 2]), np.array([0, 0.1]), g, grad_g, 1.0 / parts_per_sec, steps)


    # method to get frames
    def make_frame(t):
        ax.clear()
        it = (int) (t* parts_per_sec)

        # plotting line
        ax.plot([x[it, 0], ], [x[it, 1], ], '.', color = 'red')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        # returning numpy image
        return mplfig_to_npimage(fig)

    # creating animation
    animation = VideoClip(make_frame, duration=duration)

    # displaying animation with auto play and looping
    animation.ipython_display(fps=20, loop=True, autoplay=True)



#@njit
def nlogp_gauss(x):
    """- log p of the target distribution"""
    return 0.5 * np.sum(np.square(x))
#@njit
def grad_nlogp_gauss(x):
    return x


#@njit
def nlogp_double_well(x, params):
    a, b, c = params[0], params[1], params[2]
    return b * x[0]**4 - x[0]**2 - a*x[0]**3 - c + x[1]**2 + 0.5*x[2]**2

#@njit
def grad_nlogp_double_well(x, params):
    a, b, c = params[0], params[1], params[2]
    return np.array([4*b*x[0]**3 - 2*x[0] - 3*a*x[0]**2, 2*x[1], x[2]])


def adaptive_step_trajectory():
    t_total = 5

    Sampler = sampler.Ruthless(nlogp_gauss, grad_nlogp_gauss, d, step ='Leapfrog')
    np.random.seed(0)
    Sampler.dt = 0.00005
    t, samples = Sampler.sample(free_time=t_total, num_bounces=1)
    plt.plot(samples[:, 0], samples[:, 1], color = 'black', label= r'Leapfrog ($\Delta t = 0.00005$)')
    print(len(samples))
    v_initial = np.sqrt(np.sum(np.square(samples[1] - samples[0]))) / Sampler.dt


    Sampler = sampler.Ruthless(nlogp_gauss, grad_nlogp_gauss, d, step ='Adaptive Leapfrog')
    np.random.seed(0)
    dt_initial = 0.001
    Sampler.dx = v_initial *dt_initial #initial momentum is normalized to 1
    Sampler.dt = dt_initial
    t, samples = Sampler.sample(free_time=t_total, num_bounces=1)#[::100, :]
    print(len(samples))
    dxes = np.sqrt(np.sum(np.square(samples[1:] - samples[:-1]), axis = 1))
    plt.plot(samples[:, 0], samples[:, 1], ':.', color = 'tab:blue', label= r'Adaptive Leapfrog ($\langle \Delta t \rangle = 0.005$)')


    Sampler = sampler.Ruthless(nlogp_gauss, grad_nlogp_gauss, d, step ='EPSSP')
    np.random.seed(0)
    Sampler.dt = 3 * t_total / len(samples)
    print(Sampler.dt)
    t, samples = Sampler.sample(free_time=t_total, num_bounces=1)#[::100, :]
    plt.plot(samples[:, 0], samples[:, 1], ':.', color = 'gold', label= r'Leapfrog ($\Delta t = 0.005$)')

    plt.legend()
    #custom_legend_order([0, 2, 1], loc = 4)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    #plt.savefig+('adaptive_step_trajectory', dpi = 1200)
    plt.show()


def custom_legend_order(order, loc = 'best'):
    """ order is the permutation of the legend entries, e.g. order = [1, 2, 0] """

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()


    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc = loc)


def hyperparameter_tuning():
    t, X = Sampler.sample(3000, 10)

    np.save('g_3000_10.npy', X)
    #x = np.random.normal(size = 300*200)

    eta, err_eta = eta200_with_error(X, np.ones(d))
    print(eta, err_eta)


d = 3
b = 0.1
a = 0.03
xmin = (3 * a + np.sqrt(9 * a * a + 32 * b)) / (8 * b)
c = xmin ** 2 * (b * xmin ** 2 - a * xmin - 1)
params = [a, b, c]

Sampler = sampler.Seperable(nlogp_gauss, grad_nlogp_gauss, d, step ='Yoshida')

np.random.seed(0)
Sampler.dt = 0.1

#plot_trajectory(Sampler)
#check_posterior(Sampler)
#variance_tst()

#variance(Sampler)

hyperparameter_tuning()


#for 3d gaussian with seperable:
#  leapfrog: