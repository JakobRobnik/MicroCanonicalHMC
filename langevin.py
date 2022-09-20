import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import special_ortho_group
from scipy import interpolate
from scipy.stats import linregress


tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def update_momentum(u, eta):
    unew = u + np.random.normal(size= len(u)) * eta
    return unew / np.sqrt(np.sum(np.square(unew)))




def decoherence_time(d, eta):
    repeat = 200
    times = np.empty(repeat)
    for i in range(repeat):
        u = np.concatenate(([1.0, ], np.zeros(d - 1)))
        time = 0
        while u[0] > 0.0:  # = < u_i, u_0 >
            u = update_momentum(u, eta)
            time += 1
        times[i] = time

    return [np.average(times), np.std(times) / np.sqrt(repeat - 1)]


def find_eta(tau, e, t):
    i = 0
    while (i < len(e)) and (t[i] > tau):
        i += 1
    i -= 1
    return e[i] + (e[i+1] - e[i]) * (tau - t[i]) / (t[i+1] - t[i])


def decoherence_time_plot():
    dimensions = [10, 50, 100, 500, 1000]
    np.random.seed(0)
    etas = np.logspace(-2, 0.5, 20)
    eta_opt = np.empty(len(dimensions))
    alpha = 1.5
    plt.figure(figsize = (15, 5))
    plt.subplot(1, 2, 1)
    for i in range(len(dimensions)):
        d= dimensions[i]
        X = np.array([decoherence_time(d, eta) for eta in etas])

        plt.errorbar(etas, X[:, 0], yerr = X[:, 1], fmt = '.:', capsize=1.5, label = 'd =' +str(d), color = tab_colors[i])
        tau = alpha * np.sqrt(d)

        eta_opt[i] = np.exp(find_eta(np.log(tau), np.log(etas), np.log(X[:, 0])))
        plt.plot([eta_opt[i], ], [tau, ], '*', markersize = 15, color = tab_colors[i])


    plt.legend()
    plt.xlabel(r'$\eta$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('decoherence time')

    plt.subplot(1, 2, 2)
    print(eta_opt)
    for i in range(len(dimensions)):
        plt.plot([dimensions[i], ], [eta_opt[i], ], '*', markersize = 10, color = tab_colors[i])


    res = linregress(np.log(dimensions), np.log(eta_opt))

    plt.title(r'$\eta = \eta_0 d^{-\beta}$,' + r'     $\eta_0 = $' + str(np.round(np.exp(res.intercept), 2)) + r', $\beta = $' + str(np.round(-res.slope, 2)))
    plt.plot(dimensions, np.exp(res.intercept) * np.power(dimensions, res.slope), ':', color='black')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('d')
    plt.ylabel(r'$\eta$')

    plt.savefig('decoherence_time.png')
    plt.show()

def decoherence_plot(d, eta):
    num = 300
    u = np.concatenate(([1.0, ], np.zeros(d - 1)))
    s = np.empty(num)
    for i in range(num):
        s[i] = u[0] # = < u_i, u_0 >
        u = update_momentum(u, eta)

    plt.plot(s)
    plt.xlabel('n')
    plt.ylabel(r'$u_n \cdot u_0$')
    plt.savefig('decoherence.png')
    plt.show()



def angle_distribution():
    p = [1.0, 0.0]

    num = 10000000
    etas = [0.1, 0.4, 1, 100]

    for eta in etas:
        P = p + np.random.normal(size= (num, 2)) * eta
        phi = np.arctan2(P[:, 1], P[:, 0])
        plt.hist(phi, bins = 300, density= True, histtype='step', label = r'$\eta = $ {0}'.format(eta))

    plt.legend()
    plt.title(r'${\bf{\Pi}}_{n+1} = \frac{{\bf{\Pi}}_{n} + \eta {\bf{z}}}{\Vert {\bf{\Pi}}_{n} + \eta {\bf{z}} \Vert}, \quad z \sim N(0, 1)$')
    plt.xlabel(r'$\phi({\bf{\Pi}}_n, {\bf{\Pi}}_{n+1})$')
    plt.ylabel(r'$p(\phi)$')
    plt.xticks([0, np.pi/4.0, np.pi/2.0, np.pi*3.0/4.0, np.pi, -np.pi/4.0, -np.pi/2.0, -np.pi*3.0/4.0, -np.pi],
               [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3 \pi}{4}$', r'$\pi$',
                r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{3 \pi}{4}$', r'$-\pi$'])

    plt.yscale('log')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(4e-4, 7)
    plt.savefig('MomentumWalk', dpi = 400)
    plt.show()


decoherence_time_plot()