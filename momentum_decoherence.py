import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import special_ortho_group
from scipy import interpolate
from scipy.stats import linregress
import jax
import jax.numpy as jnp

tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


### playing around with the momentum decoherence mechanism (checking the validity of the exponential decay formula for the generalized MCHMC) """

def update_momentum(carry, eta):
    u, key = carry
    key, subkey = jax.random.split(key)
    unew = u + jax.random.normal(subkey, u.shape, u.dtype) * eta
    return (unew / jnp.sqrt(jnp.sum(jnp.square(unew))), key), u[0]


def evolve(d, eta, nsteps, batch):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch)
    u0 = jnp.zeros(d)
    u0 = u0.at[0].set(1.0)

    results = jax.vmap(lambda seed: jax.lax.scan(lambda carry, _: update_momentum(carry, eta), init = (u0, seed), xs= None, length= nsteps)[1])(keys)

    return jnp.average(results, axis = 0), jnp.std(results, axis = 0) / jnp.sqrt(batch - 1)


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


def tau_decoherence(d, nu):
    print(d, nu)
    steps = (int)((1.0 / nu)**2) #heuristic
    s, err = evolve(d, nu, steps, 10000)
    mask = s > 1e-2
    time = np.arange(steps)
    x, y = time[mask], -np.log(s[mask])
    optimal_slope = np.dot(y, x) / np.dot(x, x)
    return optimal_slope



def tau_decoherence_plot(d, nu):
    print(d, nu)
    steps = 40#(int)((1.0 / nu)**2) #heuristic
    s, err = evolve(d, nu, steps, 1000000)
    #mask = s > 1e-2
    time = np.arange(steps)
    #x, y = time[mask], -np.log(s[mask])
    x, y = time, -np.log(s)

    optimal_slope = np.dot(y, x) / np.dot(x, x)
    plt.plot(time, s, '.', color='black')
    plt.plot(x, np.exp(-optimal_slope *x), color = 'tab:orange')
    plt.xlabel('n')
    plt.ylabel(r'$u_n \cdot u_0$')
    plt.yscale('log')
    plt.savefig('exp decay.png')
    plt.show()


def grid_scan():
    d= [50, 100, 500, 1000]
    nu = np.array([0.01, 0.03, 0.06, 0.1, 0.15])
    X = [[tau_decoherence(dd, nn) for nn in nu] for dd in d]
    np.save('grid_scan', X)


def analyze_grid_scan():
    d= [50, 100, 500, 1000]
    nu = np.array([0.01, 0.03, 0.06, 0.1, 0.15])

    X = np.load('grid_scan.npy')

    for i in range(len(X)):
        plt.plot(nu, X[i, :], 'o:', label = 'd = ' +str(d[i]))
        plt.plot(nu, 0.5 * np.log(1 + d[i] * np.square(nu)), color = 'black')

    plt.plot([], [], color= 'black', label = r'$\frac{1}{2} \, \log ( 1 + d \nu^2 )$')
    plt.legend()
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\lambda$')
    plt.savefig('umeritev')
    plt.show()



#analyze_grid_scan()
