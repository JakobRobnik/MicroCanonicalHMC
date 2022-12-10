import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats import norm
import jax.numpy as jnp
import jax
import myHMC

### Figure 1 in MCHMC paper ###



def norm_pdf(x, mu, sigma):
    return jnp.exp(-0.5 * jnp.square(x - mu) / sigma**2) / np.sqrt(2 * jnp.pi * sigma**2)

def L(x):
    f, mu, sigma2 = 0.2, 1.5, 0.6

    return - jnp.log(f * norm_pdf(x, -mu, sigma2) + (1-f) * norm_pdf(x, mu, 1))

grad_L = jax.grad(lambda x: L(x)[0])


class Target():

    def __init__(self):
        self.d = 1
        self.grad_nlogp = grad_L


def canonical_hamiltonian(x, p):
    return 0.5*np.square(p) + L(x)

def microcanonical_hamiltonian(x, p):
    return 0.5*np.log(np.square(p)) + L(x)

Pi = lambda x, E: np.exp(E - L(x))

def MCHMC_trajectory(X, dt, E):
    indexes, signs = [0, ],  [1, ]
    t = 0
    sign = 1
    i = 0

    while i >= 0:
        t += np.abs(X[i+1] - X[i]) * Pi((X[i] + X[i+1])*0.5, E)
        if t > dt:
            indexes.append(i)
            signs.append(sign)
            t = 0.0

        if i == len(X)-2:
            sign = -1

        i += sign

    return indexes, signs


fig = plt.figure(figsize= (12, 9))
spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[1, 1],  height_ratios=[3, 1])

ff = 22
lw, fmt = 1.3, 10
lw_pdf = 5
xmin, xmax= -4, 5.2

yloc_title = 1.1
yloc = 1.03



### canonical HMC ###

plt.sca(fig.add_subplot(spec[0]))
plt.title(r'Canonical HMC', fontsize = ff + 2, y = yloc_title, color = 'tab:orange')
plt.text(0.5, yloc, r'$p({\bf{x}}, {\bf{\Pi}}) \propto e^{-H({\bf{x}}, {\bf{\Pi}})}$', fontsize = ff+2, transform=plt.gca().transAxes, horizontalalignment= 'center')
x = np.linspace(xmin, xmax, 100)

p = np.linspace(-3, 3, 100)
X, P = np.meshgrid(x, p)
H = canonical_hamiltonian(X, P)
plt.contourf(X, P, np.exp(-H), cmap= 'Greys', levels = 20)

hmc = myHMC.Sampler(Target(), 0.01)
xhmc, phmc = hmc.sample(jnp.array([1.0, ]), 100 * 50, hmc.eps * 100, jax.random.PRNGKey(0))
plt.plot(xhmc, phmc, '-', lw = lw, color = 'tab:orange')
plt.plot(xhmc[::10], phmc[::10], '.',  markersize = fmt, color = 'tab:orange')


plt.xlabel(r'$\bf{x}$', fontsize= ff)
plt.ylabel(r'$\bf{\Pi}$', fontsize= ff)
plt.xticks([])
plt.yticks([])

plt.sca(fig.add_subplot(spec[2]))
plt.plot(x, np.exp(-L(x)), lw = lw_pdf, color= 'tab:orange')
plt.xlabel(r'$\bf{x}$', fontsize= ff)
plt.ylabel(r'$p({\bf{x}})$', fontsize= ff)
plt.xticks([])
plt.yticks([])
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)



### microcanonical HMC ###

plt.sca(fig.add_subplot(spec[1]))
plt.title('Microcanonical HMC', fontsize = ff+2, y = yloc_title, color = 'tab:blue')
plt.text(0.5, yloc, r'$p({\bf{x}}, {\bf{\Pi}}) \propto \delta(H({\bf{x}}, {\bf{\Pi}}) - E)$', fontsize = ff+2, transform=plt.gca().transAxes, horizontalalignment= 'center')
x = np.linspace(-4, 5.2, 100)

p = np.linspace(-3, 3, 100)
X, P = np.meshgrid(x, p)
#H = microcanonical_hamiltonian(X, P)
E = np.linspace(0.1, 10, 20)

for e in E:
    if e == E[2]:
        continue
    plt.plot(x, Pi(x, e), color = 'black', alpha = 0.3)
    plt.plot(x, -Pi(x, e), color='black', alpha=0.3)

plt.plot(x, Pi(x, E[2]), lw = lw, color = 'tab:blue')
plt.plot(x, -Pi(x, E[2]), lw = lw, color='tab:blue')


indexes, signs = MCHMC_trajectory(x, 0.05, E[2])
plt.plot(x[indexes], signs * Pi(x, E[2])[indexes], '.', markersize = fmt, color = 'tab:blue')

plt.xlabel(r'$\bf{x}$', fontsize= ff)
plt.ylabel(r'$\bf{\Pi}$', fontsize= ff)
plt.xticks([])
plt.yticks([])
plt.ylim(-2, 2)
plt.xlim(xmin, xmax)

plt.sca(fig.add_subplot(spec[3]))
plt.plot(x, np.exp(-L(x)), lw = lw_pdf, color = 'tab:blue')
plt.xlabel(r'$\bf{x}$', fontsize= ff)
plt.ylabel(r'$p({\bf{x}})$', fontsize= ff)
plt.xticks([])
plt.yticks([])
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig('submission/Canonical_vs_Microcanonical.pdf')


plt.show()