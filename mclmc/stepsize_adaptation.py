import jax
import jax.numpy as jnp

from jax.scipy.stats import norm
from jaxopt import Bisection

from scipy.special import beta as beta_func
import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8-talk', 'img/style.mplstyle'])



def trust_reparam(x, sig):
    beta = (x * (1-x**2))/sig**2
    alpha = beta * x / (1 - x)
    return alpha, beta, beta_func(1 + alpha, 1 + beta)

def trust(x, alpha, beta, norm):
    return jnp.power(x, alpha) * jnp.power(1-x, beta) / norm

def visualize_trust():

    t = jnp.linspace(0, 1, 100)

    for sig in [1, 0.7, 0.4, 0.2]:
        plt.plot(t, trust(t, *beta_params(0.65, sig)), lw = 5, label = r'$\sigma = $' + str(sig), color = plt.cm.cividis_r(sig-0.1))
    plt.plot([0.65, 0.65], [0, 4.5], color = 'black', alpha = 0.5)
    plt.xlabel('acceptance rate')
    plt.ylabel('trust weight')
    plt.legend()
    plt.savefig('trust.png')
    plt.close()

fig = plt.figure(figsize = (15, 10))
ax = fig.subplots(1, 2)


def next_filter(acc_prob, eps, acc_prob_wanted, gamma):
    """given the acceptance probabilities and stepsizes, predict the next best stepsize"""
    N = len(eps)
    W  = (1 - gamma**N) / (1-gamma)
    forgetting = jnp.power(gamma, jnp.arange(N-1, -1, -1))
    trust_params = trust_reparam(acc_prob_wanted, sig= 0.7)
    c = norm.ppf(0.5 * acc_prob)
    
    def F(x):
        acc = 2 * norm.cdf(c * jnp.square(x / eps)) 
        weights = forgetting * trust(acc, *trust_params)
        return jnp.average(acc, weights= weights) - acc_prob_wanted
    
    # F is (approximately?) monotonically increasing
    lower, upper = 0.1, 3
    X = jnp.linspace(lower, upper, 1000)
    #print(F(lower), F(upper))
    
    ax[1].set_title('Root function')
    ax[1].plot(X, [F(xx) for xx in X], lw = 5, color = 'teal')
    ax[1].set_xlabel(r"$\epsilon'$")
    ax[1].set_ylabel(r"$a(\epsilon') - \alpha$")
    
    x0 = Bisection(F, lower = lower, upper = upper).run().params
    
    ax[1].plot([x0, ], [F(x0), ], '*', markersize= 25, color = 'teal')
    
    # plt.plot(eps, 2 * norm.cdf(c * jnp.square(x0 /eps)), '.')
    # plt.savefig('debug.png')
    # plt.close()
    return x0
    
    

key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
N = 20
stepsize = jnp.square(jax.random.normal(key1, shape = (N,)))
C = 0.3
acc = jnp.min(jnp.array([2 * norm.cdf( - C * jnp.square(stepsize)) * jnp.abs(1 + jax.random.normal(key2, shape = (N, )) * 0.1), jnp.ones(N)]), axis = 0)
acc0 = 0.65

eps0 = next_filter(acc, stepsize, acc0, 0.9)
print(2 * norm.cdf( - C * jnp.square(eps0)))
ax[0].set_title('data')
ax[0].plot(stepsize, acc, '.', markersize= 15, color = 'tab:blue')
ax[0].set_xlabel(r"$\epsilon_n$")
ax[0].set_ylabel(r"$P_n$")
ax[0].plot(stepsize, acc0 * jnp.ones(N), '-', color = 'black', alpha = 0.5, lw= 2)
ax[0].plot([eps0, ], [acc0, ], '*', color = 'teal', markersize= 25)

plt.savefig('adaptation.png')
plt.close()