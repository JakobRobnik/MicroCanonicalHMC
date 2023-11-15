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



def predictor(data, counter, acc_prob_wanted, gamma = 0.9):
    """given the acceptance probabilities and stepsizes, predict the next best stepsize"""
    
    
    eps, acc_prob = data
    N = len(eps)
    n = jnp.arange(N)
    forgetting = jnp.power(gamma, counter - n[::-1]) * (n < counter)
    trust_params = trust_reparam(acc_prob_wanted, sig= 0.7)
    c = norm.ppf(0.5 * acc_prob)
    
    def F(x):
        acc = 2 * norm.cdf(c * jnp.square(x / eps)) 
        weights = forgetting * trust(acc, *trust_params)
        return jnp.average(acc, weights= weights) - acc_prob_wanted
    
    # F is approximately monotonically decreasing
    lower, upper = 1, 20
    #print(F(lower), F(upper))
    
    x0 = Bisection(F, lower = lower, upper = upper).run().params
    return x0



def dual_averaging(acc_prob, state, acc_prob_wanted = 0.7, t0 = 10, gamma= 0.05, kappa= 0.75):
    """ (copied from numpyro)
    Dual Averaging is a scheme to solve convex optimization problems. It
    belongs to a class of subgradient methods which uses subgradients (which
    lie in a dual space) to update states (in primal space) of a model. Under
    some conditions, the averages of generated parameters during the scheme are
    guaranteed to converge to an optimal value. However, a counter-intuitive
    aspect of traditional subgradient methods is "new subgradients enter the
    model with decreasing weights" (see reference [1]). Dual Averaging scheme
    resolves that issue by updating parameters using weights equally for
    subgradients, hence we have the name "dual averaging".

    This class implements a dual averaging scheme which is adapted for Markov
    chain Monte Carlo (MCMC) algorithms. To be more precise, we will replace
    subgradients by some statistics calculated at the end of MCMC trajectories.
    Following [2], we introduce some free parameters such as ``t0`` and
    ``kappa``, which is helpful and still guarantees the convergence of the
    scheme.

    **References:**

    1. *Primal-dual subgradient methods for convex problems*,
       Yurii Nesterov
    2. *The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo*,
       Matthew D. Hoffman, Andrew Gelman

    :param int t: The current time step.

    :param float accept_prob: Acceptance probability of the current trajectory.

    :param state: Current state of the adapt scheme.

    :param int t0: A free parameter introduced in reference [2] that stabilizes
        the initial steps of the scheme. Defaults to 10.
    :param float kappa: A free parameter introduced in reference [2] that
        controls the weights of steps of the scheme. For a small ``kappa``, the
        scheme will quickly forget states from early steps. This should be a
        number in :math:`(0.5, 1]`. Defaults to 0.75.
    :param float gamma: A free parameter introduced in reference [1] which
        controls the speed of the convergence of the scheme. Defaults to 0.05.
    
    :return: new state of the adapt scheme.

    """

    
    g = acc_prob_wanted - acc_prob
    
    x_t, x_avg, g_avg, t = state
    
    t += 1
    # g_avg = (g_1 + ... + g_t) / t
    g_avg = (1 - 1 / (t + t0)) * g_avg + g / (t + t0)
    # According to formula (3.4) of [1], we have
    #     x_t = argmin{ g_avg . x + loc_t . |x - x0|^2 },
    # hence x_t = x0 - g_avg / (2 * loc_t),
    # where loc_t := beta_t / t, beta_t := (gamma/2) * sqrt(t).
    x_t = - (t**0.5) / gamma * g_avg
    # weight for the new x_t
    weight_t = t ** (-kappa)
    x_avg = (1 - weight_t) * x_avg + weight_t * x_t
    
    return jnp.array([x_t, x_avg, g_avg, t])
    
    



