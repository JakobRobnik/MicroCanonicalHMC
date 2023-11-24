import jax
import jax.numpy as jnp

from .correlation_length import ess_corr



# all tuning functions are wrappers, recieving some parameters and returning a function
# func(dyn, hyp, num_total_steps) -> (dyn, hyp) 




def run(dyn, hyp, schedule, num_steps):
    
    _dyn, _hyp = dyn, hyp
    for program in schedule:
        _dyn, _hyp = program(_dyn, _hyp, num_steps)
        
    return _dyn, _hyp
 
 

 
def nan_reject(x, u, l, g, xx, uu, ll, gg, eps, eps_max, dK):
    """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""
    
    nonans = jnp.all(jnp.isfinite(xx))
    _x, _u, _l, _g, _eps, _dk = jax.tree_util.tree_map(lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old), 
                                                       (xx, uu, ll, gg, eps_max, dK), 
                                                       (x, u, l, g, eps * 0.8, 0.))
    
    return nonans, _x, _u, _l, _g, _eps, _dk
    
 


def tune12(dynamics, d,
           diag_precond,
           frac, varEwanted, sigma_xi, neff):
    
    gamma = (neff - 1.0) / (neff + 1.0)
    
    def _step(dyn_old, hyp, state_adaptive):

        W, F, eps_max = state_adaptive

        # dynamics
        dyn_new, energy_change = dynamics(dyn_old, hyp)

        # step updating
        success, x, u, l, g, eps_max, energy_change = nan_reject(dyn_old['x'], dyn_old['u'], dyn_old['l'], dyn_old['g'], 
                                                                      dyn_new['x'], dyn_new['u'], dyn_new['l'], dyn_new['g'], 
                                                                      hyp['eps'], eps_max, energy_change)

        dyn = {'x': x, 'u': u, 'l': l, 'g': g, 'key': dyn_new['key']}
        
        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = ((energy_change**2) / (d * varEwanted)) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * sigma_xi)))  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.
        
        F = gamma * F + w * (xi/jnp.power(hyp['eps'], 6.0)),
        W = gamma * W + w,
                      
        eps = jnp.power(F/W, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (eps > eps_max) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        hyp_new = {'L': hyp['L'], 'eps': eps, 'sigma': hyp['sigma']}
        
        return dyn, hyp_new, (W, F, eps_max), success


    def update_kalman(x, state, outer_weight, success, eps):
        W, F1, F2 = state
        w = outer_weight * eps * success
        zero_prevention = 1-outer_weight
        F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        W += w
        return (W, F1, F2)


    def step(state, outer_weight):
        """one adaptive step of the dynamics"""
        dyn, hyp, params, kalman_state = state
        dyn, hyp, params, success = _step(dyn, hyp, params)
        kalman_state = update_kalman(dyn['x'], kalman_state, outer_weight, success, hyp['eps'])

        return (dyn, hyp, params, kalman_state), None


    def func(_dyn, _hyp, num_steps):

        num_steps1, num_steps2 = (num_steps * frac).astype(int)
            
        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        #initial state
        params = (jnp.power(_hyp['eps'], -6.0) * 1e-5, 1e-5, jnp.inf)
        kalman_state = (0., jnp.zeros(d), jnp.zeros(d))
    
        # run the steps
        state = jax.lax.scan(step, init= (_dyn, _hyp, params, kalman_state), xs= outer_weights, length= num_steps1 + num_steps2)[0]
        dyn, hyp, params, kalman_state = state
        
        # determine L
        if num_steps2 != 0.:
            _, F1, F2 = kalman_state
            variances = F2 - jnp.square(F1)
            sigma2 = jnp.average(variances)

            # optionally we do the diagonal preconditioning (and readjust the stepsize)
            if diag_precond:

                # diagonal preconditioning
                sigma = jnp.sqrt(variances)
                L = jnp.sqrt(d)

                #readjust the stepsize
                steps = num_steps2 // 3 #we do some small number of steps
                state = jax.lax.scan(step, init= state, xs= jnp.ones(steps), length= steps)
                dyn, hyp, params, kalman_state = state
            
            else:
                L = jnp.sqrt(sigma2 * d)
                sigma = hyp['sigma']
            
        hyp = {'L': L, 'sigma': sigma, 'eps': hyp['eps']}
        
        return dyn, hyp 

    return func





def tune3(step, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""
    

    def sample_full(num_steps, _dyn, hyp):
        """Stores full x for each step. Used in tune2."""

        def _step(state, useless):
            dyn_old = state
            dyn_new = step(dyn_old, hyp)
            
            return dyn_new, dyn_new['x']

        return jax.lax.scan(_step, init=_dyn, xs=None, length=num_steps)


    def func(dyn, hyp, num_steps):
        steps = (num_steps * frac).astype(int)
        
        dyn, X = sample_full(steps, dyn, hyp)
        ESS = ess_corr(X) # num steps / effective sample size
        Lnew = Lfactor * hyp['eps'] / ESS # = 0.4 * length corresponding to one effective sample

        return dyn, {'L': Lnew, 'eps': hyp['eps'], 'sigma': hyp['sigma']}


    return func






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

    Args:
        acc_prob: Acceptance probability of the current sample
        
        state: (log eps, log eps avg, E[acc prob wanted - acc prob], current time step)
            
        acc_prob_wanted: desired average acceptance rate
        
        t0: A free parameter introduced in reference [2] that stabilizes
            the initial steps of the scheme. Defaults to 10.
        kappa: A free parameter introduced in reference [2] that
            controls the weights of steps of the scheme. For a small ``kappa``, the
            scheme will quickly forget states from early steps. This should be a
            number in :math:`(0.5, 1]`. Defaults to 0.75.
        gamma: A free parameter introduced in reference [1] which
            controls the speed of the convergence of the scheme. Defaults to 0.05.
    
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




def dual_averaging(step, frac, acc_prob_wanted):
    
    def func(dyn, hyp, num_steps):
        
        steps = (num_steps * frac).astype(int)

        
        adap = jnp.zeros(4)
        
        
        def _step(_state, useless):
            dyn, hyp, adap = _state
            dyn, acc = step(dyn, hyp)
            adap = dual_averaging(acc, adap, acc_prob_wanted)
            hyp = {'L': hyp['L'], 'eps': jnp.exp(adap[0]), 'sigma': hyp['sigma']}
            return (dyn, hyp, adap), None
        
        
        dyn, hyp, adap = jax.lax.scan(_step, init= (dyn, hyp, adap), xs=None, length= steps)[0]
        
        hyp = {'L': hyp['L'], 'eps': jnp.exp(adap[1]), 'sigma': hyp['sigma']}

        #acc = adap[2] # this is the acceptance rate that we got
        
        return dyn, hyp
    
    return func