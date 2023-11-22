import jax
import jax.numpy as jnp

from .correlation_length import ess_corr



# all tuning functions are wrappers, recieving some parameters and returning a function
# func(dyn, hyp, num_total_steps) -> (dyn, hyp) 




def run(dyn, hyp, schedule, num_steps):
    
    _dyn, _hyp = dyn, hyp
    extra_params = None

    for program in schedule:
        _dyn, _hyp, extra_params = program(_dyn, _hyp, num_steps, extra_params)
        
    return _dyn, _hyp
 
 

 
def nan_reject(x, u, l, g, xx, uu, ll, gg, eps, eps_max, dK):
    """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""
    
    nonans = jnp.all(jnp.isfinite(xx))
    _x, _u, _l, _g, _eps, _dk = jax.tree_util.tree_map(lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old), 
                                                       (xx, uu, ll, gg, eps_max, dK), 
                                                       (x, u, l, g, eps * 0.8, 0.))
    
    return nonans, _x, _u, _l, _g, _eps, _dk
    
 


def tune12(dynamics, d, adjust, 
           diag_precond, frac, 
           varEwanted = 1e-3, sigma_xi = 1.5, neff = 150, # these parameters will have no effect if adjust = True
           acc_prob_wanted = 0.7): # these parameters will have no effect if adjust = False
    
    gamma = (neff - 1.0) / (neff + 1.0)
    
    if adjust:
        adaptive_state = jnp.zeros(4)
        _step = dual_averaging      
        
    else:
        adaptive_state = (0., 0., jnp.inf)
        _step = predictor
        
    
    def predictor(dyn_old, hyp, adaptive_state):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
            Designed for the unadjusted MCHMC"""
        
        W, F, eps_max = adaptive_state

        # dynamics
        dyn_new, energy_change = dynamics(dyn_old, hyp)

        # step updating
        success, x, u, l, g, eps_max, energy_change = nan_reject(dyn_old['x'], dyn_old['u'], dyn_old['l'], dyn_old['g'], 
                                                                      dyn_new['x'], dyn_new['u'], dyn_new['l'], dyn_new['g'], 
                                                                      hyp['eps'], eps_max, energy_change)

        dyn = {'x': x, 'u': u, 'l': l, 'g': g, 'key': dyn_new['key']}
        
        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (jnp.square(energy_change) / (d * varEwanted)) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * sigma_xi)))  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        F = gamma * F + w * (xi/jnp.power(hyp['eps'], 6.0))
        W = gamma * W + w
        eps = jnp.power(F/W, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (eps > eps_max) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        hyp_new = {'L': hyp['L'], 'eps': eps, 'sigma': hyp['sigma']}
        
        return dyn, hyp_new, (W, F, eps_max), success

    def dual_averaging():
        

    def update_kalman(x, state, outer_weight, success, eps):
        """kalman filter to estimate the size of the posterior"""
        W, F1, F2 = state
        w = outer_weight * eps * success
        zero_prevention = 1-outer_weight
        F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        W += w
        return (W, F1, F2)


    def step(state, outer_weight):
        """does one step of the dynamcis and updates the estimate of the posterior size and optimal stepsize"""
        dyn, hyp, adaptive_state, kalman_state = state
        dyn, hyp, adaptive_state, success = _step(dyn, hyp, adaptive_state)
        kalman_state = update_kalman(dyn['x'], kalman_state, outer_weight, success, hyp['eps'])

        return (dyn, hyp, adaptive_state, kalman_state), None


    def func(_dyn, _hyp, num_steps, adjust):
        
        num_steps1, num_steps2 = (num_steps * frac).astype(int)
            
        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        #initial state
        
        kalman_state = (0., jnp.zeros(d), jnp.zeros(d))

        # run the steps
        state = jax.lax.scan(step, init= (_dyn, _hyp, adaptive_state, kalman_state), xs= outer_weights, length= num_steps1 + num_steps2)[0]
        dyn, hyp, adaptive_state, kalman_state = state
        
        # determine L
        if num_steps2 != 0.:
            _, F1, F2 = kalman_state
            variances = F2 - jnp.square(F1)
            L = jnp.sqrt(jnp.sum(variances))

            # optionally we do the diagonal preconditioning (and readjust the stepsize)
            if diag_precond:

                # diagonal preconditioning
                sigma = jnp.sqrt(variances)
                L = jnp.sqrt(d)

                #readjust the stepsize
                steps = num_steps2 // 3 #we do some small number of steps
                state = jax.lax.scan(step, init= state, xs= jnp.ones(steps), length= steps)[0]
                dyn, hyp, adaptive_state, kalman_state = state
            
            else:
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
            dyn_new, _ = step(dyn_old, hyp)
            
            return dyn_new, dyn_new['x']

        return jax.lax.scan(_step, init=_dyn, xs=None, length=num_steps)


    def func(dyn, hyp, num_steps, extra_params):
        steps = (num_steps * frac).astype(int)
        
        dyn, X = sample_full(steps, dyn, hyp)
        ESS = ess_corr(X) # num steps / effective sample size
        Lnew = Lfactor * hyp['eps'] / ESS # = 0.4 * length corresponding to one effective sample

        return dyn, {'L': Lnew, 'eps': hyp['eps'], 'sigma': hyp['sigma']}, extra_params


    return func






def dual_averaging(acc_prob, state, acc_prob_wanted = 0.7, t0 = 10, gamma= 0.05, kappa= 0.75):
    """taken from numpyro, see documentation there"""

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