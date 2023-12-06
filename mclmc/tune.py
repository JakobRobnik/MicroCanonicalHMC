import jax
import jax.numpy as jnp
from typing import NamedTuple

from .dynamics import State
from .correlation_length import ess_corr



class Hyperparameters(NamedTuple):
    """Tunable parameters"""

    L: float
    eps: float
    sigma: any


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
           diag_precond, frac, 
           varEwanted = 1e-3, sigma_xi = 1.5, neff = 150):
           
    gamma_forget = (neff - 1.0) / (neff + 1.0)
    
    
    def predictor(dyn_old, hyp, adaptive_state):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
            Designed for the unadjusted MCHMC"""
        
        W, F, eps_max = adaptive_state

        # dynamics
        dyn_new, energy_change = dynamics(dyn_old, hyp)

        # step updating
        success, x, u, l, g, eps_max, energy_change = nan_reject(dyn_old.x, dyn_old.u, dyn_old.l, dyn_old.g, 
                                                                      dyn_new.x, dyn_new.u, dyn_new.l, dyn_new.g, 
                                                                      hyp.eps, eps_max, energy_change)

        dyn = State(x, u, l, g, dyn_new.key)
        
        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (jnp.square(energy_change) / (d * varEwanted)) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * sigma_xi)))  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        F = gamma_forget * F + w * (xi/jnp.power(hyp.eps, 6.0))
        W = gamma_forget * W + w
        eps = jnp.power(F/W, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (eps > eps_max) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        hyp_new = Hyperparameters(hyp.L, eps, hyp.sigma)
        
        return dyn, hyp_new, hyp_new, (W, F, eps_max), success


    def update_kalman(x, state, outer_weight, success, eps):
        """kalman filter to estimate the size of the posterior"""
        W, F1, F2 = state
        w = outer_weight * eps * success
        zero_prevention = 1-outer_weight
        F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        W += w
        return (W, F1, F2)


    adap0 = (0., 0., jnp.inf)
    _step = predictor
        
        
    def step(state, outer_weight):
        """does one step of the dynamcis and updates the estimate of the posterior size and optimal stepsize"""
        dyn, hyp, _, adaptive_state, kalman_state = state
        dyn, hyp, hyp_final, adaptive_state, success = _step(dyn, hyp, adaptive_state)
        kalman_state = update_kalman(dyn.x, kalman_state, outer_weight, success, hyp.eps)

        return (dyn, hyp, hyp_final, adaptive_state, kalman_state), None


    def func(_dyn, _hyp, num_steps):
        
        num_steps1, num_steps2 = jnp.rint(num_steps * frac).astype(int)
            
        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        #initial state
        
        kalman_state = (0., jnp.zeros(d), jnp.zeros(d))

        # run the steps
        state = jax.lax.scan(step, init= (_dyn, _hyp, _hyp, adap0, kalman_state), xs= outer_weights, length= num_steps1 + num_steps2)[0]
        dyn, _, hyp, adap, kalman_state = state
        
        # determine L
        L = hyp.L
        sigma = hyp.sigma
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
                dyn, _, hyp, adap, kalman_state = state
            else:
                sigma = hyp.sigma
        
        return dyn, Hyperparameters(L, hyp.eps, sigma)

    return func




def tune3(step, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""
    

    def sample_full(num_steps, _dyn, hyp):
        """Stores full x for each step. Used in tune2."""

        def _step(state, useless):
            dyn_old = state
            dyn_new, _ = step(dyn_old, hyp)
            
            return dyn_new, dyn_new.x

        return jax.lax.scan(_step, init=_dyn, xs=None, length=num_steps)


    def func(dyn, hyp, num_steps):
        steps = jnp.rint(num_steps * frac).astype(int)
        
        dyn, X = sample_full(steps, dyn, hyp)
        ESS = ess_corr(X) # num steps / effective sample size
        Lnew = Lfactor * hyp.eps / ESS # = 0.4 * length corresponding to one effective sample

        return dyn, Hyperparameters(Lnew, hyp.eps, hyp.sigma)


    return func

