import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os, sys
import time
sys.path.append('../blackjax/')
from ensemble.main import targets
import blackjax
from jax.debug import callback
from arviz import psislw


models = [t[0] for t in targets]
model_names = [model.name for model in models]


def multi_path_slow(model, num_chains, rng_key= jax.random.key(42)):
    """Count the number of logp calls. 
    Pathfinder is here run in a for loop which makes it very slow, so this is only intended for counting the number of gradients."""

    print(model.name)
    init_key, run_key, resample_key  = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, num_chains)
    run_keys = jax.random.split(run_key, num_chains)
            
    def register_call():
        global calls
        calls += 1

    def _log_density(x):
        callback(register_call)
        return model.logdensity_fn(x)

    pf = blackjax.pathfinder(_log_density)


    def single_run(key, init):
        key1, key2 = jax.random.split(key)
            
        state, info = pf.approximate(key1, init, maxiter=30)

    # run the algorithm
    init = jax.vmap(model.sample_init)(init_keys)
    
    all_calls = np.empty(num_chains)
    for i in range(num_chains):
        global calls
        calls= 0
        
        single_run(run_keys[i], init[i])
        
        all_calls[i] = calls
        print(i, calls)

    return all_calls


def multi_path(model, num_chains, num_samples, rng_key= jax.random.key(42)):
    """Multi-path Pathfinder, algorithm 2 in the paper"""
    
    init_key, run_key, resample_key  = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, num_chains)
    run_keys = jax.random.split(run_key, num_chains)
    
    pf = blackjax.pathfinder(model.logdensity_fn)
    
    
    def single_run(key, init):
        key1, key2 = jax.random.split(key)
            
        state, info = pf.approximate(key1, init, maxiter=30)

        # obtain samples from the estimate of the posterior
        samples, logq = pf.sample(key2, state, num_samples)
        
        # importance weights
        logp = jax.vmap(model.logdensity_fn)(samples)
        log_weights = logp - logq
        
        return samples, log_weights 
    
    
    multiple_run = jax.pmap(single_run)

    # run multiple optimizations
    init = jax.pmap(model.sample_init)(init_keys) # initial conditions
    samples, log_weights = multiple_run(run_keys, init)
    samples = jnp.concatenate(samples)
    log_weights = jnp.concatenate(log_weights)

    # pareto smoothing    
    log_weights = jnp.array(psislw(np.array(log_weights))[0])

    # importance resampling
    samples = jax.random.choice(resample_key, samples, (len(samples), ), p = jnp.exp(log_weights))

    return samples



def _convergence(model):
    
    print(model.name)

    # run pathfinder
    num_chains = 64
    samples = multi_path(model, num_chains= num_chains, num_samples= 4096//num_chains)

    # compute the bias of the estimate
    observables, contract= blackjax.adaptation.ensemble_mclmc.bias(model)
    x2 = jax.vmap(observables)(samples)
    E_x2 = jnp.average(x2, axis= 0)
    b2 = contract(E_x2)
    
    return b2


def convergence():
    
    b = np.array([_convergence(model) for model in models])
    df = pd.DataFrame(b, columns= ['bmax', 'bavg']) # save the results
    df['name'] = model_names
    df.to_csv('ensemble/submission/pathfinder_convergence.csv', sep= '\t', index=False)
    
    
def cost():
    
    grad_calls = np.array([multi_path_slow(model, num_chains= 64) for model in models])
    df = pd.DataFrame(grad_calls.T, columns= model_names) # save the results
    df.to_csv('ensemble/submission/pathfinder_cost.csv', sep= '\t', index=False)
    
    
    
if __name__ == '__main__':
    
    #cost()
    convergence()