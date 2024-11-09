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




def _main(model):
    
    t0 = time.time()

    # count the number of logp calls
    global calls
    calls= 0

    def register_call():
        global calls
        calls += 1

    def logp(x):
        callback(register_call)
        return model.logdensity_fn(x)


    # initialize pathfinder
    init_key, approx_key, sample_key = jax.random.split(jax.random.key(42), 3)
    init = model.sample_init(init_key)
    pf = blackjax.pathfinder(logp)

    # run the algorithm
    state, info = pf.approximate(approx_key, init, maxiter=30)
    
    # obtain samples from the estimate of the posterior
    samples, _ = pf.sample(sample_key, state, 4096)
        
    # compute the bias of the estimate
    observables, contract= blackjax.adaptation.ensemble_mclmc.bias(model)
    x2 = jax.vmap(observables)(samples)
    E_x2 = jnp.average(x2, axis= 0)
    b2 = contract(E_x2)
    
    # measure how long did the algorithm take
    jax.block_until_ready(b2)
    t1 = time.time()
    dt= t1-t0
    
    return (*b2, calls, dt)



if __name__ == '__main__':
    
    b = np.array([_main(t[0]) for t in targets])
    
    df = pd.DataFrame(b, columns= ['bmax', 'bavg', 'grads', 'time']) # save the results
    df['name'] = [target[0].name for target in targets]
    df.to_csv('ensemble/pathfinder_data.csv', sep= '\t', index=False)