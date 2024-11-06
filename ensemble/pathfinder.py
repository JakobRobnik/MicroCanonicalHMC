import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os, sys
sys.path.append('../blackjax/')
from ensemble.main import targets
import blackjax




def _main(model):

    init_key, approx_key, sample_key = jax.random.split(jax.random.key(42), 3)
    init = model.sample_init(init_key)

    pf = blackjax.pathfinder(model.logdensity_fn)
    state, info = pf.approximate(approx_key, init)
    print(info)
    samples, _ = pf.sample(sample_key, state, 4096)
    
    print(samples.shape)
    
    observables, contract= blackjax.adaptation.ensemble_mclmc.bias(model)
    num_grads = 30
    
    x2 = jax.vmap(observables)(observables)
    print(x2.shape)
    E_x2 = jnp.average(x2, axis= 0)
    b2 = contract(E_x2)
    
    print(b2)
    return (*b2, num_grads)



if __name__ == '__main__':
    
    
    for t in targets[:1]:
        _main(t[0]) 
        
    # b = np.array([_main(t[0]) for t in targets])
    
    # df = pd.DataFrame(b, columns= ['bmax', 'bavg', 'grads']) # save the results
    # df['name'] = [target[0].name for target in targets]
    # df.to_csv('ensemble/pathfinder_data.csv', sep= '\t', index=False)