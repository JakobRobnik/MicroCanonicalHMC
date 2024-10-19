import jax
import jax.numpy as jnp
from collections import namedtuple
from benchmarks.inference_models import Banana

import sys, os
sys.path.append('../blackjax/')
from blackjax.mcmc import mclmc
from blackjax.mcmc.integrators import isokinetic_leapfrog
from blackjax.util import run_eca

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

print(len(jax.devices()))#, jax.lib.xla_bridge.get_backend().platform)
 #jax.extend.backend.get_backend().platform)

mesh = jax.sharding.Mesh(jax.devices(), 'chains')

target = Banana()
kernel = mclmc.build_kernel(logdensity_fn= target.logdensity_fn, integrator= isokinetic_leapfrog)

key = jax.random.key(0)
num_chains = 256
num_steps = 5

def sequential_kernel(key, state, hyperparameters):
    return state + eps, acc_rate


adap = namedtuple('uMCLMC_adaptation', ['hyperparameters', 'time'])

class Initialization:
    
    def __init__(self):
        None
    def sequential_init(self, key):
        return jnp.array([0., 1., 2.])
    
    def summary_statistics(self, x):
        return x
    
    def ensemble_init(self, state, theta):
        return state

class Adaptation:
    
    def __init__(self):        
        self.initial_state = adap(hyperparameters = 0.1, time= 0)
    
    def summary_statistics(self, x, info):
        return {'x': x, 'acc_rate': info}
           
    def update(self, adaptation_state, Etheta, key_adaptation):

        return adap(hyperparameters = 0.2, time= adaptation_state.time + 1), Etheta
    
final_state, final_adaptation_state, info = run_eca(key, sequential_kernel, Initialization(), Adaptation(), num_steps, num_chains, mesh)

#print(final_state.shape)
#print(info['acc_rate'].shape)
print(info['x'])


#shifter --image=reubenharry/cosmo:1.0 python3 -m ensemble.importing




