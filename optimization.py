import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import ESH
import myHMC
from targets import *

key = jax.random.PRNGKey(0)
d = 50
target = IllConditionedGaussian(d, 10000.0)

key, subkey = jax.random.split(key)
x0 = 3.0*jax.random.normal(subkey, shape=(d, ), dtype='float64')
loss0 = target.nlogp(x0)

plt.figure(figsize=(15, 10))

#HMC (no resampling)
sampler = myHMC.Sampler(target, eps=0.01)
steps = 200
X = sampler.sample(x0, steps, steps *sampler.eps * 2, key)

plt.plot(np.insert(target.nlogp(X), 0, loss0) + 0.5*d *np.log(2*np.pi), color = 'tab:red', label = 'HMC (no resampling)')

#HMC (tuned)
sampler.eps = 0.1
X = sampler.sample(x0, steps, 0.15, key)
plt.plot(np.insert(target.nlogp(X), 0, loss0) + 0.5*d *np.log(2*np.pi), color = 'tab:orange', label = 'HMC (hand-tuned)')

#MCHMC
sampler = ESH.Sampler(target, eps=1.5)
steps = 200
X, W = sampler.sample(x0, steps, steps *sampler.eps * 2, key)
plt.plot(np.insert(target.nlogp(X), 0, loss0) + 0.5*d *np.log(2*np.pi), color = 'tab:blue', label = 'MCHMC (no bounces)')
plt.plot([0, len(X)], np.ones(2) * d * 0.5 * (1 + np.log(2 * np.pi)), ':', color = 'black', label = 'entropy')


plt.yscale('log')
plt.legend()
plt.xlabel('# gradient evaluations')
plt.ylabel('- log p')
plt.savefig("optimization.png")
plt.show()