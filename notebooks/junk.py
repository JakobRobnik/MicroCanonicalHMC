import jax
import jax.numpy as jnp
import numpy as np
from sampling.sampler import Sampler
from benchmarks.benchmarks_mchmc import *



target = IllConditionedGaussian(d = 100, condition_number= 100.)
sampler = Sampler(target, L = jnp.sqrt(target.d), eps = 1., diagonal_preconditioning= False, frac_tune1 = 0, frac_tune2 = 0, frac_tune3 = 0)
sampler.sample(10000, output = 'normal')


exit()
hmc = pd.read_csv('submission/MCHMC/Table HMC_LF.csv')
mchmc = pd.read_csv('submission/MCHMC/Table MCHMC q = 0.csv', sep = '\t')
#mchmc = pd.read_csv('submission/MCHMC/Table generalized_LF_q=0.csv')#, sep = '\t')
print(mchmc)
d = np.array([100, 50, 36, 20, 51, 2400])
print(hmc['L']/hmc['eps'])
print(mchmc['alpha']*np.sqrt(d)/mchmc['eps'])

print(hmc['eps'])
print(mchmc['eps'] / np.sqrt(d))
