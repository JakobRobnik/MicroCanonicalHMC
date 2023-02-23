import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import os

num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from applications.lattice_field_theories.theories import phi4
from sampling.sampler import Sampler
from sampling.sampler import find_crossing
from sampling.grid_search import search_wrapper

dir = os.path.dirname(os.path.realpath(__file__))
#params_critical_line = pd.read_csv(dir + '/theories/phi4_parameters.csv')




def parallel_run(function, values):
    parallel_function= jax.pmap(jax.vmap(function))
    results = jnp.array(parallel_function(values.reshape(num_cores, len(values) // num_cores)))
    return results.reshape([len(values), ] + [results.shape[i] for i in range(2, len(results.shape))])



def get_params(side):
    """parameters from https://arxiv.org/pdf/2207.00283.pdf"""
    return np.array(params_critical_line[params_critical_line['L'] == side][['lambda']])[0][0] # = lambda



def ground_truth():

    sides = [6, 8, 10, 12, 14]

    def chi(lam):
        target = phi4.Theory(side, lam)
        sampler = Sampler(target, L= np.sqrt(target.d) * 1, eps= np.sqrt(target.d) * 0.2)
        phibar, burnin = sampler.sample(1000000)
        return phi4.reduce_chi(target.susceptibility2(phibar[burnin:, 0]), side)



    for i in range(len(sides)):
        side = sides[i]
        print('side = ' + str(side))
        lam = phi4.unreduce_lam(phi4.reduced_lam, side)
        chis= [chi(ll) for ll in lam]
        np.save(dir + '/phi4results/mchmc/ground_truth/chi/L'+str(side)+'.npy', chis)




class ess_with_psd:

    def __init__(self, side):
        self.lambda_array = phi4.unreduce_lam(phi4.reduced_lam, side)
        data = np.load(dir + '/phi4results/hmc/ground_truth/psd/L'+str(side)+'.npy')
        self.ground = np.median(data, axis = 1)
        self.side = side


    def ess(self, alpha, beta, index_lam, steps = 10000):

        target = phi4.Theory(self.side, self.lambda_array[index_lam])
        sampler = Sampler(target, L=np.sqrt(target.d) * alpha, eps=np.sqrt(target.d) * beta, integrator='MN')

        phi, E, burnin = sampler.sample(steps, num_chains= num_cores, output= 'full')
        phi_reshaped = phi.reshape(phi.shape[0], phi.shape[1], target.L, target.L)
        P = jax.vmap(jax.vmap(target.psd))(phi_reshaped)

        b2_all = np.empty((len(P), steps))

        for ichain in range(len(P)):
            Pchain = np.cumsum(P[ichain, burnin[ichain]:, :, :], axis= 0) / np.arange(1, 1 + steps-burnin[ichain])[:, None, None]
            b2_part = np.average(np.square(1.0 - (Pchain / self.ground[index_lam][None, :, :])), axis = (1, 2))
            b2_part = np.concatenate((np.ones(burnin[ichain]) * 500, b2_part))
            b2_all[ichain] = b2_part

        b2_sq = np.median(b2_all, axis = 0)

        nsteps = find_crossing(b2_sq, 0.01)

        return 200.0 / nsteps



def grid_search(side, index_lam):

    ess_explorer = ess_with_psd(side)

    score_function = lambda a, e: ess_explorer.ess(a, e, index_lam)

    ess, alpha, beta = search_wrapper(score_function, 0.5, 5, 0.1, 1.0, parallel= False, save_name= dir + '/phi4results/mchmc/grid_search/L'+str(side)+'/lambda_'+str(index_lam) + '.png')

    return ess, alpha.item(), beta.item()



def compute_ess():
    sides = [6, 8, 10, 12, 14]
    ess, alpha, beta = np.zeros(len(phi4.reduced_lam)), np.zeros(len(phi4.reduced_lam)), np.zeros(len(phi4.reduced_lam))

    for i in range(1, len(sides)):
        side = sides[i]

        for j in range(len(phi4.reduced_lam)):
            ess[j], alpha[j], beta[j] = grid_search(side, j)

        df = pd.DataFrame(np.array([phi4.reduced_lam, ess, alpha, beta]).T, columns = ['reduced lambda', 'ess', 'alpha', 'beta'])
        print(df)
        df.to_csv(dir+'/phi4results/mchmc/grid_search/L'+str(side)+'/all.npy')



#compute_ess()
ground_truth()