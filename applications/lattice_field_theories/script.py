import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import os

from applications.lattice_field_theories.theories import phi4
from sampling.sampler import Sampler
from sampling.sampler import find_crossing
from sampling.grid_search import search_wrapper


#set the number of cores
num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

#data
dir = os.path.dirname(os.path.realpath(__file__))
params_critical_line = pd.read_csv(dir + '/theories/phi4_parameters.csv')

reduced_lam = np.linspace(-2.5, 7.5, 18)



def parallel_run(function, values):
    parallel_function= jax.pmap(jax.vmap(function))
    results = jnp.array(parallel_function(values.reshape(num_cores, len(values) // num_cores)))
    return results.reshape([len(values), ] + [results.shape[i] for i in range(2, len(results.shape))])



def get_params(side):
    """parameters from https://arxiv.org/pdf/2207.00283.pdf"""
    return np.array(params_critical_line[params_critical_line['L'] == side][['lambda']])[0][0] # = lambda


def unreduce_lam(reduced_lam, side):
    """see Fig 3 in https://arxiv.org/pdf/2207.00283.pdf"""
    return 4.25 * (reduced_lam * np.power(side, -1.0) + 1.0)


def reduce_chi(chi, side):
    return chi * np.power(side, -7.0/4.0)


def ground_truth(sides):

    sides = [12, ]
    #sides = [6, 8, 10, 12, 14]
    reduced_chi = np.empty((len(sides), len(reduced_lam)))

    def chi(lam):
        target = phi4.Theory(side, lam)
        sampler = Sampler(target, L=np.sqrt(target.d) * 1, eps=np.sqrt(target.d) * 0.005)
        phibar, E = sampler.sample(10000000, output='energy')
        return reduce_chi(target.susceptibility2(phibar))


    for i in range(len(sides)):
        side = sides[i]
        lam = unreduce_lam(reduced_lam, side)
        chis= parallel_run(chi, lam)

        reduced_chi[i, :] = chis

    df = pd.DataFrame(reduced_chi.T, columns=['L = ' + str(side) for side in sides])
    df['reduced lam'] = reduced_lam

    df.to_csv('phi4_results/reduced_chi2.csv')




class ess_with_chi:

    def __init__(self, side):
        self.lambda_array = unreduce_lam(reduced_lam, side)
        data = pd.read_csv(dir + '/theories/phi4_ground_truth.csv')
        print(data)
        self.ground = data['L = ' + str(side)] #ground truth reduced chi(2) as computed by very long chains
        self.side = side


    def ess(self, alpha, beta, index_lam, steps = 100000):

        target = phi4.Theory(self.side, self.lambda_array[index_lam])
        sampler = Sampler(target, L=np.sqrt(target.d) * alpha, eps=np.sqrt(target.d) * beta, integrator='LF')

        phibar = sampler.sample(steps, num_chains=num_cores*10, remove_burn_in= True)
        chi = reduce_chi(target.susceptibility2_full(phibar), self.side)
        chi0 = self.ground[index_lam]
        error = jnp.average(jnp.abs(chi - chi0) / chi0, axis = 0)
        nsteps = find_crossing(error, 0.1)

        return 200.0 / nsteps




def grid_search(side, index_lam):

    ess_explorer = ess_with_chi(side)

    score_function = lambda a, e: ess_explorer.ess(a, e, index_lam)

    search_wrapper(score_function, 2, 20, 0.4, 0.05, parallel= False, show= True)


def ess_lam(side):
    ess_explorer = ess_with_chi(side)

    ESS = np.empty(len(reduced_lam))
    for i in range(len(reduced_lam)):
        ESS[i] = ess_explorer.ess(3.0, 0.1, i)
        print(i, ESS[i])

    np.save('phi4_results/ESS_L'+str(side) + '.npy', ESS)


def gerdes_tabel1():

    side= 12
    lam = get_params(side)
    target = phi4.Theory(side, lam)

    sampler = Sampler(target, L = np.sqrt(target.d)*1.0, eps = np.sqrt(target.d) * 0.01, integrator= 'LF')

    phi, E = sampler.sample(10000, output = 'full', random_key= jax.random.PRNGKey(0))

    L = jax.vmap(target.nlogp)(phi)
    from sampling.sampler import burn_in_ending
    print(burn_in_ending(L))
    plt.plot(L)
    plt.show()


    phibar, E = sampler.sample(100000, output = 'energy', random_key= jax.random.PRNGKey(0))

    burnin = 1000
    phi, E = phibar[burnin:], E[burnin:]
    plt.plot(E)
    plt.show()
    print(np.std(E)**2/target.d)

    plt.plot(target.susceptibility2_full(phibar))
    plt.show()



#from NUTS.nuts import nuts6 as nuts_sample


def hmc(target, samples, samples_adapt):
    theta0 = target.prior_draw(jax.random.PRNGKey(0))
    delta = 0.2

    samples, lnprob, epsilon = nuts_sample(target.grad_nlogp, samples, samples_adapt, theta0, delta, progress=True)

    return samples




print(jax.local_device_count())
print(jax.lib.xla_bridge.get_backend().platform)


#grid_search(6, 17)


#phibar = hmc(target, 5000, 5000)



#plot_gerdes_fig3()

#ess_lam(12)
#gerdes_tabel1()
