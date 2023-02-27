import matplotlib.pyplot as plt
import numpy as np
import os
import jax

num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

import numpyro
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS
#numpyro.set_platform("gpu")

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from applications.lattice_field_theories.theories import phi4
from applications.lattice_field_theories.theories import gauge_theory

dir = os.path.dirname(os.path.realpath(__file__))


def mchmc_target_to_numpyro(target):
    """given mchmc target outputs the corresponding numpyto model"""

    class distribution(numpyro.distributions.Distribution):
        """Custom defined distribution, see https://forum.pyro.ai/t/creating-a-custom-distribution-in-numpyro/3332/3"""

        support = constraints.real_vector

        def __init__(self, *args):
            self.target = target(*args)
            self.log_prob = lambda x: -self.target.nlogp(x)
            super().__init__(event_shape=(self.target.d,))

        def sample(self, key, sample_shape=()):
            raise NotImplementedError

    def model(*args):
        x = numpyro.sample('x', distribution(*args))

    return model


phi4_model = mchmc_target_to_numpyro(phi4.Theory)
U1_model = mchmc_target_to_numpyro(gauge_theory.Theory)


def nuts(L, lam, num_samples, num_chains, num_warmup = 500, thinning= 1, full= True, psd= True):

    # setup
    theory = phi4.Theory(L, lam)
    nuts_setup = NUTS(phi4_model, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)
    sampler = MCMC(nuts_setup, num_warmup=num_warmup, num_samples=num_samples, num_chains= num_chains, progress_bar=False, thinning= thinning)

    key = jax.random.PRNGKey(42)
    key, prior_key = jax.random.split(key)
    x0 = jax.vmap(theory.prior_draw)(jax.random.split(prior_key, num_chains))

    # run
    sampler.warmup(key, L, lam, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    burn_in_steps = np.sum(np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype = int), axis = 1)

    sampler.run(key, L, lam, extra_fields=['num_steps'])

    phi = np.array(sampler.get_samples(group_by_chain= True)['x']).reshape(num_chains, num_samples//thinning, L, L)

    steps = np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype=int)

    if psd:

        PSD = np.average(theory.psd(phi), axis = 1)

        if full:
            return burn_in_steps, steps, np.cumsum(theory.psd(phi), axis= 1) / np.arange(1, 1+num_samples//thinning)[None, :, None, None]
        else:
            return PSD

    else:
        phi_bar = np.average(phi, axis=2)

        if full:
            chi = theory.susceptibility2_full(phi_bar)
            return burn_in_steps, steps, chi

        else:
            return theory.susceptibility2(phi_bar)



def nuts_u1(L, beta, num_samples, num_chains, num_warmup = 500, thinning= 1):

    # setup
    theory = gauge_theory.Theory(L, beta)
    nuts_setup = NUTS(U1_model, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)
    sampler = MCMC(nuts_setup, num_warmup=num_warmup, num_samples=num_samples, num_chains= num_chains, progress_bar=False, thinning= thinning)

    key = jax.random.PRNGKey(42)
    key, prior_key = jax.random.split(key)
    x0 = jax.vmap(theory.prior_draw)(jax.random.split(prior_key, num_chains))

    # run
    sampler.warmup(key, L, beta, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    burn_in_steps = np.sum(np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype = int), axis = 1)

    sampler.run(key, L, beta, extra_fields=['num_steps'])
    links = np.array(sampler.get_samples()['x'])
    steps = np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype=int)
    Q = jax.vmap(theory.topo_charge)(links)

    plt.plot(np.cumsum(steps), Q, '.')
    plt.xlabel('gradient evaluations')
    plt.ylabel('topological charge')
    plt.show()



def ground_truth_nuts():

    sides = [6, 8, 10, 12, 14]
    thinning = [10, 10, 100, 100, 100]

    for i in range(1, len(sides)):
        side = sides[i]
        print('side = ' + str(side))
        lam = phi4.unreduce_lam(phi4.reduced_lam, side)
        data = np.empty((len(phi4.reduced_lam), num_cores, side, side))

        for j in range(len(lam)):
            print(str(j) + '/' +str(len(lam)))
            data[j] = nuts(L= side, lam= lam[j], num_samples= 1000*thinning[i], num_chains= num_cores,
                           num_warmup= 2000, thinning = thinning[i], full= False, psd= True)

        np.save('phi4results/hmc/ground_truth/psd/L'+str(side)+'.npy', data)


def join_ground_truth_arrays():
    dir = 'phi4results/hmc/ground_truth/'
    sides = [6, 8, 10, 12, 14]

    arr = np.array([np.load(dir + 'L' + str(side) + '.npy') for side in sides])
    np.save(dir + 'all.npy', arr)


def quartiles(data, axis):

    val = np.expand_dims(np.median(data, axis), axis)
    shape = np.array(np.shape(data), dtype = int)
    shape[axis] = shape[axis]//2

    upper = np.median(data[data - val > 0].reshape(shape), axis = axis)
    lower = np.median(data[data - val < 0].reshape(shape), axis = axis)

    return val, lower, upper


def compute_ess():


    ess = np.empty((len(phi4.reduced_lam), 2))

    sides = [6, 8, 10, 12, ]#14]
    repeat = 5

    for i in range(3, len(sides)):
        side= sides[i]
        PSD0 = np.median(np.load(dir + '/phi4results/hmc/ground_truth/psd/L' + str(side) + '.npy'), axis= 1)
        print('side = ' + str(side))
        lam = phi4.unreduce_lam(phi4.reduced_lam, side)
        num_samples, thinning = 3000, 1

        for j in range(len(lam)):
            print(str(j) + '/' +str(len(lam)))

            num_burnin = 0
            steps = np.zeros(num_samples//thinning)
            b2_sq = np.zeros(num_samples//thinning)

            for rep in range(repeat):
                burnin, _steps, PSD = nuts(L= side, lam= lam[j], num_samples= num_samples,
                                          num_chains= num_cores, thinning= 1, full= True, psd= True)
                num_burnin += np.average(burnin)
                steps += np.average(np.cumsum(_steps, axis= 1), axis= 0)
                b2_sq += np.average(np.square(1 - (PSD/PSD0[None, None, j, :, :])), axis= (0, 2, 3))

            b2_sq /= repeat
            steps /= repeat
            num_burnin /= repeat

            index= np.argmax(b2_sq < 0.01)

            if index == 0:
                num_steps = np.inf
                plt.plot(np.sqrt(b2_sq))
                plt.plot(np.arange(len(b2_sq)), np.ones(len(b2_sq))*0.1, color = 'black')
                plt.xlabel('hmc steps')
                plt.ylabel(r'$b_2$')
                plt.yscale('log')
                plt.show()

            else:
                num_steps = steps[index]

            print(num_steps)
            ess[j, 0] = 200.0 / num_steps
            ess[j, 1] = 200.0 / (num_steps + num_burnin)

        np.save('phi4results/hmc/ess/psd/L'+str(side)+'.npy', ess)



nuts_u1(L= 16, beta= 2.0, num_samples= 1000, num_chains= 1, num_warmup= 500, thinning= 1)
