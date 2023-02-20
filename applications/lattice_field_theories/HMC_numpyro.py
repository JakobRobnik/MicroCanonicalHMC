import numpy as np
import os
import jax
import jax.numpy as jnp

num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from numpyro.infer import MCMC, NUTS

from applications.lattice_field_theories.theories import phi4



def nuts(L, lam, num_samples, num_chains, num_warmup = 500, full= True):

    # setup
    theory = phi4.Theory(L, lam)
    nuts_setup = NUTS(phi4.model, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)
    sampler = MCMC(nuts_setup, num_warmup=num_warmup, num_samples=num_samples, num_chains= num_chains, progress_bar=False)

    key = jax.random.PRNGKey(42)
    key, prior_key = jax.random.split(key)
    x0 = jax.vmap(theory.prior_draw)(jax.random.split(prior_key, num_chains))

    # run
    sampler.warmup(key, L, lam, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    burn_in_steps = np.sum(np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype = int), axis = 1)

    sampler.run(key, L, lam, extra_fields=['num_steps'])

    phi = np.array(sampler.get_samples(group_by_chain= True)['phi'])

    #phi = np.array(numpyro_samples['phi']).reshape(num_samples, num_chains, L ** 2)
    phi_bar = np.average(phi, axis = 2)


    steps = np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype=int)

    if full:
        chi = theory.susceptibility2_full(phi_bar)
        return burn_in_steps, steps, chi

    else:
        return theory.susceptibility2(phi_bar)




def ground_truth_nuts():

    sides = [6, 8, 10, 12, 14]
    #sides = [6, ]
    reduced_chi = np.empty((len(sides), len(phi4.reduced_lam), num_cores))

    for i in range(len(sides)):
        side= sides[i]
        print('side = ' + str(side))
        lam = phi4.unreduce_lam(phi4.reduced_lam, side)
        for j in range(len(lam)):
            print(str(j) + '/' +str(len(lam)))
            reduced_chi[i, j, :] = phi4.reduce_chi(nuts(L= side, lam= lam[j], num_samples= 10000, num_chains= num_cores, full= False), side)

    np.save('phi4results/ground_truth.npy', reduced_chi)


def ess():

    sides = [6, ]
    #sides = [6, ]
    reduced_chi = np.empty((len(sides), len(phi4.reduced_lam)))

    for i in range(len(sides)):
        side= sides[i]
        print('side = ' + str(side))
        lam = phi4.unreduce_lam(phi4.reduced_lam, side)
        for j in range(len(lam)):
            print(str(j) + '/' +str(len(lam)))

            burnin, steps, chi = nuts(L= side, lam= lam[j], num_samples= 10000, num_chains= 100, full= True)
            chi = phi4.reduced_chi(chi, side)
            print(np.shape(chi))

            exit()


    np.save('phi4results/ground_truth.npy', reduced_chi)


ground_truth_nuts()
