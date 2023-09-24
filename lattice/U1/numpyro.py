import matplotlib.pyplot as plt
import numpy as np
import os
import jax
import jax.numpy as jnp

num_cores = 6 #specific to my PC
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(num_cores)

from numpyro.infer import MCMC, NUTS
#numpyro.set_platform("gpu")

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from lattice.U1 import theory
from HMC.mchmc_to_numpyro import mchmc_target_to_numpyro


dir = os.path.dirname(os.path.realpath(__file__))

U1_model = mchmc_target_to_numpyro(theory.Theory)


def sample(L, beta, num_samples, num_chains, num_warmup = 500, thinning= 1):
    """do sampling with NUTS in numpyro"""

    # setup
    theory = theory.Theory(L, beta)
    nuts_setup = NUTS(U1_model, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False)
    sampler = MCMC(nuts_setup, num_warmup=num_warmup, num_samples=num_samples, num_chains= num_chains, progress_bar=False, thinning= thinning)

    key = jax.random.PRNGKey(42)
    key, prior_key = jax.random.split(key)
    x0 = jax.vmap(theory.prior_draw)(jax.random.split(prior_key, num_chains))

    # run
    sampler.warmup(key, L, beta, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    burn_in_steps = np.sum(np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype = int), axis = 1)

    sampler.run(key, L, beta, extra_fields=['num_steps'])
    links = np.array(sampler.get_samples(group_by_chain= True)['x'])

    steps = np.array(sampler.get_extra_fields(group_by_chain= True)['num_steps'], dtype=int)
    print(np.median(steps))
    Q = jax.pmap(jax.vmap(theory.topo_charge))(links)

    chi = jnp.average(jnp.square(Q), axis = 1) / L**2

    return chi

    # plt.plot(np.cumsum(steps), Q, '.')
    # plt.xlabel('gradient evaluations')
    # plt.ylabel('topological charge')
    # plt.show()


def susceptibility_plot():
    """show topological susceptibility as a function of beta with NUTS"""

    beta = np.arange(1, 11)
    chi = np.array([sample(L= 8, beta= b, num_samples= 10000, num_chains= num_cores, num_warmup= 500, thinning= 1) for b in beta])

    plt.plot(beta, chi, 'o-', label = 'NUTS')
    plt.legend()
    plt.xlabel(r'$\beta$')
    plt.ylabel("topological susceptibility")
    plt.savefig('U1_nuts.png')
    plt.show()



susceptibility_plot()