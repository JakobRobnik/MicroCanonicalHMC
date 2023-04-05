
#Sparse logistic regression fitted to the German credit data
#We use the version implemented in the inference-gym: https://pypi.org/project/inference-gym/
#In some part we directly use their tutorial: https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb


import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import random
from numpyro.infer import MCMC, NUTS

import HMC.benchmarks_numpyro as targets
import sampling.benchmark_targets as MCHMC_targets


#to get ground truth, run the following from the source directory of inference-gym:
#python -m inference_gym.tools.get_ground_truth --target= GermanCreditNumericSparseLogisticRegression

target = gym.targets.GermanCreditNumericSparseLogisticRegression()

target = gym.targets.VectorModel(target, flatten_sample_transformations=True)


identity_fn = target.sample_transformations['identity']

def target_nlog_prob_fn(z):
    x = target.default_event_space_bijector(z)
    return -(target.unnormalized_log_prob(x) + target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims=1))

target_nlog_prob_grad_fn = jax.grad(target_nlog_prob_fn)


def map_solution():

    def map_objective_fn(z):
      x = target.default_event_space_bijector(z)
      return -target.unnormalized_log_prob(x)

    map_objective_grad_fn = jax.grad(map_objective_fn)


    # MAP solution

    def optimize(z_init, objective_fn, objective_grad_fn, learning_rate, num_steps):
        def opt_step(z):
            objective = objective_fn(z)
            z = z - learning_rate * objective_grad_fn(z)
            return z, objective

        return jax.lax.scan(lambda z, _: opt_step(z), init=z_init, xs=None, length=num_steps)


    z_map, objective_trace = optimize(
        z_init=jnp.zeros(target.default_event_space_bijector.inverse_event_shape(target.event_shape)),
        objective_fn=map_objective_fn, objective_grad_fn=map_objective_grad_fn, learning_rate=0.001, num_steps=200, )


    return z_map



class Target():

    def __init__(self):
        self.d = 51
        identity_fn = target.sample_transformations['identity']
        self.variance = jnp.square(identity_fn.ground_truth_standard_deviation) + jnp.square(identity_fn.ground_truth_mean) #in the transformed coordinates

        xmap = map_solution()
        self.transform = lambda x: target.default_event_space_bijector(x + xmap)
        self.nlogp = lambda x: target_nlog_prob_fn(x + xmap)
        self.grad_nlogp = lambda x: (target_nlog_prob_fn(x + xmap), target_nlog_prob_grad_fn(x + xmap))

    # def prior_draw(self, key):
    #     return jnp.zeros(self.d)

    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64') * 0.5



def ground_truth(key_num):

    # setup
    nuts_setup = NUTS(target, adapt_step_size=True, adapt_mass_matrix=True)#, step_size= stepsize)

    sampler = MCMC(nuts_setup, num_warmup= 10000, num_samples= 10000 * 10, num_chains= 1, progress_bar= True, thinning=10) #will return num_samples / thinning

    # run
    key = random.PRNGKey(key_num)
    key, prior_key = random.split(key)
    MCHMC_target = Target()

    x0 = MCHMC_target.transform(MCHMC_target.prior_draw(prior_key))


    #run
    sampler.warmup(key, init_params=x0, extra_fields=['num_steps'], collect_warmup=True)
    warmup_calls = np.sum(np.array(sampler.get_extra_fields()['num_steps'], dtype=int))
    sampler.run(key, extra_fields=['num_steps'])

    # get results
    numpyro_samples = sampler.get_samples()

    steps = np.array(sampler.get_extra_fields()['num_steps'], dtype=int)

    s= np.array(numpyro_samples['s'])
    sigma= np.array(numpyro_samples['sigma'])
    nu= np.array(numpyro_samples['nu'])

    if posterior_band:
        volatility = np.sort(np.exp(s), axis=0)
        np.save('data/stochastic_volatility/NUTS_posterior_band.npy', [volatility[len(volatility) // 4, :], volatility[len(volatility) // 2, :], volatility[3 * len(volatility) // 4, :]])

    #np.savez('data/stochastic_volatility/NUTS_samples.npz', s= s, sigma = sigma, nu= nu)

    if ground_truth:
        second_moment = np.empty(len(s[0]) + 2)
        var_second_moment = np.empty(len(second_moment))

        # estimate the second moments
        second_moment[:-2] = np.average(np.square(s), axis=0)
        second_moment[-2] = np.average(np.square(sigma))
        second_moment[-1] = np.average(np.square(nu))

        #estimate the variance of the second moments
        var_second_moment[:-2] = np.std(np.square(s), axis=0)**2
        var_second_moment[-2] = np.std(np.square(sigma))**2
        var_second_moment[-1] = np.std(np.square(nu))**2

        np.save('data/stochastic_volatility/ground_truth'+str(key_num)+'.npy', [second_moment, var_second_moment])


    else:
        X = np.empty((len(nu), len(s[0]) + 2))
        X[:, :-2] = s
        X[:, -2] = sigma
        X[:, -1] = nu

        variance_true = np.load('data/stochastic_volatility/ground_truth_moments.npy')

        B = bias(X, np.ones(len(X)), variance_true)

        ess, n_crossing = ess_cutoff_crossing(B, steps)
        print(ess, ess / (1 + ess* warmup_calls / 200.0))

        return ess, ess / (1 + ess* warmup_calls / 200.0)





def richard_results():
    import arviz as az
    folder = 'Tests/data/german_credit/'

    hmc_data = az.from_netcdf(folder + 'inference_data_german_credit_mcmc.nc')
    tau = np.array(hmc_data['posterior']['tau'])
    lam = np.array(hmc_data['posterior']['lam'])
    beta = np.array(hmc_data['posterior']['beta'])
    hmc_steps = np.array(hmc_data['sample_stats']['n_steps'])
    tunning = np.loadtxt(folder + 'german_credit_warmup_n_steps.txt')
    tunning_steps = np.sum(tunning, axis = 1)


    X = np.concatenate([[tau, ], np.swapaxes(lam.T, 1, 2), np.swapaxes(beta.T, 1, 2)])

    var = np.average(np.average(np.square(X), axis = 2), axis = 1)

    bias = np.sqrt(np.average(np.square(((np.cumsum(np.square(X), axis = 2) / np.arange(1, 10001)).T - var) / var), axis=2).T)

    ess = np.empty(10)
    ess_with_tunning = np.empty(10)
    for i in range(len(bias)):
        j = 0
        while bias[i, j] > 0.1:
            j += 1
        ess[i]= 200 / np.sum(hmc_steps[i, :j+1])
        ess_with_tunning[i] = 200 / (np.sum(hmc_steps[i, :j + 1]) + tunning_steps[i])

    print('ESS = {0}, ESS (with tunning) = {1}'.format(np.average(ess), np.average(ess_with_tunning)))


if __name__ == '__main__':

    richard_results()