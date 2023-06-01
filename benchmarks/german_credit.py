
#Sparse logistic regression fitted to the German credit data
#We use the version implemented in the inference-gym: https://pypi.org/project/inference-gym/
#In some part we directly use their tutorial: https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb

import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np
import os

from HMC.mchmc_to_numpyro import mchmc_target_to_numpyro
from NUTS import sample_nuts

dirr = os.path.dirname(os.path.realpath(__file__))

name = 'german_credit'
target_base = gym.targets.GermanCreditNumericSparseLogisticRegression()
prior_distribution = target_base.prior_distribution()

target = gym.targets.VectorModel(target_base, flatten_sample_transformations=True)

identity_fn = target.sample_transformations['identity']

def target_nlog_prob_fn(z):
    x = target.default_event_space_bijector(z)
    return -(target.unnormalized_log_prob(x) + target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims=1))

target_nlog_prob_grad_fn = jax.grad(target_nlog_prob_fn)



class Target():

    def __init__(self):
        self.d = 51
        self.name = name

        data = np.load(dirr+'/ground_truth/'+name+'/ground_truth.npy')
        self.second_moments, self.variance_second_moments = data[0], data[1]

        #xmap = np.load(dirr+'/ground_truth/'+name+'/map.npy')
        self.transform = lambda x: target.default_event_space_bijector(x)
        self.nlogp = lambda x: target_nlog_prob_fn(x)
        self.grad_nlogp = lambda x: (target_nlog_prob_fn(x), target_nlog_prob_grad_fn(x))


    # def prior_draw(self, key):
    #     x = prior_distribution.sample(seed= key)
    #     w = x['unscaled_weights']
    #     ls = x['local_scales']
    #     gs = x['global_scale']
    #     return jnp.concatenate((jnp.log(ls), jnp.ones(1) * jnp.log(gs), w))


    # def prior_draw(self, key):
    #     key1, key2 = jax.random.split(key)
    #     weights = jax.random.normal(key1, shape = (25, ))
    #     scales = jax.random.gamma(key2, a=  0.5, shape = (26, )) / 0.5
    #     return jnp.concatenate((jnp.log(scales), weights))
    
    #fix the global hierarchical parameter
    # def prior_draw(self, key):
    #     key1, key2 = jax.random.split(key)
    #     weights = jax.random.normal(key1, shape = (25, ))
    #     scales = jax.random.gamma(key2, a=  0.5, shape = (25, )) / 0.5
    #     return jnp.concatenate((jnp.log(scales), jnp.zeros(1), weights))
    
    #fix scale parameters
    def prior_draw(self, key):
        weights = jax.random.normal(key, shape = (25, ))
        return jnp.concatenate((jnp.zeros(26), weights))


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
        objective_fn= map_objective_fn, objective_grad_fn=map_objective_grad_fn, learning_rate=0.001, num_steps=2000, )

    import matplotlib.pyplot as plt
    plt.plot(objective_trace - objective_trace[-1], '.-')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()

    np.save('ground_truth/'+name+'/map.npy', z_map)



def ground_truth(key_num):
    key = jax.random.PRNGKey(key_num)
    mchmc_target = Target()
    numpyro_target = mchmc_target_to_numpyro(Target)
    samples, steps, steps_warmup = sample_nuts(numpyro_target, mchmc_target, None, 10000, 10000, 20, random_key=key, progress_bar= True)

    z = np.array(samples['x'])
    x = jax.vmap(mchmc_target.transform)(z)

    second_moments = jnp.average(jnp.square(x), axis = 0)
    variance_second_moments = jnp.std(jnp.square(x), axis = 0)**2

    np.save('ground_truth/'+name+'/ground_truth_'+str(key_num) +'.npy', [second_moments, variance_second_moments])



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


def joint_ground_truth():

    data = np.array([np.load('ground_truth/'+name+'/ground_truth_'+str(i)+'.npy') for i in range(3)])

    truth = np.median(data, axis = 0)
    np.save('ground_truth/'+name+'/ground_truth.npy', truth)

    for i in range(3):
        bias_d = np.square(data[i, 0] - truth[0]) / truth[1]
        print(np.sqrt(np.average(bias_d)), np.sqrt(np.max(bias_d)))



if __name__ == '__main__':

    ground_truth(2)

    #joint_ground_truth()
