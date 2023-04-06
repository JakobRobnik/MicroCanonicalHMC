
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

from HMC.mchmc_to_numpyro import mchmc_target_to_numpyro
from NUTS import sample_nuts


#to get ground truth, run the following from the source directory of inference-gym:
#python -m inference_gym.tools.get_ground_truth --target= GermanCreditNumericSparseLogisticRegression

target = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations()
name = 'brownian'

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
        self.d = 32
        self.name = name

        data = np.load('data/'+name+'/ground_truth.npy')
        self.second_moments, self.variance_second_moments = data[0], data[1]

        xmap = map_solution()
        self.transform = lambda x: target.default_event_space_bijector(x + xmap)
        self.nlogp = lambda x: target_nlog_prob_fn(x + xmap)
        self.grad_nlogp = lambda x: (target_nlog_prob_fn(x + xmap), target_nlog_prob_grad_fn(x + xmap))

    # def prior_draw(self, key):
    #     return jnp.zeros(self.d)

    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64') * 0.05



def ground_truth(key_num):
    key = jax.random.PRNGKey(key_num)
    mchmc_target = Target()
    numpyro_target = mchmc_target_to_numpyro(Target)
    samples, steps, steps_warmup = sample_nuts(numpyro_target, mchmc_target, None, 10000, 10000, 20, random_key=key, progress_bar= True)

    z = np.array(samples['x'])
    x = jax.vmap(mchmc_target.transform)(z)

    second_moments = jnp.average(jnp.square(x), axis = 0)
    variance_second_moments = jnp.std(jnp.square(x), axis = 0)**2

    np.save('../data/'+name+'/ground_truth_'+str(key_num) +'.npy', [second_moments, variance_second_moments])




if __name__ == '__main__':

    #ground_truth(2)

    data = np.array([np.load('../data/'+name+'/ground_truth_'+str(i)+'.npy') for i in range(3)])

    truth = np.median(data, axis = 0)
    np.save('../data/'+name+'/ground_truth.npy', truth)

    for i in range(3):
        bias_d = np.square(data[i, 0] - truth[0]) / truth[1]
        print(np.sqrt(np.average(bias_d)), np.sqrt(np.max(bias_d)))
