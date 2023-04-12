import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np
import os

from HMC.mchmc_to_numpyro import mchmc_target_to_numpyro
from benchmarks.benchmarks_mchmc import random_walk
from NUTS import sample_nuts


dirr = os.path.dirname(os.path.realpath(__file__))

#to get ground truth, run the following from the source directory of inference-gym:
#python -m inference_gym.tools.get_ground_truth --target= GermanCreditNumericSparseLogisticRegression

target_base = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations()
name = 'brownian'

target = gym.targets.VectorModel(target_base, flatten_sample_transformations=True)

prior_distribution = target_base.prior_distribution()

identity_fn = target.sample_transformations['identity']

def target_nlog_prob_fn(z):
    x = target.default_event_space_bijector(z)
    return -(target.unnormalized_log_prob(x) + target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims=1))




class Target():

    def __init__(self):
        self.d = 32
        self.name = name

        ground_truth_moments = np.load(dirr+'/ground_truth/'+name+'/ground_truth.npy')
        self.second_moments, self.variance_second_moments = ground_truth_moments[0], ground_truth_moments[1]

        key_data = jax.random.PRNGKey(500)
        self.data = self.prior_draw(key_data)[2:]
        self.observable = jnp.concatenate((jnp.ones(10), jnp.zeros(10), jnp.ones(10)))

        #xmap = np.load(dirr+'/ground_truth/'+name+'/map.npy')
        self.transform = lambda x: target.default_event_space_bijector(x)
        self.nlogp = lambda x: target_nlog_prob_fn(x)
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        sigma = jnp.log(1.0 + jnp.exp(x[:2]))
        lik = 0.5 * jnp.sum(self.observable * jnp.square(x[2:] - self.data) / sigma[1]**2)
        prior_x = 0.5 * jnp.sum(self.observable * jnp.square(x[2:]) / sigma[0]**2)
        prior_sigma = 

        return lik + prior_x + prior_sigma


    # def inv_softplus(self, x):
    #     y = jnp.log(jnp.exp(x)-1.0)
    #     mask = x > 50
    #     return mask * x + (1-mask) * jnp.nan_to_num(y)
    #

    def prior_draw(self, key):
        """draws x from the prior"""

        key_walk, key_sigma = jax.random.split(key)

        sigma = jnp.exp(jax.random.normal(key_sigma, shape= (2, ))*2)

        walk = random_walk(key_walk, self.d - 2) * sigma[0]

        return jnp.concatenate((self.inv_softplus(sigma), walk))


    # def prior_draw(self, key):
    #     x = prior_distribution.sample(seed=key)
    #     scale= jnp.array([x.innovation_noise_scale, x.observation_noise_scale])
    #
    #     x1 = jnp.array([x.x_0, x.x_1, x.x_2, x.x_3, x.x_4, x.x_5, x.x_6, x.x_7, x.x_8, x.x_9,
    #                     x.x_10, x.x_11, x.x_12, x.x_13, x.x_14, x.x_15, x.x_16, x.x_17, x.x_18, x.x_19,
    #                     x.x_20, x.x_21, x.x_22, x.x_23, x.x_24, x.x_25, x.x_26, x.x_27, x.x_28, x.x_29])
    #     return
    #


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
        objective_fn=map_objective_fn, objective_grad_fn=map_objective_grad_fn, learning_rate=0.00005, num_steps=4000, )


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

    xsq = jnp.square(jax.vmap(mchmc_target.transform)(np.array(samples['x'])))

    second_moments = jnp.average(xsq, axis = 0)
    variance_second_moments = jnp.std(xsq, axis = 0)**2

    np.save('ground_truth/'+name+'/ground_truth_'+str(key_num) +'.npy', [second_moments, variance_second_moments])


def join_ground_truth():
    data = np.array([np.load('ground_truth/'+name+'/ground_truth_'+str(i)+'.npy') for i in range(3)])

    truth = np.median(data, axis = 0)
    np.save('ground_truth/'+name+'/ground_truth.npy', truth)

    for i in range(3):
        bias_d = np.square(data[i, 0] - truth[0]) / truth[1]
        print(np.sqrt(np.average(bias_d)), np.sqrt(np.max(bias_d)))


if __name__ == '__main__':
    key = jax.random.PRNGKey(8)
    t = Target()
    x = jax.vmap(t.prior_draw)(jax.random.split(key, 300))
    moments = jnp.average(jnp.square(t.transform(x)), axis=0)
    print(moments)
    #ground_truth(2)
    #join_ground_truth()