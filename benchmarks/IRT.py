import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np
import os

#from NUTS import sample_nuts

dirr = os.path.dirname(os.path.realpath(__file__))

target_base = gym.targets.SyntheticItemResponseTheory()
name= 'IRT'

target = gym.targets.VectorModel(target_base, flatten_sample_transformations=True)
prior_distribution = target_base.prior_distribution()

identity_fn = target.sample_transformations['identity']


def target_nlog_prob_fn(z):
    x = target.default_event_space_bijector(z)
    return -(target.unnormalized_log_prob(x) + target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims=1))

target_nlog_prob_grad_fn = jax.grad(target_nlog_prob_fn)



class Target():

    def __init__(self):
        self.d = 501
        self.name= name

        data = np.load(dirr+'/ground_truth/'+name+'/ground_truth.npy')
        self.second_moments, self.variance_second_moments = data[0], data[1]

        #xmap = np.load(dirr+'/ground_truth/'+name+'/map.npy')
        self.transform = lambda x: target.default_event_space_bijector(x)
        self.nlogp = lambda x: target_nlog_prob_fn(x)
        self.grad_nlogp = lambda x: (target_nlog_prob_fn(x), target_nlog_prob_grad_fn(x))

    # def prior_draw(self, key):
    #     return jnp.zeros(self.d)

    def prior_draw(self, key):

        x = prior_distribution.sample(seed=key)
        question = x['question_difficulty']
        meanstudent = x['mean_student_ability']
        student = x['centered_student_ability']

        return jnp.concatenate((question, meanstudent * jnp.ones(1), student))



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
        objective_fn=map_objective_fn, objective_grad_fn=map_objective_grad_fn, learning_rate=0.0002, num_steps=1000, )

    import matplotlib.pyplot as plt
    plt.plot(objective_trace - objective_trace[-1], '.-')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()

    np.save('ground_truth/'+name+'/map.npy', z_map)



def ground_truth(key_num):
    
    from HMC.mchmc_to_numpyro import mchmc_target_to_numpyro

    key = jax.random.PRNGKey(key_num)
    mchmc_target = Target()
    numpyro_taget = mchmc_target_to_numpyro(Target)

    samples, steps, steps_warmup = sample_nuts(numpyro_taget, mchmc_target, None, 10000, 10000, 20, random_key=key, progress_bar= True)

    z = np.array(samples['x'])
    x = jax.vmap(mchmc_target.transform)(z)

    second_moments = jnp.average(jnp.square(x), axis = 0)
    variance_second_moments = jnp.std(jnp.square(x), axis = 0)**2

    np.save('ground_truth/'+name+'/ground_truth_'+str(key_num) +'.npy', [second_moments, variance_second_moments])



def joint_ground_truth():

    data = np.array([np.load('ground_truth/'+name+'/ground_truth_'+str(i)+'.npy') for i in range(3)])

    truth = np.median(data, axis = 0)
    np.save('ground_truth/'+name+'/ground_truth.npy', truth)

    for i in range(3):
        bias_d = np.square(data[i, 0] - truth[0]) / truth[1]
        print(np.sqrt(np.average(bias_d)), np.sqrt(np.max(bias_d)))


if __name__ == '__main__':

    kkey = jax.random.PRNGKey(0)
    key = jax.random.split(kkey, 100)
    t = Target()

    x = jax.vmap(t.prior_draw)(key)
    g = jax.vmap(lambda x: t.grad_nlogp(x)[1])(x)

    print(jnp.average(x * g, axis=0))

    #Target().prior_draw(jax.random.PRNGKey(0))
