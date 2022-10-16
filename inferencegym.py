#Examples from the inference gym: https://pypi.org/project/inference-gym/
#In some part we directly use their tutorial: https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb


import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import ESH
import CTV

#to get ground truth, run the following from the source directory of inference-gym:


#python -m inference_gym.tools.get_ground_truth --target= GermanCreditNumericSparseLogisticRegression



class Target():

    def __init__(self, name):

        if name == 'German Credit':
            self.d = 51
            target = gym.targets.GermanCreditNumericSparseLogisticRegression()
            target = gym.targets.VectorModel(target, flatten_sample_transformations=True)


        elif name == 'Stohastic Volatility': #for some reason works very slow, so I implemented it myself in targets.py
            self.d = 2519
            target = gym.targets.VectorizedStochasticVolatilitySP500()

        else:
            print('The target name does not match any of the available target names.')
            exit()



        self.target = target

        identity_fn = target.sample_transformations['identity']


        def target_nlog_prob_fn(x):
            y = target.default_event_space_bijector(x)
            return -(target.unnormalized_log_prob(y) + target.default_event_space_bijector.forward_log_det_jacobian(y, event_ndims=1))

        self.nlogp = target_nlog_prob_fn
        self.grad_nlogp = jax.grad(target_nlog_prob_fn)

        self.variance = jnp.square(identity_fn.ground_truth_standard_deviation) + jnp.square(identity_fn.ground_truth_mean) #in the transformed coordinates

        self.transform = target.default_event_space_bijector



    def prior_draw(self, key):
        return jax.random.normal(key, shape = (self.d, ), dtype = 'float64')


    def map_solution(self):

        map_objective_fn = lambda x: -self.target.unnormalized_log_prob(self.target.default_event_space_bijector(x))

        map_objective_grad_fn = jax.grad(map_objective_fn)
        learning_rate = 0.001

        return jax.lax.scan(lambda x, _: ( x - learning_rate * map_objective_grad_fn(x), None),
                            init=jnp.zeros(self.target.default_event_space_bijector.inverse_event_shape(self.target.event_shape)),
                            xs=None, length=200)[0]




def posterior():
    eps = 0.1
    esh = ESH.Sampler(Target=Target(), eps=eps)
    L = 1.5 * np.sqrt(esh.Target.d)

    key = jax.random.PRNGKey(0)
    x0 = map_solution()
    X, W = esh.sample(x0, 1000000, L, key, prerun=0)

    np.savez('Tests/data/german_credit/mchmc.npz', x= X[:, [0, 1, 26]], w= W)


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
    #ess()

    from numpyro.examples.datasets import SP500, load_dataset
    _, fetch = load_dataset(SP500, shuffle=False)
    dates, returns = fetch()

    print(p)

    # names= ['German Credit', 'Stohastic Volatility']
    # target = Target(names[1])
    #
    # import time
    # t0 = time.time()
    # for i in range(10):
    #     target.nlogp(jnp.zeros(target.d))
    #     print(time.time()-t0)
    #     t0 = time.time()

    #richard_results()