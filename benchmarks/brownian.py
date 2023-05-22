import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt

from HMC.mchmc_to_numpyro import mchmc_target_to_numpyro
from benchmarks.benchmarks_mchmc import random_walk
from NUTS import sample_nuts


dirr = os.path.dirname(os.path.realpath(__file__))
name = 'brownian'


class Target():
    """
    log sigma_i ~ N(0, 2)
    log sigma_obs ~N(0, 2)

    x ~ RandomWalk(0, sigma_i)
    x_observed = (x + noise) * mask
    noise ~ N(0, sigma_obs)
    mask = 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
    """

    def __init__(self):
        self.num_data = 30
        self.d = self.num_data + 2
        self.name = name

        ground_truth_moments = np.load(dirr+'/ground_truth/'+name+'/ground_truth.npy')
        self.second_moments, self.variance_second_moments = ground_truth_moments[0], ground_truth_moments[1]

        self.data = jnp.array([0.21592641, 0.118771404, -0.07945447, 0.037677474, -0.27885845, -0.1484156, -0.3250906, -0.22957903, -0.44110894, -0.09830782,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               -0.8786016, -0.83736074, -0.7384849, -0.8939254, -0.7774566, -0.70238715, -0.87771565, -0.51853573, -0.6948214, -0.6202789])
        #sigma_obs = 0.15, sigma_i = 0.1


        self.observable = jnp.concatenate((jnp.ones(10), jnp.zeros(10), jnp.ones(10)))
        self.num_observable = jnp.sum(self.observable) # = 20
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        #y = softplus_to_log(x[:2])

        lik = 0.5 * jnp.exp(-2*x[1]) * jnp.sum(self.observable * jnp.square(x[2:] - self.data)) + x[1] * self.num_observable
        prior_x = 0.5 * jnp.exp(-2*x[0]) * (x[2]**2 + jnp.sum(jnp.square(x[3:] - x[2:-1]))) + x[0] * self.num_data
        prior_logsigma = 0.5 * jnp.sum(jnp.square(x / 2.0))

        return lik + prior_x + prior_logsigma

    def transform(self, x):
        return jnp.concatenate((jnp.exp(x[:2]), x[2:]))


    def prior_draw_optimize(self, key):
        """draws x from the prior"""

        return jax.scipy.optimize.minimize(self.nlogp, x0 = self.prior_draw(key), method = 'BFGS', options = {'maxiter': 500}).x

    def prior_draw(self, key):
        key_walk, key_sigma = jax.random.split(key)

        log_sigma = jax.random.normal(key_sigma, shape= (2, ))*2 #log sigma_i, log sigma_obs
        #log_sigma = jnp.log(np.array([0.1, 0.15])) + jax.random.normal(key_sigma, shape=(2,)) *0.1#*0.05# log sigma_i, log sigma_obs

        walk = random_walk(key_walk, self.d - 2) * jnp.exp(log_sigma[0])

        return jnp.concatenate((log_sigma, walk))


    def generate_data(self, key):

        key_walk, key_sigma, key_noise = jax.random.split(key, 3)

        log_sigma = jax.random.normal(key_sigma, shape=(2,)) * 2  # log sigma_i, log sigma_obs

        walk = random_walk(key_walk, self.d - 2) * jnp.exp(log_sigma[0])
        noise = jax.random.normal(key_noise, shape = (self.d - 2, )) * jnp.exp(log_sigma[1])

        return walk + noise


def ground_truth(key_num):
    key = jax.random.PRNGKey(key_num)
    mchmc_target = Target()
    numpyro_target = mchmc_target_to_numpyro(Target)
    samples, steps, steps_warmup = sample_nuts(numpyro_target, mchmc_target, None, 10000, 100000, 20, random_key=key, progress_bar= True)

    x = np.array(samples['x'])
    xsq = jnp.square(jax.vmap(mchmc_target.transform)(x))

    second_moments = jnp.average(xsq, axis = 0)
    variance_second_moments = jnp.std(xsq, axis = 0)**2

    np.save('ground_truth/'+name+'/ground_truth_'+str(key_num) +'.npy', [second_moments, variance_second_moments])
    np.save('ground_truth/'+name+'/chain_'+str(key_num) +'.npy', x)


def join_ground_truth():
    data = np.array([np.load('ground_truth/'+name+'/ground_truth_'+str(i)+'.npy') for i in range(3)])

    truth = np.median(data, axis = 0)
    np.save('ground_truth/'+name+'/ground_truth.npy', truth)

    for i in range(3):
        bias_d = np.square(data[i, 0] - truth[0]) / truth[1]
        print(np.sqrt(np.average(bias_d)), np.sqrt(np.max(bias_d)))


def plot_hierarchical():
    x= np.load('ground_truth/'+name+'/chain_1.npy')
    print(x.shape)
    sigi = np.exp(x[:, 0])
    sigo = np.exp(x[:, 1])
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 10))
    plt.hexbin(sigi, sigo, cmap = 'cividis')
    plt.plot([0.1, ], [0.15, ], '*', color = 'gold', markersize = 20)
    plt.xlim(0.04, 0.25)
    plt.ylim(0.04, 0.25)
    plt.title('Hyper parameters')
    plt.xlabel(r'$\sigma_{\mathrm{rw}}$')
    plt.ylabel(r'$\sigma_{\mathrm{obs}}$')
    plt.xticks([0.05, 0.1, 0.15, 0.2, 0.25])
    plt.yticks([0.05, 0.1, 0.15, 0.2, 0.25])
    plt.savefig('hierarchical_posterior.png')
    plt.show()


def plot_walk():
    x = np.sort(np.load('ground_truth/' + name + '/chain_1.npy')[:, 2:], axis = 0)
    n = len(x)
    xavg = x[n//2]
    xp, xm = x[3 * n // 4], x[n // 4]

    plt.plot(Target().data, 'o', color='tab:red', label = 'data')

    plt.plot(xavg, color = 'tab:blue', label = 'posterior')
    plt.fill_between(np.arange(len(xm)), xm, xp, color = 'tab:blue', alpha = 0.3)

    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.savefig('walk_posterior.png')
    plt.show()


def map():
    chains = 10
    from optimization.adam import optimize_adam
    from scipy.optimize import minimize
    t = Target()
    def store(x):
        X.append(x[0])
        Y.append(x[1])

    x0 = jax.vmap(t.prior_draw)(jax.random.split(jax.random.PRNGKey(0), chains))
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 10))

    for i in range(chains):
        X = []
        Y = []
        #opt = minimize(t.grad_nlogp, jac = True, x0 = x0[i], method = 'BFGS', callback = store, options = {'maxiter': 5000})
        #opt = minimize(t.grad_nlogp, jac = True, x0 = x0[i], method = 'L-BFGS-B', callback = store, options = {'maxiter': 1000, 'maxcor': 50})
        opt = minimize(t.grad_nlogp, jac = True, x0 = x0[i], method = 'Newton-CG', callback = store, options = {'maxiter': 1000})

        print(len(X))
        plt.plot(X, Y, '.-', color = 'black', alpha = 0.5)
        plt.plot(X[0], Y[0], 'o', color='tab:red')
        plt.plot(X[-1], Y[-1], 'o', color='tab:blue')


    plt.plot(jnp.log(jnp.array([0.1, ])), jnp.log(jnp.array([0.15, ])), '*', color='gold', markersize=20)
    plt.xlabel(r'$\log \sigma_{\mathrm{rw}}$')
    plt.ylabel(r'$\log \sigma_{\mathrm{obs}}$')
    plt.show()


def adam_comparisson():
    from optimization.adam import optimize_adam
    chains = 5
    t = Target()

    x0 = jax.vmap(t.prior_draw)(jax.random.split(jax.random.PRNGKey(0), chains))
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 10))

    #MCHMC
    data = np.load('brownian_samples.npy')
    xmchmc = data[0]

    #ADAM
    for i in range(chains):
        l, x = optimize_adam(t.grad_nlogp, x0[i], 100000, lr = 0.05, trace= True)
        X, Y = x[:, 0], x[:, 1]

        plt.plot(X, Y, '-', color='tab:red', alpha=0.5)
        plt.plot(X[0], Y[0], 'o', color='tab:red')
        #plt.plot(X[-1], Y[-1], 'o', color='tab:blue')


        plt.plot(xmchmc[i, :,  0], xmchmc[i, :, 1], '-', alpha = 0.5, color = 'tab:blue')
        plt.plot(xmchmc[i, 0, 0], xmchmc[i, 0, 1], 'o', color='tab:blue')

    plt.plot([], [], 'o-', color = 'tab:blue', label = 'MCLMC')
    plt.plot([], [], 'o-', color='tab:red', label='ADAM')

    plt.plot(jnp.log(jnp.array([0.1, ])), jnp.log(jnp.array([0.15, ])), '*', color='gold', markersize=20, label = 'true parameters')
    plt.legend()
    plt.xlabel(r'$\log \sigma_{\mathrm{rw}}$')
    plt.ylabel(r'$\log \sigma_{\mathrm{obs}}$')
    plt.savefig('adam_comparisson.png')
    plt.show()



def adam_comparisson2():
    from optimization.adam import optimize_adam
    chains = 5
    t = Target()

    x0 = jax.vmap(t.prior_draw)(jax.random.split(jax.random.PRNGKey(0), chains))
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 10))

    #MCHMC
    data = np.load('brownian_samples.npy')
    xmchmc = data[0]

    #ADAM
    shift = 1e4
    for i in range(chains):
        l, x = optimize_adam(t.grad_nlogp, x0[i], 100000, lr = 0.05, trace= True)

        plt.plot(l + shift, '-', color='tab:red')


        plt.plot(jax.vmap(t.nlogp)(xmchmc[i]) + shift, '-', color = 'tab:blue')

    plt.plot([], [], 'o-', color = 'tab:blue', label = 'MCLMC')
    plt.plot([], [], 'o-', color='tab:red', label='ADAM')

    plt.legend()
    plt.xlabel('gradient calls')
    plt.ylabel(r'$-\log p$')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('adam_comparisson2.png')
    plt.show()




def mchmc():
    from sampling.sampler import Sampler

    target = Target()
    target.transform = lambda x: x
    num_chains = 32
    key, key_prior = jax.random.split(jax.random.PRNGKey(0))
    keys = jax.random.split(key_prior, num_chains)

    sampler = Sampler(target, integrator='LF', L=0.8, eps=0.1, frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0,
                      diagonal_preconditioning=False)

    # initial condition
    x0 = jax.vmap(target.prior_draw)(keys)

    x = sampler.sample(10000, num_chains, x0, random_key = key, output='normal')

    print(x.shape)
    g = jax.vmap(jax.vmap(lambda z: target.grad_nlogp(z)[1]))(x)
    print(g.shape)
    np.save('brownian_samples.npy', np.array([x, g]))


    #plt.plot(x[:, -1, 0], x[:, -1, 1], '.')
    # for m in range(chains):
    #     plt.plot(x[m, :, 0], x[m, :, 1], '-')
        # for k in range(1):
        #     plt.plot(x[m, 500 * k, 0], x[m, 500 * k, 1], '.', color='tab:red')



if __name__ == '__main__':
    adam_comparisson2()
    #plott()
    #mchmc()
    #ground_truth(2)
    #plot_hierarchical()
    #join_ground_truth()