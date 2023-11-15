import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
from optimization.adam import optimize_adam
from mclmc.sampler import Sampler
from benchmarks.brownian import Target

plt.rcParams.update({'font.size': 25})


key = jax.random.PRNGKey(42)

target = Target()
#target.transform = lambda x: x # the transform is the identity in the computations here
xtrue = jnp.log(jnp.array([0.1, 0.15]))
residual = lambda x: jnp.sqrt(jnp.sum(jnp.square(x[:, :2]-xtrue), axis = -1))

    


def adam(x0, steps, lr):
    l, x = optimize_adam(target.grad_nlogp, x0, steps, lr, trace= True)
    return x[:, :2], l


def mchmc(x0, steps, L, eps, key):
    
    sampler = Sampler(target, integrator='LF', L=L, eps=eps, frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, diagonal_preconditioning=False)
    x = sampler.sample(steps, 1, x0, random_key = key, output='normal')

    return x[:, :2], jax.vmap(target.nlogp)(x)




def convergence():
    
    chains = 3
    steps = 10000
    lr = 0.001
    L, eps = 4., 0.1
    n = np.arange(steps)+1
    keys = jax.random.split(key, 2 * chains)
    x0 = jax.vmap(target.prior_draw)(keys[chains:])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (15, 15))
    ac, mc = 'tab:red', 'tab:blue'
    
    for i in range(chains):

        x1, l1 = adam(x0[i], steps, lr)
        x2, l2 = mchmc(x0[i], steps, L, eps, keys[i])

        # -logp
        ax1.plot(n, l1, '-', color= ac)
        ax1.plot(n, l2, '-', color = mc)

        # distance (in hyperparameter space) from the truth
        ax2.plot(n, residual(x1), '-', color= ac)
        ax2.plot(n, residual(x2), '-', color = mc)

        # trajectory in the hyperparameter space
        ax3.plot(x1[:, 0], x1[:, 1], '.-', color= ac)
        ax3.plot(x1[[0, steps-1], 0], x1[[0, steps-1], 1], 'o', color= ac)
        ax3.plot(x2[:, 0], x2[:, 1], '.-', color= mc)
        ax3.plot(x2[[0, steps-1], 0], x2[[0, steps-1], 1], 'o', color= mc)  

    # for the legend
    ax1.plot([], [], 'o-', color = mc, label = 'MCLMC')
    ax1.plot([], [], 'o-', color= ac, label='ADAM')

    ax1.legend()
    ax1.set_xlabel('gradient calls')
    ax1.set_ylabel(r'$-\log p$')
    
    ax2.set_xlabel('gradient calls')
    ax2.set_ylabel(r'$\vert x - x_{\mathrm{true}} \vert$')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
          
    ax3.plot(xtrue[0], xtrue[1], '*', color='gold', markersize=20, label = 'true parameters')
    ax3.set_xlabel(r'$\log \sigma_{\mathrm{rw}}$')
    ax3.set_ylabel(r'$\log \sigma_{\mathrm{obs}}$')
    plt.tight_layout()
    plt.savefig('adam_comparisson_lr='+str(lr)+'.png')
    plt.close()


#convergence()


sampler = Sampler(target, integrator='LF', L=4, eps=0.01, frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, diagonal_preconditioning=False)
sampler.sample(1000000, 1, output='ess')
