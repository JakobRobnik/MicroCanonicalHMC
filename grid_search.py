import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp



def search_wrapper(ess_function, amin, amax, epsmin, epsmax):

    #A = jnp.array([1e20, ]) #without bounces
    A = jnp.logspace(np.log10(amin), np.log10(amax), 6)
    epsilon = jnp.logspace(np.log10(epsmin), np.log10(epsmax), 6)

    results1 = search_step(ess_function, A, epsilon)

    plt.figure(figsize= (15, 10))
    plt.subplot(1, 2, 1)
    ess, i, j = visualize(results1, A, epsilon, show = True)

    if (i == 0) or (i == 5) or (j == 0) or (j == 5):
        plt.show()

    else:
        A = jnp.logspace(np.log10(A[i-1]), np.log10(A[i+1]), 6)
        epsilon = jnp.logspace(np.log10(epsilon[j-1]), np.log10(epsilon[j+1]), 6)
        results2 = search_step(ess_function, A, epsilon)

        plt.subplot(1, 2, 2)
        ess, i, j = visualize(results2, A, epsilon, show=True)

        plt.show()

    return ess, A[i], epsilon[j]


def search_step(ess_function, A, epsilon):
    return jax.vmap(lambda a: jax.pmap(lambda e: ess_function(a, e))(epsilon))(A)


def visualize(ess_arr, A, epsilon, show):

    I = np.argmax(ess_arr)
    eps_best = epsilon[I % (len(epsilon))]
    alpha_best = A[I // len(epsilon)]
    ess_best = np.max(ess_arr)
    print(ess_best)

    if show:
        ax = plt.gca()
        cax = ax.matshow(ess_arr)
        plt.colorbar(cax)
        plt.title(r'ESS = {0} ($\alpha$ = {1}, $\epsilon$ = {2})'.format(np.round(ess_best, 4), *np.round([alpha_best, eps_best], 2)))


        ax.set_xticklabels([''] + [str(e)[:4] for e in epsilon])
        ax.set_yticklabels([''] + [str(a)[:4] for a in A])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(r'$\epsilon$')
        ax.set_ylabel(r'$\alpha$')
        ax.invert_yaxis()

    return ess_best, I // len(epsilon), I % (len(epsilon))
