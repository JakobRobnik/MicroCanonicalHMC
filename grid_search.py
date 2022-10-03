import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp



def search_wrapper(ess_function, Lmin, Lmax, epsmin, epsmax):

    L = jnp.logspace(np.log10(Lmin), np.log10(Lmax), 6)
    epsilon = jnp.logspace(np.log10(epsmin), np.log10(epsmax), 6)

    results1 = search_step(ess_function, L, epsilon)

    plt.figure(figsize= (15, 10))
    plt.subplot(1, 2, 1)
    ess, i, j = visualize(results1, L, epsilon, show = True)

    if (i == 0) or (i == 5) or (j == 0) or (j == 5):
        plt.show()

    else:
        L = jnp.logspace(np.log10(L[i-1]), np.log10(L[i+1]), 6)
        epsilon = jnp.logspace(np.log10(epsilon[i-1]), np.log10(epsilon[i+1]), 6)
        results2 = search_step(ess_function, L, epsilon)

        plt.subplot(1, 2, 2)
        ess, i, j = visualize(results2, L, epsilon, show=True)

        plt.show()

    return ess, L[i], epsilon[j]


def search_step(ess_function, L, epsilon):
    return jax.vmap(lambda l: jax.pmap(lambda e: ess_function(l, e))(epsilon))(L)


def visualize(ess_arr, L, epsilon, show):

    I = np.argmax(ess_arr)
    eps_best = epsilon[I % (len(epsilon))]
    L_best = L[I // len(epsilon)]
    ess_best = np.max(ess_arr)
    print(ess_best)

    if show:
        ax = plt.gca()
        cax = ax.matshow(ess_arr)
        plt.colorbar(cax)
        plt.title(r'ESS = {0} (with optimal L = {1}, $\epsilon$ = {2})'.format(ess_best, *np.round([L_best, eps_best], 2)))

        e_rounded = np.round(epsilon, 2)
        l_rounded = np.round(L, 2)
        ax.set_xticklabels([''] + [str(e) for e in e_rounded])
        ax.set_yticklabels([''] + [str(l) for l in l_rounded])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("L")
        ax.invert_yaxis()

    return ess_best, I // len(epsilon), I % (len(epsilon))
