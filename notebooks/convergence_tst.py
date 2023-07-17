import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from sampling.sampler import find_crossing


num_chains = (128, 32)

crossing = jax.vmap(find_crossing, (0, None))


def splitR(x):
    """See: https://arxiv.org/pdf/2110.13017.pdf, Definition 8."""

    X = x.reshape(num_chains[0], num_chains[1], x.shape[1])
    avg = jnp.average(X, axis=1)  # average for each superchain
    avg2 = jnp.average(jnp.square(X), axis=1)
    var = avg2 - jnp.square(avg)  # variance for each superchain

    # identify outlier superchains
    dev = jnp.max(jnp.abs(avg - jnp.median(avg)), axis=1)
    perm = jnp.argsort(dev)

    Avg = avg[perm][:len(avg) * 7 // 10]
    Var = var[perm][:len(avg) * 7 // 10]
    avg_all = jnp.average(Avg, axis=0)  # average over all chains
    var_between = jnp.sum(jnp.square(Avg - avg_all), axis=0) / (len(Avg) - 1)

    var_within = jnp.average(Var, axis=0)

    R = jnp.sqrt(1. + var_between / var_within)

    return jnp.max(R)  # , perm

def avg_deviation(x):

    X = x.reshape(num_chains[0], num_chains[1], x.shape[1])
    avg = jnp.average(X, axis=1)  # average for each superchain
    second_moment = jnp.average(jnp.square(X), axis=1)
    var = second_moment - jnp.square(avg)
    dev = jnp.sqrt(jnp.average(jnp.square(avg - jnp.median(avg)) / var, axis=1))
    return dev



def virial_loss(x, g, key):
    """loss^2 = Tr[(1 - V)^T (1 - V)] / d
        where Vij = <xi gj> is the matrix of virials.
        Loss is computed with the Hutchinson's trick."""

    z = jax.random.rademacher(key, (hutchinson_repeat, x.shape[-1])) # <z_i z_j> = delta_ij
    X = z - (g @ z.T).T @ x / x.shape[0]
    return jnp.sqrt(jnp.average(jnp.square(X)))


def split_virial_loss(x, g, key):
    reshaped = (num_chains[0], num_chains[1], x.shape[1])
    X = x.reshape(reshaped)
    G = g.reshape(reshaped)
    virials = jax.vmap(virial_loss)(X, G, jax.random.split(key, num_chains[0]))

    return virials



x, g = np.load('benchmarks/brownian_samples.npy')
hutchinson_repeat = 100


x = jax.random.normal(jax.random.PRNGKey(0), shape= (100, 100))
print(virial_loss(x, x, jax.random.PRNGKey(3)))

exit()

cutoff = 30
cutoff_dev = 5

keys = jax.random.split(jax.random.PRNGKey(42), x.shape[1])
virials = jax.vmap(split_virial_loss)(jnp.swapaxes(x, 0, 1), jnp.swapaxes(g, 0, 1), keys).T

n = crossing(virials, cutoff)*10


dev = jax.vmap(avg_deviation)(jnp.swapaxes(x, 0, 1)).T
ndev= crossing(dev, cutoff_dev)*10


plt.plot(ndev, n, '.')
plt.plot([0, 0], [6000, 6000], '-', alpha = 0.5, color = 'black')
plt.xlabel("steps for group average convergence")
plt.ylabel("steps for virial convergence")
plt.savefig('criteria_correlation')
plt.show()


plt.plot([0, 6000], np.ones(2) * cutoff, '-', alpha = 0.5, color = 'black')
for i in range(num_chains[0]//2):
    plt.plot(np.arange(0, 6000, 10), virials[i])

plt.yscale('log')
plt.savefig('virial.png')
plt.show()


plt.plot([0, 6000], np.ones(2) * cutoff_dev, '-', alpha = 0.5, color = 'black')
for i in range(num_chains[0]//2):
    plt.plot(np.arange(0, 6000, 10), dev[i])

plt.yscale('log')
plt.savefig('dev.png')
plt.show()
