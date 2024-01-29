import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import blackjax


def anneal_target(logp, shift, mass):

    def logp2(x):
        return jnp.logaddexp(logp(x), jnp.log(mass) - 0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(jnp.square(x-shift)))

    return logp2


def evidence(x, final_shift, mass):

    M, d = x.shape

    # count the number of particles in the standard Gaussian mode

    def is_in_mode2(y):
        dist = jnp.sum(jnp.square(y - final_shift))
        prob = jax.scipy.stats.chi2.sf(dist, d)
        return prob > 1 / M

    mode2 = jax.vmap(is_in_mode2)(x)
    num2 = jnp.sum(mode2)
    print(M- num2, num2)
    print(mode2)

    return mass * (M - num2) / num2


def shift_annealing(logp, sampler, init, mass, rng_key):
    """Estimates the evidence: int e^logp(x) dx by shift annealing.
        Args:
            logp: a function with mode close to x = 0 and preconditioned to be as close to the standard Gaussian as possible (doesn't actually need to be a Gaussian)
            sampler: a function with signature: (logp, initial condition for n particles, random key) -> final state of n particles
            initial condition: location of n particles. 2n particles will be run. shape = (n, d)
            mass: float. The annealing target is q(x | mu) = e^logp(x) + mass * N(mu, 1). mass should be the estimate of the evidence that we have before running this algorithm
        Returns:
            evidence: float
    """
    d = init.shape[1]
    key_a, key_init, key_sampler = jax.random.split(rng_key, 3)
    
    ### random unit vector for a:
    a = jax.random.normal(key_a, shape= (d,))
    a /= jnp.sqrt(jnp.sum(jnp.square(a)))
    
    ### annealing schedule
    schedule = jnp.sqrt(d) * jnp.linspace(0., 1.5, 20)

    ### initial condition (half of particles from the given distribution, the other half from the added Gaussian)
    y = jax.random.normal(key_init, shape= init.shape) + schedule[0] * a
    x = jnp.concatenate((init, y))

    ### run the algorithm
    key = jax.random.split(key_sampler, len(schedule))
    folder = 'data/shift_annealing/stn/'
    jnp.save(folder + '0.npy', x)
    for i in range(len(schedule)):
        print(str(i) + '/' + str(len(schedule)))
        target = anneal_target(logp, schedule[i] * a, mass)
        x = sampler(target, x, key[i])
        jnp.save(folder + str(i+1) + '.npy', x)
    
    ev= evidence(x, schedule[-1] * a, mass)
    print(ev)
    
    return ev
    


def prepare_mclmc(num_steps, stepsize, L):

    def mclmc(logp, initial_position, rng_key):

        alg = blackjax.mclmc(logp, L, stepsize)

        run = lambda key, x: blackjax.util.run_inference_algorithm(rng_key= key, initial_state_or_position= x, inference_algorithm=alg, num_steps=num_steps, progress_bar= False, transform = lambda state: None)[0].position

        return jax.vmap(run)(jax.random.split(rng_key, initial_position.shape[0]), initial_position)
    
    return mclmc


def IllConditionedGaussian(d, condition_number):
    """Gaussian distribution. Covariance matrix has eigenvalues equally spaced in log-space, going from 1/condition_bnumber^1/2 to condition_number^1/2."""

    eigs = jnp.logspace(-0.5 * jnp.log10(condition_number), 0.5 * jnp.log10(condition_number), d) # eigenvalues of the covariance matrix
    eigs /= jnp.average(eigs)
    Hessian = 1. / eigs

    N = - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * eigs))
    logp = lambda x: -0.5 * jnp.dot(Hessian, jnp.square(x)) + N

    posterior_draw = lambda key: (jax.random.normal(key, shape=(d,)) * jnp.sqrt(eigs))

    return logp, posterior_draw


d, condition_number = 100, 1.
logp, posterior_draw = IllConditionedGaussian(d, condition_number)


key = jax.random.PRNGKey(42)
num_particles = 500

key_prior, key_sampling = jax.random.split(key)
init = jax.vmap(posterior_draw)(jax.random.split(key_prior, num_particles//2))


mclmc = prepare_mclmc(100000, 2., jnp.sqrt(d))

m = shift_annealing(logp, mclmc, init, 0.5, key_sampling)
