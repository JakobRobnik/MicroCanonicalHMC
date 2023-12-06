from jax.config import config
config.update("jax_enable_x64", True)
import math
import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.mclmc import init, build_kernel, noneuclidean_mclachlan
import mclmc
import jax.numpy as jnp
import jax

from mclmc.sampler import Sampler, Target
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState, mclmc_find_L_and_step_size
import mclmc.sampler
print(Sampler.sample)

def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key, zero_keys=False
):
    
    state = sampling_algorithm.init(initial_val)
    return run_kernel(sampling_algorithm.step, state, zero_keys=zero_keys, rng_key=rng_key, num_steps=num_steps)

def run_kernel(kernel, state, zero_keys, rng_key, num_steps):
    keys = jax.random.split(rng_key, num_steps)
    if zero_keys:
        keys = jnp.array([jax.random.PRNGKey(SEED)]*num_steps)
    _, info = jax.lax.scan(lambda s, k: (kernel(k, s)), state, keys)
    return info

SEED = 0
# USE ipython, for some reason version of something is different in ipython
# for this to pass, the MCLMC repo's prngkeys must all be set to SEED
# the seeds in tuning also need to be set to SEED in blackjax
def test_mclmc_sampler():
    # Set up your test inputs
    num_steps = 1000
    num_chains = 1
    dim = 2
    key = jax.random.PRNGKey(SEED)


    initial_position = jnp.array([1., 1.,])
    logdensity_fn = lambda x: -0.5 * jnp.sum(jnp.square(x))

    mclmc = blackjax.mcmc.mclmc.mclmc(
        logdensity_fn=logdensity_fn,
        transform=lambda x: x,
        L=math.sqrt(dim), step_size=math.sqrt(dim) * 0.4,
        seed=SEED
    )

    blackjax_mclmc_result = run_sampling_algorithm(
        sampling_algorithm=mclmc,
        num_steps=num_steps,
        initial_val=initial_position,
        rng_key=key,
        zero_keys=True,
    )

    blackjax_mclmc_samples = blackjax_mclmc_result.transformed_position
    # print(blackjax_mclmc_samples)
    # raise Exception

    target_simple = Target(d = dim, nlogp=lambda x : -logdensity_fn(x))
    native_mclmc_samples = Sampler(Target=target_simple,L=math.sqrt(dim), eps=math.sqrt(dim) * 0.4, frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0).sample(num_steps, x_initial = initial_position, random_key=key)
    
    # print(blackjax_mclmc_samples.shape)
    # Assert that the number of samples is correct
    assert blackjax_mclmc_samples.shape == (num_steps, dim)
    assert native_mclmc_samples.shape == (num_steps, dim)

    # Assert that the samples are equal
    print(blackjax_mclmc_samples, native_mclmc_samples)
    assert jnp.allclose(blackjax_mclmc_samples, native_mclmc_samples)


    # now with tuning
    print("\n\n\nTUNING\n\n\n")

    native_mclmc_sampler = Sampler(Target=target_simple)
    native_mclmc_samples = native_mclmc_sampler.sample(num_steps, x_initial = initial_position, random_key=key) 
    
    print("\nNATIVE MCLMC PARAMS")
    print(native_mclmc_sampler.hyp)
    
    kernel = build_kernel(logdensity_fn=logdensity_fn, integrator=noneuclidean_mclachlan, transform=lambda x: x)

    # run mclmc with tuning and get result
    initial_state = init(x_initial=initial_position, logdensity_fn=logdensity_fn, rng_key=key)
    blackjax_state_after_tuning, blackjax_mclmc_sampler_params = mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=key,

    )
    print("\nBLACKJAX MCLMC PARAMS")
    print(blackjax_mclmc_sampler_params)

    assert jnp.allclose(blackjax_mclmc_sampler_params.L,native_mclmc_sampler.hyp.L) and jnp.allclose(blackjax_mclmc_sampler_params.step_size,native_mclmc_sampler.hyp.eps)


    blackjax_mclmc_result = run_kernel(lambda key, state: kernel(L=blackjax_mclmc_sampler_params.L, step_size=blackjax_mclmc_sampler_params.step_size, rng_key=key, state=state), state=blackjax_state_after_tuning, zero_keys=True, num_steps=num_steps, rng_key=jax.random.PRNGKey(SEED))


    print("shapes", native_mclmc_samples.shape, blackjax_mclmc_result.transformed_position.shape)
    print("native mclmc post tuning samples", native_mclmc_samples[-1:])
    print("blackjax mclmc post tuning samples", blackjax_mclmc_result.transformed_position[-1:])

    assert jnp.allclose(native_mclmc_samples, blackjax_mclmc_result.transformed_position)







def run_mclmc(logdensity_fn,num_steps, initial_position, key):

    init_key, part1_key, part2_key, run_key = jax.random.split(key, 4)
    
    initial_state = init(x_initial=initial_position, logdensity_fn=logdensity_fn, rng_key=key)

    blackjax_state_after_tuning, blackjax_mclmc_sampler_params = mclmc_find_L_and_step_size(
        kernel=build_kernel(logdensity_fn=logdensity_fn, integrator=noneuclidean_mclachlan, transform=lambda x: x),
        num_steps=num_steps,
        state=initial_state,
        part1_key=key,
        part2_key=key,
    )

    keys = jax.random.split(key, num_steps)

    kernel = build_kernel(logdensity_fn=logdensity_fn, integrator=noneuclidean_mclachlan, transform=lambda x: x)
    
    _, blackjax_mclmc_result = jax.lax.scan(
        f=lambda state, key: kernel(L=blackjax_mclmc_sampler_params.L, step_size=blackjax_mclmc_sampler_params.step_size, rng_key=key, state=state), 
        xs=keys, 
        init=blackjax_state_after_tuning)

    # (lambda key, state: kernel(L=blackjax_mclmc_sampler_params.L, step_size=blackjax_mclmc_sampler_params.step_size, rng_key=key, state=state), state=blackjax_state_after_tuning, zero_keys=True, num_steps=num_steps, rng_key=jax.random.PRNGKey(SEED))

    return blackjax_mclmc_result.transformed_position

# out = run_mclmc(logdensity_fn=lambda x: -0.5 * jnp.sum(jnp.square(x)), num_steps=1000, initial_position=jnp.array([1., 1.]), key=jax.random.PRNGKey(0))
# print(out.shape, out)

test_mclmc_sampler()
