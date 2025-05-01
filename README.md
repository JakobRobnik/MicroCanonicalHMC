# Benchmarking Samplers 

This repository exists to benchmark sampling algorithms implemented in blackjax, currently focusing on the family of Microcanonical Hamiltonian Monte Carlo algorithms.

## Example usage

The core functionality is provided by the function `benchmark`, which takes a distribution from the repository's selection of distributions (defined up to a differentiable log pdf), a sampling algorithm (like NUTS), and returns some simple metrics. For example:

```python
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=Gaussian(ndims=100,condition_number=100),
    sampler=nuts(integrator_type="velocity_verlet", preconditioning=False),
    key=jax.random.PRNGKey(0), 
    n=10000,
    batch=num_chains,
)

print(f"\nGradient calls for NUTS to reach RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
```

This will return:

> Gradient calls for NUTS to reach RMSE of X^2 of 0.1: 8105.0810546875 (avg over 128 chains and dimensions)

(This is NUTS without a mass matrix, and using dual averaging targeting 0.8 acceptance probability to tune the step size).

You can then compare to another algorithm, like Microcanonical Langevin Monte Carlo (MCLMC):

```python
num_chains = 128
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=Gaussian(ndims=100,condition_number=100),
    sampler=unadjusted_mclmc(integrator_type="mclachlan", preconditioning=False),
    key=jax.random.PRNGKey(0), 
    n=10000,
    batch=num_chains,
)

print(f"\nGradient calls for MCLMC to reach standardized RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
```

You'll get 

> Gradient calls for MCLMC to reach standardized RMSE of X^2 of 0.1: 978.0 (avg over 128 chains and dimensions)

See the file `benchmarks/example.py` for this example in full, with imports and so on.

## What this repository provides

This repository is intended to be used in conjunction with blackjax, which is a library of sampling methods written in Jax.

Blackjax is a developer facing library, and while it provides both the tuning procedures and the algorithms for e.g. NUTS, it doesn't put these together into a single function for you. This repository does that (benchmarks/sampling_algorithms.py).

It also provides a library of distributions (benchmarks/inference_models.py) and scripts to calculate metrics (benchmarks/metrics.py) like autocorrelation and time to reach a target RMSE.

## Microcanonical Langevin Monte Carlo

Microcaninical Langevin Monte Carlo (MCLMC) is intended as a alternative for HMC and NUTS. On the metric described above (gradient calls to reach a target RMSE of X^2) and on many other metrics, we have found that MCLMC gets better results than NUTS, *but are very interested in hearing about cases where this is not true*. As a simple example, here's a Gaussian with a high condition number:

```python
model = Gaussian(ndims=100,condition_number=1e5)
num_chains = 128
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=model,
    sampler=unadjusted_mclmc(integrator_type="mclachlan", preconditioning=False, num_windows=3,),
    key=jax.random.PRNGKey(1), 
    n=20000,
    batch=num_chains,  
)
```

> Gradient calls for MCLMC to reach standardized RMSE of X^2 of 0.1: 19058.0 (avg over 128 chains and dimensions)

This is with mass matrix set to the identity. Meanwhile for NUTS (also with mass matrix as the identity), we get:

> Gradient calls for NUTS to reach standardized RMSE of X^2 of 0.1: 204317.8125 (avg over 128 chains and dimensions)

In higher dimensional Gaussians, we find that MCLMC has even more of an advantage over NUTS.


For an open-source implementation, please refer to 
https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html, and for the details of the algorithm, see 
https://microcanonical-monte-carlo.netlify.app/.

## Contact

If you encounter any issues do not hesitate to contact us at jakob_robnik@berkeley.edu.

