# Benchmarking Samplers 

This repository exists to benchmark sampling algorithms implemented in blackjax, currently focusing on the family of Microcanonical Hamiltonian Monte Carlo algorithms.

## Example usage

The core functionality is provided by the function `benchmark`, which takes a distribution from the repository's selection of distributions (defined up to a differentiable log pdf), a sampling algorithm (like NUTS), and returns some simple metrics. For example:

```python
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=Gaussian(100,100),
    sampler=nuts(integrator_type="velocity_verlet", preconditioning=False),
    key=jax.random.PRNGKey(0), 
    n=10000,
    batch=num_chains,
)

print(f"\nGradient calls for NUTS to reach RMSE of X^2 of 0.1: {grads_to_low_avg} (avg over {num_chains} chains and dimensions)")
```

This will return:

> Gradient calls for NUTS to reach RMSE of X^2 of 0.1: 8105.0810546875 (avg over 128 chains and dimensions)

You can then compare to another algorithm, like Microcanonical Langevin Monte Carlo (MCLMC):

```python
num_chains = 128
ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
    model=Gaussian(100,100),
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

## Microcanonical Hamiltonian Monte Carlo

For an open-source implementation, please refer to 
https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
for the details of the algorithm 
https://microcanonical-monte-carlo.netlify.app/.

## Contact

If you encounter any issues do not hesitate to contact us at jakob_robnik@berkeley.edu.

