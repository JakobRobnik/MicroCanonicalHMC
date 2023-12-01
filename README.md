# MicroCanonical Hamiltonian Monte Carlo (MCHMC)

## Installation 

`pip install mclmc`

## Overview

![poster](img/github_poster.png)


You can check out the tutorials:
- [getting started](notebooks/tutorials/intro_tutorial.ipynb): sampling from a standard Gaussian
- [advanced tutorial](notebooks/tutorials/advanced_tutorial.ipynb): sampling the hierarchical Stochastic Volatility model for the S&P500 returns data

Julia implementation is available [here](https://github.com/JaimeRZP/MicroCanonicalHMC.jl).

The associated papers are:
- [method and benchmark tests](https://arxiv.org/abs/2212.08549)
- [formulation as a stochastic process and first application to the lattice field theory](https://arxiv.org/abs/2303.18221)

The code is still in active development, so let us know if you encounter any issues, including bad sampling performance, and we will do our best to help you out.
You can submit a github issue or contact us at jakob_robnik@berkeley.edu .

## Frequently asked questions:

### How can I sample with MCHMC if my parameters have bounds?
Check out [this tutorial](notebooks/tutorials/Constraints.ipynb).

### How does cost of producing one sample in HMC compare to the cost of one sample in MCHMC?
MCHMC samples are less costly. What is relevant for the computational time is the number of gradient evaluations used. Each sample in MCHMC is two gradient evaluations (1 gradient evaluation if leapfrog integrator is used instead of minimal norm integrator). Each sample in HMC is L gradient evaluations (where L is the number of leapfrog steps per sample), which can be quite large for hard targets (in default NUTS setting up to 1024).

### Is MCHMC just some weird projection of HMC onto the constant energy surface?
No, the Hamiltonian dynamics of both methods are different (the particles move differently). Below is the motion of MCHMC particles for the Rosenbrock target distribution.





![ensamble](img/rosenbrock.gif)
