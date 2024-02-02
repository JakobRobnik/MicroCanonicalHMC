# Microcanonical Hamiltonian Monte Carlo (MCHMC)

For details, please refer to https://microcanonical-monte-carlo.netlify.app/.

## Usage

This repository is currently deprecated, in favor of the implementation in Blackjax: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html.

A Julia implementation is also available [here](https://github.com/JaimeRZP/MicroCanonicalHMC.jl).

## Overview

![poster](img/github_poster.png)

## Frequently asked questions:

### How does cost of producing one sample in HMC compare to the cost of one sample in MCHMC?

MCHMC samples are less costly. What is relevant for the computational time is the number of gradient evaluations used. Each sample in MCHMC is two gradient evaluations (1 gradient evaluation if leapfrog integrator is used instead of minimal norm integrator). Each sample in HMC is L gradient evaluations (where L is the number of leapfrog steps per sample), which can be quite large for hard targets (in default NUTS setting up to 1024).

### Is MCHMC just some projection of HMC onto the constant energy surface?

No, the Hamiltonian dynamics of both methods are different (the particles move differently). Below is the motion of MCHMC particles for the Rosenbrock target distribution.

![ensamble](img/rosenbrock.gif)
