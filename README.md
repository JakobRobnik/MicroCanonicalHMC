# MicroCanoncial Hamiltonian Monte Carlo (MCHMC)

MCHMC is a sampler from an arbitrary distribution with an available gradient of the probability density. Unlike the canonical HMC it does not require the equilibration between the different energy levels. Instead it tunes the Hamiltonian function such that the marginal of the microcanonical ensamble over the momentum variables gives the desired target distribution. Consequently it is severalfold more efficient in terms of the required gradient calls for the same posterior quality.

You can check out the tutorials:
- [getting started](intro_tutorial.ipynb): sampling from a standard Gaussian (sequential sampling)
- [ensamble](Ensamble_tutorial.ipynb): sampling from the Rosenbrock function (ensamble sampling, in progress)

Julia implementation is available [here](https://github.com/JaimeRZP/MicroCanonicalHMC.jl).

The associated papers are:
- [method and benchmark tests](https://arxiv.org/abs/2212.08549)
- [formulation as a stochastic process and first application to the lattice field theory](https://arxiv.org/abs/2303.18221)

If you have any questions do not hesitate to contact me at jakob_robnik@berkeley.edu

![ensamble](plots/movies/rosenbrock.gif)
