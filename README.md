# MicroCanoncial Hamiltonian Monte Carlo (MCHMC)

MCHMC is a sampler from an arbitrary distribution with an available gradient of the probability density. Unlike the canonical HMC it does not require the equilibration between the different energy levels. Instead it tunes the Hamiltonian function such that the marginal of the microcanonical ensamble over the momentum variables gives the desired target distribution. Consequently it is severalfold more efficient in terms of the required gradient calls for the same posterior quality.

The accompanying paper is available [here](https://arxiv.org/abs/2212.08549).
You can check out the [getting started tutorial](simple_tutorial.ipynb) (sampling from a standard Gaussian target). 

If you have any questions do not hesitate to contact me at jakob_robnik@berkeley.edu
