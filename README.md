**MicroCanoncial Hamiltonian Monte Carlo (MCHMC)** is a constant energy sampler from an arbitrary distribution with a differentiable probability density.


Curently the files are:


- ESH.py: implementation of the [ESH algorithm](https://arxiv.org/pdf/2111.02434.pdf)
- HMC.py: applying NUTS to our examples
- bias.py: computes ESS by monitoring the bias
- check_distribution and check_integrator: kind of obsolete, were used to study the properties of the integrator and sampler
- parallel.py: convenience functions for parallel computing with mpi4py
- sampler.py: our implementation of sampling with standard T+V, position-dependent-mass and BI Hamiltonians. Not competitive yet
- simple_tests.py: hyperparameter tuning, condition number dependence, funnel...
- graphs_tests.py: for visualizing results from simple_tests.py
- targets.py: example distributions that we sample from


**Example**:
To compute the frequency of bounce dependence in parallel, you can run from the terminal:
```
mpiexec -n 4 python3 simple_tests.py 
```
and visualize the results by running graphs_tests.py

If you have any questions do not hesitate to ask me.
