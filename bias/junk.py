import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.inference_models import Gaussian, rng_inference_gym_icg



target1= Gaussian(ndims= 100, condition_number= 100., eigenvalues='log')
target2= Gaussian(ndims= 100, condition_number= 100., eigenvalues='outliers')

