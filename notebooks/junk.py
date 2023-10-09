import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw

from scipy.integrate import odeint

num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)
