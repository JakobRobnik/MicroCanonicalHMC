import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw

from scipy.integrate import odeint


a = 0.05
field = lambda t, u: 2 * np.sqrt(u * np.exp(u) - a * np.exp(2*u))

urange = np.linspace(-lambertw(-a, 0), -lambertw(-a, -1), 500)

plt.plot(urange, field(None, urange))
plt.show()