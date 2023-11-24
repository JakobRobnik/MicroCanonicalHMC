import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt





def uniform_halton(float_index, max_bits=10):
  float_index = jnp.asarray(float_index)
  bit_masks = 2**jnp.arange(max_bits, dtype=float_index.dtype)
  return jnp.einsum('i,i->', jnp.mod((float_index + 1) // bit_masks, 2), 0.5 / bit_masks)

plt.plot([uniform_halton(i + 0.) for i in range(100)])

plt.show()
