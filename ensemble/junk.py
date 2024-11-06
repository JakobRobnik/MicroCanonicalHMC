import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

x = np.arange(12).reshape(6, 2)

targets = ['a', 'b', 'c', 'd', 'e', 'f']
columns = ['bavg', 'bmax']

df = pd.DataFrame(x, columns= columns)
df['name'] = targets

print(df)
