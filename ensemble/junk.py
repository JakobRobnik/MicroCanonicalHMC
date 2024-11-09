import jax
import jax.numpy as jnp
import pandas as pd


pf= pd.read_csv('ensemble/pathfinder_data.csv', sep= '\t')
pf = pf[pf['name'] == 'GermanCredit']
bavg, bmax, grads = pf[['bavg', 'bmax', 'grads']].to_numpy()[0]
print(grads)