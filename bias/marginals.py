import jax
import blackjax
import numpy as np
import jax.numpy as jnp

from benchmarks.inference_models import *
from mclmc import run_mclmc
from benchmarks.truth import nuts

scratch = '/pscratch/sd/j/jrobnik/mchmc/bias/' # The chains produced here are long. To save memeory in $HOME, let's store results in $PSCRATCH


accuracy = 0.01
eevpd = 4 * accuracy**3

def do_mclmc(model, indices):
    samples = run_mclmc(model, 10**7, desired_energy_var= eevpd, transform= lambda x: x[indices])
    np.save(scratch + model.name + '/mclmc_b=1e-2.npy', samples)

def do_nuts(model, indices):
    model.transform = lambda x: x[indices]
    samples = nuts(model, 5 * 10**6)
    np.save(scratch + model.name + '/nuts.npy', samples)


funnel = (Funnel_with_Data(), jnp.array([0, -1]))
brownian = (Brownian(), jnp.arange(Brownian().ndims))
# do_mclmc(*funnel)
#do_mclmc(*brownian)

do_nuts(*funnel)
#nuts(*brownian)