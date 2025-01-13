import os

# change JAX GPU memory preallocation fraction
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'

# !nvidia-smi --query-gpu=gpu_name --format=csv,noheader

import numpy as np
import matplotlib.pyplot as plt

import jax
# jax.config.update("jax_debug_nans", True)


import jax.numpy as jnp
from jax import jit
import numpy

import blackjax
from datetime import date
from blackjax.util import run_inference_algorithm
from jax import jit, checkpoint, custom_vjp
from functools import partial


rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))


def rfftnfreq_2d(shape, spacing, dtype=np.float64):
    """Broadcastable "``sparse``" wavevectors for ``numpy.fft.rfftn``.

    Parameters
    ----------
    shape : tuple of int
        Shape of ``rfftn`` input.
    spacing : float or None, optional
        Grid spacing. None is equivalent to a 2π spacing, with a wavevector period of 1.
    dtype : dtype_like

    Returns
    -------
    kvec : list of jax.numpy.ndarray
        Wavevectors.

    """
    freq_period = 1
    if spacing is not None:
        freq_period = 2 * np.pi / spacing

    kvec = []
    for axis, s in enumerate(shape[:-1]):
        k = np.fft.fftfreq(s).astype(dtype) * freq_period
        kvec.append(k)

    k = np.fft.rfftfreq(shape[-1]).astype(dtype) * freq_period
    kvec.append(k)

    kvec = np.meshgrid(*kvec, indexing='ij', sparse=True)

    return kvec


##### Spatial parameters

nc = 128*4
bs = 100

ptcl_spacing = bs/nc
ptcl_grid_shape = (nc,) * 2



kvec = rfftnfreq_2d(ptcl_grid_shape, ptcl_spacing)
k = jnp.sqrt(sum(k**2 for k in kvec))

from scipy import interpolate

eps=10**(-6)

def ps_test(k):
    return (k+eps)**(-1)

nmodes = 10
kbins = np.linspace(10**(-0.5),k.max(),nmodes)
pk_vals_0 = ps_test((kbins[1:]+kbins[:-1])/2)


bp = jnp.hstack([pk_vals_0])


kvec = rfftnfreq_2d(ptcl_grid_shape, ptcl_spacing)
k = jnp.sqrt(sum(k**2 for k in kvec))
kplot = (kbins[1:]+kbins[:-1])/2

kbins[0] = -0.1
kbins[-1] = 100

def power3d(k,theta):
    val = jnp.interp(k,jnp.array(kplot),theta)
    return val

Plin = power3d(k, pk_vals_0)

nx = nc
ny = nc

kx = jnp.ones(k.shape)*kvec[0]
ky = jnp.ones(k.shape)*kvec[1]
from jax import jit, checkpoint, custom_vjp

@jit
@checkpoint
def linear_modes(modes, theta):
    kvec = rfftnfreq_2d(ptcl_grid_shape, ptcl_spacing)
    k = jnp.sqrt(sum(k**2 for k in kvec))

    Plin = power3d(k, theta)
    
    if jnp.isrealobj(modes):
        modes = jnp.fft.rfftn(modes, norm='ortho')
    modes *= jnp.sqrt(Plin)#*jnp.sqrt(10)
    return modes


def gen_map_den2d(theta,z):
    modes_e = z[:nx*ny].reshape((nx,ny))
    
    theta_e = theta[:nmodes-1]
    
    phi_e = linear_modes(modes_e, theta_e)
  
    return jnp.fft.irfftn(phi_e )#Need to check normalization



error_val=1.0

def sample_x_z(key, θ):
        keys = jax.random.split(key, 2)
        z = jax.random.normal(keys[0], (nx*ny,))
        x = (gen_map_den2d(θ,z) + error_val*jax.random.normal(keys[1], (nx,ny))) #1*jax.random.normal(keys[1], (32**3,)).reshape((32,32,32))
        return (x, z)


stheta = jnp.array(bp)

(xx,zz) = sample_x_z(rng_key,stheta)


x = xx

n_bp = nmodes-1

def logLike_MCHMC(v):
        z = v[n_bp:]
        
        #θ = stheta
        θ = v[:n_bp]
        #θ=θ.at[1].set(v[1])
        return logLike(z,θ)+logPrior(θ)

def logLike(z, θ):
        return -(jnp.sum((x - gen_map_den2d(θ,z))**2) + jnp.sum(z**2.0))

def logPrior(θ): #20% error on power spectra prior
        return -1/2.*jnp.sum(((θ-jnp.array(bp))/(0.2*bp))**2)
    


(xx,zz) = sample_x_z(rng_key,jnp.array(bp))


### import time

initial_position = numpy.random.rand(
    nx*ny+n_bp)

initial_position[:n_bp] = stheta
initial_position[n_bp:]= zz

initial_position= jnp.array(initial_position)

init_key, run_key = jax.random.split(rng_key, 2)


logLike(zz,stheta)

logLike_MCHMC(initial_position)

test = jax.value_and_grad(logLike_MCHMC)

def run_mclmc(logdensity_fn, num_steps, initial_position, key, transform):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )


    # build the kernel
    kernel = lambda inverse_mass_matrix : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix,
    )
    # jax.debug.print("{x} state before tuning", x=initial_state)

    # find values for L and step_size
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        # frac_tune2=0.0,
        # frac_tune3=0.0,
        diagonal_preconditioning=False,
    )
    # jax.debug.print("{x} state after tuning", x=(blackjax_state_after_tuning))
    jax.debug.print("{x} params after tuning", x=(blackjax_mclmc_sampler_params))

    # L = 0.47224498
    # step_size = 1.1806124


    # # use the quick wrapper to build a new kernel with the tuned parameters
    # sampling_alg = blackjax.mclmc(
    #     logdensity_fn,
    #     # L=blackjax_mclmc_sampler_params.L,
    #     # step_size=blackjax_mclmc_sampler_params.step_size,
    #     L=L,
    #     step_size=step_size,
    # )

    # # run the sampler
    # _, samples = blackjax.util.run_inference_algorithm(
    #     rng_key=run_key,
    #     initial_state=initial_state,
    #     # initial_state=initial_state,
    #     inference_algorithm=sampling_alg,
    #     num_steps=num_steps,
    #     transform=transform,
    #     progress_bar=True,
    # )

    # return samples


# run the algorithm on a high dimensional gaussian, and show two of the dimensions

sample_key, rng_key = jax.random.split(jax.random.PRNGKey(0))
samples = run_mclmc(
    logdensity_fn=logLike_MCHMC,
    num_steps=300,
    initial_position=initial_position,
    key=sample_key,
    transform=lambda state, info: state.position,
)


# samples.mean()

# def f(x, y):
#   return x / y
# jax.jit(f)(0., 0.)  # ==> raises FloatingPointError exception!
