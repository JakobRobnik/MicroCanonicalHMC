import sys

sys.path.insert(0, '../../')
sys.path.insert(0, './')

# from sampling.annealing import Sampler
import jax
import jax.numpy as jnp
from sampling.annealing import Annealing
from sampling.sampler import Sampler, Target
import sampling.old_annealing as A

temp_schedule = jnp.array([3.0, 2.0, 1.0])


def test_annealing_comparison():
    nlogp = lambda x: 0.5*jnp.sum(jnp.square(x))
    target = Target(d = 10, nlogp=nlogp)
    target.prior_draw = lambda key : jax.random.normal(key, shape = (10, ), dtype = 'float64') 
    sampler = Sampler(target)

    annealer = Annealing(sampler)
    annealer_old = A.Sampler(target)
    samples = annealer.sample(steps_at_each_temp = 1000, tune_steps= 100, num_chains= 100, temp_schedule = temp_schedule)
    samples_old = annealer_old.sample(steps_at_each_temp = 1000, tune_steps= 100, num_chains= 100, temp_schedule = temp_schedule)
    assert jnp.array_equal(samples[0][-1, -1, :, :], samples_old), "Old and new annealer code should give same result"

