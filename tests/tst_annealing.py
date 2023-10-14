from sampling.annealing import Sampler
import jax
import jax.numpy as jnp


temp_schedule = jnp.array([3.0, 2.0, 1.0])


class StandardNormal:
    """Standard Normal distribution in d dimensions"""

    def __init__(self, d):
        self.d = d
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * jnp.sum(jnp.square(x))

    def prior_draw(self, key):
        return jax.random.normal(key, shape=(self.d,), dtype="float64") * jnp.sqrt(
            temp_schedule[0]
        )  # start from the distribution at high temperature


target = StandardNormal(d=100)

sampler = Sampler(target)

x = sampler.sample(
    steps_at_each_temp=1000, tune_steps=100, num_chains=100, temp_schedule=temp_schedule
)


x1 = jnp.average(x, axis=0)
x2 = jnp.average(jnp.square(x), axis=0)

print(x1)
print(x2)
