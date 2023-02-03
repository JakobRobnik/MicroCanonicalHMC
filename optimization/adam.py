import jax.numpy as jnp


class Adam:

    def __init__(self, grad0, beta = 0.999):
        """beta = beta2 in the original paper"""
        self.steps = 1
        self.beta = beta
        self.grad_sq = jnp.square(grad0) * (1 - beta)

    def sigma_estimate(self):
        """correcting for the bias, see https://arxiv.org/pdf/1412.6980.pdf """
        unbiased_grad_sq = self.grad_sq / (1 - self.beta**self.steps)
        return jnp.sqrt(1.0 / unbiased_grad_sq)

    def step(self, grad):
        self.grad_sq = self.beta * self.grad_sq + (1 - self.beta) * jnp.square(grad)
        self.steps = self.steps + 1