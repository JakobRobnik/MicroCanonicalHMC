import jax
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





def optimize_adam(function, x0, steps= 100, lr= 1e-3, beta1= 0.9, beta2= 0.999, eps= 1e-8, trace = False):

    def step(state):
        """one step of the adam algorithm"""
        x, m1, m2, t = state
        t += 1
        f, gradf = function(x)
        m1 = beta1 * m1 + (1-beta1) * gradf
        m2 = beta2 * m2 + (1 - beta2) * jnp.square(gradf)
        M1 = m1 / (1-jnp.power(beta1, t))
        M2 = m2 / (1 - jnp.power(beta2, t))
        x -= lr * M1 / (jnp.sqrt(M2) + eps)
        return x, m1, m2, t, f

    def step_normal(state, useless):
        x, m1, m2, t, f = step(state)
        return (x, m1, m2, t), None

    def step_track(state, useless):
        x, m1, m2, t, f = step(state)
        return (x, m1, m2, t), (f, x)

    state = (x0, jnp.zeros(len(x0)), jnp.zeros(len(x0)), 0) #intialization

    if trace: # return the value of the function and the parameter values for each step of the optimization
        return jax.lax.scan(step_track, init= state, xs= None, length= steps)[1]

    else: # only return the final value of the parameters
        return jax.lax.scan(step_normal, init= state, xs=None, length=steps)[0][0]


def tst_rosenbrock():
    a, b = 1., 10.
    func = lambda x: jnp.square(a - x[0]) + b * jnp.square(x[1] -jnp.square(x[0]))
    grad= jax.value_and_grad(func)

    f, x = optimize_adam(grad, jnp.array([-1., 0.]), lr = 0.01, trace= True, steps= 1000)

    import matplotlib.pyplot as plt
    plt.plot(f)
    plt.yscale('log')
    plt.show()
