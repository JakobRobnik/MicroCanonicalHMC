import numpy as np
import matplotlib.pyplot as plt

L = 10
C = np.array([-0.466516, -0.834376, -0.714529, -0.0245586, 0.238837, 0.0143649, 0.271003, -0.374538, 0.873564, -0.370258, 0.891462, -0.665239, 0.810546, 0.198216, -0.816637, -0.195351, -0.573181, 0.251745, 0.647615, 0.201654])
T = 1.0

bins= 100
dx = L/bins

eps = 0.5*(np.sqrt(5)-1) * L

def asign_bin(x):
    return ((x / L) * bins).astype(int)

def weight(y):
    return np.exp(-np.sum([C[n] * np.sin(2 * np.pi * (n + 1) * y) for n in range(20)]) / T)

def compute_chi2(rho1, rho2):
    return np.sum(np.square(rho1 - rho2)) * dx


def compute_rho(num):
    x = np.linspace(0, L, num, endpoint= False)
    bin_indices = asign_bin(x)
    V = np.sum([C[i] * np.sin(2 * np.pi * (i+1) * x / L) for i in range(20)], axis = 0)
    rho = np.zeros(bins)
    for i in range(num):
        rho[bin_indices[i]] += np.exp(- V[i] / T)
    rho /= (np.sum(rho) * dx)
    return rho


def quasi_random(n):
    X = np.empty(n)
    X[0] = np.random.uniform(0, L)
    for i in range(1, n):
        x = X[i-1] + eps
        y = x/L
        y -= (int)(y)
        X[i] = y * L
    return X



rho0 = compute_rho(1000000)


def mchmc(steps):

    def step(state, useless):
        x, prob, W = state

        x += eps
        y = x/L
        y -= (int)(y)
        w = weight(y)

        bin = (int)(y * bins)
        prob *= W / (W + w)
        prob[bin] += w / (W + w)
        W += w


        return (x, prob, W), compute_chi2(rho0, prob/dx)

    x = np.random.uniform(0, L)
    w = weight(x/L)
    bin = (int)(x/dx)
    prob = np.zeros(bins)
    prob[bin] += 1.0
    state = (x, prob, w)
    chi2 = np.empty(steps)

    for i in range(steps):
        state, chi2[i] = step(state, None)

    return chi2


chi2 = mchmc(100000)
plt.plot(np.sqrt(chi2))
plt.xlabel('steps')
plt.ylabel(r'$\chi$')
plt.yscale('log')
plt.xscale('log')
plt.savefig('sinus20_large_steps.png')
plt.show()