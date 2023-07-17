import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


a = 30.
m = 0.3

def p(x):
    return (1- m) *norm.pdf(x, 0.0, 1.0) + m * norm.pdf(x, a, 2.0)



def shift_annealing():


    x = np.linspace(-40, 40,  400)

    def get_ind(xmin, xmax):
        return (x < xmax) & (x > xmin)

    num = 5
    plt.figure(figsize= (6, 4 * num))

    beta = [0, 0.1, 0.2, 0.4, 1]
    for i in range(num):
        plt.subplot(num, 1, i+1)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        b = beta[i]
        plt.title(r'$\beta = $' + str(b), y = 0.85, color = 'royalblue')
        p1 = p(x)
        p2 = p(x + (1 - b) * a)
        plt.plot(x, p1, color= 'tab:blue', alpha = 0.5)
        plt.plot(x, p2, color='tab:red', alpha = 0.5)
        plt.plot(x, p1 + p2, color = 'black', lw = 1)
        mask1 = get_ind(-10, 20 + 20 * (i == num-1))
        plt.fill_between(x[mask1], np.zeros(len(x))[mask1], (p1 + p2)[mask1], color = 'black', alpha = 0.1)
        plt.ylim(0, 1.2 * np.max(p1 + p2))
        #if i != num-1:
        plt.xticks([])

        plt.yticks([])

    plt.tight_layout()
    plt.savefig('shift_annealing.png')
    plt.show()


def temperature_annealing():
    x = np.linspace(-10, 50,  400)

    num = 5
    plt.figure(figsize= (6, 4 * num))

    beta = [0, 0.01, 0.05, 0.1, 1]
    for i in range(num):
        plt.subplot(num, 1, i+1)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        b = beta[i]
        plt.title(r'$\beta = $' + str(b), y = 0.85, color= 'royalblue')
        p1 = p(x)**b
        plt.plot(x, p1, color = 'tab:blue', lw = 1)

        plt.ylim(0, 1.2 * np.max(p1))

        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig('temperature_annealing.png')
    plt.show()

shift_annealing()