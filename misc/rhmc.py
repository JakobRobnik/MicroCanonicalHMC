#import jax
#import jax.numpy as np
from typing import NamedTuple, Any
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, shgo, differential_evolution
import pickle
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"

#jax.config.update('jax_platform_name', 'cpu') # we will use cpu here (because we can reserve so many), this line is only to avoid jax warning

M = 20 # number of knots


class Knots(NamedTuple):
    """parameters of the RQ spline"""

    x: Any#jax.Array # locations of knots
    y: Any#jax.Array # values at knots
    dy: Any#jax.Array # slopes at knots


def _RQSpline(x, knots):
    m = np.searchsorted(knots.x, x) - 1 #which x is in which bin
    x1, x2, y1, y2, dy1, dy2 = knots.x[m], knots.x[m + 1], knots.y[m], knots.y[m + 1], knots.dy[m], knots.dy[m + 1]
    s = (y2 - y1) / (x2 - x1)
    sigma = dy2 + dy1 - 2 * s
    ksi = (x - x1) / (x2 - x1)
    return y1, y2, dy1, dy2, s, ksi, sigma

def RQSpline(x, knots):
    """computes the spline at locations x given parameters of the spline = knots"""
    y1, y2, dy1, dy2, s, ksi, sigma = _RQSpline(x, knots)
    return y1 + (y2 - y1) * (s * np.square(ksi) + dy1 * ksi*(1.0-ksi)) / (s + sigma*ksi*(1.0-ksi))


def RQSpline_derivative(x, knots):
    """computes the spline at locations x given parameters of the spline = knots"""
    y1, y2, dy1, dy2, s, ksi, sigma = _RQSpline(x, knots)
    return np.square(s) * (dy2 * np.square(ksi) + 2 * s * ksi*(1. - ksi) + dy1 * np.square(1-ksi)) / np.square(s + sigma*ksi*(1.0-ksi))


def sample(knots):
    """Returns samples x with inverse transform sampling"""
    U = np.linspace(0, 1, 500)
    return RQSpline(U, knots)


def ESS(knots):
    """ESS for the worst direction"""
    inv_sigma = np.linspace(1., 50., 10000)
    x = sample(knots)
    avg_time = np.average(x)
    rho = np.max(np.average(np.square(np.cos(np.outer(x, inv_sigma))), axis = 0))

    return (0.5 * np.pi / avg_time) * (1 - np.square(rho)) / (1 + np.square(rho))


def opt_params_to_knots(params):

  return Knots(x= np.linspace(0, 1, M, endpoint=True),  # these are fixed throughout
               y= np.cumsum(params[:M]),
               dy= params[M:]
               )

def loss(params):
    knots = opt_params_to_knots(params)
    return - ESS(knots)



def global_optimization(file_out):
    
    
    delta_y_max = 2.
    dy_max = 2* delta_y_max
    bounds = [(0., delta_y_max) for i in range(M)]+ [(0, dy_max) for i in range(M)]
    
    #opt = differential_evolution(loss, bounds = bounds, options= {'workers': -1})          
    opt = differential_evolution(loss, bounds = bounds, workers= -1)

    save_opt(opt, file_out)
    
    

def save_opt(opt, file_out):

    with open(file_out + '.pickle', 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    knots = opt_params_to_knots(opt['x'])
    plott(knots, file_out)

def load_opt(file):

    with open(file + '.pickle', 'rb') as handle:
        opt = pickle.load(handle)

    knots = opt_params_to_knots(opt['x'])
    plott(knots, file)


def plott(knots, file_out):
    u = np.linspace(0, 1, 500)
    x = RQSpline(u, knots)
    pdf = 1./RQSpline_derivative(u, knots)

    plt.figure(figsize = (10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(knots.y, knots.x, 'o', color = 'black')
    plt.plot(x, u, color= 'teal')
    plt.ylabel('CDF')
    plt.xlabel('x')
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.plot(x, pdf, color= 'teal')
    plt.ylabel('p(x)')
    plt.xlabel('x')
    plt.ylim(0, np.max(pdf)*1.05)
    plt.savefig(file_out + '.png')
    plt.close()
    
    
if __name__ == '__main__':
    
    file_out = 'de_M'+str(M)
    global_optimization(file_out)
    #load_opt(file_out)