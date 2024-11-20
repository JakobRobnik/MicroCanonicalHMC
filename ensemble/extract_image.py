import numpy as np
from scipy.interpolate import splev


# this is currently only done for stochastic volatility

third_party_methods = [['MEADS', 'green'], ['ChEES', 'blue'], ['NUTS', 'orange']]


x_unit = 1000
log10b2_init = 4
y_unit = 1

def grads2x(grads):
    phase, period = 77.5, 196.4
    return phase + period * grads / x_unit

def y2bias(y):
    phase, period = 24., 75.9
    slope = - y_unit / period
    log10b2 = slope * (y - phase) + log10b2_init
    return np.power(10, log10b2)   


def load_spline(name):    
    x = np.load(name)
    return (x['spline0'], x['spline1'], x['spline2'])

def save_spline(name, spline):
    np.savez(name, spline0= spline[0], spline1= spline[1], spline2= spline[2])
    

def imported_plot(n, spline_loc):
    spline = load_spline(spline_loc)
    x = grads2x(n)
    y = splev(x, spline)
    b2 = y2bias(y)
    return b2