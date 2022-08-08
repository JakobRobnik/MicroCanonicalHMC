import numpy as np
import matplotlib.pyplot as plt
import ESH
import parallel
from targets import *


def bias(X, w, var_true):
    """ Bias = average over dimensions ( variance(samples) - variance(true) )^2 / variance(true)^2 )
        Args:
            X: samples, an array of shape (num_samples, d)
            w: weight of each sample, an array of shape (num_samples, )
            var_true: an exact variance of each dimension, an array of shape (d, )
        Returns:
            variance bias as a function of number of samples (in the way they are ordered)
    """
    var = (np.cumsum((np.square(X.T) * w), axis = 1) / np.cumsum(w)).T

    return np.sqrt(np.average(np.square((var - var_true) / var_true), axis = 1))


def moving_median_trend(flux, band_half_width, average, borders):
    """smoothing by a moving average method.
        flux -> series to be smoothened
        band_half_width -> integer half width of the moving window
        average -> function for comuting the average, e.g. np.average, np.median...
        if borders == True the points near the border are computed using smaller window
        if borders == False the points with indexes 0:band_half_width+1 have the same value and points len(flux) - band_half_width -1 : len(flux) have the same value
    """
    smooth = np.zeros(len(flux))
    indeks_min = band_half_width
    indeks_maks = len(flux)-1-band_half_width #chosen in such a way that when averiging we do not have out of range
    for i in range(indeks_min, indeks_maks+1):
        smooth[i] = average(flux[i-band_half_width: i+band_half_width+1])

    #bordering regions
    if borders:
        for i in range(indeks_min): #bordering region at the beggining of the series
            smooth[i] = average(flux[: 2*i+1])
        for i in range(indeks_maks+1, len(flux)): #bordering region at the end of the series
            smooth[i] = average(flux[-len(flux)+1+2*i:])

    else:
        for i in range(indeks_min): #bordering region at the beggining of the series
            smooth[i] = smooth[indeks_min]
        for i in range(indeks_maks+1, len(flux)): #bordering region at the end of the series
            smooth[i] = smooth[indeks_maks]

    return smooth


def estimate_ess(X, w, var_true):
    """estimates (effecitve sample size) / (sample size)"""
    variance_bias = bias(X, w, var_true)
    # plt.plot(variance_bias, color= 'black')
    # plt.plot([0, len(variance_bias)], [0.1, 0.1], color = 'tab:red')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('# evaluations')
    # plt.ylabel('bias')
    # plt.savefig('bias100.png')
    # plt.show()
    flag = variance_bias < 0.1
    fraction = np.cumsum(flag) / (np.arange(len(flag))+1) #fraction of time points which have bias^2 < 0.01 as a function of time

    # plt.plot(fraction, '.')
    # plt.plot([0, len(variance_bias)], [0.5, 0.5], color = 'tab:red')
    # plt.xscale('log')
    # plt.xlabel('# evaluations')
    # plt.ylabel('fraction of points with bias < 0.1')
    # plt.savefig('fraction100.png')
    # plt.show()

    i = len(fraction) -1
    while fraction[i] > 0.5:
        i-=1
        if i < 0:
            return 0

    #the first time for which the fraction is smaller than 0.5 (starting from the above)
    return 2* 200.0 / i



def ess_bootstrap(X, w, var_true):
    """reduces the ess estimate variance"""
    num = 100
    ess = np.empty(num)

    for i in range(num):
        perm = np.random.permutation(len(X))
        ess[i] = estimate_ess(X[perm, :], w[perm], var_true)

    return np.median(ess)#, np.std(ess)/ np.sqrt(num)




def compute_bounce_frequency(n):
    nn = [1, 10, 100, 1000, 10000, 100000]

    d = 50
    total_num = 1000000
    num_bounces = nn[n]
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps=0.1)
    x0 = esh.Target.draw(1)[0]

    X, w = esh.sample(x0, total_num //num_bounces, total_num)
    np.save('Tests/bounce_frequency/X_half_sphere_'+str(num_bounces)+'.npy', X)
    np.save('Tests/bounce_frequency/w_half_sphere_'+str(num_bounces)+'.npy', w)


def compute_eps_dependence(n):
    eps_arr = [0.05, 0.1, 0.5, 1, 2]
    eps = eps_arr[n]
    d = 50
    total_num = 1000000
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps=eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    X, w = esh.sample(x0, 100, total_num)
    np.save('Tests/eps/X'+str(n)+'.npy', X)
    np.save('Tests/eps/w'+str(n)+'.npy', w)


def compute_kappa(n):
    kappa = ([1, 10, 100, 1000])[n]
    d = 50
    total_num = 1000000
    esh = ESH.Sampler(Target=IllConcitionedGaussian(d=d, condition_number=kappa), eps=0.1)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    X, w = esh.sample(x0, 100, total_num)
    np.save('Tests/kappa/X'+str(kappa)+'.npy', X)
    np.save('Tests/kappa/w'+str(kappa)+'.npy', w)


def compute_energy(n):
    eps_arr = [0.05, 0.1, 0.5, 1, 2]
    eps = eps_arr[n]
    d = 50
    total_num = 1000000
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps=eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    t, X, P, E = esh.trajectory(x0, total_num)
    np.save('Tests/energy/E'+str(n)+'.npy', E)



if __name__ == '__main__':
    parallel.run_void(compute_kappa, 4, 1)

