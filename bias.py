import numpy as np


### general functions for computing the bias and for estimating the effective sample size ###


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


def ess_cutoff_crossing(b, step_cost):

    cutoff = 0.1

    n_crossing = 0
    while b[n_crossing] > cutoff:
        n_crossing += 1
        if n_crossing == len(b):
            return 0, n_crossing

    return 200.0 / np.sum(step_cost[:n_crossing + 1]), n_crossing


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



# def get_ess():
#     kappa = [1, 10, 100, 1000]
#     d = 50
#     ess_arr = np.zeros(len(kappa))
#
#     for n in range(len(kappa)):
#         target = IllConditionedGaussian(d=d, condition_number=kappa[n])
#
#         X = np.load('Tests/kappa/X' + str(kappa[n]) + '.npy')
#         w = np.load('Tests/kappa/w' + str(kappa[n]) + '.npy')
#
#
#         n_crossing = cutoff_crossing(X, w, target.variance)
#
#         ess_arr[n] = 200.0 / n_crossing
#
#     print(ess_arr)



