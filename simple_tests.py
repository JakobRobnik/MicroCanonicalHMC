
import numpy as np
import matplotlib.pyplot as plt
import ESH
import parallel
from targets import *
import bias


def compute_free_time(n):
    free_steps_arr = (np.linspace(50, 250, 18)).astype(int)##[1, 10, 100, 1000, 10000, 100000]

    d = 100
    eps = 0.1
    free_steps = free_steps_arr[n]
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps=eps)
    x0 = esh.Target.draw(1)[0] #we draw an initial condition from the target

    ess = esh.ess(x0, free_steps)


    return [ess, free_steps, eps, d]

    # np.save('Tests/bounce_frequency/X_fine'+str(free_steps)+'.npy', X)
    # np.save('Tests/bounce_frequency/w_fine'+str(free_steps)+'.npy', w)


def compute_eps_dependence(n):
    eps_arr = np.logspace(np.log10(0.05), np.log10(0.5), 18)
    eps = eps_arr[n]
    d = 100
    free_time = 15.0
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps=eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]
    free_steps = (int)(free_time / eps)
    ess = esh.ess(x0, free_steps)

    return [ess, free_steps, eps, d]


def compute_kappa(n):
    kappa = ([1, 10, 100, 1000])[n]
    d = 50
    total_num = 1000000
    esh = ESH.Sampler(Target=IllConditionedGaussian(d=d, condition_number=kappa), eps=0.1)
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
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps= eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    t, X, P, E = esh.trajectory(x0, total_num)
    np.save('Tests/energy/E'+str(n)+'.npy', E)


if __name__ == '__main__':

    #to run in parallel run from the terminal:
    # mpiexec -n 6 python3 simple_tests.py
    #compute_bounce_frequency(2)
    parallel.run_collect(compute_free_time, 6, 3, 'working/', 'Tests/free_time1')

