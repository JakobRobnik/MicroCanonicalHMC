import numpy as np
import matplotlib.pyplot as plt
import sys

import ESH
import parallel
from targets import *
import bias


def compute_free_time(n):
    #free_steps_arr = (np.linspace(50, 250, 18)).astype(int)
    free_steps_arr = [1, 10, 100, 1000]
    #free_steps_arr = (np.linspace(50, 250, 18)).astype(int)  ##[1, 10, 100, 1000, 10000, 100000]

    d = 100
    kappa = 100
    eps = 0.5
    free_steps = free_steps_arr[n]
    esh = ESH.Sampler(Target=IllConditionedGaussian(d=d, condition_number = kappa), eps=eps)
    x0 = esh.Target.draw(1)[0] #we draw an initial condition from the target

    ess = esh.ess(x0, free_steps)

    return [ess, free_steps, eps, d, kappa]

    # np.save('Tests/bounce_frequency/X_fine'+str(free_steps)+'.npy', X)
    # np.save('Tests/bounce_frequency/w_fine'+str(free_steps)+'.npy', w)


def compute_eps(n):
    eps_arr = np.logspace(np.log10(0.1), np.log10(16), 6)
    eps = eps_arr[n]
    d = 100
    free_time = 16.0
    esh = ESH.Sampler(Target=StandardNormal(d=d), eps=eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]
    free_steps = (int)(free_time / eps)
    ess = esh.ess(x0, free_steps)

    return [ess, free_steps, eps, d]


def compute_kappa(n):
    kappa = np.logspace(0, 3, 18)[n]#([1, 10, 100, 1000])[n]
    d = 100
    eps, free_steps = 1, 16
    esh = ESH.Sampler(Target=IllConditionedGaussian(d=d, condition_number=kappa), eps=eps)
    np.random.seed(0)
    x0 = esh.Target.draw(1)[0]

    ess = esh.ess(x0, free_steps)

    return [ess, free_steps, eps, d, kappa]


def compute_dimension(n):
    d = ([2, 3, 5, 10])[n]  # ([1, 10, 100, 1000])[n]

    eps, free_steps = 1.0, 16
    esh = ESH.Sampler(Target= StandardNormal(d=d), eps=eps)
    np.random.seed(0)
    num_averaging = 10
    x0 = esh.Target.draw(num_averaging)

    ess, ess_upper, ess_lower = esh.ess_with_averaging(x0, free_steps)

    return [ess, ess_upper, ess_lower, free_steps, eps, d]


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


def compute_mode_mixing(n):
    mu = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])[n]

    eps, free_steps = 1.0, 16
    d = 50
    esh = ESH.Sampler(Target= BiModal(d=d, mu= mu), eps=eps)
    np.random.seed(0)

    avg_island_size = esh.mode_mixing(free_steps)
    print(avg_island_size)
    sys.stdout.flush()

    return [avg_island_size, free_steps, eps, d, mu]





if __name__ == '__main__':

    #parallel run:
    parallel.run_collect(compute_mode_mixing, num_cores= 4, runs= 2, working_folder= 'working/', name_results= 'Tests/mode_mixing')

