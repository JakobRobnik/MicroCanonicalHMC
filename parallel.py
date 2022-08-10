from mpi4py import MPI
import time
import numpy as np
import sys
import traceback
import shutil
import os


def run_void(f, num_threads, runs, begin = 0, args = None):
    """runs = how many calls of f does each thread make.
    f(n) will be evaluated for n in range(begin, begin + num_threads * runs)"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        t1 = time.time()

    for i in range(runs):
        print(begin + rank + num_threads*i)
        sys.stdout.flush()
        if args != None:
            f(begin + rank + num_threads*i, *args)
        else:
            f(begin + rank + num_threads * i)

    comm.Barrier()

    if rank == 0:
        print("Total time: " + str(np.round((time.time() - t1) / 60.0, 2)) + " min")


def run_collect(f, num_cores, runs, working_folder, name_results):
    """function f(num_iteration) to be executed in parallel.
       should return an array of results, which will be stored in 'name_results' line by line"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if os.path.isdir(working_folder):
            shutil.rmtree(working_folder)
        os.mkdir(working_folder)
    comm.Barrier()

    def ff(i):
        np.save(working_folder + str(i) + '.npy', f(i))

    run_void(ff, num_cores, runs)

    if rank == 0:
        X = []
        for i in range(runs*num_cores):
            X.append(np.load(working_folder + str(i) + '.npy'))
        np.save(name_results + '.npy', X)
