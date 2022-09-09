from mpi4py import MPI
import time
import numpy as np
import sys
import traceback
import shutil
import os


def run_void(f, runs, begin = 0, args = None):
    """runs = how many calls of f does each thread make.
    f(n) will be evaluated for n in range(begin, begin + num_threads * runs)"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_cores = comm.Get_size()

    if rank == 0:
        t1 = time.time()

    for i in range(runs):
        print(begin + rank + num_cores*i)
        sys.stdout.flush()
        if args != None:
            f(begin + rank + num_cores*i, *args)
        else:
            f(begin + rank + num_cores * i)

    comm.Barrier()

    if rank == 0:
        print("Total time: " + str(np.round((time.time() - t1) / 60.0, 2)) + " min")


def run_collect(f, runs, working_folder, name_results):
    """function f(num_iteration) to be executed in parallel.
       should return an array of results, which will be stored in 'name_results' line by line"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_cores = comm.Get_size()

    if rank == 0:
        if os.path.isdir(working_folder):
            shutil.rmtree(working_folder)
        os.mkdir(working_folder)
    comm.Barrier()

    def ff(i):
        #np.save(working_folder + str(i) + '.npy', f(i, *args))
        np.save(working_folder + str(i) + '.npy', f(i))

    run_void(ff, runs)

    if rank == 0:
        X = []
        for i in range(runs*num_cores):
            X.append(np.load(working_folder + str(i) + '.npy'))
        np.save(name_results + '.npy', X)
