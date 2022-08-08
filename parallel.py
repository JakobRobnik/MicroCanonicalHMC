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
        print(i)
        sys.stdout.flush()
        if args != None:
            f(begin + rank + num_threads*i, *args)
        else:
            f(begin + rank + num_threads * i)

    comm.Barrier()

    if rank == 0:
        print("Total time: " + str(np.round((time.time() - t1) / 60.0, 2)) + " min")