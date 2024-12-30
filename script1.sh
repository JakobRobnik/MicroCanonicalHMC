#!/bin/bash

#SBATCH -A m4031_g
#SBATCH -N 1
#SBATCH --image=reubenharry/cosmo:1.0
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -J reubentest
#SBATCH -t 04:00:00

shifter python3 -m benchmarks.benchmark_hard_problems
