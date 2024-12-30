#!/bin/bash

#SBATCH -A m4031
#SBATCH -N 1
#SBATCH --image=reubenharry/cosmo:1.0
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J reubentest
#SBATCH -t 08:00:00

shifter python3 -m benchmarks.benchmark_omelyan
