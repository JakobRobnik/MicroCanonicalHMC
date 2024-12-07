#!/bin/bash

#SBATCH -N 1
#SBATCH --image=reubenharry/cosmo:1.0
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J sequential
#SBATCH -t 04:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu
#SBATCH --array=0-1
 
cd ..

shifter python3 -m ensemble.sequential $SLURM_ARRAY_TASK_ID

