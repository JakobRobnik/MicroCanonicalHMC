#!/bin/bash

#SBATCH -N 1
#SBATCH --image=reubenharry/cosmo:1.0
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J bias
#SBATCH -t 05:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu



# load environment
cd ..

shifter python3 -m bias.main 1 0 1 8

#python3 -m bias.main 0 5 1 8

#python3 -m bias.main 1 5 5 8
#python3 -m bias.main 1 5 20 8

#shifter python3 -m bias.main 1 $SLURM_ARRAY_TASK_ID 1 8
#SB ATCH --array=0-1