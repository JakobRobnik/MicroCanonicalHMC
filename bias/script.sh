#!/bin/bash

#SBATCH -A m4031
#SBATCH -N 1
#SBATCH --image=reubenharry/cosmo:1.0
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J MCLMC
#SBATCH -t 02:30:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu


# load environment
cd ..

shifter python3 -m bias.main 0 0 1 7
#python3 -m bias.main 0 5 1 8

#python3 -m bias.main 1 5 5 8
#python3 -m bias.main 1 5 20 8
