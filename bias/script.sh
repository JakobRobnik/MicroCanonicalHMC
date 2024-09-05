#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J funnelMCLMC
#SBATCH -t 01:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu


# load environment
cd ..
module load python
conda activate jaxenv


python3 -m bias.main 0 5 1 8
#python3 -m bias.main 0 5 1 8

#python3 -m bias.main 1 5 5 8
#python3 -m bias.main 1 5 20 8
