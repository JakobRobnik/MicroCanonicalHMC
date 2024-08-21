#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J biasMCLMC
#SBATCH -t 00:15:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu


# load environment
cd ..
source ../black.sh

python3 -m bias.main 0 3 1

