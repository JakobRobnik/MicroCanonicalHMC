#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J biasHMC
#SBATCH -t 01:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu


# load environment
cd ..
source ../black.sh

python3 -m bias.main 1 2 9
python3 -m bias.main 1 3 9
python3 -m bias.main 1 4 9