#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J bias
#SBATCH -t 02:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu


# load environment
source ../black.sh
  
python3 -m notebooks.bias 0 3 1
python3 -m notebooks.bias 0 4 1

python3 -m notebooks.bias 1 3 5
python3 -m notebooks.bias 1 4 5

python3 -m notebooks.bias 1 2 9
python3 -m notebooks.bias 0 0 1
