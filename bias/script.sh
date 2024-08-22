#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J bias
#SBATCH -t 01:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu


# load environment
cd ..
source ../black.sh

for sampler in 0 1
do
    for target in 2 3 4
    do
        for L in {1..9}
        do
            python3 -m bias.main $sampler $target $L
        done
    done
done