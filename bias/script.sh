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

for k in 0 1
do
    for j in 2 3 4
    do
        for i in {1..9}
        do
            python3 -m bias.main $k $j $i
        done
    done
done