#!/bin/bash
#SBATCH --image=ghcr.io/nvidia/jax:jax
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --constraint=cpu

srun shifter --module=cpu python3 ./slow_parallelization.py