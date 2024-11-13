#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=jax_distributed_container_demo
#SBATCH --output=output.out
#SBATCH --account=m4031_g
#SBATCH --image=ghcr.io/nvidia/jax:jax
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu

# Run script
srun shifter --module=gpu python3 ./distributed.py
