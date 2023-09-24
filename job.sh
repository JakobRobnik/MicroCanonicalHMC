#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q preempt
#SBATCH -t 00:15:00
#SBATCH -A m_4031g

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
python3 -m applications.lattice_field_theories.potential
#srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1  python3 -m applications.lattice_field_theories.potential
