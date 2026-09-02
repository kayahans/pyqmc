#!/bin/bash
#SBATCH --job-name abvmc
#SBATCH -p QMCREGULAR
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -A qmc
#SBATCH --ntasks-per-node=128
#SBATCH -t 24:00:00
#SBATCH -o pyqmc.out
#SBATCH -e pyqmc.err
#SBATCH --export=ALL
#SBATCH --mem=0

module load  tbb/2021.10.0 compiler-rt/2023.2.1 mpi/openmpi-x86_64 mkl

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mpirun -n 128 python -u -m mpi4py.futures run.py

