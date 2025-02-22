#!/bin/bash
#SBATCH --mail-type=FAIL


module load GCC/12.3.0
source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate
python mcmc_const.py

echo Finishing job
exit 0
