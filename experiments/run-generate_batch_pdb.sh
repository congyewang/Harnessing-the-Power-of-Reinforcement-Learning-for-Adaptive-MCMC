#!/bin/bash
#SBATCH --job-name=run-generate_batch_pdb
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_batch_pdb.py

cd whole_results
rm -rf test-*

echo "Finished Job"
exit 0
