#!/bin/bash
#SBATCH --job-name=run-generate_naive_step_size
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_naive_step_size.py

cd whole_results_naive_step_size
rm -rf test-*
cd -

bash submit_naive_step_size.sh

echo "Finished Job"
exit 0
