#!/bin/bash
#SBATCH --job-name=run-generate_naive_step_size
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_naive_step_size_001.py

cd whole_results_naive_step_size_001
rm -rf test-*
cd -

bash submit_naive_step_size_001.sh

echo "Finished Job"
exit 0
