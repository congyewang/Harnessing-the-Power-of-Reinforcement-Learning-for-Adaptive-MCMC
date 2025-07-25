#!/bin/bash
#SBATCH --job-name=run-generate_discount_factor
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_discount_factor.py

cd whole_results_discount_factor
rm -rf test-*
cd -

bash submit_discount_factor.sh

echo "Finished Job"
exit 0
