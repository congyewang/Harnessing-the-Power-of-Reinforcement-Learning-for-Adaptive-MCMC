#!/bin/bash
#SBATCH --job-name=run-generate_batch_discount_factor
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_batch_discount_factor.py

cd whole_results
rm -rf test-*
cd -

bash submit_batch_discount_factor.sh

echo "Finished Job"
exit 0
