#!/bin/bash
#SBATCH --job-name=run-generate_sensitivity
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_sensitivity.py

cd sensitivity
rm -rf test-*
cd -

bash submit_sensitivity.sh

echo "Finished Job"
exit 0
