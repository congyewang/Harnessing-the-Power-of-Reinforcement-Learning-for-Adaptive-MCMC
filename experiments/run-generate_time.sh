#!/bin/bash
#SBATCH --job-name=run-generate_time
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_time.py

cd whole_results
rm -rf test-*
cd -

bash submit_time.sh

echo "Finished Job"
exit 0
