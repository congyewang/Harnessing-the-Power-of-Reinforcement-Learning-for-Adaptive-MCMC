#!/bin/bash
#SBATCH --job-name=run-generate_actor_888
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_actor_888.py

cd whole_results_actor_888
rm -rf test-*
cd -

bash submit_actor_888.sh

echo "Finished Job"
exit 0
