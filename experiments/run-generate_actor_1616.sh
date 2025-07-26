#!/bin/bash
#SBATCH --job-name=run-generate_actor_1616
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python generate_actor_1616.py

cd whole_results_actor_1616
rm -rf test-*
cd -

bash submit_actor_1616.sh

echo "Finished Job"
exit 0
