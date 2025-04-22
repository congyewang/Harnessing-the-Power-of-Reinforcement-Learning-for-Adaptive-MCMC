#!/bin/bash
#SBATCH --job-name=flex-{{ critic_learning_rate }}_{{ actor_learning_rate }}_{{ tau }}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python search_run_{{ critic_learning_rate }}_{{ actor_learning_rate }}_{{ tau }}.py

curl -d "✅ SLURM Job ${SLURM_JOB_NAME} Finished Successfully" https://ntfy.greenlimes.top/asus

echo "Finished Job"
exit 0
