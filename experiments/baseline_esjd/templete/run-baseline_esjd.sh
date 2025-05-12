#!/bin/bash
#SBATCH --job-name=baseline_esjd-{{ model_name }}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

python baseline_esjd_run_{{ model_name }}.py

curl -d "✅ SLURM Job ${SLURM_JOB_NAME} Finished Successfully" https://ntfy.greenlimes.top/asus

echo "Finished Job"
exit 0
