#!/bin/bash
#SBATCH --job-name=laplace_4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL

module load GCC

source /mnt/nfs/home/c2029946/Code/PythonProjects/pyrlmala/.venv/bin/activate

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Running on node: ${SLURMD_NODENAME}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Allocated MEM: 32G"
echo "Starting python script..."

python -c "
import psutil, os
mem = psutil.virtual_memory()
print(f'Node Total Mem: {mem.total / 1024 / 1024:.2f} MB')
"

python const_run.py

curl -d "✅ SLURM Job ${SLURM_JOB_ID} Finished Successfully" https://ntfy.greenlimes.top/asus

echo "Finished Job"
exit 0
