#!/bin/bash
#SBATCH --job-name=const-{{ model_name }}-{{ rl_algorithm }}_{{ mcmc_env }}_seed_{{ random_seed }}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=2-00:00:00
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

python run_pdb_{{ rl_algorithm }}_{{ mcmc_env }}_seed_{{ random_seed }}.py

curl -d "✅ SLURM Job ${SLURM_JOB_NAME} Finished Successfully" https://ntfy.greenlimes.top/asus

echo "Finished Job"
exit 0
