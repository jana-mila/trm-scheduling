#!/bin/bash
#SBATCH --job-name=trm-jobshop
#SBATCH --output=logs/%x-%j.txt
#SBATCH --error=logs/%x-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=8G

mkdir -p logs

cd $SLURM_SUBMIT_DIR

echo "Running from $(SLURM_SUBMIT_DIR)"

module load python/3.10
source .venv/bin/activate
srun python train.py "$@"