#!/bin/bash
#SBATCH --job-name=AE_VAE
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

module use /opt/insy/modulefiles
module load cuda/12.4

cd "$PROJECT_DIR"

# Map array index -> model
MODELS=("AE" "VAE")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID})"

srun apptainer exec \
    --nv \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python AE/training.py --model "$MODEL" --binary