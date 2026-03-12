#!/bin/sh
#SBATCH --job-name=AE
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

# Models to train
MODELS=("ZINBAE" "ZINBVAE")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: Training ${MODEL}"
srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/training/training.py \
    --model "${MODEL}" \
    --use_conv \
    --filename "BinSize10_MovingAvgFalse_ZINB_" \
    --results_subdir "this_is_exp+1" \
    --epochs 100 \
    --sample_fraction 1.0 \
    --denoise_percent 0.3 \
