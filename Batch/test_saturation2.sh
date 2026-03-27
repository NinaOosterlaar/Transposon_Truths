#!/bin/sh
#SBATCH --job-name=saturation
#SBATCH --partition=general
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

# Map array index to k value (0->1, 1->2, 2->3, etc.)
K_VALUE=$((SLURM_ARRAY_TASK_ID + 1))

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

echo "Running saturation test for k=${K_VALUE} (array task ${SLURM_ARRAY_TASK_ID})"

srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/test_AE/test_saturation.py --k ${K_VALUE}