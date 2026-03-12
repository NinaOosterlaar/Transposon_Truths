#!/bin/sh
#SBATCH --job-name=AE
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0

set -euo pipefail

NOISE_LEVEL=0.15
MU_OFFSETs=(0)
MU_OFFSET="${MU_OFFSETs[$SLURM_ARRAY_TASK_ID]}"

echo "Running with noise_level=${NOISE_LEVEL}"
echo "Running with mu_offset=${MU_OFFSET}"

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/main.py --noise_level "$NOISE_LEVEL" --mu_offset "$MU_OFFSET"