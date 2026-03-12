#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

BINS=(5 10 20 50 75 100)
MA_FLAGS=("--no_moving_average")

BIN_IDX=$((SLURM_ARRAY_TASK_ID / ${#MA_FLAGS[@]}))
MA_IDX=$((SLURM_ARRAY_TASK_ID % ${#MA_FLAGS[@]}))

BIN=${BINS[$BIN_IDX]}
MA_FLAG=${MA_FLAGS[$MA_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: --bin ${BIN} ${MA_FLAG}"
srun apptainer exec \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/preprocessing/preprocessing.py \
    --bin "$BIN" \
    --zinb_mode \
    "${MA_FLAG}" \
    --train_val_test_split 0.8 0.1 0.1