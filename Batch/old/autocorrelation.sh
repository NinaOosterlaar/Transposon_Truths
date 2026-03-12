#!/bin/bash
#SBATCH --job-name=autocorrelation
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

ZERO_FLAGS=("--zeros" "")
ZERO_FLAG=${ZERO_FLAGS[$SLURM_ARRAY_TASK_ID]}

echo "Running preprocessing with zeros flag: ${ZERO_FLAGS[$SLURM_ARRAY_TASK_ID]}"

srun apptainer exec \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python Signal_processing/autocorrelation.py "Data/distances_with_zeros" --max_lag 2000 $ZERO_FLAG --output_folder "Signal_processing/results/autocorrelation/uncombined" --plot