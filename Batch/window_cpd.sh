#!/bin/bash
#SBATCH --job-name=window_performance
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=36:00:00
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_window_perf_%A.out
#SBATCH --error=slurm_window_perf_%A.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

echo "Starting window performance analysis on 10 synthetic datasets"
echo "Job ID: $SLURM_JOB_ID"

srun apptainer exec \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python Signal_processing/final/window_performance.py


