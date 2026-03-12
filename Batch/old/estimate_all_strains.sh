#!/bin/sh
#SBATCH --job-name=ZINB_strains
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=36:00:00
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_estimate_strains_%j.out
#SBATCH --error=slurm_estimate_strains_%j.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

echo "Starting ZINB parameter estimation for all strains..."
echo "Window size: 2000"

srun apptainer exec \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python Signal_processing/ZINB_MLE/estimate_all_strains.py

echo "ZINB estimation completed!"
