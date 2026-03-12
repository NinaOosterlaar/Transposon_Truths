#!/bin/bash
#SBATCH --job-name=linear
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

srun apptainer exec \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python Utils/reader.py
