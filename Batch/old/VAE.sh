#!/bin/bash
#SBATCH --job-name=VAE
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --qos=long
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"

# Your project directory on the cluster
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

module use /opt/insy/modulefiles  # If not already
module load cuda/12.4

cd "$PROJECT_DIR"

srun apptainer exec \
    --nv \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python AE/training.py --model VAE