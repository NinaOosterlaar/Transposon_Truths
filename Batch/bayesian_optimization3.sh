#!/bin/sh
#SBATCH --job-name=1bayesOpt
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=36:00:00
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_1bayesopt_%j.out
#SBATCH --error=slurm_1bayesopt_%j.err

set -euo pipefail



export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"


srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/training/bayesian_hyperparameter.py \
    --n_calls 150 \
    --n_initial_points 20 \
    --random_state 42 \
    --n_jobs 1 \
    --metric "zinb_nll" \

