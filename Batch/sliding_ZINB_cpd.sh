#!/bin/bash
#SBATCH --job-name=sliding_ZINB_cpd
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

DATASET_NAME="SATAY_synthetic"
INPUT_FILE="Signal_processing/sample_data/SATAY_synthetic.csv"
THETAS=("0" "0.5" "1" "5")  # Example theta values for different runs

THETA_GLOBAL="${THETAS[$SLURM_ARRAY_TASK_ID]}"

echo "Running sliding ZINB CPD on dataset: ${DATASET_NAME}"
echo "Input file: ${INPUT_FILE}"
echo "Theta global: ${THETA_GLOBAL}"

srun apptainer exec \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python Signal_processing/sliding_mean/sliding_ZINB_CPD.py \
    "$INPUT_FILE" \
    --dataset_name "${DATASET_NAME}" \
    --n_workers 4 \
    --output_folder "Signal_processing/results/sliding_mean/sliding_ZINB_CPD/${THETA_GLOBAL}" \
    --theta_global "${THETA_GLOBAL}"

echo "Finished processing ${DATASET_NAME}"