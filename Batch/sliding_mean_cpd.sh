#!/bin/bash
#SBATCH --job-name=sliding_mean_cpd
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=04:00:00
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

# Array of dataset names and their corresponding files
DATASETS=("pretty_data" "realistic_data")
INPUT_FILES=("Signal_processing/sample_data/pretty_data.csv" "Signal_processing/sample_data/realistic_data.csv")

DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}
INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}

echo "Running sliding mean CPD on dataset: ${DATASET_NAME}"
echo "Input file: ${INPUT_FILE}"

srun apptainer exec \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python Signal_processing/sliding_mean/sliding_mean_CPD.py \
    "$INPUT_FILE" \
    --output_folder "Signal_processing/results/sliding_mean_CPD/${DATASET_NAME}" \
    --dataset_name "${DATASET_NAME}"

echo "Finished processing ${DATASET_NAME}"
