#!/bin/sh
#SBATCH --job-name=reconstruct_cpd
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_reconstruct_cpd_%j.out
#SBATCH --error=slurm_reconstruct_cpd_%j.err

set -euo pipefail

# Model to use for reconstruction
MODEL_PATH="AE/results/models/ZINBAE_layers1600_ep144_noise0.150_muoff0.000_all.pt"

# Input and output directories
TEST_CPD_FOLDER="Data/test_CPD"
OUTPUT_DIR="Data/reconstruction_cpd_test_all_chrom"

# Model parameters (must match training configuration)
NOISE_LEVEL=0.15
STEP_SIZE=900

echo "================================================================"
echo "Reconstructing test_CPD data using trained model"
echo "================================================================"
echo "Model: ${MODEL_PATH}"
echo "Input folder: ${TEST_CPD_FOLDER}"
echo "Output folder: ${OUTPUT_DIR}"
echo "Noise level: ${NOISE_LEVEL}"
echo "Step size: ${STEP_SIZE}"
echo "================================================================"

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/reconstruct_all_chromosomes.py \
    --model_path "$MODEL_PATH" \
    --test_cpd_folder "$TEST_CPD_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --noise_level "$NOISE_LEVEL" \
    --step_size "$STEP_SIZE"

echo "================================================================"
echo "Reconstruction complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "================================================================"
