#!/bin/sh
#SBATCH --job-name=train_recon_cpd
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_train_recon_cpd_%j.out
#SBATCH --error=slurm_train_recon_cpd_%j.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"


# Step 1: Train model on all chromosomes
echo "=========================================="
echo "STEP 1: Training model on all chromosomes"
echo "=========================================="
srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/train_all_chromosomes.py

echo ""
echo "Training completed successfully!"
echo ""

# Step 2: Reconstruct test_CPD data using the trained model
echo "=========================================="
echo "STEP 2: Reconstructing test_CPD data"
echo "=========================================="
srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/reconstruct_cpd_test.py


echo "Output locations:"
echo "  - Model: AE/results/models/"
echo "  - Reconstructions: Data/reconstruction_cpd_test/"
echo ""
