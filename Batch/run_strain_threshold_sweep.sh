#!/bin/bash
#SBATCH --job-name=threshold_sweep
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_threshold_sweep_%j.out
#SBATCH --error=slurm_threshold_sweep_%j.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

# Generate threshold list (1 to 40)
THRESHOLDS=$(seq 1 40 | tr '\n' ' ')

# Step 1: Run CPD analysis
echo "=========================================="
echo "STEP 1: Running CPD Analysis (thresholds 1-40)"
echo "=========================================="
srun apptainer exec \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python Signal_processing/sliding_mean/run_CPD_SATAY_v3_strains.py \
  --thresholds $THRESHOLDS

echo ""
echo "CPD analysis completed successfully!"
echo ""

# Step 2: Run essentiality calculation
echo "=========================================="
echo "STEP 2: Running Essentiality Calculation (thresholds 1-40)"
echo "=========================================="
srun apptainer exec \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python Signal_processing/essentiality_calculation/calculate_strain_essentiality.py \
  --thresholds $THRESHOLDS

echo ""
echo "Essentiality calculation completed successfully!"
echo ""

