#!/bin/bash
#SBATCH --job-name=recon_cpd_pipeline
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=36:00:00
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_recon_cpd_pipeline_%A.out
#SBATCH --error=slurm_recon_cpd_pipeline_%A.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

echo "Starting reconstruction CPD pipeline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project dir: $PROJECT_DIR"

run_python() {
  local title="$1"
  shift
  echo
  echo "============================================================"
  echo "$title"
  echo "============================================================"

  srun apptainer exec \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python "$@"
}

# 1) Generate reconstruction densities
run_python "Step 1/4: Generate reconstruction densities" \
  Data_exploration/densities/generate_test_cpd_densities.py

# 2) Extract +/-1000bp centromere windows for reconstruction data
run_python "Step 2/4: Extract reconstruction centromere windows" \
  Utils/extract_centromere_windows.py reconstruction

# 3) Run CPD for both ZINB and Gaussian
run_python "Step 3/4: Run CPD (ZINB + Gaussian)" \
  Signal_processing/sliding_mean/sliding_mean_reconstruction.py

# 4) Evaluate performance and generate separate plots for ZINB and Gaussian
run_python "Step 4/4: Evaluate performance (separate methods)" \
  Signal_processing/evaluation/SATAY_performance.py \
  --results_base Signal_processing/Results/reconstruction_cpd \
  --window_folder Data/reconstruction_cpd_test_all_chrom/centromere_window \
  --output_root Signal_processing/Results/reconstruction_cpd/evaluation \
  --methods ZINB Gaussian \
  --saturation_levels 0 1 2 3 4 5 6 7

echo
echo "Pipeline completed successfully."
echo "CPD results: Signal_processing/Results/reconstruction_cpd"
echo "Evaluation:  Signal_processing/Results/reconstruction_cpd/evaluation"
