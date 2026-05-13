"""
Apply sliding mean CPD on centromere window reconstruction data.

Processes reconstructed data and applies Gaussian sliding mean CPD
without preprocessing (no moving average).
"""

import numpy as np
import os
import sys
import pandas as pd
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path
from Signal_processing.CPD_algorithms.sliding_other.sliding_mean_CPD import sliding_mean_CPD


def read_centromere_window_csv(csv_path):
    """Read a centromere window CSV file and extract the value column."""
    df = pd.read_csv(csv_path)
    
    if df.empty:
        return np.array([], dtype=float)
    
    if "value" in df.columns:
        values = pd.to_numeric(df["value"], errors="coerce")
    elif len(df.columns) >= 2:
        values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    else:
        raise ValueError(f"Could not find a usable signal column in {csv_path}")
    
    values = values.dropna().to_numpy(dtype=float)
    return values


def save_results(output_path, change_points, means, sigmas, window_size, overlap, threshold):
    """Save change point detection results to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for cp in change_points:
            f.write(f"{cp}\n")
        f.write(f"means: {means}\n")
        f.write(f"sigmas: {sigmas}\n")
        f.write(f"window_size: {window_size}, overlap: {overlap}, threshold: {threshold}\n")


def process_chromosome_file(csv_path, output_dir, chromosome, window_size, overlap, 
                            thresholds, saturation_level, strain):
    """Process a single chromosome CSV file with CPD."""
    # Read the data
    data = read_centromere_window_csv(csv_path)
    
    if data.size == 0:
        print(f"    Warning: No valid data in {csv_path}")
        return
    
    # Create output directory structure
    chrom_output_dir = os.path.join(
        output_dir, 
        str(saturation_level), 
        strain,
        f"Chr{chromosome}",
        f"Chr{chromosome}_centromere_window",
        f"window{window_size}"
    )
    
    # Process for each threshold
    for threshold in thresholds:
        change_points, means, sigmas = sliding_mean_CPD(
            data, window_size, overlap, threshold
        )
        
        # Create output filename
        output_filename = f"Chr{chromosome}_centromere_window_ws{window_size}_ov{int(overlap*100)}_th{threshold:.2f}.txt"
        output_path = os.path.join(chrom_output_dir, output_filename)
        
        # Save results
        save_results(output_path, change_points, means, sigmas, window_size, overlap, threshold)


def process_strain(strain_path, strain_name, output_dir, window_size, overlap, 
                   thresholds, saturation_level):
    """Process all chromosome files for a single strain."""
    # Get all chromosome CSV files
    csv_files = sorted(Path(strain_path).glob("Chr*_centromere_window.csv"))
    
    if not csv_files:
        print(f"  Warning: No chromosome files found in {strain_path}")
        return
    
    print(f"  Processing strain {strain_name} ({len(csv_files)} chromosomes)")
    
    processed = 0
    for csv_file in csv_files:
        # Extract chromosome name from filename
        chrom_name = csv_file.stem.replace("_centromere_window", "").replace("Chr", "")
        
        process_chromosome_file(
            str(csv_file), 
            output_dir, 
            chrom_name, 
            window_size, 
            overlap, 
            thresholds,
            saturation_level,
            strain_name
        )
        processed += 1
    
    return processed


def process_saturation_level(saturation_path, saturation_level, output_dir, 
                             window_size, overlap, thresholds):
    """Process all strains for a single saturation level."""
    # Get all strain folders
    strain_folders = [
        d for d in sorted(Path(saturation_path).iterdir()) 
        if d.is_dir() and d.name.startswith("yEK23_")
    ]
    
    if not strain_folders:
        print(f"Warning: No strain folders found in {saturation_path}")
        return
    
    print(f"\nProcessing saturation level {saturation_level} ({len(strain_folders)} strains)")
    
    total_processed = 0
    for strain_folder in strain_folders:
        count = process_strain(
            str(strain_folder),
            strain_folder.name,
            output_dir,
            window_size,
            overlap,
            thresholds,
            saturation_level
        )
        if count:
            total_processed += count
    
    print(f"  Total chromosomes processed: {total_processed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply Gaussian sliding mean CPD to AE reconstruction centromere windows."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Data/reconstruction_cpd_test_all_chrom/centromere_window",
        help="Input directory containing reconstruction centromere-window saturation folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Signal_processing/results_new/Gaussian_AE_CPD",
        help="Output directory for Gaussian CPD results.",
    )
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--threshold_start", type=float, default=0.0)
    parser.add_argument("--threshold_end", type=float, default=20.0)
    parser.add_argument("--threshold_step", type=float, default=1.0)
    parser.add_argument(
        "--saturation_levels",
        type=int,
        nargs="+",
        default=list(range(8)),
        help="Saturation levels to process.",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    if args.overlap < 0 or args.overlap >= 1:
        raise ValueError("--overlap must be in [0, 1)")
    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be > 0")
    if args.window_size <= 1:
        raise ValueError("--window_size must be > 1")

    thresholds = np.arange(
        args.threshold_start,
        args.threshold_end + (args.threshold_step * 0.5),
        args.threshold_step,
        dtype=float
    )
    
    print("="*80)
    print("Gaussian Sliding Mean CPD on Reconstruction Data")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Overlap: {args.overlap}")
    print(f"Thresholds: {args.threshold_start} to {args.threshold_end} (step {args.threshold_step})")
    print(f"Saturation levels: {args.saturation_levels}")
    print(f"Preprocessing: None (raw data)")
    print("="*80)
    
    # Process each saturation level
    for saturation_level in args.saturation_levels:
        saturation_path = os.path.join(args.input_dir, str(saturation_level))
        
        if not os.path.exists(saturation_path):
            print(f"\nWarning: Saturation level {saturation_level} folder not found at {saturation_path}")
            continue
        
        process_saturation_level(
            saturation_path,
            saturation_level,
            args.output_dir,
            args.window_size,
            args.overlap,
            thresholds
        )
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
