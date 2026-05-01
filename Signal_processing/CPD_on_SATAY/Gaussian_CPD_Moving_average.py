#!/usr/bin/env python3
"""
Apply sliding mean change point detection on centromere window data.

This script:
1. Preprocesses data with a 20-point moving average
2. Processes all saturation levels (folders 0-7) in Data/test_CPD/centromere_windows
3. Processes all strains (yEK23_1 through yEK23_6) within each saturation level
4. Applies sliding_mean_CPD to all chromosome CSV files
5. Saves results in an organized structure for downstream evaluation

Author: Generated for thesis analysis
Date: April 2026
"""

import numpy as np
import os
import sys
import pandas as pd
import argparse
from pathlib import Path

# Add parent directory to path to import from sliding_mean_CPD
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path
from Signal_processing.CPD_algorithms.sliding_other.sliding_mean_CPD import sliding_mean_CPD


def moving_average(data, window_size):
    """Apply a moving average filter to smooth the data, ignoring zero values.
    
    Args:
        data: 1D numpy array of values
        window_size: Size of the moving average window
    
    Returns:
        Smoothed data (same length as input), averaging only non-zero values
    """
    if window_size <= 1:
        return data
    
    result = np.zeros_like(data, dtype=float)
    half_window = window_size // 2
    
    for i in range(len(data)):
        # Define window boundaries (centered window)
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        
        # Get window data and filter out zeros
        window_data = data[start:end]
        non_zero_values = window_data[window_data != 0]
        
        # Calculate average of non-zero values only
        if len(non_zero_values) > 0:
            result[i] = np.mean(non_zero_values)
        else:
            # If all values in window are zero, keep zero
            result[i] = 0.0
    
    return result


def read_centromere_window_csv(csv_path):
    """Read a centromere window CSV file and extract the value column.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        numpy array of values
    """
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
    """Save change point detection results to a file.
    
    Args:
        output_path: Full path to the output file
        change_points: List of detected change point positions
        means: List of window means
        sigmas: List of window standard deviations
        window_size: Window size used
        overlap: Overlap fraction used
        threshold: Threshold value used
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for cp in change_points:
            f.write(f"{cp}\n")
        f.write(f"means: {means}\n")
        f.write(f"sigmas: {sigmas}\n")
        f.write(f"window_size: {window_size}, overlap: {overlap}, threshold: {threshold}\n")


def process_chromosome_file(csv_path, output_dir, chromosome, window_size, overlap, 
                            thresholds, moving_avg_size, saturation_level, strain):
    """Process a single chromosome CSV file with CPD.
    
    Args:
        csv_path: Path to the chromosome CSV file
        output_dir: Base output directory
        chromosome: Chromosome name (e.g., "I", "II", "III")
        window_size: Sliding window size for CPD
        overlap: Window overlap fraction
        thresholds: Array of threshold values to test
        moving_avg_size: Size of moving average window for preprocessing
        saturation_level: Saturation level (0-7)
        strain: Strain name (e.g., "yEK23_1")
    """
    # Read the data
    data = read_centromere_window_csv(csv_path)
    
    if data.size == 0:
        print(f"    Warning: No valid data in {csv_path}")
        return
    
    # Apply moving average preprocessing
    smoothed_data = moving_average(data, moving_avg_size)
    
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
            smoothed_data, window_size, overlap, threshold
        )
        
        # Create output filename
        output_filename = f"Chr{chromosome}_centromere_window_ws{window_size}_ov{int(overlap*100)}_th{threshold:.2f}.txt"
        output_path = os.path.join(chrom_output_dir, output_filename)
        
        # Save results
        save_results(output_path, change_points, means, sigmas, window_size, overlap, threshold)


def process_strain(strain_path, strain_name, output_dir, window_size, overlap, 
                   thresholds, moving_avg_size, saturation_level):
    """Process all chromosome files for a single strain.
    
    Args:
        strain_path: Path to the strain folder
        strain_name: Name of the strain (e.g., "yEK23_1")
        output_dir: Base output directory
        window_size: Sliding window size for CPD
        overlap: Window overlap fraction
        thresholds: Array of threshold values
        moving_avg_size: Size of moving average window
        saturation_level: Saturation level (0-7)
    """
    # Get all chromosome CSV files
    csv_files = sorted(Path(strain_path).glob("Chr*_centromere_window.csv"))
    
    if not csv_files:
        print(f"  Warning: No chromosome files found in {strain_path}")
        return
    
    print(f"  Processing strain {strain_name} ({len(csv_files)} chromosomes)")
    
    for csv_file in csv_files:
        # Extract chromosome name from filename (e.g., "ChrI_centromere_window.csv" -> "I")
        chrom_name = csv_file.stem.replace("_centromere_window", "").replace("Chr", "")
        
        process_chromosome_file(
            str(csv_file), 
            output_dir, 
            chrom_name, 
            window_size, 
            overlap, 
            thresholds, 
            moving_avg_size,
            saturation_level,
            strain_name
        )


def process_saturation_level(saturation_path, saturation_level, output_dir, 
                             window_size, overlap, thresholds, moving_avg_size):
    """Process all strains for a single saturation level.
    
    Args:
        saturation_path: Path to the saturation level folder
        saturation_level: Saturation level number (0-7)
        output_dir: Base output directory
        window_size: Sliding window size for CPD
        overlap: Window overlap fraction
        thresholds: Array of threshold values
        moving_avg_size: Size of moving average window
    """
    # Get all strain folders
    strain_folders = [
        d for d in sorted(Path(saturation_path).iterdir()) 
        if d.is_dir() and d.name.startswith("yEK23_")
    ]
    
    if not strain_folders:
        print(f"Warning: No strain folders found in {saturation_path}")
        return
    
    print(f"\nProcessing saturation level {saturation_level} ({len(strain_folders)} strains)")
    
    for strain_folder in strain_folders:
        process_strain(
            str(strain_folder),
            strain_folder.name,
            output_dir,
            window_size,
            overlap,
            thresholds,
            moving_avg_size,
            saturation_level
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply sliding mean CPD on centromere windows with moving average preprocessing."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="Data/test_CPD/centromere_windows",
        help="Input directory containing saturation level folders (default: Data/test_CPD/centromere_windows)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="Signal_processing/results_new/Gaussian_MA_CPD",
        help="Output directory for results (default: Signal_processing/results_new/Gaussian_MA_CPD)"
    )
    parser.add_argument(
        "--window_size", 
        type=int, 
        default=100,
        help="Sliding window size for CPD (default: 100)"
    )
    parser.add_argument(
        "--overlap", 
        type=float, 
        default=0.5,
        help="Window overlap fraction, must be in [0, 1) (default: 0.5)"
    )
    parser.add_argument(
        "--threshold_start", 
        type=float, 
        default=0.0,
        help="Start of threshold range (default: 0.0)"
    )
    parser.add_argument(
        "--threshold_end", 
        type=float, 
        default=20.0,
        help="End of threshold range (default: 20.0)"
    )
    parser.add_argument(
        "--threshold_step", 
        type=float, 
        default=1.0,
        help="Step for threshold range (default: 1.0)"
    )
    parser.add_argument(
        "--moving_avg_size", 
        type=int, 
        default=20,
        help="Size of moving average window for preprocessing (default: 20)"
    )
    parser.add_argument(
        "--saturation_levels", 
        type=int, 
        nargs="+", 
        default=list(range(8)),
        help="Saturation levels to process (default: 0 1 2 3 4 5 6 7)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate arguments
    if args.overlap < 0 or args.overlap >= 1:
        raise ValueError("--overlap must be in [0, 1)")
    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be > 0")
    if args.window_size <= 1:
        raise ValueError("--window_size must be > 1")
    if args.moving_avg_size < 1:
        raise ValueError("--moving_avg_size must be >= 1")
    
    # Create threshold array
    thresholds = np.arange(
        args.threshold_start,
        args.threshold_end + (args.threshold_step * 0.5),
        args.threshold_step,
        dtype=float
    )
    
    print("="*80)
    print("Sliding Mean CPD on Centromere Windows")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Overlap: {args.overlap}")
    print(f"Thresholds: {args.threshold_start} to {args.threshold_end} (step {args.threshold_step})")
    print(f"Moving average size: {args.moving_avg_size}")
    print(f"Saturation levels: {args.saturation_levels}")
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
            thresholds,
            args.moving_avg_size
        )
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
