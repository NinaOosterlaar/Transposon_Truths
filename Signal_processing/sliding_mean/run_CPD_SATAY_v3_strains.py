"""
Run sliding_ZINB_CPD_v3_SATAY.py on all strains from Data/combined_strains.
Processes all chromosomes for each strain with their corresponding density files.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def read_count_data(csv_file):
    """Read count data from a CSV file."""
    df = pd.read_csv(csv_file)
    if "Value" in df.columns:
        return df["Value"].astype(float).to_numpy()
    if len(df.columns) < 2:
        raise ValueError(f"Expected at least two columns in {csv_file}")
    return df.iloc[:, 1].astype(float).to_numpy()


def remove_problematic_positions(data, chrom_name):
    """
    Remove known problematic positions by setting them to zero.
    
    Args:
        data: Array of count values
        chrom_name: Name of chromosome (e.g., 'ChrXV')
    
    Returns:
        cleaned_data: Data with problematic positions set to zero
        n_removed: Number of positions removed
    """
    data_cleaned = np.array(data, copy=True)
    n_removed = 0
    
    # Known problematic positions: (chromosome, position)
    problematic_positions = [
        ('ChrXV', 565596),
    ]
    
    for prob_chrom, prob_pos in problematic_positions:
        if chrom_name == prob_chrom and prob_pos < len(data_cleaned):
            if data_cleaned[prob_pos] != 0:
                data_cleaned[prob_pos] = 0
                n_removed += 1
    
    return data_cleaned, n_removed


def compute_global_outlier_threshold(strain_folder):
    """Compute 95th percentile threshold across all chromosomes in a strain (non-zero values only)."""
    all_data = []
    chromosome_files = sorted(strain_folder.glob("Chr*_distances.csv"))
    total_removed = 0
    
    for chrom_file in chromosome_files:
        try:
            # Extract chromosome name
            chrom_name = chrom_file.stem.replace("_distances", "")
            
            # Read data
            data = read_count_data(chrom_file)
            
            # Remove problematic positions
            data, n_removed = remove_problematic_positions(data, chrom_name)
            total_removed += n_removed
            
            all_data.extend(data)
        except Exception as e:
            print(f"    Warning: Could not read {chrom_file.name}: {e}")
    
    if not all_data:
        return None
    
    all_data = np.array(all_data)
    
    # Compute threshold only on non-zero values
    non_zero_data = all_data[all_data > 0]
    if len(non_zero_data) == 0:
        return None
    
    threshold = np.quantile(non_zero_data, 0.95)  # 95th percentile = top 5%
    n_outliers = np.sum(all_data > threshold)
    
    return threshold, n_outliers, len(all_data), total_removed


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run sliding_ZINB_CPD_v3_SATAY.py on all strains from Data/combined_strains."
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[3, 5, 10, 15],
        help="List of thresholds to test (e.g., 3 5 10 15). Default: [3, 5, 10, 15]",
    )
    args = parser.parse_args()
    
    # Paths
    strains_data = PROJECT_ROOT / "Data" / "combined_strains"
    centromere_base = PROJECT_ROOT / "Data_exploration" / "results" / "densities" / "centromere_strains" / "combined_Datasets_Boolean_True_bin_10000_absolute"
    nucleosome_base = PROJECT_ROOT / "Data_exploration" / "results" / "densities" / "nucleosome_strains" / "combined_Datasets_Boolean_True"
    output_base = PROJECT_ROOT / "Signal_processing" / "strains"
    cpd_script = PROJECT_ROOT / "Signal_processing" / "sliding_mean" / "sliding_ZINB_CPD_v3_SATAY.py"
    
    if not cpd_script.exists():
        print(f"Error: CPD script not found at {cpd_script}")
        return
    
    if not strains_data.exists():
        print(f"Error: Strains data folder not found at {strains_data}")
        return
    
    # Parameters
    window_sizes = [100]
    overlap = 0.5
    thresholds = args.thresholds  # Use thresholds from command line
    theta_block_size = 2000  # Local theta estimation in 2000bp blocks
    n_workers = 1
    timeout_seconds = 1800  # 30 minutes per chromosome
    
    print("=" * 80)
    print("Running sliding_ZINB_CPD_v3_SATAY.py on all strains")
    print("=" * 80)
    print(f"Input data:   {strains_data}")
    print(f"Centromeres:  {centromere_base}")
    print(f"Nucleosomes:  {nucleosome_base}")
    print(f"Results:      {output_base}")
    print(f"Script:       {cpd_script}")
    print(f"Thresholds:   {thresholds}")
    print()
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    # Get all strain folders
    strain_folders = sorted([d for d in strains_data.iterdir() if d.is_dir() and d.name.startswith("strain_")])
    
    if not strain_folders:
        print("Error: No strain folders found in Data/combined_strains")
        return
    
    print(f"Found {len(strain_folders)} strains to process:\n")
    for strain_folder in strain_folders:
        print(f"  - {strain_folder.name}")
    print()
    
    # Process each strain
    for strain_folder in strain_folders:
        strain_name = strain_folder.name
        
        print(f"\n{'='*80}")
        print(f"Processing {strain_name}")
        print(f"{'='*80}")
        
        # Find corresponding density files
        # Format: dataset-strain_FD_combined_centromere_density_Boolean_True_bin_10000_absolute.csv
        centromere_file = centromere_base / f"dataset-{strain_name}_combined_centromere_density_Boolean_True_bin_10000_absolute.csv"
        # Format: dataset-strain_FD_combined_Boolean_True_nucleosome_density.csv
        nucleosome_file = nucleosome_base / f"dataset-{strain_name}_combined_Boolean_True_nucleosome_density.csv"
        
        # Check if density files exist
        if not centromere_file.exists():
            print(f"  ⚠ Warning: Centromere density file not found:")
            print(f"    {centromere_file.name}")
            print(f"  Skipping {strain_name}")
            total_skipped += 1
            continue
        
        if not nucleosome_file.exists():
            print(f"  ⚠ Warning: Nucleosome density file not found:")
            print(f"    {nucleosome_file.name}")
            print(f"  Skipping {strain_name}")
            total_skipped += 1
            continue
        
        print(f"  ✓ Centromere file: {centromere_file.name}")
        print(f"  ✓ Nucleosome file: {nucleosome_file.name}")
        print()
        
        # Find all chromosome files
        chromosome_files = sorted(strain_folder.glob("Chr*_distances.csv"))
        
        if not chromosome_files:
            print(f"  ⚠ No chromosome files found in {strain_name}")
            total_skipped += 1
            continue
        
        # Compute global outlier threshold across all chromosomes for this strain
        print(f"  Computing global outlier threshold across all chromosomes...")
        threshold_result = compute_global_outlier_threshold(strain_folder)
        
        if threshold_result is None:
            print(f"  ⚠ Could not compute outlier threshold for {strain_name}")
            total_skipped += 1
            continue
        
        outlier_threshold, n_outliers, n_total, n_problematic = threshold_result
        if n_problematic > 0:
            print(f"  Removed {n_problematic} problematic position(s) (set to zero)")
        print(f"  Global 99th percentile (non-zero): {outlier_threshold:.1f}")
        print(f"  Total outliers to cap: {n_outliers} ({100*n_outliers/n_total:.2f}%) across all chromosomes")
        print()
        
        print(f"  Processing {len(chromosome_files)} chromosomes:")
        print(f"  Thresholds: {thresholds}")
        print()
        
        # Process each chromosome
        for chrom_file in chromosome_files:
            # Extract chromosome name (e.g., "ChrI" from "ChrI_distances.csv")
            chrom_name = chrom_file.stem.replace("_distances", "")
            
            # Output folder: Signal_processing/strains/{strain_name}/{chromosome}/
            output_folder = output_base / strain_name / chrom_name
            
            # Build command - pass thresholds as multiple arguments
            cmd = [
                sys.executable,
                str(cpd_script),
                str(chrom_file),
                "--output_folder", str(output_folder),
                "--window_sizes", str(window_sizes[0]),
                "--overlap", str(overlap),
                "--theta_block_size", str(theta_block_size),
                "--thresholds"
            ]
            # Add each threshold as a separate value
            cmd.extend([str(t) for t in thresholds])
            cmd.extend([
                "--outlier_threshold", str(outlier_threshold),
                "--n_workers", str(n_workers),
                "--nucleosome_file", str(nucleosome_file),
                "--centromere_file", str(centromere_file),
            ])
            
            # Run the command
            try:
                print(f"    Processing {chrom_name}...")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT,
                    timeout=timeout_seconds
                )
                
                if result.returncode == 0:
                    # Print key diagnostic output (theta, outliers)
                    if result.stdout:
                        for line in result.stdout.strip().split('\n'):
                            # Show lines about outlier removal and theta estimation
                            if any(keyword in line.lower() for keyword in 
                                   ['outlier', 'theta', 'estimated', 'block', 'capped']):
                                print(f"      {line}")
                    print(f"    ✓ {chrom_name} complete")
                    total_processed += 1
                else:
                    print(f"    ✗ {chrom_name} failed (exit code {result.returncode})")
                    total_errors += 1
                    if result.stderr:
                        print(f"      Error output:")
                        for line in result.stderr.strip().split('\n')[:10]:  # Show first 10 lines
                            print(f"        {line}")
                        if len(result.stderr.strip().split('\n')) > 10:
                            print(f"        ... (truncated)")
                    
            except subprocess.TimeoutExpired:
                print(f"    ✗ timeout after {timeout_seconds}s")
                total_errors += 1
            except Exception as e:
                print(f"    ✗ {str(e)[:80]}")
                total_errors += 1
        
        print(f"\n  Completed {strain_name}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓ CPD Analysis Complete!")
    print("=" * 80)
    print(f"  Chromosomes processed successfully: {total_processed}")
    print(f"  Chromosomes with errors:            {total_errors}")
    print(f"  Strains skipped:                    {total_skipped}")
    print(f"\n  Results saved to: {output_base}")
    print("=" * 80)


if __name__ == "__main__":
    main()
