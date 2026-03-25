"""
Run sliding_ZINB_CPD_v3_SATAY.py on all strains from Data/combined_strains.
Processes all chromosomes for each strain with their corresponding density files.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
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
    
    print("=" * 80)
    print("Running sliding_ZINB_CPD_v3_SATAY.py on all strains")
    print("=" * 80)
    print(f"Input data:   {strains_data}")
    print(f"Centromeres:  {centromere_base}")
    print(f"Nucleosomes:  {nucleosome_base}")
    print(f"Results:      {output_base}")
    print(f"Script:       {cpd_script}")
    print()
    
    # Parameters
    window_sizes = [100]
    overlap = 0.5
    threshold_start = 0.0
    threshold_end = 40.0
    threshold_step = 1.0
    n_workers = 1
    timeout_seconds = 1800  # 30 minutes per chromosome
    
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
        
        print(f"  Processing {len(chromosome_files)} chromosomes:")
        print()
        
        # Process each chromosome
        for chrom_file in chromosome_files:
            # Extract chromosome name (e.g., "ChrI" from "ChrI_distances.csv")
            chrom_name = chrom_file.stem.replace("_distances", "")
            
            # Output folder: Signal_processing/strains/{strain_name}/{chromosome}/
            output_folder = output_base / strain_name / chrom_name
            
            # Build command
            cmd = [
                sys.executable,
                str(cpd_script),
                str(chrom_file),
                "--output_folder", str(output_folder),
                "--window_sizes", str(window_sizes[0]),
                "--overlap", str(overlap),
                "--threshold_start", str(threshold_start),
                "--threshold_end", str(threshold_end),
                "--threshold_step", str(threshold_step),
                "--n_workers", str(n_workers),
                "--nucleosome_file", str(nucleosome_file),
                "--centromere_file", str(centromere_file),
            ]
            
            # Run the command
            try:
                print(f"    Processing {chrom_name}...", end=" ", flush=True)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT,
                    timeout=timeout_seconds
                )
                
                if result.returncode == 0:
                    print("✓")
                    total_processed += 1
                else:
                    print(f"✗ (exit code {result.returncode})")
                    total_errors += 1
                    if result.stderr:
                        print(f"      Error output:")
                        for line in result.stderr.strip().split('\n')[:10]:  # Show first 10 lines
                            print(f"        {line}")
                        if len(result.stderr.strip().split('\n')) > 10:
                            print(f"        ... (truncated)")
                    
            except subprocess.TimeoutExpired:
                print(f"✗ (timeout after {timeout_seconds}s)")
                total_errors += 1
            except Exception as e:
                print(f"✗ ({str(e)[:80]})")
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
