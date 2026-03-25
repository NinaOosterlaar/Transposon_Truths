"""
Run sliding_ZINB_CPD_v3_SATAY.py on all centromere windows from test_CPD datasets.
Uses the combined density files created for each dataset.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # Paths
    windows_base = PROJECT_ROOT / "Data" / "test_CPD" / "centromere_windows"
    test_cpd_base = PROJECT_ROOT / "Data" / "test_CPD"
    output_base = PROJECT_ROOT / "Signal_processing" / "Results" / "test_CPD_SATAY_CPD_v3"
    cpd_script = PROJECT_ROOT / "Signal_processing" / "sliding_mean" / "sliding_ZINB_CPD_v3_SATAY.py"
    
    if not cpd_script.exists():
        print(f"Error: CPD script not found at {cpd_script}")
        return
    
    print("=" * 70)
    print("Running sliding_ZINB_CPD_v3_SATAY.py on test_CPD centromere windows")
    print("=" * 70)
    print(f"Windows:  {windows_base}")
    print(f"Results:  {output_base}")
    print(f"Script:   {cpd_script}")
    print()
    
    # Parameters
    window_sizes = [100]
    overlap = 0.5
    threshold_start = 0.0
    threshold_end = 40.0
    threshold_step = 1.0
    n_workers = 1
    
    total_processed = 0
    total_skipped = 0
    
    # Process folders 0-7
    for folder_num in range(0, 8):
        windows_folder = windows_base / str(folder_num)
        
        if not windows_folder.exists():
            continue
        
        print(f"\n{'='*70}")
        print(f"Folder {folder_num}")
        print(f"{'='*70}")
        
        # Process each dataset in this folder
        for dataset_folder in sorted(windows_folder.iterdir()):
            if not dataset_folder.is_dir():
                continue
            
            dataset_name = dataset_folder.name
            
            # Find the corresponding density files
            density_folder = test_cpd_base / str(folder_num) / dataset_name
            nucleosome_density = density_folder / "combined_nucleosome_density.csv"
            centromere_density = density_folder / "combined_centromere_density_bin10000.csv"
            
            # Check if density files exist
            if not nucleosome_density.exists() or not centromere_density.exists():
                print(f"\n{dataset_name}:")
                print(f"  ⚠ Skipping - density files not found")
                if not nucleosome_density.exists():
                    print(f"    Missing: {nucleosome_density.name}")
                if not centromere_density.exists():
                    print(f"    Missing: {centromere_density.name}")
                total_skipped += 1
                continue
            
            print(f"\n{dataset_name}:")
            print(f"  Density files:")
            print(f"    ✓ {nucleosome_density.name}")
            print(f"    ✓ {centromere_density.name}")
            
            # Find all chromosome window files
            window_files = sorted(dataset_folder.glob("Chr*_centromere_window.csv"))
            
            if not window_files:
                print(f"  ⚠ No window files found")
                total_skipped += 1
                continue
            
            print(f"  Processing {len(window_files)} chromosomes...")
            
            # Process each chromosome
            for window_file in window_files:
                chrom_name = window_file.stem.replace("_centromere_window", "")
                
                # Output folder for this chromosome
                output_folder = output_base / str(folder_num) / dataset_name / chrom_name
                
                # Build command for v3
                cmd = [
                    sys.executable,
                    str(cpd_script),
                    str(window_file),
                    "--output_folder", str(output_folder),
                    "--dataset_name", f"{chrom_name}_centromere_window",
                    "--window_sizes", str(window_sizes[0]),
                    "--overlap", str(overlap),
                    "--threshold_start", str(threshold_start),
                    "--threshold_end", str(threshold_end),
                    "--threshold_step", str(threshold_step),
                    "--n_workers", str(n_workers),
                    "--nucleosome_file", str(nucleosome_density),
                    "--centromere_file", str(centromere_density),
                ]
                
                # Run the command
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=PROJECT_ROOT,
                        timeout=300  # 5 minute timeout per chromosome
                    )
                    
                    if result.returncode == 0:
                        print(f"    ✓ {chrom_name}")
                        total_processed += 1
                    else:
                        print(f"    ✗ {chrom_name} - Error (exit code {result.returncode})")
                        if result.stderr:
                            print(f"      Error output:")
                            for line in result.stderr.strip().split('\n'):
                                print(f"        {line}")
                        total_skipped += 1
                        
                except subprocess.TimeoutExpired:
                    print(f"    ✗ {chrom_name} - Timeout")
                    total_skipped += 1
                except Exception as e:
                    print(f"    ✗ {chrom_name} - {str(e)[:100]}")
                    total_skipped += 1
    
    print("\n" + "=" * 70)
    print("✓ CPD Analysis Complete!")
    print(f"  Total chromosomes processed: {total_processed}")
    print(f"  Total skipped: {total_skipped}")
    print(f"  Results saved to: {output_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
