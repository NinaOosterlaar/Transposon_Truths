"""
Run sliding_ZINB_CPD_v3_SATAY.py on all centromere windows from reconstruction_cpd_test_all_chrom.
Uses the combined density files created for each reconstruction dataset.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # Paths
    windows_base = PROJECT_ROOT / "Data" / "reconstruction_cpd_test_all_chrom" / "centromere_window"
    reconstruction_base = PROJECT_ROOT / "Data" / "reconstruction_cpd_test_all_chrom"
    results_base = PROJECT_ROOT / "Signal_processing" / "Results" / "reconstruction_cpd"
    zinb_output_base = results_base / "ZINB"
    gaussian_output_base = results_base / "Gaussian"

    zinb_script = PROJECT_ROOT / "Signal_processing" / "sliding_mean" / "sliding_ZINB_CPD_v3_SATAY.py"
    gaussian_script = PROJECT_ROOT / "Signal_processing" / "sliding_mean" / "old" / "sliding_mean_CPD.py"

    for script, name in [(zinb_script, "ZINB"), (gaussian_script, "Gaussian")]:
        if not script.exists():
            print(f"Error: {name} script not found at {script}")
            return

    print("=" * 70)
    print("Running CPD on reconstruction centromere windows (ZINB + Gaussian)")
    print("=" * 70)
    print(f"Windows:      {windows_base}")
    print(f"ZINB results: {zinb_output_base}")
    print(f"Gauss results: {gaussian_output_base}")
    print()

    # Parameters — ZINB v3
    window_sizes = [100]
    overlap = 0.5
    threshold_start = 0.0
    threshold_end = 20.0
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
            density_folder = reconstruction_base / str(folder_num) / dataset_name
            nucleosome_density = density_folder / "combined_nucleosome_density.csv"
            centromere_density = density_folder / "combined_centromere_density_bin10000.csv"

            # Check if density files exist
            if not nucleosome_density.exists() or not centromere_density.exists():
                print(f"\n{dataset_name}:")
                print("  Warning: Skipping - density files not found")
                if not nucleosome_density.exists():
                    print(f"    Missing: {nucleosome_density.name}")
                if not centromere_density.exists():
                    print(f"    Missing: {centromere_density.name}")
                total_skipped += 1
                continue

            print(f"\n{dataset_name}:")
            print("  Density files:")
            print(f"    OK {nucleosome_density.name}")
            print(f"    OK {centromere_density.name}")

            # Find all chromosome window files
            window_files = sorted(dataset_folder.glob("Chr*_centromere_window.csv"))

            if not window_files:
                print("  Warning: No window files found")
                total_skipped += 1
                continue

            print(f"  Processing {len(window_files)} chromosomes...")

            # Process each chromosome
            for window_file in window_files:
                chrom_name = window_file.stem.replace("_centromere_window", "")
                dataset_label = f"{chrom_name}_centromere_window"

                # --- ZINB v3 ---
                zinb_folder = zinb_output_base / str(folder_num) / dataset_name / chrom_name
                zinb_cmd = [
                    sys.executable,
                    str(zinb_script),
                    str(window_file),
                    "--output_folder", str(zinb_folder),
                    "--dataset_name", dataset_label,
                    "--window_sizes", str(window_sizes[0]),
                    "--overlap", str(overlap),
                    "--threshold_start", str(threshold_start),
                    "--threshold_end", str(threshold_end),
                    "--threshold_step", str(threshold_step),
                    "--n_workers", str(n_workers),
                    "--nucleosome_file", str(nucleosome_density),
                    "--centromere_file", str(centromere_density),
                ]

                # --- Gaussian sliding mean ---
                gaussian_folder = gaussian_output_base / str(folder_num) / dataset_name / chrom_name
                gaussian_cmd = [
                    sys.executable,
                    str(gaussian_script),
                    str(window_file),
                    "--output_folder", str(gaussian_folder),
                    "--dataset_name", dataset_label,
                    "--window_sizes", str(window_sizes[0]),
                    "--overlap", str(overlap),
                    "--threshold_start", str(threshold_start),
                    "--threshold_end", str(threshold_end),
                    "--threshold_step", str(threshold_step),
                ]

                for cmd, label in [(zinb_cmd, "ZINB"), (gaussian_cmd, "Gaussian")]:
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            cwd=PROJECT_ROOT,
                            timeout=300
                        )

                        if result.returncode == 0:
                            print(f"    OK {chrom_name} [{label}]")
                            total_processed += 1
                        else:
                            print(f"    ERROR {chrom_name} [{label}] - exit code {result.returncode}")
                            if result.stderr:
                                print("      Error output:")
                                for line in result.stderr.strip().split('\n'):
                                    print(f"        {line}")
                            total_skipped += 1

                    except subprocess.TimeoutExpired:
                        print(f"    ERROR {chrom_name} [{label}] - Timeout")
                        total_skipped += 1
                    except Exception as e:
                        print(f"    ERROR {chrom_name} [{label}] - {str(e)[:100]}")
                        total_skipped += 1

    print("\n" + "=" * 70)
    print("CPD Analysis Complete")
    print(f"  Total runs processed: {total_processed}")
    print(f"  Total skipped:        {total_skipped}")
    print(f"  ZINB results:     {zinb_output_base}")
    print(f"  Gaussian results: {gaussian_output_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
