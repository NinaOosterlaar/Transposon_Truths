"""
Extract ±1000 bp windows around centromeres from test_CPD datasets.
Creates one window file per chromosome per dataset, matching the format of
Signal_processing/sample_data/Centromere_region/*.csv files.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from SGD_API.yeast_architecture import Centromeres
from SGD_API.yeast_architecture import Nucleosomes


def extract_centromere_window(input_file, output_file):
    """
    Extract ±1000 bp window around the centromere from a chromosome file.
    
    Args:
        input_file: Path to Chr*_distances.csv file
        output_file: Path to save the centromere window
    """
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Find centromere position (where Centromere_Distance = 0)
    centromere_rows = df[df['Centromere_Distance'] == 0]
    
    if len(centromere_rows) == 0:
        print(f"  ⚠ No centromere found in {os.path.basename(input_file)}")
        return False
    
    centromere_pos = centromere_rows['Position'].values[0]
    
    # Extract ±1000 bp window
    window_df = df[
        (df['Position'] >= centromere_pos - 1000) & 
        (df['Position'] <= centromere_pos + 1000)
    ].copy()
    
    if len(window_df) < 2000:
        print(f"  ⚠ Window too small ({len(window_df)} rows) for {os.path.basename(input_file)}")
        return False
    
    # Re-center position so centromere is at 0
    window_df['Position'] = window_df['Position'] - centromere_pos
    
    # Rename columns to match reference format (lowercase)
    window_df = window_df.rename(columns={
        'Position': 'Position',  # Keep as is
        'Value': 'value',
        'Nucleosome_Distance': 'nucleosome_distance',
        'Centromere_Distance': 'Centromere_Distance'  # Keep as is
    })
    
    # Save window file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    window_df.to_csv(output_file, index=False)
    
    return True


def extract_centromere_window_reconstruction(input_file, output_file, centromeres, nucleosomes):
    """Extract a centered 2001-row window from reconstruction Chr*.csv files.

    Args:
        input_file: Path to Chr*.csv file with columns including position and reconstruction
        output_file: Path to save the centromere window
        centromeres: Centromeres helper object
        nucleosomes: Nucleosomes helper object
    """
    df = pd.read_csv(input_file)
    chrom_name = Path(input_file).stem
    centromere_pos = centromeres.get_middle(chrom_name)

    if centromere_pos is None:
        print(f"  Warning: No centromere found for {chrom_name}")
        return False

    if 'position' not in df.columns or 'reconstruction' not in df.columns:
        print(f"  Warning: Missing required columns in {input_file}")
        return False

    # In reconstruction outputs, 'position' is already genomic coordinate.
    # Collapse potential duplicates and sort for robust interpolation.
    signal_df = (
        df[['position', 'reconstruction']]
        .dropna()
        .groupby('position', as_index=False)['reconstruction']
        .mean()
        .sort_values('position')
    )

    if len(signal_df) < 2:
        print(f"  Warning: Not enough signal points in {chrom_name}")
        return False

    # Build exactly 2001 output points for [-1000, 1000] bp around centromere.
    target_pos = np.arange(-1000, 1001)
    target_abs_pos = target_pos + int(centromere_pos)
    x = signal_df['position'].to_numpy(dtype=float)
    y = signal_df['reconstruction'].to_numpy(dtype=float)

    # Use nearest edge values outside the sampled range to keep a full 2001-row window.
    interp_vals = np.interp(target_abs_pos, x, y, left=y[0], right=y[-1])
    nucleosome_distances = [nucleosomes.compute_distance(chrom_name, int(p)) for p in target_abs_pos]

    window_df = pd.DataFrame({
        'Position': target_pos,
        'value': interp_vals,
        'nucleosome_distance': nucleosome_distances,
    })
    window_df['Centromere_Distance'] = window_df['Position']

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    window_df.to_csv(output_file, index=False)
    return True


def main():
    # Base paths
    base_path = Path(__file__).parent.parent / "Data" / "test_CPD"
    output_base = base_path / "centromere_windows"
    print("=" * 60)
    print("Extracting Centromere Windows from test_CPD datasets")
    print("=" * 60)
    print(f"Input:  {base_path}")
    print(f"Output: {output_base}")
    print()

    total_windows = 0
    total_skipped = 0

    # Process folders 0-7
    for folder_num in range(0, 8):
        input_folder = base_path / f"{folder_num}"

        if not input_folder.exists():
            continue

        print(f"\nFolder {folder_num}:")

        # Process each dataset in this folder
        for dataset_folder in sorted(input_folder.iterdir()):
            if not dataset_folder.is_dir():
                continue

            dataset_name = dataset_folder.name
            print(f"  {dataset_name}:")

            # Find all Chr*_distances.csv files
            distance_files = sorted(dataset_folder.glob("Chr*_distances.csv"))

            if not distance_files:
                print("    No distance files found")
                continue

            # Process each chromosome
            for input_file in distance_files:
                chrom_name = input_file.stem.replace("_distances", "")
                output_file = output_base / f"{folder_num}" / dataset_name / f"{chrom_name}_centromere_window.csv"

                success = extract_centromere_window(str(input_file), str(output_file))

                if success:
                    total_windows += 1
                    print(f"    ✓ {chrom_name}")
                else:
                    total_skipped += 1

    print("\n" + "=" * 60)
    print("✓ Extraction complete!")
    print(f"  Total windows created: {total_windows}")
    print(f"  Total skipped: {total_skipped}")
    print(f"  Output location: {output_base}")
    print("=" * 60)


def main_reconstruction():
    # Base paths
    base_path = Path(__file__).parent.parent / "Data" / "reconstruction_cpd_test_all_chrom"
    output_base = base_path / "centromere_window"

    print("=" * 60)
    print("Extracting Centromere Windows from reconstruction_cpd_test_all_chrom")
    print("=" * 60)
    print(f"Input:  {base_path}")
    print(f"Output: {output_base}/[folder_num]/[strain]")
    print()

    centromeres = Centromeres()
    nucleosomes = Nucleosomes()
    total_windows = 0
    total_skipped = 0

    # Process folders 0-7
    for folder_num in range(0, 8):
        folder_path = base_path / f"{folder_num}"
        if not folder_path.exists():
            continue

        print(f"\nFolder {folder_num}:")

        # Process each strain folder (e.g. yEK23_1)
        for strain_folder in sorted(folder_path.iterdir()):
            if not strain_folder.is_dir():
                continue

            print(f"  {strain_folder.name}:")
            output_folder = output_base / f"{folder_num}" / strain_folder.name

            # Prefer canonical layout strain_folder/strain_name; fallback to other subdirs.
            canonical_chrom_dir = strain_folder / strain_folder.name
            if canonical_chrom_dir.is_dir():
                chromosome_dirs = [canonical_chrom_dir]
            else:
                chromosome_dirs = [
                    d for d in sorted(strain_folder.iterdir())
                    if d.is_dir() and d.name != "centromere_windows"
                ]
            found_chrom_dir = False

            for chromosome_dir in chromosome_dirs:
                chrom_files = sorted(chromosome_dir.glob("Chr*.csv"))
                if not chrom_files:
                    continue

                found_chrom_dir = True
                for input_file in chrom_files:
                    chrom_name = input_file.stem
                    output_file = output_folder / f"{chrom_name}_centromere_window.csv"

                    success = extract_centromere_window_reconstruction(
                        str(input_file), str(output_file), centromeres, nucleosomes
                    )

                    if success:
                        total_windows += 1
                        print(f"    Saved {chrom_name}")
                    else:
                        total_skipped += 1

            if not found_chrom_dir:
                print("    No chromosome directory with Chr*.csv found")

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print(f"  Total windows created: {total_windows}")
    print(f"  Total skipped: {total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    # Usage:
    #   python Utils/extract_centromere_windows.py reconstruction
    #   python Utils/extract_centromere_windows.py test_cpd
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "reconstruction"

    if mode == "reconstruction":
        main_reconstruction()
    elif mode == "test_cpd":
        main()
    else:
        print("Unknown mode. Use 'reconstruction' or 'test_cpd'.")
        sys.exit(1)
