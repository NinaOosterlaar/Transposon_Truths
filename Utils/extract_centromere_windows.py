"""
Extract ±1000 bp windows around centromeres from test_CPD datasets.
Creates one window file per chromosome per dataset, matching the format of
Signal_processing/sample_data/Centromere_region/*.csv files.
"""

import os
import pandas as pd
from pathlib import Path


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
    
    # Process folders 1-6
    for folder_num in range(1, 7):
        input_folder = base_path / str(folder_num)
        
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
                print(f"    No distance files found")
                continue
            
            # Process each chromosome
            for input_file in distance_files:
                chrom_name = input_file.stem.replace("_distances", "")
                output_file = output_base / str(folder_num) / dataset_name / f"{chrom_name}_centromere_window.csv"
                
                success = extract_centromere_window(str(input_file), str(output_file))
                
                if success:
                    total_windows += 1
                    print(f"    ✓ {chrom_name}")
                else:
                    total_skipped += 1
    
    print("\n" + "=" * 60)
    print(f"✓ Extraction complete!")
    print(f"  Total windows created: {total_windows}")
    print(f"  Total skipped: {total_skipped}")
    print(f"  Output location: {output_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
