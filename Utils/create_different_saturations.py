import os
import random
import itertools
import shutil
import argparse
import pandas as pd

def create_saturation_test_combinations(
    source_folder="Data/test_CPD/1",
    output_base="Data/test_CPD",
    max_combinations=6,
    random_seed=42
):
    """
    Create random combinations of datasets for saturation testing.
    
    For k in [2, 3, 4, 5], generates up to max_combinations random combinations
    of k datasets from the source folder and saves them to output_base/{k}/yEK23_{i}/.
    
    The combining method averages non-zero values per position, matching the
    behavior in AE/test_saturation.py.
    
    Args:
        source_folder (str): Path to folder containing individual datasets (folder 1)
        output_base (str): Base path for test_CPD folders
        max_combinations (int): Maximum number of combinations to create per k
        random_seed (int): Random seed for reproducibility
    """

    
    # Set random seed for reproducibility
    rng = random.Random(random_seed)
    
    # Get list of source dataset folders
    source_datasets = sorted([
        d for d in os.listdir(source_folder)
        if os.path.isdir(os.path.join(source_folder, d)) and not d.startswith('.')
    ])
    
    n_datasets = len(source_datasets)
    print(f"Found {n_datasets} source datasets in {source_folder}:")
    for ds in source_datasets:
        print(f"  {ds}")
    
    # Get list of all chromosomes from the first dataset
    first_dataset_path = os.path.join(source_folder, source_datasets[0])
    chromosomes = sorted([
        fname.split("_")[0]
        for fname in os.listdir(first_dataset_path)
        if fname.endswith("_distances.csv") and not fname.startswith("ChrM")
    ])
    
    print(f"\nFound {len(chromosomes)} chromosomes to process")
    
    # Process k=2,3,4,5
    for k in range(2, 6):
        print(f"\n{'='*60}")
        print(f"Processing k={k} (combining {k} datasets)")
        print(f"{'='*60}")
        
        # Generate all possible combinations and randomly sample
        all_combos = list(itertools.combinations(range(n_datasets), k))
        selected_combos = rng.sample(all_combos, min(max_combinations, len(all_combos)))
        
        print(f"Selected {len(selected_combos)} combinations out of {len(all_combos)} possible")
        
        # Create output folder for this k
        k_folder = os.path.join(output_base, str(k))
        
        # Process each selected combination
        for combo_idx, combo in enumerate(selected_combos, 1):
            # Get dataset names for this combination
            datasets_in_combo = [source_datasets[i] for i in combo]
            output_folder = os.path.join(k_folder, f"yEK23_{combo_idx}")
            
            print(f"\n  [{combo_idx}/{len(selected_combos)}] Creating: {output_folder}")
            print(f"    Combining: {', '.join(datasets_in_combo)}")
            
            # Remove existing folder if it exists (overwrite)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder, exist_ok=True)
            
            # Process each chromosome
            for chrom in chromosomes:
                # Read all datasets for this chromosome
                frames = []
                for dataset in datasets_in_combo:
                    csv_path = os.path.join(source_folder, dataset, f"{chrom}_distances.csv")
                    if os.path.exists(csv_path):
                        frames.append(pd.read_csv(csv_path))
                
                if not frames:
                    continue
                
                # Combine all datasets
                all_data = pd.concat(frames, ignore_index=True)
                
                # For each position, keep first-seen auxiliary columns (distances)
                aux = (
                    all_data.groupby('Position', sort=True)
                    [['Nucleosome_Distance', 'Centromere_Distance']]
                    .first()
                )
                
                # Average non-zero values per position
                def avg_nonzero(series):
                    nz = series[series != 0]
                    return nz.mean() if len(nz) > 0 else 0.0
                
                combined_values = (
                    all_data.groupby('Position', sort=True)['Value']
                    .agg(avg_nonzero)
                )
                
                # Construct output DataFrame
                result = aux.copy()
                result['Value'] = combined_values
                result = result.reset_index()[['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance']]
                
                # Save to CSV
                out_path = os.path.join(output_folder, f"{chrom}_distances.csv")
                result.to_csv(out_path, index=False)
            
            print(f"    ✓ Saved {len(chromosomes)} chromosome files")
    
    print(f"\n{'='*60}")
    print("Saturation test combinations complete!")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create higher-saturation test_CPD datasets by combining base datasets."
    )
    parser.add_argument(
        "--source_folder",
        type=str,
        default="Data/test_CPD/1",
        help="Folder containing base saturation datasets.",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="Data/test_CPD",
        help="Base folder where saturation folders are written.",
    )
    parser.add_argument(
        "--max_combinations",
        type=int,
        default=6,
        help="Maximum random dataset combinations to create per saturation level.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible combination sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_saturation_test_combinations(
        source_folder=args.source_folder,
        output_base=args.output_base,
        max_combinations=args.max_combinations,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
