import os
import pandas as pd
import numpy as np
import argparse


def create_sparser_version(input_folder, output_folder, removal_fraction):
    """
    Create a sparser version of the data by randomly removing a fraction of non-zero values.
    
    Args:
        input_folder: Path to input folder (e.g., Data/test_CPD/1/yEK23_1)
        output_folder: Path to output folder (e.g., Data/test_CPD/0.2/yEK23_1)
        removal_fraction: Fraction of non-zero values to remove (0.5 for 50%, 0.75 for 75%)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files that are chromosome distance files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('_distances.csv')]
    
    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        output_path = os.path.join(output_folder, csv_file)
        
        # Read the CSV file
        df = pd.read_csv(input_path)
        
        # Find indices where Value is non-zero
        non_zero_indices = df[df['Value'] > 0].index.tolist()
        
        # Randomly select indices to zero out
        n_to_remove = int(len(non_zero_indices) * removal_fraction)
        indices_to_remove = np.random.choice(non_zero_indices, size=n_to_remove, replace=False)
        
        # Set selected values to 0
        df.loc[indices_to_remove, 'Value'] = 0
        
        # Save the modified dataframe
        df.to_csv(output_path, index=False)
        
        print(f"  Processed {csv_file}: removed {n_to_remove} non-zero values out of {len(non_zero_indices)}")


def process_all_subfolders(base_input_folder, base_output_folder, removal_fraction, folder_name):
    """
    Process all subfolders in the input folder.
    
    Args:
        base_input_folder: Path to Data/test_CPD/1
        base_output_folder: Path to Data/test_CPD/0.2 or 0.1
        removal_fraction: Fraction to remove (0.5 or 0.75)
        folder_name: Name for logging (e.g., "0.2")
    """
    # Get all subfolders (yEK23_1, yEK23_2, etc.)
    subfolders = [f for f in os.listdir(base_input_folder) 
                  if os.path.isdir(os.path.join(base_input_folder, f)) and f.startswith('yEK23')]
    
    print(f"\nCreating folder {folder_name} (removing {removal_fraction*100}% of non-zero values):")
    print(f"Found {len(subfolders)} subfolders to process")
    
    for subfolder in sorted(subfolders):
        input_path = os.path.join(base_input_folder, subfolder)
        output_path = os.path.join(base_output_folder, subfolder)
        
        print(f"\nProcessing {subfolder}:")
        create_sparser_version(input_path, output_path, removal_fraction)


def create_lower_saturations(
    input_folder="Data/test_CPD/1",
    output_base="Data/test_CPD",
    saturations=(0.2, 0.1),
    removal_fractions=(0.5, 0.75),
    random_seed=42,
):
    """Create lower-saturation datasets by randomly zeroing non-zero insertions."""
    np.random.seed(random_seed)
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    if len(saturations) != len(removal_fractions):
        raise ValueError("saturations and removal_fractions must have the same length")

    for saturation, removal_fraction in zip(saturations, removal_fractions):
        folder_name = str(saturation)
        output_folder = os.path.join(output_base, folder_name)
        process_all_subfolders(input_folder, output_folder, removal_fraction, folder_name)

    print("\n" + "="*60)
    print("Sparse data creation complete!")
    print("="*60)


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create lower-saturation test_CPD datasets by removing non-zero insertions."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="Data/test_CPD/1",
        help="Input folder containing base saturation datasets.",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="Data/test_CPD",
        help="Base folder where sparse saturation folders are written.",
    )
    parser.add_argument(
        "--saturations",
        type=parse_float_list,
        default=[0.2, 0.1],
        help="Comma-separated output saturation folder names.",
    )
    parser.add_argument(
        "--removal_fractions",
        type=parse_float_list,
        default=[0.5, 0.75],
        help="Comma-separated fractions of non-zero insertions to remove.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible sparse sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_lower_saturations(
        input_folder=args.input_folder,
        output_base=args.output_base,
        saturations=args.saturations,
        removal_fractions=args.removal_fractions,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
