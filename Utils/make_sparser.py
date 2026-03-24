import os
import pandas as pd
import numpy as np
import sys
import shutil

# Set base directory
base_dir = "/Users/ninaoosterlaar/Documents/Repositories/Thesis"

# Set random seed for reproducibility
np.random.seed(42)

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


def main():
    # Define base paths
    input_folder = os.path.join(base_dir, "Data/test_CPD/1")
    output_folder_02 = os.path.join(base_dir, "Data/test_CPD/0.2")
    output_folder_01 = os.path.join(base_dir, "Data/test_CPD/0.1")
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist!")
        return
    
    # Create sparse versions
    # 0.2 folder: remove 50% of non-zero values
    process_all_subfolders(input_folder, output_folder_02, 0.5, "0.2")
    
    # 0.1 folder: remove 75% of non-zero values
    process_all_subfolders(input_folder, output_folder_01, 0.75, "0.1")
    
    print("\n" + "="*60)
    print("Sparse data creation complete!")
    print("="*60)
    print("\nNow generating density files...")
    print("="*60)
    
    # Add base directory to path and import density functions
    sys.path.insert(0, base_dir)
    sys.path.insert(0, os.path.join(base_dir, "Data_exploration"))
    sys.path.insert(0, os.path.join(base_dir, "Data_exploration/densities"))
    from densities import density_from_centromere, density_from_nucleosome
    import generate_test_cpd_densities
    
    # Process both new folders
    for folder_name in ['0.2', '0.1']:
        print(f"\n{'='*60}")
        print(f"Generating densities for folder {folder_name}")
        print('='*60)
        
        input_folder = os.path.join(base_dir, "Data/test_CPD", folder_name)
        temp_path = os.path.join(base_dir, "Data/test_CPD", f"_temp_densities_{folder_name}")
        
        # Create temp directory
        os.makedirs(temp_path, exist_ok=True)
        
        # Get all subfolders (yEK23_1, yEK23_2, etc.)
        subfolders = [f for f in os.listdir(input_folder) 
                      if os.path.isdir(os.path.join(input_folder, f)) and f.startswith('yEK23')]
        
        for subfolder in sorted(subfolders):
            subfolder_path = os.path.join(input_folder, subfolder)
            print(f"\nProcessing {folder_name}/{subfolder}:")
            
            # Generate centromere densities
            print(f"  - Generating centromere densities...")
            density_from_centromere(
                input_folder=subfolder_path,
                output_folder=temp_path,
                bin=10000,
                max_distance_global=None,
                min_distance_global=None,
                boolean=True
            )
            
            # Generate nucleosome densities
            print(f"  - Generating nucleosome densities...")
            density_from_nucleosome(
                input_folder=subfolder_path,
                output_folder=temp_path,
                boolean=True
            )
        
        # Now combine the per-chromosome files for each subfolder
        print(f"\n{'='*60}")
        print(f"Combining chromosomes for folder {folder_name}")
        print('='*60)
        
        for subfolder in sorted(subfolders):
            print(f"\nCombining {folder_name}/{subfolder}...")
            
            temp_subfolder_path = os.path.join(temp_path, subfolder)
            target_folder = os.path.join(input_folder, subfolder)
            
            if os.path.exists(temp_subfolder_path):
                # Combine centromere files
                print(f"  - Combining centromere densities...")
                centromere_df = generate_test_cpd_densities.combine_chromosome_centromere_files(
                    temp_subfolder_path, target_folder, bin=10000)
                if centromere_df is not None:
                    generate_test_cpd_densities.create_centromere_plot(target_folder, centromere_df, bin=10000)
                    print(f"    ✓ Saved to {target_folder}")
                
                # Combine nucleosome files
                print(f"  - Combining nucleosome densities...")
                nucleosome_df = generate_test_cpd_densities.combine_chromosome_nucleosome_files(
                    temp_subfolder_path, target_folder)
                if nucleosome_df is not None:
                    generate_test_cpd_densities.create_nucleosome_plot(target_folder, nucleosome_df)
                    print(f"    ✓ Saved to {target_folder}")
        
        # Clean up temp folder
        print(f"\nCleaning up temporary files for {folder_name}...")
        shutil.rmtree(temp_path)
    
    print("\n" + "="*60)
    print("All done! ✨")
    print("="*60)


if __name__ == "__main__":
    main()
