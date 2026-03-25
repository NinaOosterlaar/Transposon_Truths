import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from densities import density_from_centromere, density_from_nucleosome
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.plot_config import setup_plot_style

setup_plot_style()


def combine_chromosome_centromere_files(temp_dataset_folder, output_dataset_folder, bin=10000):
    """Combine per-chromosome centromere density files into one averaged file."""
    
    # Find all per-chromosome centromere density files
    density_files = [f for f in os.listdir(temp_dataset_folder) 
                    if f.endswith(f"_Boolean:True_bin:{bin}_centromere_density.csv")]
    
    if not density_files:
        print(f"    No centromere density files found")
        return None
    
    all_data = []
    for file in density_files:
        file_path = os.path.join(temp_dataset_folder, file)
        df = pd.read_csv(file_path)
        all_data.append(df[['Bin_Center', 'Density_per_bp']])
    
    # Merge all chromosomes on Bin_Center
    merged = all_data[0]
    for i in range(1, len(all_data)):
        merged = merged.merge(all_data[i], on='Bin_Center', how='outer', suffixes=('', f'_{i}'))
    
    # Calculate statistics across all Density_per_bp columns
    density_cols = [col for col in merged.columns if 'Density_per_bp' in col]
    result = pd.DataFrame()
    result['Bin_Center'] = merged['Bin_Center']
    result['mean_density'] = merged[density_cols].mean(axis=1)
    result['sd_density'] = merged[density_cols].std(axis=1)
    result['se_density'] = merged[density_cols].sem(axis=1)
    result['n_chromosomes'] = merged[density_cols].count(axis=1)
    result = result.sort_values('Bin_Center')
    
    # Save combined file
    os.makedirs(output_dataset_folder, exist_ok=True)
    output_file = os.path.join(output_dataset_folder, f"combined_centromere_density_bin{bin}.csv")
    result.to_csv(output_file, index=False)
    
    return result


def combine_chromosome_nucleosome_files(temp_dataset_folder, output_dataset_folder):
    """Combine per-chromosome nucleosome density files into one averaged file."""
    
    # Find all per-chromosome nucleosome density files
    density_files = [f for f in os.listdir(temp_dataset_folder) 
                    if f.endswith("_Boolean:_True_nucleosome_density.csv")]
    
    if not density_files:
        print(f"    No nucleosome density files found")
        return None
    
    all_data = []
    for file in density_files:
        file_path = os.path.join(temp_dataset_folder, file)
        df = pd.read_csv(file_path)
        all_data.append(df[['distance', 'density']])
    
    # Merge all chromosomes on distance
    merged = all_data[0]
    for i in range(1, len(all_data)):
        merged = merged.merge(all_data[i], on='distance', how='outer', suffixes=('', f'_{i}'))
    
    # Calculate statistics across all density columns
    density_cols = [col for col in merged.columns if 'density' in col]
    result = pd.DataFrame()
    result['distance'] = merged['distance']
    result['mean_density'] = merged[density_cols].mean(axis=1)
    result['sd_density'] = merged[density_cols].std(axis=1)
    result['se_density'] = merged[density_cols].sem(axis=1)
    result['n_chromosomes'] = merged[density_cols].count(axis=1)
    result = result.sort_values('distance')
    
    # Save combined file
    os.makedirs(output_dataset_folder, exist_ok=True)
    output_file = os.path.join(output_dataset_folder, "combined_nucleosome_density.csv")
    result.to_csv(output_file, index=False)
    
    return result


def create_centromere_plot(output_dataset_folder, density_df, bin=10000):
    """Create a line plot with error bands for centromere distance density."""
    
    dataset_name = os.path.basename(output_dataset_folder)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = density_df['Bin_Center'].abs().values
    y = density_df['mean_density'].values
    se = density_df['se_density'].values
    
    # Plot line with error band
    ax.plot(x, y, linewidth=2, color='steelblue', label='Mean density')
    ax.fill_between(x, y - se, y + se, alpha=0.3, color='steelblue', label='±1 SE')
    
    ax.set_title(f'Centromere Distance Insertion Rate - {dataset_name}\n(Averaged across all chromosomes)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance from Centromere (bp)', fontsize=12)
    ax.set_ylabel('Insertion Rate', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add vertical line at x=0 (centromere position)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Centromere')
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dataset_folder, f"combined_centromere_density_bin{bin}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_nucleosome_plot(output_dataset_folder, density_df):
    """Create a line plot with error bands for nucleosome distance density."""
    
    dataset_name = os.path.basename(output_dataset_folder)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = density_df['distance'].values
    y = density_df['mean_density'].values
    se = density_df['se_density'].values
    
    # Plot line with error band
    ax.plot(x, y, linewidth=2, color='steelblue', label='Mean density')
    ax.fill_between(x, y - se, y + se, alpha=0.3, color='steelblue', label='±1 SE')
    
    ax.set_title(f'Nucleosome Distance Insertion Rate - {dataset_name}\n(Averaged across all chromosomes)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance from Nucleosome (bp)', fontsize=12)
    ax.set_ylabel('Normalized Insertion Rate', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set axes limits
    ax.set_xlim(0, 400)
    ax.set_ylim(bottom=0)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dataset_folder, "combined_nucleosome_density.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    # Base path to the test_CPD data
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data", "test_CPD"))
    temp_path = os.path.join(base_path, "_temp_densities")
    
    # Create temp directory
    os.makedirs(temp_path, exist_ok=True)
    
    try:
        print("="*60)
        print("Step 1: Generating per-chromosome density files (temp)")
        print("="*60)
        
        # Process folders 0-7 (representing different sparsity levels)
        for folder_num in range(0, 8):
            input_folder = os.path.join(base_path, str(folder_num))
            
            if not os.path.exists(input_folder):
                print(f"Folder {input_folder} does not exist. Skipping.")
                continue
            
            print(f"\nFolder {folder_num}:")
            print(f"  - Centromere densities (bin=10000, boolean=True)...")
            density_from_centromere(
                input_folder=input_folder,
                output_folder=temp_path,
                bin=10000,
                max_distance_global=None,
                min_distance_global=None,
                boolean=True
            )
            
            print(f"  - Nucleosome densities (boolean=True)...")
            density_from_nucleosome(
                input_folder=input_folder,
                output_folder=temp_path,
                boolean=True
            )
        
        print("="*60)
        print("Combining chromosomes per dataset")
        print("="*60 + "\n")
        
        # Walk through temp folder and combine files for each dataset
        for root, dirs, files in os.walk(temp_path):
            # Check if this is a dataset folder (contains CSV files)
            csv_files = [f for f in files if f.endswith(".csv")]
            if not csv_files:
                continue
            
            # Get the relative path from temp to determine target location
            rel_path = os.path.relpath(root, temp_path)
            if rel_path == ".":
                continue
            
            # Target folder in actual test_CPD structure
            target_folder = os.path.join(base_path, rel_path)
            
            # Extract folder info for display
            path_parts = rel_path.split(os.sep)
            if len(path_parts) >= 2:
                folder_num, dataset_name = path_parts[0], path_parts[1]
                print(f"Processing {folder_num}/{dataset_name}...")
                
                # Combine centromere files
                print(f"  - Combining centromere densities...")
                centromere_df = combine_chromosome_centromere_files(root, target_folder, bin=10000)
                if centromere_df is not None:
                    create_centromere_plot(target_folder, centromere_df, bin=10000)
                    print(f"    ✓ Saved to {target_folder}")
                
                # Combine nucleosome files
                print(f"  - Combining nucleosome densities...")
                nucleosome_df = combine_chromosome_nucleosome_files(root, target_folder)
                if nucleosome_df is not None:
                    create_nucleosome_plot(target_folder, nucleosome_df)
                    print(f"    ✓ Saved to {target_folder}")
        
        print("\n" + "="*60)
        print("✓ All combined density files and plots generated!")
        print("="*60)
        print(f"\nCombined files saved to respective dataset folders in: {base_path}")
        print(f"Temp files preserved in: {temp_path}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print(f"Temp files preserved in: {temp_path} for debugging")
        raise


if __name__ == "__main__":
    main()
