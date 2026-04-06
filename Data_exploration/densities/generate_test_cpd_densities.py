import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Utils.plot_config import setup_plot_style
from Utils.SGD_API.yeast_architecture import Centromeres, Nucleosomes

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
        # Skip Step 1 - per-chromosome files already exist in temp
        # Uncomment below to regenerate per-chromosome files:
        """
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
        """
        
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


def build_per_position_signal(chrom_name, chrom_csv_path, metadata_df):
    """Convert reconstruction window-index predictions to a per-position genomic signal.

    Each window value is assigned to its full [start_pos, end_pos] span, then overlapping
    windows are averaged at each genomic position.
    """
    df = pd.read_csv(chrom_csv_path)
    chrom_meta = metadata_df[metadata_df['chromosome'] == chrom_name][['window_index', 'start_pos', 'end_pos']].copy()
    if chrom_meta.empty:
        return None

    windows = df.merge(chrom_meta, left_on='position', right_on='window_index', how='inner')
    if windows.empty:
        return None

    windows['start_pos'] = windows['start_pos'].astype(int)
    windows['end_pos'] = windows['end_pos'].astype(int)

    max_pos = int(windows['end_pos'].max())
    diff_sum = np.zeros(max_pos + 2, dtype=float)
    diff_count = np.zeros(max_pos + 2, dtype=float)

    starts = windows['start_pos'].to_numpy(dtype=int)
    ends = windows['end_pos'].to_numpy(dtype=int)
    values = windows['reconstruction'].to_numpy(dtype=float)

    np.add.at(diff_sum, starts, values)
    np.add.at(diff_sum, ends + 1, -values)
    np.add.at(diff_count, starts, 1.0)
    np.add.at(diff_count, ends + 1, -1.0)

    sum_signal = np.cumsum(diff_sum[:-1])
    count_signal = np.cumsum(diff_count[:-1])
    valid_mask = count_signal > 0
    if not valid_mask.any():
        return None

    positions = np.where(valid_mask)[0]
    reconstruction_signal = sum_signal[valid_mask] / count_signal[valid_mask]

    return pd.DataFrame({
        'genomic_position': positions,
        'reconstruction': reconstruction_signal,
    })


def compute_nucleosome_distances(chrom_name, positions, nucleosomes_obj):
    """Vectorized nearest-nucleosome distance for many genomic positions."""
    middles = np.array(nucleosomes_obj.get_middles(chrom_name), dtype=int)
    if middles.size == 0:
        return None
    middles.sort()

    positions = np.asarray(positions, dtype=int)
    idx = np.searchsorted(middles, positions, side='left')

    left_idx = np.clip(idx - 1, 0, middles.size - 1)
    right_idx = np.clip(idx, 0, middles.size - 1)

    left_dist = np.abs(positions - middles[left_idx])
    right_dist = np.abs(middles[right_idx] - positions)
    return np.minimum(left_dist, right_dist)


def process_chrom_csv_centromere(chrom_name, chrom_csv_path, temp_folder, centromeres_obj, bin=10000, boolean=True):
    """Read a reconstruction Chr.csv, compute centromere distances, and write a per-chromosome
    centromere density CSV in the format combine_chromosome_centromere_files expects.

    The 'position' column in each Chr*.csv is already a genomic bp coordinate produced by
    OutputReconstructor.reconstruct_to_dataframe (linear interpolation of window positions).
    No metadata mapping is needed.
    """
    signal_df = pd.read_csv(chrom_csv_path)
    if signal_df.empty:
        print(f"    Empty CSV for {chrom_name}. Skipping centromere density.")
        return

    signal_df['Centromere_Distance'] = signal_df['position'].apply(
        lambda pos: centromeres_obj.compute_distance(chrom_name, int(pos))
    )
    if boolean:
        signal_df['Value'] = signal_df['reconstruction'].apply(lambda x: 1 if x > 0 else 0)
    else:
        signal_df['Value'] = signal_df['reconstruction']

    max_distance = signal_df['Centromere_Distance'].max()
    min_distance = signal_df['Centromere_Distance'].min()
    data_range = max(abs(min_distance), abs(max_distance))
    n_bins_each_side = int(np.ceil(data_range / bin)) + 1
    bin_centers = np.arange(-n_bins_each_side * bin, (n_bins_each_side + 1) * bin, bin)
    bin_edges = bin_centers - bin / 2
    bin_edges = np.append(bin_edges, bin_edges[-1] + bin)

    signal_df['Distance_Bin'] = pd.cut(signal_df['Centromere_Distance'], bins=bin_edges, right=False, include_lowest=True)
    density = signal_df.groupby('Distance_Bin')['Value'].sum().reset_index()
    density['Bin_Center'] = density['Distance_Bin'].apply(lambda x: x.left + bin / 2)
    density['Density_per_bp'] = density['Value'] / bin
    density = density.sort_values('Bin_Center')

    os.makedirs(temp_folder, exist_ok=True)
    output_file = os.path.join(temp_folder, f"{chrom_name}_Boolean:{boolean}_bin:{bin}_centromere_density.csv")
    density[['Bin_Center', 'Density_per_bp']].to_csv(output_file, index=False)


def process_chrom_csv_nucleosome(chrom_name, chrom_csv_path, temp_folder, nucleosomes_obj, nucleosomes_normalization, boolean=True):
    """Read a reconstruction Chr.csv, compute nucleosome distances, and write a per-chromosome
    nucleosome density CSV in the format combine_chromosome_nucleosome_files expects.

    The 'position' column in each Chr*.csv is already a genomic bp coordinate.
    """
    signal_df = pd.read_csv(chrom_csv_path)
    if signal_df.empty:
        print(f"    Empty CSV for {chrom_name}. Skipping nucleosome density.")
        return

    if boolean:
        signal_df['Value'] = signal_df['reconstruction'].apply(lambda x: 1 if x > 0 else 0)
    else:
        signal_df['Value'] = signal_df['reconstruction']

    nuc_distances = compute_nucleosome_distances(
        chrom_name,
        signal_df['position'].to_numpy(dtype=int),
        nucleosomes_obj
    )
    if nuc_distances is None:
        print(f"    No nucleosome middles for {chrom_name}. Skipping nucleosome density.")
        return
    signal_df['Nucleosome_Distance'] = nuc_distances.astype(int)

    counts = {}
    for _, row in signal_df.iterrows():
        dist = int(row['Nucleosome_Distance'])
        counts[dist] = counts.get(dist, 0) + row['Value']

    norm = nucleosomes_normalization.get(chrom_name, {})
    normalized = {}
    for dist, val in counts.items():
        if dist in norm:
            normalized[dist] = val / norm[dist]
    for dist in norm:
        if dist not in normalized:
            normalized[dist] = 0

    os.makedirs(temp_folder, exist_ok=True)
    output_file = os.path.join(temp_folder, f"{chrom_name}_Boolean:_{boolean}_nucleosome_density.csv")
    with open(output_file, "w") as f:
        f.write("distance,density\n")
        for dist, dens in normalized.items():
            f.write(f"{dist},{dens}\n")


def main_reconstruction():
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Data", "reconstruction_cpd_test_all_chrom")
    )
    temp_path = os.path.join(base_path, "_temp_densities")

    print("Loading nucleosome normalization data...")
    nucleosomes_obj = Nucleosomes()
    centromeres_obj = Centromeres()
    nucleosomes_normalization = {}
    all_chroms = [f"Chr{r}" for r in ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI"]]
    for chrom in all_chroms:
        nucleosomes_normalization[chrom] = nucleosomes_obj.compute_exposure(chrom)

    for folder_num in range(8):
        folder_path = os.path.join(base_path, str(folder_num))
        if not os.path.exists(folder_path):
            continue
        for strain_name in os.listdir(folder_path):
            strain_path = os.path.join(folder_path, strain_name)
            if not os.path.isdir(strain_path):
                continue

            # Chromosome CSVs live one level deeper: strain_path/strain_name/ChrX.csv
            chrom_dir = os.path.join(strain_path, strain_name)
            if not os.path.isdir(chrom_dir):
                continue

            temp_dataset_folder = os.path.join(temp_path, str(folder_num), strain_name)
            output_folder = strain_path  # next to metadata.json

            print(f"\nProcessing {folder_num}/{strain_name}...")
            for csv_file in os.listdir(chrom_dir):
                if not csv_file.endswith(".csv"):
                    continue
                chrom_name = csv_file.replace(".csv", "")
                chrom_csv_path = os.path.join(chrom_dir, csv_file)

                process_chrom_csv_centromere(
                    chrom_name, chrom_csv_path, temp_dataset_folder, centromeres_obj
                )
                process_chrom_csv_nucleosome(
                    chrom_name, chrom_csv_path, temp_dataset_folder,
                    nucleosomes_obj, nucleosomes_normalization
                )

            print(f"  - Combining centromere densities...")
            centromere_df = combine_chromosome_centromere_files(temp_dataset_folder, output_folder, bin=10000)
            if centromere_df is not None:
                create_centromere_plot(output_folder, centromere_df, bin=10000)
                print(f"    Saved to {output_folder}")

            print(f"  - Combining nucleosome densities...")
            nucleosome_df = combine_chromosome_nucleosome_files(temp_dataset_folder, output_folder)
            if nucleosome_df is not None:
                create_nucleosome_plot(output_folder, nucleosome_df)
                print(f"    Saved to {output_folder}")

    print("\nDone. Combined files saved next to each metadata.json.")


if __name__ == "__main__":
    main_reconstruction()
