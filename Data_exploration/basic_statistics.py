import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns  
from tqdm import tqdm
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.plot_config import setup_plot_style, COLORS
from Utils.reader import read_wig, label_from_filename

# Set up standardized plot style
setup_plot_style()


yeast_chrom_lengths = {
    "chrI":     230218,  "chrII":    813184,  "chrIII":   316620, "chrIV":   1531933,
    "chrV":     576874,  "chrVI":    270161,  "chrVII":  1090940, "chrVIII":  562643,
    "chrIX":    439888,  "chrX":     745751,  "chrXI":    666816, "chrXII":  1078177,
    "chrXIII":  924431,  "chrXIV":   784333,  "chrXV":   1091291, "chrXVI":   948066,
    "chrM":      85779,
}
 
 
def show_counts_part_chromosome(file, chrom, start, end):
    """
    Displays counts for a specific chromosome region.
    """
    dict = read_wig(file)
    
    if chrom not in dict:
        print(f"Chromosome {chrom} not found in data.")
        return

    df = dict[chrom]
    region_df = df[(df['Position'] >= start) & (df['Position'] <= end)]

    # Plot bar plot with counts
    plt.figure(figsize=(12, 6))
    plt.bar(region_df['Position'], region_df['Value'], width=1, color='blue')
    plt.title(f'Counts for {chrom}:{start}-{end}')
    plt.xlabel('Position')
    plt.ylabel('Counts')
    plt.xlim(start, end)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
def plot_presence_transposon(folder, chrom, start, end):
    """Plot a heatmap that shows presence or absence of transposon insertions in a given chromosomal region across multiple samples.
    """
    all_counts = []
    for file in os.listdir(folder):
        if not file.endswith('.wig'):
            continue
        file_path = os.path.join(folder, file)
        dict = read_wig(file_path)
        
        if chrom not in dict:
            print(f"Chromosome {chrom} not found in {file}. Skipping.")
            continue
        df = dict[chrom]
        region_df = df[(df['Position'] >= start) & (df['Position'] <= end)]
        # Create a binary presence/absence array for the region
        presence = np.zeros(end - start + 1, dtype=int)
        for _, row in region_df.iterrows():
            pos = int(row['Position']) - start
            if 0 <= pos < len(presence):
                presence[pos] = 1  # Mark presence
        label = label_from_filename(file)
        all_counts.append((label, presence))
        
    if not all_counts:
        print("No valid data found for the specified chromosome and region.")
        return
    all_counts.sort(key=lambda x: x[0])  # Sort by label
    labels = [x[0] for x in all_counts]
    data_matrix = np.array([x[1] for x in all_counts])
    plt.figure(figsize=(12, max(6, len(labels) * 0.3)))
    sns.heatmap(data_matrix, cmap='Greys', cbar=False, yticklabels=labels)
    plt.title(f'Transposon Insertion Presence in {chrom}:{start}-{end}')
    plt.xlabel('Position in Region')
    plt.ylabel('Samples')
    plt.tight_layout()
    plt.show()
    
def plot_basic_statistics(stats, output_folder):
    # --- Total sum per file ---
    file_labels = [s[0] for s in stats]
    total_sums  = [s[1] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.bar(file_labels, total_sums, color='deeppink')
    plt.title('Total Sum of Counts per WIG File')
    plt.xlabel('Sample')
    plt.ylabel('Total Counts')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'total_counts_per_file.png'))
    else:
        plt.show()

    # --- Mean counts per bp ---
    mean_counts = [s[2] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.bar(file_labels, mean_counts, color='hotpink')
    plt.title('Mean Counts per WIG File')
    plt.xlabel('Sample')
    plt.ylabel('Mean Counts')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'mean_counts.png'))
    else:
        plt.show()

    # --- Occupied vs unoccupied sites ---
    occupied = [s[3] for s in stats]
    unoccupied = [s[4] for s in stats]

    plt.figure(figsize=(12, 6))
    plt.bar(file_labels, occupied, label="Occupied sites", color="lightskyblue")
    plt.bar(file_labels, unoccupied, bottom=occupied, label="Unoccupied sites", color="lightpink")
    plt.title('Occupied vs Unoccupied Sites per Sample')
    plt.xlabel('Sample')
    plt.ylabel('Number of Sites')
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'occupied_vs_unoccupied_sites.png'))
    else:
        plt.show()
        
        
# def create_histogram(folder_name, output_file):
#     """Create and save a histogram of counts from all WIG files in the specified folder."""
#     all_values = {}

#     for root, dirs, files in os.walk(folder_name):
#         print(f"root: {root}, files: {files}, dirs: {dirs}")
#         number_of_datasets = len(files)
#         for wig_file in files:
#             if not wig_file.endswith(".wig"): continue
            
#             # Read the wig file
#             file_path = os.path.join(root, wig_file)
#             wig_dict = read_wig(file_path) 
#             assert len(wig_dict) == 17, f"Expected 17 chromosomes, found {len(wig_dict)} in {wig_file}"

#             for chrom, df in wig_dict.items():
#                 for count in df['Value']:
#                     if count in all_values:
#                         all_values[count] += 1
#                     else:
#                         all_values[count] = 1
#         # normalize by number of datasets
#         for key in all_values:
#             all_values[key] /= number_of_datasets

    # plt.figure(figsize=(10, 6))
    # plt.hist(all_values.keys(), bins=50, weights=all_values.values(), color='purple', alpha=0.7)
    # plt.title('Histogram of Counts from All WIG Files')
    # plt.xlabel('Counts')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y')
    # plt.tight_layout() 
    # plt.savefig(output_file)
    # plt.close()

# create_histogram("Data/wiggle_format", "Data_exploration/results/histogram_counts.png")
    
def save_basic_info(folder_name, output_file, plot = False, output_folder_figures = None):
    """Saves basic statistics to a text file."""
    stats = []  # (label, total_sum, mean_count, occupied_sites, unoccupied_sites)

    genome_size = sum(yeast_chrom_lengths.values())
    
    for root, dirs, files in os.walk(folder_name):
        print(f"root: {root}, files: {files}, dirs: {dirs}")
        for wig_file in files:
            if not wig_file.endswith(".wig"): continue
            
            # Read the wig file
            file_path = os.path.join(root, wig_file)
            # file_path = os.path.join(folder_name, wig_file)
            wig_dict = read_wig(file_path) 
            assert len(wig_dict) == 17, f"Expected 17 chromosomes, found {len(wig_dict)} in {wig_file}"

            total_sum = 0
            occupied_sites = 0
            non_zero_sum = 0
            non_zero_count = 0
            all_values = []

            for chrom, df in wig_dict.items():
                # Filter out positions with values > 1 million
                high_values = df[df['Value'] > 1000000]
                if not high_values.empty:
                    for _, row in high_values.iterrows():
                        print(f"Filtered out high value in {wig_file}, {chrom}, Position {row['Position']}: {row['Value']}")
                    df = df[df['Value'] <= 1000000]
                
                total_sum += df['Value'].sum()
                if not df.empty:
                    occupied_sites += (df['Value'] > 0).sum()
                    # Calculate sum and count of non-zero values
                    non_zero_values = df[df['Value'] > 0]['Value']
                    non_zero_sum += non_zero_values.sum()
                    non_zero_count += len(non_zero_values)
                    # Collect all values for standard deviation calculation
                    all_values.extend(df['Value'].tolist())

            mean_count = total_sum / genome_size
            mean_non_zero = non_zero_sum / non_zero_count if non_zero_count > 0 else 0
            std_dev = np.std(all_values) if len(all_values) > 0 else 0
            unoccupied_sites = genome_size - occupied_sites
            label = label_from_filename(wig_file)
            density = occupied_sites / (occupied_sites + unoccupied_sites) 

            stats.append((label, total_sum, mean_count, occupied_sites, unoccupied_sites, density, mean_non_zero, std_dev))

    with open(output_file, 'w') as f:
        f.write("Sample\tTotal_Sum\tMean_Coverage_per_bp\tOccupied_Sites\tUnoccupied_Sites\tDensity\tMean_Non_Zero_Count\tStd_Dev\n")
        for s in stats:
            f.write(f"{s[0]}\t{s[1]}\t{s[2]:.6f}\t{s[3]}\t{s[4]}\t{s[5]:.6f}\t{s[6]:.6f}\t{s[7]:.6f}\n")
            
    if plot:
        plot_basic_statistics(stats, output_folder_figures)


def read_csv_combined_replicates(folder_path):
    """
    Reads CSV files from combined_replicates format and returns a dict of DataFrames, one per chromosome.
    Combined replicates have columns: Position, Value, Nucleosome_Distance, Centromere_Distance
    """
    chrom_data = {}
    
    for file in os.listdir(folder_path):
        if file.endswith('.csv') and file.startswith('Chr'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            # Extract chromosome name from filename (e.g., ChrI_distances.csv -> ChrI)
            chrom = file.split('_')[0]
            
            # Keep only Position and Value columns to match the wig format
            if 'Position' in df.columns and 'Value' in df.columns:
                chrom_data[chrom] = df[['Position', 'Value']]
    
    return chrom_data


def save_basic_info_csv(folder_name, output_file, plot=False, output_folder_figures=None):
    """
    Saves basic statistics to a text file for CSV format (combined_replicates).
    This handles CSV files with zeros included, unlike wiggle files which omit zeros.
    """
    stats = []  # (label, total_sum, mean_count, occupied_sites, unoccupied_sites)

    genome_size = sum(yeast_chrom_lengths.values())
    
    for root, dirs, files in os.walk(folder_name):
        # Check if this folder contains CSV files (not subdirectories)
        csv_files = [f for f in files if f.endswith('.csv') and f.startswith('Chr')]
        if not csv_files:
            continue
            
        print(f"Processing: {root}")
        
        # Read all CSV files for this sample
        sample_dict = read_csv_combined_replicates(root)
        
        if len(sample_dict) == 0:
            continue
            
        # Get label from the folder name
        label = label_from_filename(os.path.basename(root))

        total_sum = 0
        occupied_sites = 0
        non_zero_sum = 0
        non_zero_count = 0
        all_values = []

        for chrom, df in sample_dict.items():
            # Skip mitochondrial chromosome if desired
            if chrom == 'ChrM':
                continue
                
            # Filter out positions with values > 1 million
            high_values = df[df['Value'] > 1000000]
            if not high_values.empty:
                for _, row in high_values.iterrows():
                    print(f"Filtered out high value in {label}, {chrom}, Position {row['Position']}: {row['Value']}")
                df = df[df['Value'] <= 1000000]
            
            total_sum += df['Value'].sum()
            occupied_sites += (df['Value'] > 0).sum()
            
            # Calculate sum and count of non-zero values
            non_zero_values = df[df['Value'] > 0]['Value']
            non_zero_sum += non_zero_values.sum()
            non_zero_count += len(non_zero_values)
            
            # Collect all values for standard deviation calculation
            all_values.extend(df['Value'].tolist())

        mean_count = total_sum / genome_size
        mean_non_zero = non_zero_sum / non_zero_count if non_zero_count > 0 else 0
        std_dev = np.std(all_values) if len(all_values) > 0 else 0
        unoccupied_sites = genome_size - occupied_sites
        density = occupied_sites / genome_size if genome_size > 0 else 0

        stats.append((label, total_sum, mean_count, occupied_sites, unoccupied_sites, density, mean_non_zero, std_dev))

    # Sort stats by label for consistent output
    stats.sort(key=lambda x: x[0])

    with open(output_file, 'w') as f:
        f.write("Sample\tTotal_Sum\tMean_Coverage_per_bp\tOccupied_Sites\tUnoccupied_Sites\tDensity\tMean_Non_Zero_Count\tStd_Dev\n")
        for s in stats:
            f.write(f"{s[0]}\t{s[1]}\t{s[2]:.6f}\t{s[3]}\t{s[4]}\t{s[5]:.6f}\t{s[6]:.6f}\t{s[7]:.6f}\n")
            
    if plot:
        plot_basic_statistics(stats, output_folder_figures)
        
def data_distribution(data, sample_name, ignore_zeros=True, output_folder="Data_exploration/results/count_distribution"):
    """Plot and save the transposon count data in a histogram and a datafile.
    
    Parameters:
    - data: numpy array or list of count values
    - sample_name: name of the sample for labeling
    - ignore_zeros: if True, exclude positions with Value=0
    - output_folder: folder to save outputs
    
    Creates:
    - A histogram plot (PNG)
    - A CSV file with the distribution data
    - A percentile analysis plot and statistics
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert to numpy array if needed
    data = np.array(data)
    
    # Filter zeros if requested
    if ignore_zeros:
        data = data[data > 0]
        zero_suffix = "_no_zeros"
    else:
        zero_suffix = "_with_zeros"
    
    if len(data) == 0:
        print(f"Warning: No data for {sample_name} after filtering")
        return
    
    # Calculate percentiles
    p5 = np.percentile(data, 5)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)  # median
    p75 = np.percentile(data, 75)
    p95 = np.percentile(data, 95)
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    # Data within 5-95 percentile range
    data_5_95 = data[(data >= p5) & (data <= p95)]
    mean_5_95 = np.mean(data_5_95)
    std_5_95 = np.std(data_5_95)
    
    # Calculate distribution
    unique_counts, counts = np.unique(data, return_counts=True)
    percentages = (counts / len(data)) * 100
    
    # Save distribution data to CSV
    csv_path = os.path.join(output_folder, f"{sample_name}{zero_suffix}_distribution.csv")
    df_dist = pd.DataFrame({
        'Count': unique_counts,
        'Frequency': counts,
        'Percentage': percentages
    })
    df_dist.to_csv(csv_path, index=False)
    
    # Save percentile statistics
    stats_path = os.path.join(output_folder, f"{sample_name}{zero_suffix}_percentile_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Percentile Statistics for {sample_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total positions: {len(data)}\n")
        f.write(f"Unique count values: {len(unique_counts)}\n\n")
        f.write("Percentiles:\n")
        f.write(f"  5th percentile:  {p5:.2f}\n")
        f.write(f"  25th percentile: {p25:.2f}\n")
        f.write(f"  50th percentile (median): {p50:.2f}\n")
        f.write(f"  75th percentile: {p75:.2f}\n")
        f.write(f"  95th percentile: {p95:.2f}\n\n")
        f.write("Overall statistics:\n")
        f.write(f"  Mean: {mean_val:.2f}\n")
        f.write(f"  Std Dev: {std_val:.2f}\n\n")
        f.write("Statistics for 5-95 percentile range:\n")
        f.write(f"  Data points in range: {len(data_5_95)} ({len(data_5_95)/len(data)*100:.1f}%)\n")
        f.write(f"  Mean: {mean_5_95:.2f}\n")
        f.write(f"  Std Dev: {std_5_95:.2f}\n")
    
    # Create two subplots: full distribution and 5-95 percentile range
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full distribution
    ax1 = axes[0]
    max_count = np.max(data)
    min_count = np.min(data)
    
    # Better binning strategy based on data range
    if max_count > 1000:
        # For very large ranges, use intelligent binning
        # Use finer bins for low values, coarser for high values
        bins_low = np.arange(min_count, min(100, p95), 1)
        bins_high = np.logspace(np.log10(max(100, p95)), np.log10(max_count + 1), 30)
        bins = np.concatenate([bins_low, bins_high])
        ax1.set_xscale('log')
    elif max_count > 100:
        # For medium ranges, use adaptive binning
        bins = np.linspace(min_count, max_count, min(100, int(max_count - min_count + 1)))
    else:
        # For small ranges, use integer bins
        bins = np.arange(min_count, max_count + 2) - 0.5  # Center bins on integers
    
    ax1.hist(data, bins=bins, color='steelblue', alpha=0.7, edgecolor='black', 
             weights=np.ones(len(data)) / len(data) * 100)
    ax1.axvline(p5, color='red', linestyle='--', linewidth=2, label=f'5th percentile ({p5:.1f})')
    ax1.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'95th percentile ({p95:.1f})')
    ax1.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'Median ({p50:.1f})')
    
    ax1.set_title(f'Full Transposon Count Distribution - {sample_name}{"" if not ignore_zeros else " (excluding zeros)"}')
    ax1.set_xlabel('Transposon Count')
    ax1.set_ylabel('Percentage of Positions (%)')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Plot 2: 5-95 percentile range only
    ax2 = axes[1]
    # Use linear bins for the percentile range
    range_size = p95 - p5
    if range_size > 100:
        bins_5_95 = np.linspace(p5, p95, 100)
    elif range_size > 50:
        bins_5_95 = np.linspace(p5, p95, 50)
    else:
        # For narrow ranges, use integer bins
        bins_5_95 = np.arange(int(p5), int(p95) + 2) - 0.5
    
    ax2.hist(data_5_95, bins=bins_5_95, color='darkseagreen', alpha=0.7, edgecolor='black',
             weights=np.ones(len(data_5_95)) / len(data_5_95) * 100)
    ax2.axvline(mean_5_95, color='orange', linestyle='--', linewidth=2, label=f'Mean in range ({mean_5_95:.1f})')
    ax2.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'Median ({p50:.1f})')
    
    ax2.set_title(f'5-95 Percentile Range (n={len(data_5_95)}, {len(data_5_95)/len(data)*100:.1f}% of data)')
    ax2.set_xlabel('Transposon Count')
    ax2.set_ylabel('Percentage of Positions (%)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_folder, f"distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Create boxplot (using 5-95 percentile data for better visualization)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    bp = ax.boxplot([data_5_95], vert=True, patch_artist=True, labels=[sample_name],
                     showmeans=True, meanline=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     meanprops=dict(color='red', linewidth=2),
                     medianprops=dict(color='darkgreen', linewidth=2))
    ax.set_ylabel('Transposon Count')
    ax.set_title(f'Boxplot (5-95 percentile range) - {sample_name}{"" if not ignore_zeros else " (excluding zeros)"}\n5th: {p5:.1f}, Median: {p50:.1f}, 95th: {p95:.1f}')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save boxplot
    boxplot_path = os.path.join(output_folder, f"boxplot.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    
    print(f"  Saved: {sample_name}{zero_suffix} - Total: {len(data)}, P5-P95: [{p5:.1f}, {p95:.1f}]")


def analyze_count_dsitribution(base_folder="Data/distances_with_zeros", ignore_zeros=True, output_base_folder="Data_exploration/results/count_distribution"):
    """
    Analyzes transposon count distribution for all samples in distances_with_zeros folder.
    Creates individual plots for each sample and one combined plot for all samples together.
    Each dataset gets its own folder within the output directory.
    
    Parameters:
    - base_folder: path to distances_with_zeros folder
    - ignore_zeros: if True, exclude positions with Value=0
    - output_base_folder: base folder to save outputs
    """
    print(f"\nAnalyzing transposon count distributions (ignore_zeros={ignore_zeros})...")
    print("=" * 80)
    
    all_data = []  # To collect all data for combined analysis
    
    # Determine zero suffix for folder naming
    zero_suffix = "_no_zeros" if ignore_zeros else "_with_zeros"
    
    # Iterate through all strain folders
    for strain_folder in sorted(os.listdir(base_folder)):
        strain_path = os.path.join(base_folder, strain_folder)
        if not os.path.isdir(strain_path):
            continue
        
        print(f"\nProcessing strain: {strain_folder}")
        
        # Iterate through all sample folders within the strain
        for sample_folder in sorted(os.listdir(strain_path)):
            sample_path = os.path.join(strain_path, sample_folder)
            if not os.path.isdir(sample_path):
                continue
            
            print(f"  Sample: {sample_folder}")
            
            # Read all chromosome CSV files for this sample
            sample_data = []
            
            for csv_file in sorted(os.listdir(sample_path)):
                if not csv_file.endswith('_distances.csv'):
                    continue
                
                csv_path = os.path.join(sample_path, csv_file)
                df = pd.read_csv(csv_path)
                
                # Extract Value column
                if 'Value' in df.columns:
                    sample_data.extend(df['Value'].tolist())
            
            if len(sample_data) > 0:
                # Create sanitized sample name
                sample_name = sample_folder.replace('_merged-DpnII-NlaIII-a_trimmed.sorted.bam', '') \
                                          .replace('_merged-DpnII-NlaIII-b_trimmed.sorted.bam', '') \
                                          .replace('.bam', '')
                
                # Create sample-specific output folder
                sample_output_folder = os.path.join(output_base_folder, sample_name + zero_suffix)
                os.makedirs(sample_output_folder, exist_ok=True)
                
                # Plot distribution for this sample
                data_distribution(sample_data, sample_name, ignore_zeros=ignore_zeros, output_folder=sample_output_folder)
                
                # Add to combined data
                all_data.extend(sample_data)
    
    # Create combined plot for all samples
    if len(all_data) > 0:
        print("\n" + "=" * 80)
        print("Creating combined distribution for all samples...")
        combined_output_folder = os.path.join(output_base_folder, "all_samples_combined" + zero_suffix)
        os.makedirs(combined_output_folder, exist_ok=True)
        data_distribution(all_data, "all_samples_combined", ignore_zeros=ignore_zeros, output_folder=combined_output_folder)
    
    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {output_base_folder}")
    print(f"Total datasets processed: {len(all_data)}")


# Example usage:
# For wiggle format files (without zeros):
# save_basic_info("Data/old/wiggle_format", "Data_exploration/results/basic_statistics_wiggle.txt", plot=False, output_folder_figures="Data_exploration/figures")

# For combined_replicates CSV files (with zeros):
# save_basic_info_csv("Data/combined_strains", "Data_exploration/results/basic_statistics_combined_strains.txt", plot=False, output_folder_figures="Data_exploration/figures")

# For analyzing count distributions in distances_with_zeros:
analyze_count_dsitribution(ignore_zeros=True)
# analyze_count_dsitribution(ignore_zeros=False)