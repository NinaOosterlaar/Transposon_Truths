import json
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.reader import read_csv_file_with_distances

def find_outliers(input_folder, output_file, percentile=95, min_value_threshold=None, filtered_output_file=None):
    """Find all the outliers in the count data, 
    and save the position and value per dataset and per chromosome in a json file.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing the data
    output_file : str
        Path to save all outlier statistics
    percentile : float
        Percentile threshold for outlier detection
    min_value_threshold : float, optional
        Minimum value threshold for filtered output (e.g., 10000)
    filtered_output_file : str, optional
        Path to save filtered outlier statistics (only outliers above min_value_threshold)
    """
    transposon_data = read_csv_file_with_distances(input_folder)
    outlier_stats = {}
    outlier_stats_filtered = {}
    
    for dataset in transposon_data:
        outlier_stats[dataset] = {}
        if min_value_threshold is not None:
            outlier_stats_filtered[dataset] = {}
            
        for chrom in transposon_data[dataset]:
            df = transposon_data[dataset][chrom]
            if 'Value' not in df.columns:
                continue
            values = df['Value'].values
            non_zero_values = values[values > 0]
            if len(non_zero_values) == 0:
                print(f"Warning: No non-zero values found for {dataset} - {chrom}")
                continue
            threshold = np.percentile(non_zero_values, percentile)
            outlier_indices = np.where(values > threshold)[0]
            outlier_values = values[outlier_indices].tolist()
            outlier_positions = df.iloc[outlier_indices]['Position'].tolist()
            
            # Store all outliers
            outlier_stats[dataset][chrom] = {
                'threshold': threshold,
                'outliers': [
                    {'position': pos, 'value': val} 
                    for pos, val in zip(outlier_positions, outlier_values)
                ]
            }
            print(f"{dataset} - {chrom}: Found {len(outlier_indices)} outliers above {threshold:.2f}")
            
            # Store filtered outliers if threshold is specified
            if min_value_threshold is not None:
                filtered_outliers = [
                    {'position': pos, 'value': val} 
                    for pos, val in zip(outlier_positions, outlier_values)
                    if val > min_value_threshold
                ]
                if filtered_outliers:
                    outlier_stats_filtered[dataset][chrom] = {
                        'threshold': threshold,
                        'outliers': filtered_outliers
                    }
                    print(f"  -> {len(filtered_outliers)} outliers above {min_value_threshold}")

    # Save outlier statistics to JSON file
    with open(output_file, 'w') as f:
        json.dump(outlier_stats, f, indent=4)
    print(f"\nAll outliers saved to {output_file}")
    
    # Save filtered outliers if requested
    if min_value_threshold is not None and filtered_output_file is not None:
        with open(filtered_output_file, 'w') as f:
            json.dump(outlier_stats_filtered, f, indent=4)
        print(f"Filtered outliers (value > {min_value_threshold}) saved to {filtered_output_file}")

def find_cross_dataset_outliers(json_file, position_window=5, output_file=None):
    """
    Find outliers from different datasets that are within a specified distance from each other.
    Excludes comparisons between technical replicates of the same sample.
    
    Parameters:
    -----------
    json_file : str
        Path to the JSON file containing outlier statistics
    position_window : int
        Maximum distance between outlier positions to be considered neighbors (default: 5)
    output_file : str
        Optional path to save the results
    
    Returns:
    --------
    dict : Dictionary containing flagged outlier clusters
    """
    
    def get_replicate_group(dataset_name):
        """Determine which technical replicate group a dataset belongs to."""
        dataset_lower = dataset_name.lower()
        # FD7 replicates
        if 'fd7' in dataset_lower:
            return 'FD7_group'
        # FD9 replicates
        elif 'fd9' in dataset_lower:
            return 'FD9_group'
        # dnrp1-1 replicates
        elif 'dnrp1-1' in dataset_lower:
            return 'dnrp1-1_group'
        # dnrp1-2 replicates
        elif 'dnrp1-2' in dataset_lower:
            return 'dnrp1-2_group'
        # If not a known replicate group, return the dataset name itself
        else:
            return dataset_name
    
    print(f"Loading outlier data from {json_file}...")
    with open(json_file, 'r') as f:
        outlier_stats = json.load(f)
    
    # Organize outliers by chromosome and position for efficient lookup
    flagged_clusters = {}
    
    # Track flagged outliers for percentage calculation
    flagged_outliers = set()  # Set of (dataset, chrom, position) tuples
    total_outliers = 0
    
    # Create overlap matrix for heatmap - track number of overlaps between dataset pairs
    all_datasets = sorted(outlier_stats.keys())
    overlap_matrix = pd.DataFrame(0, index=all_datasets, columns=all_datasets)
    
    # Get all chromosomes present in the data
    all_chromosomes = set()
    for dataset in outlier_stats:
        all_chromosomes.update(outlier_stats[dataset].keys())
    
    # Count total outliers
    for dataset in outlier_stats:
        for chrom in outlier_stats[dataset]:
            outliers = outlier_stats[dataset][chrom].get('outliers', [])
            total_outliers += len(outliers)
    
    print(f"Found {len(all_chromosomes)} chromosomes across {len(outlier_stats)} datasets")
    print(f"Total outliers across all datasets: {total_outliers}")
    
    # Process each chromosome
    for chrom in sorted(all_chromosomes):
        print(f"\nProcessing {chrom}...")
        
        # Collect all outliers for this chromosome across all datasets
        outliers_by_dataset = {}
        for dataset in outlier_stats:
            if chrom in outlier_stats[dataset]:
                outliers = outlier_stats[dataset][chrom].get('outliers', [])
                if outliers:
                    outliers_by_dataset[dataset] = outliers
        
        if len(outliers_by_dataset) < 2:
            print(f"  Skipping {chrom} - only {len(outliers_by_dataset)} dataset(s) with outliers")
            continue
        
        # Find clusters of outliers from different datasets
        clusters = []
        datasets = list(outliers_by_dataset.keys())
        
        # Compare each pair of datasets
        for i, dataset1 in enumerate(datasets):
            for dataset2 in datasets[i+1:]:
                # Skip if both datasets are technical replicates of the same sample
                group1 = get_replicate_group(dataset1)
                group2 = get_replicate_group(dataset2)
                if group1 == group2:
                    continue
                
                outliers1 = outliers_by_dataset[dataset1]
                outliers2 = outliers_by_dataset[dataset2]
                
                # Check each outlier in dataset1 against all outliers in dataset2
                for outlier1 in outliers1:
                    pos1 = outlier1['position']
                    val1 = outlier1['value']
                    
                    for outlier2 in outliers2:
                        pos2 = outlier2['position']
                        val2 = outlier2['value']
                        
                        # Check if positions are within the window
                        if abs(pos1 - pos2) <= position_window:
                            cluster = {
                                'chromosome': chrom,
                                'datasets': {
                                    dataset1: {'position': pos1, 'value': val1},
                                    dataset2: {'position': pos2, 'value': val2}
                                },
                                'distance': abs(pos1 - pos2)
                            }
                            clusters.append(cluster)
                            
                            # Track these outliers as flagged
                            flagged_outliers.add((dataset1, chrom, pos1))
                            flagged_outliers.add((dataset2, chrom, pos2))
                            
                            # Update overlap matrix
                            overlap_matrix.loc[dataset1, dataset2] += 1
                            overlap_matrix.loc[dataset2, dataset1] += 1
        
        if clusters:
            flagged_clusters[chrom] = clusters
            print(f"  Found {len(clusters)} flagged outlier pairs")
    
    # Summary
    total_flags = sum(len(clusters) for clusters in flagged_clusters.values())
    num_flagged_outliers = len(flagged_outliers)
    percentage_flagged = (num_flagged_outliers / total_outliers * 100) if total_outliers > 0 else 0
    
    # Print detailed results
    for chrom in sorted(flagged_clusters.keys()):
        print(f"\n{chrom}:")
        for i, cluster in enumerate(flagged_clusters[chrom], 1):
            datasets_info = cluster['datasets']
            dataset_names = list(datasets_info.keys())
            print(f"  Cluster {i}:")
            for dataset in dataset_names:
                info = datasets_info[dataset]
                print(f"    {dataset}: position {info['position']}, value {info['value']:.2f}")
            print(f"    Distance: {cluster['distance']} bp")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total outliers: {total_outliers}")
    print(f"  Outliers with cross-dataset flags: {num_flagged_outliers} ({percentage_flagged:.2f}%)")
    print(f"  Total flagged outlier pairs: {total_flags}")
    print(f"  Chromosomes with flagged outliers: {len(flagged_clusters)}")
    print(f"{'='*60}")
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(flagged_clusters, f, indent=4)
        print(f"\nResults saved to {output_file}")
    
    # Create heatmap of dataset overlaps
    if len(all_datasets) > 1:
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        sns.heatmap(overlap_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                    square=True, linewidths=0.5, cbar_kws={'label': 'Number of overlapping outliers'},
                    ax=ax)
        
        ax.set_title('Cross-Dataset Outlier Overlaps\n(Outliers within 5bp of each other)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save heatmap
        if output_file:
            heatmap_file = output_file.replace('.json', '_heatmap.png')
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {heatmap_file}")
        
        plt.close()
    
    return flagged_clusters, overlap_matrix

if __name__ == "__main__":
    input_folder = "Data/distances_with_zeros_new"
    output_file = "Data_exploration/results/outliers/outlier_stats.json"
    filtered_output_file = "Data_exploration/results/outliers/outlier_stats_filtered_10k.json"
    percentile = 99.99 # Percentile threshold
    min_value_threshold = 10000  # Minimum value for filtered output

    # Step 1: Find outliers (comment out if already done)
    # find_outliers(input_folder, output_file, percentile, 
    #               min_value_threshold=min_value_threshold, 
    #               filtered_output_file=filtered_output_file)
    
    # Step 2: Find cross-dataset outliers
    cross_dataset_output = "Data_exploration/results/outliers/cross_dataset_outliers_filtered_10k.json"
    find_cross_dataset_outliers(filtered_output_file, position_window=5, output_file=cross_dataset_output)