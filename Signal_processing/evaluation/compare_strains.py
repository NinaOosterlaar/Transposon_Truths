
"""
Compare change point detection results across different SATAY strains.

Computes Jaccard Index and Adjusted Rand Index between all strain pairs
at specified thresholds to measure overlap in detected change points.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS
from Signal_processing.evaluation.evaluation import (
    calculate_jaccard_index, 
    adjusted_rand_index,
    jaccard_index_with_tolerance,
    mean_nearest_breakpoint_distance
)

# Set up standardized plot style
setup_plot_style()

# Chromosome lengths in base pairs
chromosome_length = {
    "ChrI": 230218,
    "ChrII": 813184,
    "ChrIII": 316620,
    "ChrIV": 1531933,
    "ChrV": 576874,
    "ChrVI": 270161,
    "ChrVII": 1090940,
    "ChrVIII": 562643,
    "ChrIX": 439888,
    "ChrX": 745751,
    "ChrXI": 666816,
    "ChrXII": 1078171,
    "ChrXIII": 924431,
    "ChrXIV": 784333,
    "ChrXV": 1091291,
    "ChrXVI": 948066,
}

# All chromosomes in order
ALL_CHROMOSOMES = list(chromosome_length.keys())

# List of strains to process
# Comment out any strains you want to exclude from the analysis
STRAINS = [
    'strain_FD',
    'strain_dnrp',
    'strain_yEK19',
    'strain_yEK23',
    'strain_yTW001',
    'strain_yWT03a',
    'strain_yWT04a',
    'strain_ylic137'
]


def load_changepoints_for_strain(strain_name, threshold, window_size=100, overlap=50, base_path=None):
    """
    Load change points for all chromosomes of a given strain at a specific threshold.
    
    Parameters
    ----------
    strain_name : str
        Name of the strain (e.g., 'strain_FD', 'strain_dnrp')
    threshold : float
        Threshold value to load
    window_size : int, optional
        Window size parameter (default: 100)
    overlap : int, optional
        Overlap percentage (default: 50)
    base_path : Path, optional
        Base path to the strains directory. If None, uses default.
        
    Returns
    -------
    dict
        Dictionary with chromosome names as keys and sets of change points as values
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "strains"
    
    strain_path = base_path / strain_name
    changepoints = {}
    
    for chrom in ALL_CHROMOSOMES:
        # Path to change point file
        cp_file = (strain_path / chrom / f"{chrom}_distances" / f"window{window_size}" / 
                   f"{chrom}_distances_ws{window_size}_ov{overlap}_th{threshold:.2f}.txt")
        
        if cp_file.exists():
            # Read change points (one per line), skip metadata lines
            cps = []
            with open(cp_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and lines containing metadata (scores:, theta_global:, etc.)
                    if line and not any(line.startswith(prefix) for prefix in ['scores:', 'theta_global:', 'window_size:']):
                        try:
                            cps.append(int(line))
                        except ValueError:
                            # Skip lines that aren't valid integers
                            continue
            changepoints[chrom] = set(cps)
        else:
            # Empty set if file doesn't exist
            changepoints[chrom] = set()
    
    return changepoints


def compute_aggregated_jaccard(strain_cps_a, strain_cps_b):
    """
    Compute Jaccard Index aggregated across all chromosomes.
    
    Parameters
    ----------
    strain_cps_a : dict
        Change points for strain A (chromosome -> set of positions)
    strain_cps_b : dict
        Change points for strain B (chromosome -> set of positions)
        
    Returns
    -------
    float
        Jaccard Index aggregated across all chromosomes
    """
    return calculate_jaccard_index(strain_cps_a, strain_cps_b)


def compute_aggregated_ari(strain_cps_a, strain_cps_b):
    """
    Compute Adjusted Rand Index aggregated across all chromosomes.
    
    For each chromosome:
    - Converts change points to segmentation labels
    - Computes ARI
    Then aggregates using weighted average by chromosome length.
    
    Parameters
    ----------
    strain_cps_a : dict
        Change points for strain A (chromosome -> set of positions)
    strain_cps_b : dict
        Change points for strain B (chromosome -> set of positions)
        
    Returns
    -------
    float
        Weighted average ARI across all chromosomes
    """
    ari_values = []
    weights = []
    
    for chrom in ALL_CHROMOSOMES:
        n_points = chromosome_length[chrom]
        cps_a = sorted(list(strain_cps_a.get(chrom, set())))
        cps_b = sorted(list(strain_cps_b.get(chrom, set())))
        
        ari = adjusted_rand_index(cps_a, cps_b, n_points)
        ari_values.append(ari)
        weights.append(n_points)
    
    # Weighted average by chromosome length
    total_length = sum(weights)
    weighted_ari = sum(a * w for a, w in zip(ari_values, weights)) / total_length
    
    return weighted_ari


def compute_aggregated_jaccard_with_tolerance(strain_cps_a, strain_cps_b, tol=100):
    """
    Compute Jaccard Index with tolerance aggregated across all chromosomes.
    
    Uses one-to-one matching within tolerance window.
    
    Parameters
    ----------
    strain_cps_a : dict
        Change points for strain A (chromosome -> set of positions)
    strain_cps_b : dict
        Change points for strain B (chromosome -> set of positions)
    tol : int, optional
        Tolerance window in base pairs (default: 100)
        
    Returns
    -------
    float
        Jaccard Index with tolerance
    """
    return jaccard_index_with_tolerance(strain_cps_a, strain_cps_b, tol)


def compute_aggregated_nearest_breakpoint_distance(strain_cps_a, strain_cps_b):
    """
    Compute mean nearest-breakpoint distance (asymmetric) between strains.
    
    Returns distances in both directions (A→B and B→A).
    
    Parameters
    ----------
    strain_cps_a : dict
        Change points for strain A (chromosome -> set of positions)
    strain_cps_b : dict
        Change points for strain B (chromosome -> set of positions)
        
    Returns
    -------
    dict
        Dictionary with 'a_to_b' and 'b_to_a' keys containing mean distances
    """
    return mean_nearest_breakpoint_distance(strain_cps_a, strain_cps_b, chromosome_length)


def load_essentiality_for_strain(strain_name, threshold, window_size=100, overlap=50, base_path=None):
    """
    Load essentiality z-scores for all chromosomes of a given strain at a specific threshold.
    
    Creates a position-to-z-score mapping by reading segment_mu CSV files.
    
    Parameters
    ----------
    strain_name : str
        Name of the strain (e.g., 'strain_FD', 'strain_dnrp')
    threshold : float
        Threshold value to load
    window_size : int, optional
        Window size parameter (default: 100)
    overlap : int, optional
        Overlap percentage (default: 50)
    base_path : Path, optional
        Base path to the strains directory. If None, uses default.
        
    Returns
    -------
    dict
        Dictionary with chromosome names as keys and numpy arrays (position-to-z-score) as values.
        Positions with NaN z-scores are set to NaN.
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "strains"
    
    strain_path = base_path / strain_name
    essentiality_data = {}
    
    files_loaded = 0
    files_missing = 0
    total_valid_positions = 0
    
    for chrom in ALL_CHROMOSOMES:
        # Path to segment_mu CSV file
        segment_file = (strain_path / chrom / f"{chrom}_distances" / f"window{window_size}" / 
                       "segment_mu" / f"{chrom}_distances_ws{window_size}_ov{overlap}_th{threshold:.2f}_segment_mu.csv")
        
        if not segment_file.exists():
            # No essentiality data for this chromosome - create array of NaNs
            chrom_length = chromosome_length[chrom]
            essentiality_data[chrom] = np.full(chrom_length, np.nan)
            files_missing += 1
            continue
        
        # Read segment data
        try:
            df = pd.read_csv(segment_file)
            chrom_length = chromosome_length[chrom]
            
            # Initialize array with NaN
            z_score_array = np.full(chrom_length, np.nan)
            
            # Fill in z-scores for each segment
            for _, row in df.iterrows():
                start = int(row['start_index'])
                end = int(row['end_index_exclusive'])
                z_score = row['mu_z_score']
                
                # Assign z-score to all positions in this segment
                if start < chrom_length and end <= chrom_length:
                    z_score_array[start:end] = z_score
            
            essentiality_data[chrom] = z_score_array
            files_loaded += 1
            
            # Count valid (non-NaN) positions
            n_valid = np.sum(~np.isnan(z_score_array))
            total_valid_positions += n_valid
            
        except Exception as e:
            print(f"    Warning: Could not load {strain_name} {chrom} threshold {threshold}: {e}")
            chrom_length = chromosome_length[chrom]
            essentiality_data[chrom] = np.full(chrom_length, np.nan)
            files_missing += 1
    
    # Print summary
    if files_loaded > 0 or files_missing > 0:
        print(f"    {strain_name}: loaded {files_loaded} chromosomes, missing {files_missing}, "
              f"valid positions: {total_valid_positions:,}")
    
    return essentiality_data


def compute_essentiality_correlation(strain_ess_a, strain_ess_b, method='pearson'):
    """
    Compute correlation of essentiality z-scores between two strains.
    
    Concatenates all chromosomes and computes correlation across all genomic positions.
    Positions where either strain has NaN are excluded.
    
    Parameters
    ----------
    strain_ess_a : dict
        Essentiality data for strain A (chromosome -> z-score array)
    strain_ess_b : dict
        Essentiality data for strain B (chromosome -> z-score array)
    method : str, optional
        'pearson' or 'spearman' (default: 'pearson')
        
    Returns
    -------
    float
        Correlation coefficient. Returns NaN if insufficient valid data.
    """
    # Concatenate all chromosomes in order
    all_z_a = []
    all_z_b = []
    
    for chrom in ALL_CHROMOSOMES:
        z_a = strain_ess_a.get(chrom, np.array([]))
        z_b = strain_ess_b.get(chrom, np.array([]))
        
        # Ensure same length
        min_len = min(len(z_a), len(z_b))
        if min_len > 0:
            all_z_a.extend(z_a[:min_len])
            all_z_b.extend(z_b[:min_len])
    
    # Convert to numpy arrays
    all_z_a = np.array(all_z_a)
    all_z_b = np.array(all_z_b)
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(all_z_a) | np.isnan(all_z_b))
    
    n_valid = np.sum(valid_mask)
    
    if n_valid < 2:
        # Not enough valid data points
        return np.nan
    
    z_a_valid = all_z_a[valid_mask]
    z_b_valid = all_z_b[valid_mask]
    
    # Check for zero variance (all values identical)
    if np.std(z_a_valid) == 0 or np.std(z_b_valid) == 0:
        print(f"    Warning: Zero variance detected. Valid points: {n_valid}, "
              f"Strain A: mean={np.mean(z_a_valid):.4f} std={np.std(z_a_valid):.4f}, "
              f"Strain B: mean={np.mean(z_b_valid):.4f} std={np.std(z_b_valid):.4f}")
        return np.nan
    
    # Compute correlation
    try:
        if method == 'pearson':
            corr, p_value = pearsonr(z_a_valid, z_b_valid)
        elif method == 'spearman':
            corr, p_value = spearmanr(z_a_valid, z_b_valid)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'.")
        
        # Debug: print if correlation is suspiciously high
        if abs(corr) > 0.99 and n_valid > 10:
            print(f"    High correlation detected: {method}={corr:.6f}, n_valid={n_valid}, "
                  f"p-value={p_value:.6e}")
        
        return corr
    except Exception as e:
        print(f"Warning: Could not compute {method} correlation: {e}")
        return np.nan


def compute_pairwise_metrics(strain_names, threshold, tolerance=100):
    """
    Compute all similarity metrics for all pairs of strains at a given threshold.
    
    Parameters
    ----------
    strain_names : list of str
        List of strain names to compare
    threshold : float
        Threshold value to use
    tolerance : int, optional
        Tolerance window for Jaccard index (default: 100 bp)
        
    Returns
    -------
    dict
        Dictionary containing DataFrames for each metric:
        - 'jaccard': Jaccard Index (exact matching)
        - 'jaccard_tol': Jaccard Index with tolerance
        - 'ari': Adjusted Rand Index
        - 'nbp_dist_a_to_b': Nearest-breakpoint distance A→B (asymmetric)
        - 'nbp_dist_b_to_a': Nearest-breakpoint distance B→A (asymmetric)
        - 'ess_corr_pearson': Essentiality correlation (Pearson)
        - 'ess_corr_spearman': Essentiality correlation (Spearman)
    """
    n_strains = len(strain_names)
    jaccard_matrix = np.ones((n_strains, n_strains))
    jaccard_tol_matrix = np.ones((n_strains, n_strains))
    ari_matrix = np.ones((n_strains, n_strains))
    nbp_dist_a_to_b_matrix = np.zeros((n_strains, n_strains))
    nbp_dist_b_to_a_matrix = np.zeros((n_strains, n_strains))
    ess_corr_pearson_matrix = np.zeros((n_strains, n_strains))
    ess_corr_spearman_matrix = np.zeros((n_strains, n_strains))
    
    # Load change points for all strains
    strain_changepoints = {}
    for strain in strain_names:
        strain_changepoints[strain] = load_changepoints_for_strain(strain, threshold)
    
    # Load essentiality data for all strains
    print(f"  Loading essentiality data for threshold {threshold}...")
    strain_essentiality = {}
    for strain in strain_names:
        strain_essentiality[strain] = load_essentiality_for_strain(strain, threshold)
    
    # Compute pairwise metrics
    for i, strain_a in enumerate(strain_names):
        for j, strain_b in enumerate(strain_names):
            if i < j:  # Only compute upper triangle
                # Exact Jaccard Index
                jaccard = compute_aggregated_jaccard(
                    strain_changepoints[strain_a],
                    strain_changepoints[strain_b]
                )
                
                # Jaccard Index with tolerance
                jaccard_tol = compute_aggregated_jaccard_with_tolerance(
                    strain_changepoints[strain_a],
                    strain_changepoints[strain_b],
                    tol=tolerance
                )
                
                # Adjusted Rand Index
                ari = compute_aggregated_ari(
                    strain_changepoints[strain_a],
                    strain_changepoints[strain_b]
                )
                
                # Nearest-breakpoint distance (asymmetric)
                nbp_dist = compute_aggregated_nearest_breakpoint_distance(
                    strain_changepoints[strain_a],
                    strain_changepoints[strain_b]
                )
                
                # Essentiality correlations (symmetric)
                ess_pearson = compute_essentiality_correlation(
                    strain_essentiality[strain_a],
                    strain_essentiality[strain_b],
                    method='pearson'
                )
                
                ess_spearman = compute_essentiality_correlation(
                    strain_essentiality[strain_a],
                    strain_essentiality[strain_b],
                    method='spearman'
                )
                
                # Fill symmetric metrics in both triangles
                jaccard_matrix[i, j] = jaccard
                jaccard_matrix[j, i] = jaccard
                
                jaccard_tol_matrix[i, j] = jaccard_tol
                jaccard_tol_matrix[j, i] = jaccard_tol
                
                ari_matrix[i, j] = ari
                ari_matrix[j, i] = ari
                
                ess_corr_pearson_matrix[i, j] = ess_pearson
                ess_corr_pearson_matrix[j, i] = ess_pearson
                
                ess_corr_spearman_matrix[i, j] = ess_spearman
                ess_corr_spearman_matrix[j, i] = ess_spearman
                
                # Fill asymmetric metrics
                nbp_dist_a_to_b_matrix[i, j] = nbp_dist['a_to_b']
                nbp_dist_b_to_a_matrix[i, j] = nbp_dist['b_to_a']
                nbp_dist_a_to_b_matrix[j, i] = nbp_dist['b_to_a']  # Transpose
                nbp_dist_b_to_a_matrix[j, i] = nbp_dist['a_to_b']  # Transpose
    
    # Fill diagonal for correlation matrices (strain with itself = 1.0)
    for i in range(n_strains):
        ess_corr_pearson_matrix[i, i] = 1.0
        ess_corr_spearman_matrix[i, i] = 1.0
    
    # Convert to DataFrames with proper labels
    results = {
        'jaccard': pd.DataFrame(jaccard_matrix, index=strain_names, columns=strain_names),
        'jaccard_tol': pd.DataFrame(jaccard_tol_matrix, index=strain_names, columns=strain_names),
        'ari': pd.DataFrame(ari_matrix, index=strain_names, columns=strain_names),
        'nbp_dist_a_to_b': pd.DataFrame(nbp_dist_a_to_b_matrix, index=strain_names, columns=strain_names),
        'nbp_dist_b_to_a': pd.DataFrame(nbp_dist_b_to_a_matrix, index=strain_names, columns=strain_names),
        'ess_corr_pearson': pd.DataFrame(ess_corr_pearson_matrix, index=strain_names, columns=strain_names),
        'ess_corr_spearman': pd.DataFrame(ess_corr_spearman_matrix, index=strain_names, columns=strain_names)
    }
    
    return results
def plot_heatmap(matrix_df, metric_name, threshold, output_dir, metric_short_name=None):
    """
    Create and save a heatmap for a similarity/distance matrix.
    
    Parameters
    ----------
    matrix_df : pd.DataFrame
        Similarity or distance matrix (strain x strain)
    metric_name : str
        Full name of the metric for display
    threshold : float
        Threshold value used
    output_dir : Path
        Directory to save the figure
    metric_short_name : str, optional
        Short name for file naming (if None, derived from metric_name)
    """
    # Clean strain names for display (remove 'strain_' prefix)
    display_names = [name.replace('strain_', '') for name in matrix_df.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine color limits and format based on metric type
    if 'Jaccard' in metric_name:
        vmin, vmax = 0, 1
        cmap = 'YlOrRd'
        fmt = '.3f'
    elif 'ARI' in metric_name or 'Rand Index' in metric_name:
        vmin, vmax = 0, 1
        cmap = 'Blues'
        fmt = '.3f'
    elif 'Correlation' in metric_name or 'Essentiality' in metric_name:
        # Correlation metrics - range from 0 to 1
        vmin, vmax = 0, 1
        cmap = 'Blues'
        fmt = '.3f'
    elif 'Distance' in metric_name or 'NBP' in metric_name:
        # Distance metrics - lower is better, use reversed colormap
        vmin, vmax = 0, None  # Let data determine max
        cmap = 'YlOrRd_r'  # Reversed: yellow=far, red=close
        fmt = '.0f'  # Integer formatting for base pairs
    else:
        # Default
        vmin, vmax = None, None
        cmap = 'viridis'
        fmt = '.3f'
    
    # Create heatmap
    sns.heatmap(matrix_df.values, 
                annot=True, 
                fmt=fmt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'label': metric_name},
                xticklabels=display_names,
                yticklabels=display_names,
                linewidths=0.5,
                linecolor='gray',
                ax=ax)

    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    
    # Set title and labels
    ax.set_title(f'{metric_name} Between Strains\n(Threshold = {threshold:.1f})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Strain', fontsize=12, fontweight='bold')
    ax.set_ylabel('Strain', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if metric_short_name is None:
        if 'Jaccard' in metric_name and 'tolerance' in metric_name.lower():
            metric_short_name = 'jaccard_tol'
        elif 'Jaccard' in metric_name:
            metric_short_name = 'jaccard'
        elif 'ARI' in metric_name or 'Rand Index' in metric_name:
            metric_short_name = 'ari'
        elif 'NBP' in metric_name or 'Nearest' in metric_name:
            # Extract direction from metric name
            if 'A→B' in metric_name or 'A->B' in metric_name:
                metric_short_name = 'nbp_dist_a_to_b'
            else:
                metric_short_name = 'nbp_dist_b_to_a'
        else:
            metric_short_name = 'metric'
    
    output_file = output_dir / f'{metric_short_name}_heatmap_threshold_{threshold:.1f}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved heatmap: {output_file.name}")


def plot_changepoint_counts(strain_names, thresholds, output_dir):
    """
    Create bar plots showing the number of change points for each strain at each threshold.
    
    Parameters
    ----------
    strain_names : list of str
        List of strain names to analyze
    thresholds : list of float
        Threshold values to process
    output_dir : Path
        Directory to save the figures
    """
    print("Counting change points for all strains...")
    print()
    
    # Count change points for each strain and threshold
    counts = {threshold: {} for threshold in thresholds}
    
    for threshold in thresholds:
        for strain in strain_names:
            strain_cps = load_changepoints_for_strain(strain, threshold)
            # Sum across all chromosomes
            total_cps = sum(len(cps) for cps in strain_cps.values())
            counts[threshold][strain] = total_cps
    
    # Create a bar plot for each threshold
    for threshold in thresholds:
        # Clean strain names for display (remove 'strain_' prefix)
        display_names = [name.replace('strain_', '') for name in strain_names]
        strain_counts = [counts[threshold][strain] for strain in strain_names]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars
        bars = ax.bar(display_names, strain_counts, color=COLORS['blue'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Labels and title
        ax.set_xlabel('Strain', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Change Points', fontsize=12, fontweight='bold')
        ax.set_title(f'Change Point Counts per Strain (Threshold = {threshold:.1f})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        
        # Add grid for easier reading
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = output_dir / f'changepoint_counts_threshold_{threshold:.1f}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Threshold {threshold:.1f}:")
        for strain, count in counts[threshold].items():
            print(f"  {strain:20s}: {count:5d} change points")
        print(f"  Saved: {output_file.name}\n")
    
    print()


def main():
    """
    Main function to compare strains across multiple thresholds.
    """
    # Configuration
    strain_names = STRAINS  # Use the module-level constant
    
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5]  # Thresholds to analyze
    tolerance = 100  # bp tolerance for Jaccard with tolerance
    
    # Create output directory
    output_dir = Path(__file__).parent / "compare_strains2"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Comparing Change Point Detection Results Across Strains")
    print("=" * 80)
    print(f"Strains: {len(strain_names)}")
    for strain in strain_names:
        print(f"  - {strain}")
    print(f"\nThresholds: {thresholds}")
    print(f"Tolerance: {tolerance} bp")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()
    
    # Plot change point counts for all strains at all thresholds
    plot_changepoint_counts(strain_names, thresholds, output_dir)
    
    # Store all results for CSV export
    all_results = {
        'jaccard': {},
        'jaccard_tol': {},
        'ari': {},
        'nbp_dist_a_to_b': {},
        'nbp_dist_b_to_a': {},
        'ess_corr_pearson': {},
        'ess_corr_spearman': {}
    }
    
    # Process each threshold
    for threshold in thresholds:
        print(f"Processing threshold {threshold:.1f}...")
        
        # Compute pairwise metrics (returns dict of DataFrames)
        metrics = compute_pairwise_metrics(strain_names, threshold, tolerance=tolerance)
        
        # Store results
        for metric_key, df in metrics.items():
            all_results[metric_key][threshold] = df
        
        # Create heatmaps for all metrics
        plot_heatmap(metrics['jaccard'], 'Jaccard Index (Exact)', threshold, output_dir, 'jaccard')
        plot_heatmap(metrics['jaccard_tol'], f'Jaccard Index (Tolerance {tolerance}bp)', threshold, output_dir, 'jaccard_tol')
        plot_heatmap(metrics['ari'], 'Adjusted Rand Index', threshold, output_dir, 'ari')
        plot_heatmap(metrics['nbp_dist_a_to_b'], 'Mean Nearest-Breakpoint Distance (Row→Column)', threshold, output_dir, 'nbp_dist_a_to_b')
        plot_heatmap(metrics['ess_corr_pearson'], 'Essentiality Correlation (Pearson)', threshold, output_dir, 'ess_corr_pearson')
        plot_heatmap(metrics['ess_corr_spearman'], 'Essentiality Correlation (Spearman)', threshold, output_dir, 'ess_corr_spearman')
        # Optionally plot the reverse direction
        # plot_heatmap(metrics['nbp_dist_b_to_a'], 'Mean Nearest-Breakpoint Distance (Column→Row)', threshold, output_dir, 'nbp_dist_b_to_a')
        
        print(f"  Threshold {threshold:.1f} complete\n")
    
    # Save results to CSV files
    print("Saving results to CSV files...")
    
    for metric_name, threshold_dict in all_results.items():
        for threshold, df in threshold_dict.items():
            csv_file = output_dir / f'{metric_name}_threshold_{threshold:.1f}.csv'
            df.to_csv(csv_file)
        print(f"  Saved {metric_name} results for {len(thresholds)} thresholds")
    
    print()
    print("=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()