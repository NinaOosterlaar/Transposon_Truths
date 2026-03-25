
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
    """
    n_strains = len(strain_names)
    jaccard_matrix = np.ones((n_strains, n_strains))
    jaccard_tol_matrix = np.ones((n_strains, n_strains))
    ari_matrix = np.ones((n_strains, n_strains))
    nbp_dist_a_to_b_matrix = np.zeros((n_strains, n_strains))
    nbp_dist_b_to_a_matrix = np.zeros((n_strains, n_strains))
    
    # Load change points for all strains
    strain_changepoints = {}
    for strain in strain_names:
        strain_changepoints[strain] = load_changepoints_for_strain(strain, threshold)
    
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
                
                # Fill symmetric metrics in both triangles
                jaccard_matrix[i, j] = jaccard
                jaccard_matrix[j, i] = jaccard
                
                jaccard_tol_matrix[i, j] = jaccard_tol
                jaccard_tol_matrix[j, i] = jaccard_tol
                
                ari_matrix[i, j] = ari
                ari_matrix[j, i] = ari
                
                # Fill asymmetric metrics
                nbp_dist_a_to_b_matrix[i, j] = nbp_dist['a_to_b']
                nbp_dist_b_to_a_matrix[i, j] = nbp_dist['b_to_a']
                nbp_dist_a_to_b_matrix[j, i] = nbp_dist['b_to_a']  # Transpose
                nbp_dist_b_to_a_matrix[j, i] = nbp_dist['a_to_b']  # Transpose
    
    # Convert to DataFrames with proper labels
    results = {
        'jaccard': pd.DataFrame(jaccard_matrix, index=strain_names, columns=strain_names),
        'jaccard_tol': pd.DataFrame(jaccard_tol_matrix, index=strain_names, columns=strain_names),
        'ari': pd.DataFrame(ari_matrix, index=strain_names, columns=strain_names),
        'nbp_dist_a_to_b': pd.DataFrame(nbp_dist_a_to_b_matrix, index=strain_names, columns=strain_names),
        'nbp_dist_b_to_a': pd.DataFrame(nbp_dist_b_to_a_matrix, index=strain_names, columns=strain_names)
    }
    
    return results
def plot_heatmap(matrix_df, metric_name, threshold, output_dir):
    """
    Create and save a heatmap for a similarity matrix.
    
    Parameters
    ----------
    matrix_df : pd.DataFrame
        Similarity matrix (strain x strain)
    metric_name : str
        Name of the metric ('Jaccard Index' or 'Adjusted Rand Index')
    threshold : float
        Threshold value used
    output_dir : Path
        Directory to save the figure
    """
    # Clean strain names for display (remove 'strain_' prefix)
    display_names = [name.replace('strain_', '') for name in matrix_df.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine color limits based on metric
    if 'Jaccard' in metric_name:
        vmin, vmax = 0, 1
        cmap = 'YlOrRd'
        fmt = '.3f'
    else:  # ARI
        vmin, vmax = -1, 1
        cmap = 'RdYlGn'
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
    metric_short = 'jaccard' if 'Jaccard' in metric_name else 'ari'
    output_file = output_dir / f'{metric_short}_heatmap_threshold_{threshold:.1f}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved heatmap: {output_file.name}")

def main():
    """
    Main function to compare strains across multiple thresholds.
    """
    # Configuration
    strain_names = [
        'strain_FD',
        'strain_dnrp',
        'strain_yEK19',
        'strain_yEK23',
        'strain_yTW001',
        'strain_yWT03a',
        'strain_yWT04a',
        'strain_ylic137'
    ]
    
    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    
    # Create output directory
    output_dir = Path(__file__).parent / "compare_strains"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Comparing Change Point Detection Results Across Strains")
    print("=" * 80)
    print(f"Strains: {len(strain_names)}")
    for strain in strain_names:
        print(f"  - {strain}")
    print(f"\nThresholds: {thresholds}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()
    
    # Store all results for CSV export (as multi-index with threshold levels)
    all_jaccard_results = {}
    all_ari_results = {}
    
    # Process each threshold
    for threshold in thresholds:
        print(f"Processing threshold {threshold:.1f}...")
        
        # Compute pairwise metrics
        jaccard_df, ari_df = compute_pairwise_metrics(strain_names, threshold)
        
        # Store results with threshold as key
        all_jaccard_results[threshold] = jaccard_df
        all_ari_results[threshold] = ari_df
        
        # Create heatmaps
        plot_heatmap(jaccard_df, 'Jaccard Index', threshold, output_dir)
        plot_heatmap(ari_df, 'Adjusted Rand Index', threshold, output_dir)
        
        print(f"  Threshold {threshold:.1f} complete\n")
    
    # Save results to CSV files
    print("Saving results to CSV files...")
    
    # Jaccard results - save each threshold separately
    for threshold, jaccard_df in all_jaccard_results.items():
        jaccard_csv = output_dir / f'jaccard_index_threshold_{threshold:.1f}.csv'
        jaccard_df.to_csv(jaccard_csv)
    print(f"  Saved Jaccard Index results for {len(thresholds)} thresholds")
    
    # ARI results - save each threshold separately
    for threshold, ari_df in all_ari_results.items():
        ari_csv = output_dir / f'ari_threshold_{threshold:.1f}.csv'
        ari_df.to_csv(ari_csv)
    print(f"  Saved ARI results for {len(thresholds)} thresholds")
    
    print()
    print("=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()