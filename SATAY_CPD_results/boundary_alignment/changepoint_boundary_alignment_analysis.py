import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from Utils.SGD_API.yeast_genes import SGD_Genes
from boundary_alignment_utils import (
    extract_all_boundaries,
    extract_change_points,
    compute_distances_changepoint_to_boundary,
    compute_genes_with_nearby_changepoints,
    compute_changepoints_with_nearby_boundaries,
    aggregate_distance_statistics,
    compute_distances_boundary_to_changepoint,
    aggregate_boundary_to_cp_statistics
)


# ================================================================================
# CONFIGURATION
# ================================================================================

# Available thresholds
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Strains to analyze
STRAINS = ['FD', 'yEK19', 'yEK23']

# Window size for proximity analysis (±window bp)
WINDOW_SIZE = 100

# Analysis parameters
WINDOW_ANALYSIS = 100  # Window size for windowing analysis
OVERLAP = 50  # Overlap used in segment analysis

# Paths
GENE_INFO_PATH = Path(__file__).parent.parent / "Utils" / "SGD_API" / "architecture_info" / "yeast_genes_with_info.json"
STRAINS_DATA_PATH = Path(__file__).parent.parent / "SATAY_CPD_results" / "CPD_SATAY_results"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "boundary_alignment"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ================================================================================
# MAIN ANALYSIS PIPELINE
# ================================================================================

def main():
    """Main analysis pipeline."""
    
    # ============================================================================
    # Step 1: Load gene database and extract boundaries
    # ============================================================================
    gene_info_file = Path(__file__).parent.parent / "Utils" / "SGD_API" / "architecture_info" / "yeast_genes_with_info.json"
    sgd_genes = SGD_Genes(gene_list_with_info=str(gene_info_file))
    boundaries_df = extract_all_boundaries(sgd_genes)
    
    # ============================================================================
    # Step 2: Process each threshold (save incrementally)
    # ============================================================================
    all_distance_stats = []
    all_genes_with_cp = []
    all_cps_with_boundary = []
    all_boundary_to_cp_stats = []
    
    for idx, threshold in enumerate(THRESHOLDS, 1):
        threshold_start = datetime.now()
        
        # Process each strain separately (more memory efficient)
        threshold_distances = []
        threshold_genes_stats = []
        threshold_cps_stats = []
        threshold_boundary_distances = []
        
        for strain in STRAINS:
            changepoints_strain = extract_change_points(
                STRAINS_DATA_PATH,
                threshold=threshold,
                strain=strain,
                window_size=WINDOW_ANALYSIS,
                overlap=OVERLAP
            )
            
            if len(changepoints_strain) == 0:
                continue
            
            # Compute distances
            distances_df = compute_distances_changepoint_to_boundary(
                changepoints_strain, boundaries_df
            )
            threshold_distances.append(distances_df)
            
            # Aggregate statistics
            stats = aggregate_distance_statistics(distances_df)
            all_distance_stats.append(stats)
            
            # Window-based analyses
            genes_stats = compute_genes_with_nearby_changepoints(
                sgd_genes, changepoints_strain, window=WINDOW_SIZE
            )
            threshold_genes_stats.append(genes_stats)
            all_genes_with_cp.append(genes_stats)
            
            cps_stats = compute_changepoints_with_nearby_boundaries(
                changepoints_strain, boundaries_df, window=WINDOW_SIZE
            )
            threshold_cps_stats.append(cps_stats)
            all_cps_with_boundary.append(cps_stats)
            
            # Compute boundary-to-changepoint distances (reverse computation)
            boundary_distances_df = compute_distances_boundary_to_changepoint(
                boundaries_df, changepoints_strain
            )
            threshold_boundary_distances.append(boundary_distances_df)
        
        # Save threshold results
        distance_file = OUTPUT_DIR / f"distances_threshold_{threshold:.1f}.csv"
        if threshold_distances:
            combined_distances = pd.concat(threshold_distances, ignore_index=True)
            combined_distances.to_csv(distance_file, index=False)
        
        # Aggregate and save boundary-to-changepoint statistics
        if threshold_boundary_distances:
            combined_boundary_distances = pd.concat(threshold_boundary_distances, ignore_index=True)
            boundary_stats = aggregate_boundary_to_cp_statistics(combined_boundary_distances)
            all_boundary_to_cp_stats.append(boundary_stats)
        
        threshold_elapsed = (datetime.now() - threshold_start).total_seconds()
    
    # ============================================================================
    # Step 3: Combine results and save
    # ============================================================================
    
    # Distance statistics
    if len(all_distance_stats) > 0:
        distance_stats_df = pd.concat(all_distance_stats, ignore_index=True)
        output_path = OUTPUT_DIR / "distance_stats_per_threshold.csv"
        distance_stats_df.to_csv(output_path, index=False)
    
    # Genes with nearby change points
    if len(all_genes_with_cp) > 0:
        genes_with_cp_df = pd.concat(all_genes_with_cp, ignore_index=True)
        output_path = OUTPUT_DIR / "genes_with_nearby_cp.csv"
        genes_with_cp_df.to_csv(output_path, index=False)
    
    # Change points with nearby boundaries
    if len(all_cps_with_boundary) > 0:
        cps_with_boundary_df = pd.concat(all_cps_with_boundary, ignore_index=True)
        output_path = OUTPUT_DIR / "cps_with_nearby_boundary.csv"
        cps_with_boundary_df.to_csv(output_path, index=False)
    
    # Boundary-to-changepoint distance statistics
    if len(all_boundary_to_cp_stats) > 0:
        boundary_to_cp_stats_df = pd.concat(all_boundary_to_cp_stats, ignore_index=True)
        output_path = OUTPUT_DIR / "boundary_to_cp_distance_stats.csv"
        boundary_to_cp_stats_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
