"""
Change Point to Gene Boundary Alignment Analysis

This script analyzes how well change points align with gene, CDS, and protein domain boundaries.
For different thresholds, it computes:
1. Mean/median distances from change points to nearest boundaries
2. Percentage of genes with nearby change points (within window)
3. Percentage of change points with nearby boundaries (within window)

Results are analyzed per-strain and combined across all strains.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from Utils.SGD_API.yeast_genes import SGD_Genes
from results.boundary_alignment_utils import (
    extract_all_boundaries,
    extract_change_points,
    compute_distances_changepoint_to_boundary,
    compute_genes_with_nearby_changepoints,
    compute_changepoints_with_nearby_boundaries,
    aggregate_distance_statistics,
    compute_distances_boundary_to_changepoint,
    aggregate_boundary_to_cp_statistics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
STRAINS_DATA_PATH = Path(__file__).parent.parent / "Signal_processing" / "strains"
OUTPUT_DIR = Path(__file__).parent / "boundary_alignment"
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================================================================
# MAIN ANALYSIS PIPELINE
# ================================================================================

def main():
    """Main analysis pipeline."""
    logger.info("=" * 80)
    logger.info("Change Point to Boundary Alignment Analysis")
    logger.info("=" * 80)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Thresholds to analyze: {THRESHOLDS}")
    logger.info(f"Strains: {STRAINS}")
    logger.info(f"Window size: {WINDOW_SIZE} bp")
    logger.info("")
    
    # ============================================================================
    # Step 1: Load gene database and extract boundaries
    # ============================================================================
    logger.info("Step 1: Loading gene database and extracting boundaries...")
    gene_info_file = Path(__file__).parent.parent / "Utils" / "SGD_API" / "architecture_info" / "yeast_genes_with_info.json"
    sgd_genes = SGD_Genes(gene_list_with_info=str(gene_info_file))
    boundaries_df = extract_all_boundaries(sgd_genes)
    logger.info(f"Total boundaries extracted: {len(boundaries_df)}")
    logger.info("")
    
    # ============================================================================
    # Step 2: Process each threshold (save incrementally)
    # ============================================================================
    all_distance_stats = []
    all_genes_with_cp = []
    all_cps_with_boundary = []
    all_boundary_to_cp_stats = []
    
    for idx, threshold in enumerate(THRESHOLDS, 1):
        logger.info(f"\n[{idx}/{len(THRESHOLDS)}] Processing threshold: {threshold}")
        logger.info("-" * 80)
        
        threshold_start = datetime.now()
        
        # Process each strain separately (more memory efficient)
        threshold_distances = []
        threshold_genes_stats = []
        threshold_cps_stats = []
        threshold_boundary_distances = []
        
        for strain in STRAINS:
            logger.info(f"  [{strain}] Extracting change points...")
            changepoints_strain = extract_change_points(
                STRAINS_DATA_PATH,
                threshold=threshold,
                strain=strain,
                window_size=WINDOW_ANALYSIS,
                overlap=OVERLAP
            )
            
            if len(changepoints_strain) == 0:
                logger.warning(f"    No change points for strain {strain}")
                continue
            
            logger.info(f"    Found {len(changepoints_strain)} change points")
            
            # Compute distances
            logger.info(f"    Computing distances to boundaries...")
            distances_df = compute_distances_changepoint_to_boundary(
                changepoints_strain, boundaries_df
            )
            threshold_distances.append(distances_df)
            
            # Aggregate statistics
            stats = aggregate_distance_statistics(distances_df)
            all_distance_stats.append(stats)
            
            logger.info(f"    Computed {len(distances_df)} distance records")
            
            # Window-based analyses
            logger.info(f"    Computing genes with nearby CPs (window={WINDOW_SIZE})...")
            genes_stats = compute_genes_with_nearby_changepoints(
                sgd_genes, changepoints_strain, window=WINDOW_SIZE
            )
            threshold_genes_stats.append(genes_stats)
            all_genes_with_cp.append(genes_stats)
            
            logger.info(f"    Computing CPs with nearby boundaries (window={WINDOW_SIZE})...")
            cps_stats = compute_changepoints_with_nearby_boundaries(
                changepoints_strain, boundaries_df, window=WINDOW_SIZE
            )
            threshold_cps_stats.append(cps_stats)
            all_cps_with_boundary.append(cps_stats)
            
            # Compute boundary-to-changepoint distances (reverse computation)
            logger.info(f"    Computing distances from boundaries to CPs...")
            boundary_distances_df = compute_distances_boundary_to_changepoint(
                boundaries_df, changepoints_strain
            )
            threshold_boundary_distances.append(boundary_distances_df)
        
        # Save threshold results
        distance_file = OUTPUT_DIR / f"distances_threshold_{threshold:.1f}.csv"
        if threshold_distances:
            combined_distances = pd.concat(threshold_distances, ignore_index=True)
            combined_distances.to_csv(distance_file, index=False)
            logger.info(f"  Saved distances to {distance_file.name}")
        
        # Aggregate and save boundary-to-changepoint statistics
        if threshold_boundary_distances:
            combined_boundary_distances = pd.concat(threshold_boundary_distances, ignore_index=True)
            boundary_stats = aggregate_boundary_to_cp_statistics(combined_boundary_distances)
            all_boundary_to_cp_stats.append(boundary_stats)
            logger.info(f"  Computed boundary-to-CP stats: {len(boundary_stats)} records")
        
        threshold_elapsed = (datetime.now() - threshold_start).total_seconds()
        logger.info(f"  Threshold {threshold} completed in {threshold_elapsed:.1f}s")
    
    # ============================================================================
    # Step 3: Combine results and save
    # ============================================================================
    logger.info("\nStep 3: Combining results and saving...")
    
    # Distance statistics
    if len(all_distance_stats) > 0:
        distance_stats_df = pd.concat(all_distance_stats, ignore_index=True)
        output_path = OUTPUT_DIR / "distance_stats_per_threshold.csv"
        distance_stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved distance statistics to: {output_path}")
        
        # Print summary
        logger.info("\nDistance Statistics Summary:")
        for boundary_type in distance_stats_df['boundary_type'].unique():
            subset = distance_stats_df[distance_stats_df['boundary_type'] == boundary_type]
            if len(subset) > 0:
                mean_dist = subset['mean_distance'].mean()
                logger.info(f"  {boundary_type}: mean={mean_dist:.1f} bp")
    
    # Genes with nearby change points
    if len(all_genes_with_cp) > 0:
        genes_with_cp_df = pd.concat(all_genes_with_cp, ignore_index=True)
        output_path = OUTPUT_DIR / "genes_with_nearby_cp.csv"
        genes_with_cp_df.to_csv(output_path, index=False)
        logger.info(f"Saved genes analysis to: {output_path}")
    
    # Change points with nearby boundaries
    if len(all_cps_with_boundary) > 0:
        cps_with_boundary_df = pd.concat(all_cps_with_boundary, ignore_index=True)
        output_path = OUTPUT_DIR / "cps_with_nearby_boundary.csv"
        cps_with_boundary_df.to_csv(output_path, index=False)
        logger.info(f"Saved change points analysis to: {output_path}")
    
    # Boundary-to-changepoint distance statistics
    if len(all_boundary_to_cp_stats) > 0:
        boundary_to_cp_stats_df = pd.concat(all_boundary_to_cp_stats, ignore_index=True)
        output_path = OUTPUT_DIR / "boundary_to_cp_distance_stats.csv"
        boundary_to_cp_stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved boundary-to-CP distance statistics to: {output_path}")
        
        # Print summary
        logger.info("\nBoundary-to-CP Distance Summary:")
        for boundary_type in boundary_to_cp_stats_df['boundary_type'].unique():
            subset = boundary_to_cp_stats_df[boundary_to_cp_stats_df['boundary_type'] == boundary_type]
            if len(subset) > 0:
                mean_dist = subset['mean_distance'].mean()
                logger.info(f"  {boundary_type}: mean={mean_dist:.1f} bp")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
