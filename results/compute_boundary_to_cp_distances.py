"""
Quick script to compute boundary-to-changepoint distances from existing data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from Utils.SGD_API.yeast_genes import SGD_Genes
from results.boundary_alignment_utils import (
    extract_all_boundaries,
    extract_change_points,
    compute_distances_boundary_to_changepoint,
    aggregate_boundary_to_cp_statistics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
STRAINS = ['FD', 'yEK19', 'yEK23']
WINDOW_ANALYSIS = 100
OVERLAP = 50

# Paths
GENE_INFO_PATH = Path(__file__).parent.parent / "Utils" / "SGD_API" / "architecture_info" / "yeast_genes_with_info.json"
STRAINS_DATA_PATH = Path(__file__).parent.parent / "Signal_processing" / "strains"
OUTPUT_DIR = Path(__file__).parent / "boundary_alignment"

def main():
    logger.info("Loading gene database...")
    sgd_genes = SGD_Genes(gene_list_with_info=str(GENE_INFO_PATH))
    
    logger.info("Extracting boundaries...")
    boundaries_df = extract_all_boundaries(sgd_genes)
    logger.info(f"Extracted {len(boundaries_df)} boundaries")
    
    all_stats = []
    
    for threshold in THRESHOLDS:
        logger.info(f"\nProcessing threshold {threshold}...")
        
        for strain in STRAINS:
            logger.info(f"  Strain {strain}...")
            
            # Extract change points
            changepoints_df = extract_change_points(
                STRAINS_DATA_PATH,
                threshold=threshold,
                strain=strain,
                window_size=WINDOW_ANALYSIS,
                overlap=OVERLAP
            )
            
            if len(changepoints_df) == 0:
                logger.warning(f"    No change points found")
                continue
            
            logger.info(f"    Found {len(changepoints_df)} change points")
            
            # Compute boundary-to-changepoint distances
            logger.info(f"    Computing distances from boundaries to CPs...")
            boundary_distances_df = compute_distances_boundary_to_changepoint(
                boundaries_df, changepoints_df
            )
            
            # Aggregate statistics
            stats = aggregate_boundary_to_cp_statistics(boundary_distances_df)
            all_stats.append(stats)
            
            logger.info(f"    Computed {len(stats)} statistical records")
    
    # Combine and save
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        output_path = OUTPUT_DIR / "boundary_to_cp_distance_stats.csv"
        combined_stats.to_csv(output_path, index=False)
        logger.info(f"\nSaved to: {output_path}")
        logger.info(f"Total records: {len(combined_stats)}")
        
        # Print summary
        for boundary_type in combined_stats['boundary_type'].unique():
            subset = combined_stats[combined_stats['boundary_type'] == boundary_type]
            mean_dist = subset['mean_distance'].mean()
            logger.info(f"  {boundary_type}: mean={mean_dist:.1f} bp")
    else:
        logger.warning("No data computed!")

if __name__ == "__main__":
    main()
