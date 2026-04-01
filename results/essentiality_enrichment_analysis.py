"""
Main Script for Essentiality Enrichment Analysis

Position-level analysis of change point detection results:
- Expands segments to individual genomic positions
- Classifies positions by gene essentiality
- Bins by mu_z_score
- Computes enrichment statistics
- Creates visualization plots

Author: Generated for thesis analysis
Date: 2026-04-01
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gene_overlap_classifier import PositionClassifier
from position_level_analysis import PositionLevelAnalyzer
from plotting_functions import create_all_plots


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('essentiality_enrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the analysis."""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    GENE_DB_PATH = BASE_DIR / 'Utils/SGD_API/architecture_info/yeast_genes_with_info.json'
    SIGNAL_PROCESSING_PATH = BASE_DIR / 'Signal_processing/strains'
    OUTPUT_DIR = BASE_DIR / 'results/essentiality_enrichment'
    
    # Analysis parameters
    STRAINS = ['FD', 'yEK19', 'yEK23']
    THRESHOLD = 3.0
    N_BINS = 15
    BIN_EDGES = None  # If None, auto-generate; otherwise provide np.array
    
    # Chromosomes to include
    CHROMOSOMES = [
        'ChrI', 'ChrII', 'ChrIII', 'ChrIV', 'ChrV', 'ChrVI',
        'ChrVII', 'ChrVIII', 'ChrIX', 'ChrX', 'ChrXI', 'ChrXII',
        'ChrXIII', 'ChrXIV', 'ChrXV', 'ChrXVI'
    ]
    
    # Visualization options
    SHOW_PLOTS = False  # Set to True to display plots interactively
    SAVE_PLOTS = True
    SAVE_CSV = True


def validate_paths(config: Config) -> bool:
    """
    Validate that all required paths exist.
    
    Args:
        config: Configuration object
    
    Returns:
        True if all paths valid, False otherwise
    """
    logger.info("Validating paths...")
    
    if not config.GENE_DB_PATH.exists():
        logger.error(f"Gene database not found: {config.GENE_DB_PATH}")
        return False
    
    if not config.SIGNAL_PROCESSING_PATH.exists():
        logger.error(f"Signal processing directory not found: {config.SIGNAL_PROCESSING_PATH}")
        return False
    
    # Create output directory if it doesn't exist
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    
    return True


def analyze_all_strains(config: Config):
    """
    Run complete analysis pipeline for all strains.
    
    Args:
        config: Configuration object
    """
    logger.info("="*80)
    logger.info("ESSENTIALITY ENRICHMENT ANALYSIS")
    logger.info("="*80)
    logger.info(f"Threshold: {config.THRESHOLD}")
    logger.info(f"Number of bins: {config.N_BINS}")
    logger.info(f"Strains: {', '.join(config.STRAINS)}")
    logger.info(f"Chromosomes: {', '.join(config.CHROMOSOMES)}")
    logger.info("="*80)
    
    # Validate paths
    if not validate_paths(config):
        logger.error("Path validation failed. Exiting.")
        return
    
    # Load gene database
    logger.info("\nLoading gene database...")
    try:
        gene_db = PositionClassifier(str(config.GENE_DB_PATH))
        stats = gene_db.get_statistics()
        logger.info(f"✓ Loaded {stats['total_genes']} genes")
        logger.info(f"  - Essential: {stats['essential_genes']}")
        logger.info(f"  - Non-essential: {stats['non_essential_genes']}")
        logger.info(f"  - Chromosomes: {stats['chromosomes']}")
    except Exception as e:
        logger.error(f"Failed to load gene database: {e}")
        return
    
    # Create analyzer
    analyzer = PositionLevelAnalyzer(gene_db)
    
    # Process each strain
    all_summaries = {}
    
    for strain in config.STRAINS:
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING STRAIN: {strain}")
        logger.info("="*80)
        
        try:
            # Perform analysis
            summary_df = analyzer.analyze_strain(
                base_path=config.SIGNAL_PROCESSING_PATH,
                strain=strain,
                threshold=config.THRESHOLD,
                bin_edges=config.BIN_EDGES,
                n_bins=config.N_BINS
            )
            
            all_summaries[strain] = summary_df
            
            # Save CSV
            if config.SAVE_CSV:
                csv_path = config.OUTPUT_DIR / f'strain_{strain}_enrichment_summary.csv'
                summary_df.to_csv(csv_path, index=False)
                logger.info(f"✓ Saved summary to: {csv_path}")
            
            # Create plots
            if config.SAVE_PLOTS:
                logger.info(f"Generating plots for strain {strain}...")
                create_all_plots(
                    summary_df,
                    output_dir=config.OUTPUT_DIR,
                    strain=strain,
                    show_plots=config.SHOW_PLOTS
                )
                logger.info(f"✓ Plots saved to: {config.OUTPUT_DIR}")
            
            # Print summary statistics
            logger.info("\nSummary Statistics:")
            logger.info("-" * 80)
            display_cols = [
                'bin_center', 'total_positions',
                'essential_gene_percent', 'non_essential_gene_percent', 'outside_gene_percent',
                'essential_gene_enrichment'
            ]
            logger.info(summary_df[display_cols].to_string(index=False))
            
        except Exception as e:
            logger.error(f"Error processing strain {strain}: {e}", exc_info=True)
            continue
    
    # Create comparative summary
    if all_summaries:
        logger.info("\n" + "="*80)
        logger.info("COMPARATIVE SUMMARY")
        logger.info("="*80)
        
        for strain, summary in all_summaries.items():
            total_pos = summary['total_positions'].sum()
            total_essential = summary['essential_gene_count'].sum()
            pct_essential = 100 * total_essential / total_pos if total_pos > 0 else 0
            
            logger.info(f"\n{strain}:")
            logger.info(f"  Total positions: {total_pos:,}")
            logger.info(f"  Essential positions: {total_essential:,} ({pct_essential:.2f}%)")
            
            # Find bin with highest essential enrichment
            max_enrich_idx = summary['essential_gene_enrichment'].idxmax()
            max_enrich_row = summary.loc[max_enrich_idx]
            logger.info(f"  Max essential enrichment: {max_enrich_row['essential_gene_enrichment']:.2f}x "
                       f"at bin {max_enrich_row['bin_center']:.2f}")
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {config.OUTPUT_DIR}")
    logger.info(f"Log file: essentiality_enrichment.log")


def main():
    """Main entry point."""
    config = Config()
    
    # Optional: Override configuration via command line or environment
    # (could add argparse here for flexibility)
    
    try:
        analyze_all_strains(config)
    except KeyboardInterrupt:
        logger.warning("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
