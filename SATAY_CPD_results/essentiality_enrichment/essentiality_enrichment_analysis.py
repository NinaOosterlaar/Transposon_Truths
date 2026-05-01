import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from gene_overlap_classifier import PositionClassifier
from position_level_analysis import PositionLevelAnalyzer
from plotting_functions import create_all_plots


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the analysis."""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    GENE_DB_PATH = BASE_DIR / 'Utils/SGD_API/architecture_info/yeast_genes_with_info.json'
    SIGNAL_PROCESSING_PATH = BASE_DIR / 'SATAY_CPD_results/CPD_SATAY_results'
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
    if not config.GENE_DB_PATH.exists():
        print(f"Gene database not found: {config.GENE_DB_PATH}", file=sys.stderr)
        return False
    
    if not config.SIGNAL_PROCESSING_PATH.exists():
        print(f"Signal processing directory not found: {config.SIGNAL_PROCESSING_PATH}", file=sys.stderr)
        return False
    
    # Create output directory if it doesn't exist
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    return True


def analyze_all_strains(config: Config):
    """
    Run complete analysis pipeline for all strains.
    
    Args:
        config: Configuration object
    """
    # Validate paths
    if not validate_paths(config):
        print("Path validation failed. Exiting.", file=sys.stderr)
        return
    
    # Load gene database
    try:
        gene_db = PositionClassifier(str(config.GENE_DB_PATH))
    except Exception as e:
        print(f"Failed to load gene database: {e}", file=sys.stderr)
        return
    
    # Create analyzer
    analyzer = PositionLevelAnalyzer(gene_db)
    
    # Process each strain
    all_summaries = {}
    
    for strain in config.STRAINS:
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
            
            # Create plots
            if config.SAVE_PLOTS:
                create_all_plots(
                    summary_df,
                    output_dir=config.OUTPUT_DIR,
                    strain=strain,
                    show_plots=config.SHOW_PLOTS
                )
            
        except Exception as e:
            print(f"Error processing strain {strain}: {e}", file=sys.stderr)
            continue


def main():
    """Main entry point."""
    config = Config()
    
    # Optional: Override configuration via command line or environment
    # (could add argparse here for flexibility)
    
    try:
        analyze_all_strains(config)
    except KeyboardInterrupt:
        print("Analysis interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
