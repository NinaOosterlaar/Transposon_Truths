import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional

# Add workspace root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from gene_overlap_classifier import PositionClassifier
from position_level_analysis import PositionLevelAnalyzer
from plotting_functions import create_all_plots


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the analysis."""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    GENE_DB_PATH = BASE_DIR / 'Utils/SGD_API/architecture_info/yeast_genes_with_info.json'
    SIGNAL_PROCESSING_PATH = BASE_DIR / 'SATAY_CPD_results/CPD_SATAY_results'
    OUTPUT_DIR = BASE_DIR / 'results/essentiality_enrichment'
    
    # Analysis parameters
    STRAINS = ['yEK19']
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


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CPD segment essentiality scores with SGD essentiality annotations."
    )
    parser.add_argument("--gene_db_path", default=str(Config.GENE_DB_PATH), help="Path to yeast gene annotation JSON.")
    parser.add_argument("--signal_processing_path", default=str(Config.SIGNAL_PROCESSING_PATH), help="Path to CPD SATAY result folders.")
    parser.add_argument("--output_dir", default=str(Config.OUTPUT_DIR), help="Folder where CSVs and plots are written.")
    parser.add_argument("--strains", nargs="+", default=Config.STRAINS, help="Strains to analyze.")
    parser.add_argument("--threshold", type=float, default=Config.THRESHOLD, help="CPD threshold to analyze.")
    parser.add_argument("--n_bins", type=int, default=Config.N_BINS, help="Number of bins for enrichment plots.")
    parser.add_argument("--chromosomes", nargs="+", default=Config.CHROMOSOMES, help="Chromosomes to include.")
    parser.add_argument("--show_plots", type=str_to_bool, default=Config.SHOW_PLOTS, help="Show plots interactively.")
    parser.add_argument("--save_plots", type=str_to_bool, default=Config.SAVE_PLOTS, help="Save plots.")
    parser.add_argument("--save_csv", type=str_to_bool, default=Config.SAVE_CSV, help="Save CSV summaries.")
    return parser.parse_args()


def config_from_args(args) -> Config:
    config = Config()
    config.GENE_DB_PATH = Path(args.gene_db_path)
    config.SIGNAL_PROCESSING_PATH = Path(args.signal_processing_path)
    config.OUTPUT_DIR = Path(args.output_dir)
    config.STRAINS = args.strains
    config.THRESHOLD = args.threshold
    config.N_BINS = args.n_bins
    config.CHROMOSOMES = args.chromosomes
    config.SHOW_PLOTS = args.show_plots
    config.SAVE_PLOTS = args.save_plots
    config.SAVE_CSV = args.save_csv
    return config


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

    print("Running essentiality enrichment analysis")
    print(f"Gene annotations: {config.GENE_DB_PATH}")
    print(f"CPD results: {config.SIGNAL_PROCESSING_PATH}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Strains: {', '.join(config.STRAINS)}")
    print(f"Threshold: {config.THRESHOLD}")
    print(f"Bins: {config.N_BINS}")
    print(f"Save CSV: {config.SAVE_CSV}")
    print(f"Save plots: {config.SAVE_PLOTS}")
    
    # Load gene database
    try:
        print("\nLoading gene database...")
        gene_db = PositionClassifier(str(config.GENE_DB_PATH))
        stats = gene_db.get_statistics()
        print(f"Loaded {stats.get('total_genes', 'unknown')} genes")
    except Exception as e:
        print(f"Failed to load gene database: {e}", file=sys.stderr)
        return
    
    # Create analyzer
    analyzer = PositionLevelAnalyzer(gene_db)
    
    # Process each strain
    all_summaries = {}
    
    for strain in config.STRAINS:
        try:
            print(f"\nProcessing strain {strain}...")
            # Perform analysis
            summary_df = analyzer.analyze_strain(
                base_path=config.SIGNAL_PROCESSING_PATH,
                strain=strain,
                threshold=config.THRESHOLD,
                bin_edges=config.BIN_EDGES,
                n_bins=config.N_BINS
            )
            print(f"Generated enrichment summary with {len(summary_df)} bins")
            
            all_summaries[strain] = summary_df
            
            # Save CSV
            if config.SAVE_CSV:
                csv_path = config.OUTPUT_DIR / f'strain_{strain}_enrichment_summary.csv'
                summary_df.to_csv(csv_path, index=False)
                print(f"Saved enrichment summary: {csv_path}")
            
            # Create plots
            if config.SAVE_PLOTS:
                create_all_plots(
                    summary_df,
                    output_dir=config.OUTPUT_DIR,
                    strain=strain,
                    show_plots=config.SHOW_PLOTS
                )
                print(f"Saved enrichment plots in: {config.OUTPUT_DIR}")
            
        except Exception as e:
            print(f"Error processing strain {strain}: {e}", file=sys.stderr)
            continue

    print("\nEssentiality enrichment analysis complete.")


def main(args=None):
    """Main entry point."""
    config = config_from_args(parse_args() if args is None else args)
    
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
