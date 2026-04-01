"""
Position-Level Binning and Enrichment Analysis

This module performs position-level analysis by:
1. Loading segment data with mu_z_scores
2. Expanding segments to individual positions
3. Classifying each position (essential/non-essential/outside)
4. Binning positions by mu_z_score
5. Computing counts, percentages, and enrichment values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from gene_overlap_classifier import PositionClassifier


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionLevelAnalyzer:
    """
    Analyzes genomic segments at position-level resolution.
    """
    
    def __init__(self, gene_database: PositionClassifier):
        """
        Initialize analyzer with gene database.
        
        Args:
            gene_database: Loaded PositionClassifier instance
        """
        self.gene_db = gene_database
        self.classes = ['outside_gene', 'non_essential_gene', 'essential_gene']
    
    def load_segment_data(
        self,
        base_path: Path,
        strain: str,
        threshold: float = 3.0,
        chromosomes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load segment_mu data for a specific strain and threshold.
        
        Path structure: Signal_processing/strains/strain_{strain}/{chrom}/{chrom}_distances/window100/segment_mu/{chrom}_distances_ws100_ov50_th{threshold:.2f}_segment_mu.csv
        
        Args:
            base_path: Base path to Signal_processing/strains/
            strain: Strain name (e.g., 'FD', 'yEK19', 'yEK23')
            threshold: Threshold value to filter (e.g., 3.0)
            chromosomes: List of chromosomes to include (default: all ChrI-ChrXVI)
        
        Returns:
            DataFrame with all segments
        """
        if chromosomes is None:
            chromosomes = [f'Chr{num}' for num in ['I', 'II', 'III', 'IV', 'V', 'VI', 
                                                     'VII', 'VIII', 'IX', 'X', 'XI', 'XII',
                                                     'XIII', 'XIV', 'XV', 'XVI']]
        
        all_segments = []
        strain_path = base_path / f"strain_{strain}"
        
        for chrom in chromosomes:
            # Build the full path to segment_mu file
            segment_file = (strain_path / chrom / f"{chrom}_distances" / "window100" / 
                          "segment_mu" / f"{chrom}_distances_ws100_ov50_th{threshold:.2f}_segment_mu.csv")
            
            if not segment_file.exists():
                logger.warning(f"File not found: {segment_file}")
                continue
            
            try:
                df = pd.read_csv(segment_file)
                
                # The chromosome column should already be there
                if 'chromosome' not in df.columns:
                    df['chromosome'] = chrom
                
                all_segments.append(df)
                logger.info(f"Loaded {len(df)} segments from {chrom}")
                
            except Exception as e:
                logger.error(f"Error loading {segment_file}: {e}")
                continue
        
        if not all_segments:
            raise ValueError(f"No segment data found for strain {strain}")
        
        combined = pd.concat(all_segments, ignore_index=True)
        logger.info(f"Total segments loaded: {len(combined)}")
        
        return combined
    
    def expand_segment_to_positions(
        self,
        start: int,
        end: int,
        chromosome: str,
        score: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expand a segment to individual positions and classify each.
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            chromosome: Chromosome name
            score: mu_z_score for this segment
        
        Returns:
            Tuple of (positions array, classifications array)
        """
        positions = np.arange(start, end)
        classifications = self.gene_db.classify_positions_batch(chromosome, positions)
        
        return positions, classifications
    
    def bin_and_aggregate(
        self,
        segments_df: pd.DataFrame,
        bin_edges: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Perform position-level binning and aggregation.
        
        Process:
        1. For each segment, expand to positions
        2. Classify each position
        3. Bin by segment's mu_z_score
        4. Aggregate position counts per bin per class
        
        Args:
            segments_df: DataFrame with columns: start_index, end_index_exclusive, 
                         mu_z_score, chromosome
            bin_edges: Custom bin edges (if None, auto-generate)
            n_bins: Number of bins (if bin_edges not provided)
        
        Returns:
            DataFrame with bin statistics
        """
        # Determine bin edges
        if bin_edges is None:
            min_score = segments_df['mu_z_score'].min()
            max_score = segments_df['mu_z_score'].max()
            bin_edges = np.linspace(min_score, max_score, n_bins + 1)
        
        logger.info(f"Using {len(bin_edges)-1} bins from {bin_edges[0]:.2f} to {bin_edges[-1]:.2f}")
        
        # Initialize counters: bin_idx -> class -> count
        bin_counts = defaultdict(lambda: defaultdict(int))
        
        # Process each segment
        for idx, row in segments_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing segment {idx+1}/{len(segments_df)}")
            
            score = row['mu_z_score']
            chromosome = row['chromosome']
            start = int(row['start_index'])
            end = int(row['end_index_exclusive'])
            
            # Determine which bin this score belongs to
            bin_idx = np.digitize(score, bin_edges) - 1
            
            # Handle edge cases
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= len(bin_edges) - 1:
                bin_idx = len(bin_edges) - 2
            
            # Expand segment to positions and classify
            positions, classifications = self.expand_segment_to_positions(
                start, end, chromosome, score
            )
            
            # Count classifications
            unique_classes, counts = np.unique(classifications, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                bin_counts[bin_idx][cls] += count
        
        # Convert to DataFrame
        results = []
        for bin_idx in range(len(bin_edges) - 1):
            bin_start = bin_edges[bin_idx]
            bin_end = bin_edges[bin_idx + 1]
            bin_center = (bin_start + bin_end) / 2
            
            counts = bin_counts[bin_idx]
            
            row_data = {
                'bin_idx': bin_idx,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'bin_center': bin_center,
                'outside_gene_count': counts.get('outside_gene', 0),
                'non_essential_gene_count': counts.get('non_essential_gene', 0),
                'essential_gene_count': counts.get('essential_gene', 0),
            }
            
            results.append(row_data)
        
        df = pd.DataFrame(results)
        
        # Calculate total positions per bin
        df['total_positions'] = (
            df['outside_gene_count'] + 
            df['non_essential_gene_count'] + 
            df['essential_gene_count']
        )
        
        # Calculate percentages within each bin
        for cls in ['outside_gene', 'non_essential_gene', 'essential_gene']:
            df[f'{cls}_percent'] = 100 * df[f'{cls}_count'] / df['total_positions']
            df[f'{cls}_percent'] = df[f'{cls}_percent'].fillna(0)
        
        # Calculate overall baseline fractions
        total_positions_all = df['total_positions'].sum()
        overall_fractions = {}
        for cls in ['outside_gene', 'non_essential_gene', 'essential_gene']:
            total_cls = df[f'{cls}_count'].sum()
            overall_fractions[cls] = total_cls / total_positions_all if total_positions_all > 0 else 0
        
        logger.info(f"\nOverall baseline fractions:")
        for cls, frac in overall_fractions.items():
            logger.info(f"  {cls}: {frac:.4f} ({frac*100:.2f}%)")
        
        # Calculate enrichment values
        for cls in ['outside_gene', 'non_essential_gene', 'essential_gene']:
            bin_fraction = df[f'{cls}_count'] / df['total_positions']
            baseline = overall_fractions[cls]
            df[f'{cls}_enrichment'] = bin_fraction / baseline if baseline > 0 else 0
            df[f'{cls}_enrichment'] = df[f'{cls}_enrichment'].fillna(0)
            df[f'{cls}_log2_enrichment'] = np.log2(df[f'{cls}_enrichment'].replace(0, np.nan))
        
        return df
    
    def analyze_strain(
        self,
        base_path: Path,
        strain: str,
        threshold: float = 3.0,
        bin_edges: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Complete analysis pipeline for one strain.
        
        Args:
            base_path: Base path to Signal_processing/strains/
            strain: Strain name
            threshold: Score threshold
            bin_edges: Custom bin edges
            n_bins: Number of bins
        
        Returns:
            Summary DataFrame with all statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing strain: {strain}")
        logger.info(f"{'='*60}")
        
        # Load segment data
        segments_df = self.load_segment_data(base_path, strain, threshold)
        
        # Perform binning and aggregation
        summary_df = self.bin_and_aggregate(segments_df, bin_edges, n_bins)
        
        return summary_df


def main():
    """Test the position-level analyzer."""
    import os
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    gene_db_path = base_dir / 'Utils/SGD_API/architecture_info/yeast_genes_with_info.json'
    signal_processing_path = base_dir / 'Signal_processing/strains'
    
    # Load gene database
    print("Loading gene database...")
    gene_db = PositionClassifier(str(gene_db_path))
    print(f"Loaded {gene_db.get_statistics()['total_genes']} genes")
    
    # Create analyzer
    analyzer = PositionLevelAnalyzer(gene_db)
    
    # Test with one strain
    strain = 'FD'
    print(f"\nAnalyzing strain {strain}...")
    
    summary = analyzer.analyze_strain(
        base_path=signal_processing_path,
        strain=strain,
        threshold=3.0,
        n_bins=10
    )
    
    print("\nSummary Statistics:")
    print(summary[['bin_center', 'total_positions', 
                   'essential_gene_percent', 'non_essential_gene_percent', 
                   'outside_gene_percent']].to_string())
    
    # Save results
    output_file = base_dir / f'results/strain_{strain}_enrichment_test.csv'
    summary.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
