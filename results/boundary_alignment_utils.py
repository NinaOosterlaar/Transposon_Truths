"""
Utilities for analyzing alignment between change points and gene/protein boundaries.

This module provides functions to:
1. Extract gene, CDS, and protein domain boundaries from the SGD database
2. Extract change points from segment_mu files
3. Compute distances between change points and boundaries
4. Analyze overlap statistics within configurable windows
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import logging

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from Utils.SGD_API.yeast_genes import SGD_Genes

logger = logging.getLogger(__name__)


def convert_chromosome_name(chrom_name: str) -> str:
    """
    Convert chromosome name from SGD format to segment file format.
    E.g., 'Chromosome_V' -> 'ChrV', 'Chromosome_XVI' -> 'ChrXVI'
    """
    if chrom_name.startswith('Chromosome_'):
        return 'Chr' + chrom_name.split('_')[1]
    return chrom_name


def extract_all_boundaries(sgd_genes: SGD_Genes) -> pd.DataFrame:
    """
    Extract all gene, CDS, and protein domain boundaries from SGD database.
    
    Args:
        sgd_genes: Instance of SGD_Genes class with loaded gene data
        
    Returns:
        DataFrame with columns: [chromosome, position, boundary_type, gene_id, 
                                 gene_name, is_essential, domain_name]
    """
    boundaries = []
    
    for gene_id, gene_info in sgd_genes.genes.items():
        # Convert chromosome name from 'Chromosome_V' to 'ChrV' format
        chromosome_sgd = gene_info['location']['chromosome']
        chromosome = convert_chromosome_name(chromosome_sgd)
        gene_name = gene_info.get('gene_name', gene_id)
        is_essential = gene_info.get('essentiality', False)  # Note: field is 'essentiality' not 'is_essential'
        
        # Gene boundaries
        gene_start = gene_info['location']['start']
        gene_end = gene_info['location']['end']
        
        boundaries.append({
            'chromosome': chromosome,
            'position': gene_start,
            'boundary_type': 'gene_start',
            'gene_id': gene_id,
            'gene_name': gene_name,
            'is_essential': is_essential,
            'domain_name': None
        })
        
        boundaries.append({
            'chromosome': chromosome,
            'position': gene_end,
            'boundary_type': 'gene_end',
            'gene_id': gene_id,
            'gene_name': gene_name,
            'is_essential': is_essential,
            'domain_name': None
        })
        
        # Protein domain boundaries (PF domains only)
        # Note: protein_domains structure is {domain_name: {'start': [list], 'end': [list]}}
        if 'protein_domains' in gene_info and gene_info['protein_domains']:
            for domain_name, domain_info in gene_info['protein_domains'].items():
                if domain_name.startswith('PF'):
                    # Domain positions are stored as lists (can have multiple instances)
                    starts = domain_info.get('start', [])
                    ends = domain_info.get('end', [])
                    
                    # Each domain can have multiple start/end positions
                    for start_pos, end_pos in zip(starts, ends):
                        if start_pos is not None and end_pos is not None:
                            boundaries.append({
                                'chromosome': chromosome,
                                'position': start_pos,
                                'boundary_type': 'pf_domain_start',
                                'gene_id': gene_id,
                                'gene_name': gene_name,
                                'is_essential': is_essential,
                                'domain_name': domain_name
                            })
                            
                            boundaries.append({
                                'chromosome': chromosome,
                                'position': end_pos,
                                'boundary_type': 'pf_domain_end',
                                'gene_id': gene_id,
                                'gene_name': gene_name,
                                'is_essential': is_essential,
                                'domain_name': domain_name
                            })
    
    # Create DataFrame with explicit columns to handle empty case
    if boundaries:
        df = pd.DataFrame(boundaries)
    else:
        df = pd.DataFrame(columns=['chromosome', 'position', 'boundary_type', 'gene_id', 
                                   'gene_name', 'is_essential', 'domain_name'])
    
    logger.info(f"Extracted {len(df)} boundaries from {len(sgd_genes.genes)} genes")
    if len(df) > 0:
        logger.info(f"  Gene boundaries: {len(df[df['boundary_type'].str.contains('gene')])}")
        logger.info(f"  PF domain boundaries: {len(df[df['boundary_type'].str.contains('pf_domain')])}")
    
    return df


def extract_change_points(
    strains_data_path: Path,
    threshold: float,
    strain: Optional[str] = None,
    window_size: int = 100,
    overlap: int = 50
) -> pd.DataFrame:
    """
    Extract change points (segment boundaries) from segment_mu files.
    
    Args:
        strains_data_path: Path to Signal_processing/strains directory
        threshold: Threshold value to filter files
        strain: Specific strain to extract (None = all strains)
        window_size: Window size used in analysis
        overlap: Overlap used in analysis
        
    Returns:
        DataFrame with columns: [chromosome, position, strain, threshold]
    """
    change_points = []
    
    # Determine which strains to process
    if strain is not None:
        strains_to_process = [strain]
    else:
        strains_to_process = ['FD', 'yEK19', 'yEK23']
    
    chromosomes = [f'Chr{num}' for num in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 
                                             'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 
                                             'XIV', 'XV', 'XVI']]
    
    for strain_name in strains_to_process:
        for chrom in chromosomes:
            file_path = (strains_data_path / f"strain_{strain_name}" / chrom / 
                        f"{chrom}_distances" / f"window{window_size}" / "segment_mu" /
                        f"{chrom}_distances_ws{window_size}_ov{overlap}_th{threshold:.2f}_segment_mu.csv")
            
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Extract both start and end positions as change points
                for _, row in df.iterrows():
                    # Segment start is a change point
                    change_points.append({
                        'chromosome': chrom,
                        'position': int(row['start_index']),
                        'strain': strain_name,
                        'threshold': threshold,
                        'segment_id': row.get('segment_id', None)
                    })
                    
                    # Segment end is a change point
                    change_points.append({
                        'chromosome': chrom,
                        'position': int(row['end_index_exclusive']),
                        'strain': strain_name,
                        'threshold': threshold,
                        'segment_id': row.get('segment_id', None)
                    })
                    
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
    
    df = pd.DataFrame(change_points)
    
    # Remove duplicate positions (adjacent segments share boundaries)
    if len(df) > 0:
        df = df.drop_duplicates(subset=['chromosome', 'position', 'strain'], keep='first')
    
    logger.info(f"Extracted {len(df)} change points for threshold={threshold}, strain={strain}")
    
    return df


def compute_distances_changepoint_to_boundary(
    changepoints_df: pd.DataFrame,
    boundaries_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each changepoint, compute distance to nearest boundary of each type.
    Optimized version using vectorized operations.
    
    Args:
        changepoints_df: DataFrame with change point positions
        boundaries_df: DataFrame with all boundaries
        
    Returns:
        DataFrame with columns: [chromosome, cp_position, strain, threshold,
                                boundary_type, nearest_distance, nearest_gene_id, 
                                nearest_gene_name, nearest_is_essential]
    """
    results = []
    
    # For each (chromosome, threshold, strain) combination
    for (chrom, thresh, strain), cp_group in changepoints_df.groupby(['chromosome', 'threshold', 'strain']):
        bd_chr = boundaries_df[boundaries_df['chromosome'] == chrom]
        
        if len(bd_chr) == 0:
            logger.warning(f"No boundaries found for chromosome {chrom}")
            continue
        
        cp_positions = cp_group['position'].values
        
        # Process each boundary type separately
        for boundary_type in boundaries_df['boundary_type'].unique():
            bd_type = bd_chr[bd_chr['boundary_type'] == boundary_type].copy()
            
            if len(bd_type) == 0:
                continue
            
            bd_positions = bd_type['position'].values
            
            # Vectorized distance computation using broadcasting
            # Shape: (num_change_points, num_boundaries)
            distances = np.abs(cp_positions[:, np.newaxis] - bd_positions[np.newaxis, :])
            
            # Find nearest boundary for each change point
            nearest_indices = np.argmin(distances, axis=1)
            nearest_distances = np.min(distances, axis=1)
            
            # Extract boundary information for nearest boundaries
            bd_type_reset = bd_type.reset_index(drop=True)
            nearest_boundaries = bd_type_reset.iloc[nearest_indices]
            
            # Create results for this group
            group_results = pd.DataFrame({
                'chromosome': chrom,
                'cp_position': cp_positions,
                'strain': strain,
                'threshold': thresh,
                'boundary_type': boundary_type,
                'nearest_distance': nearest_distances,
                'nearest_gene_id': nearest_boundaries['gene_id'].values,
                'nearest_gene_name': nearest_boundaries['gene_name'].values,
                'nearest_is_essential': nearest_boundaries['is_essential'].values
            })
            
            results.append(group_results)
        
        logger.debug(f"Processed {len(cp_positions)} change points for {chrom}, threshold={thresh}, strain={strain}")
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['chromosome', 'cp_position', 'strain', 'threshold',
                                    'boundary_type', 'nearest_distance', 'nearest_gene_id',
                                    'nearest_gene_name', 'nearest_is_essential'])


def compute_genes_with_nearby_changepoints(
    sgd_genes: SGD_Genes,
    changepoints_df: pd.DataFrame,
    window: int = 100
) -> pd.DataFrame:
    """
    For each gene, check if any gene boundary has a nearby change point within window.
    
    Args:
        sgd_genes: Instance of SGD_Genes class
        changepoints_df: DataFrame with change point positions
        window: Window size (±window bp)
        
    Returns:
        DataFrame with columns: [threshold, strain, is_essential, total_genes, 
                                genes_with_nearby_cp, percentage]
    """
    results = []
    
    # Group by threshold and strain
    for (threshold, strain), cp_group in changepoints_df.groupby(['threshold', 'strain']):
        
        # Separate essential and non-essential genes
        for is_essential in [True, False]:
            genes_with_cp = 0
            total_genes = 0
            
            for gene_id, gene_info in sgd_genes.genes.items():
                # Use 'essentiality' field, not 'is_essential'
                if gene_info.get('essentiality', False) != is_essential:
                    continue
                
                total_genes += 1
                
                # Convert chromosome name from 'Chromosome_V' to 'ChrV'
                chromosome_sgd = gene_info['location']['chromosome']
                chromosome = convert_chromosome_name(chromosome_sgd)
                gene_start = gene_info['location']['start']
                gene_end = gene_info['location']['end']
                
                # Get change points on same chromosome
                cp_chr = cp_group[cp_group['chromosome'] == chromosome]
                
                if len(cp_chr) == 0:
                    continue
                
                cp_positions = cp_chr['position'].values
                
                # Check if any change point is near gene start or end
                distances_to_start = np.abs(cp_positions - gene_start)
                distances_to_end = np.abs(cp_positions - gene_end)
                
                if np.any(distances_to_start <= window) or np.any(distances_to_end <= window):
                    genes_with_cp += 1
            
            percentage = (genes_with_cp / total_genes * 100) if total_genes > 0 else 0
            
            results.append({
                'threshold': threshold,
                'strain': strain,
                'is_essential': is_essential,
                'total_genes': total_genes,
                'genes_with_nearby_cp': genes_with_cp,
                'percentage': percentage
            })
    
    return pd.DataFrame(results)


def compute_changepoints_with_nearby_boundaries(
    changepoints_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    window: int = 100
) -> pd.DataFrame:
    """
    For each change point, check if any boundary is within window.
    
    Args:
        changepoints_df: DataFrame with change point positions
        boundaries_df: DataFrame with all boundaries
        window: Window size (±window bp)
        
    Returns:
        DataFrame with columns: [threshold, strain, boundary_type, total_cps,
                                cps_with_nearby_boundary, percentage]
    """
    results = []
    
    # Group by threshold and strain
    for (threshold, strain), cp_group in changepoints_df.groupby(['threshold', 'strain']):
        
        # For each boundary type
        for boundary_type in boundaries_df['boundary_type'].unique():
            bd_type = boundaries_df[boundaries_df['boundary_type'] == boundary_type]
            
            cps_with_boundary = 0
            total_cps = len(cp_group)
            
            for _, cp_row in cp_group.iterrows():
                chromosome = cp_row['chromosome']
                cp_pos = cp_row['position']
                
                # Get boundaries on same chromosome
                bd_chr = bd_type[bd_type['chromosome'] == chromosome]
                
                if len(bd_chr) == 0:
                    continue
                
                bd_positions = bd_chr['position'].values
                
                # Check if any boundary is within window
                distances = np.abs(bd_positions - cp_pos)
                
                if np.any(distances <= window):
                    cps_with_boundary += 1
            
            percentage = (cps_with_boundary / total_cps * 100) if total_cps > 0 else 0
            
            results.append({
                'threshold': threshold,
                'strain': strain,
                'boundary_type': boundary_type,
                'total_cps': total_cps,
                'cps_with_nearby_boundary': cps_with_boundary,
                'percentage': percentage
            })
    
    return pd.DataFrame(results)


def aggregate_distance_statistics(distances_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and median distances aggregated by threshold, strain, and boundary type.
    
    Args:
        distances_df: DataFrame from compute_distances_changepoint_to_boundary
        
    Returns:
        DataFrame with columns: [threshold, strain, boundary_type, mean_distance, 
                                median_distance, std_distance, count]
    """
    stats = distances_df.groupby(['threshold', 'strain', 'boundary_type'])['nearest_distance'].agg([
        ('mean_distance', 'mean'),
        ('median_distance', 'median'),
        ('std_distance', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    return stats


def compute_distances_boundary_to_changepoint(
    boundaries_df: pd.DataFrame,
    changepoints_df: pd.DataFrame
) -> pd.DataFrame:
    """
    REVERSE COMPUTATION: For each boundary, compute distance to nearest change point.
    This is the opposite of compute_distances_changepoint_to_boundary.
    
    Args:
        boundaries_df: DataFrame with all boundaries
        changepoints_df: DataFrame with change point positions
        
    Returns:
        DataFrame with columns: [chromosome, boundary_type, boundary_position, 
                                gene_id, gene_name, is_essential, threshold, strain,
                                nearest_distance]
    """
    results = []
    
    # For each (chromosome, threshold, strain) combination in changepoints
    for (chrom, thresh, strain), cp_group in changepoints_df.groupby(['chromosome', 'threshold', 'strain']):
        bd_chr = boundaries_df[boundaries_df['chromosome'] == chrom]
        
        if len(bd_chr) == 0:
            continue
        
        cp_positions = cp_group['position'].values
        
        # Process each boundary type separately
        for boundary_type in ['gene_start', 'gene_end']:  # Only gene boundaries
            bd_type = bd_chr[bd_chr['boundary_type'] == boundary_type].copy()
            
            if len(bd_type) == 0:
                continue
            
            bd_positions = bd_type['position'].values
            
            # Vectorized distance computation
            # Shape: (num_boundaries, num_change_points)
            distances = np.abs(bd_positions[:, np.newaxis] - cp_positions[np.newaxis, :])
            
            # Find nearest change point for each boundary
            nearest_distances = np.min(distances, axis=1)
            
            # Create results for this group
            bd_type_reset = bd_type.reset_index(drop=True)
            group_results = pd.DataFrame({
                'chromosome': chrom,
                'boundary_type': boundary_type,
                'boundary_position': bd_positions,
                'gene_id': bd_type_reset['gene_id'].values,
                'gene_name': bd_type_reset['gene_name'].values,
                'is_essential': bd_type_reset['is_essential'].values,
                'threshold': thresh,
                'strain': strain,
                'nearest_distance': nearest_distances
            })
            
            results.append(group_results)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['chromosome', 'boundary_type', 'boundary_position',
                                    'gene_id', 'gene_name', 'is_essential', 
                                    'threshold', 'strain', 'nearest_distance'])


def aggregate_boundary_to_cp_statistics(boundary_distances_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and median distances from boundaries to change points,
    aggregated by threshold, strain, and boundary type.
    
    Args:
        boundary_distances_df: DataFrame from compute_distances_boundary_to_changepoint
        
    Returns:
        DataFrame with columns: [threshold, strain, boundary_type, mean_distance, 
                                median_distance, std_distance, count]
    """
    stats = boundary_distances_df.groupby(['threshold', 'strain', 'boundary_type'])['nearest_distance'].agg([
        ('mean_distance', 'mean'),
        ('median_distance', 'median'),
        ('std_distance', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    return stats
