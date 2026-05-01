import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from gene_reader import geneClassifier
from transposon_reader import fit_centromere_bias_from_rates, read_strain_data
from calculate_essentiality import process_genes, calculate_fitness
from Utils.plot_config import COLORS, setup_plot_style

# Set up consistent plot styling
setup_plot_style()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters."""
    
    # Paths
    STRAIN_DATA_PATH = "Data/combined_strains/strain_yEK19"
    GENE_DB_PATH = "Utils/SGD_API/architecture_info/yeast_genes_with_info.json"
    SEGMENTS_BASE_PATH = Path("SATAY_CPD_results/CPD_SATAY_results")
    OUTPUT_DIR = Path("Enzo_kingma/results/essentiality_comparison")
    
    # Analysis parameters
    STRAIN = "yEK19"
    THRESHOLD = 3.0
    MU_Z = 0.25
    MIN_INSERTIONS = 5
    
    # Chromosomes
    CHROMOSOMES = [f'Chr{x}' for x in ['I', 'II', 'III', 'IV', 'V', 'VI',
                                        'VII', 'VIII', 'IX', 'X', 'XI', 'XII',
                                        'XIII', 'XIV', 'XV', 'XVI']]


# ============================================================================
# STEP 1: Load gene-level fitness scores
# ============================================================================

def load_gene_fitness_scores(config):
    """
    Load and calculate gene-level fitness scores.
    
    Returns:
        pd.DataFrame with gene-level fitness scores
    """
    print("STEP 1: Loading gene-level fitness scores...")
    
    # Load gene classifier and strain data
    classifier = geneClassifier(config.GENE_DB_PATH)
    chr_data = read_strain_data(config.STRAIN_DATA_PATH)
    
    # Fit centromere bias (pass the dictionary, not concatenated data)
    coeffs_rate, rate_poly, insertion_rate, rate_df = fit_centromere_bias_from_rates(chr_data)
    
    # Process genes
    gene_stats_df = process_genes(chr_data, classifier, insertion_rate, 
                                   min_insertions=config.MIN_INSERTIONS)
    gene_stats_df, median_log_mean = calculate_fitness(gene_stats_df)
    
    print(f"  Loaded {len(gene_stats_df)} genes")
    print(f"  {gene_stats_df['valid_for_fitness'].sum()} valid for fitness calculation")
    
    return gene_stats_df


# ============================================================================
# STEP 2: Load position-level segments
# ============================================================================

def load_position_level_segments(config):
    """
    Load position-level segment data with mu_z scores.
    
    Returns:
        pd.DataFrame with segment data
    """
    print("\nSTEP 2: Loading position-level segments...")
    
    all_segments = []
    strain_path = config.SEGMENTS_BASE_PATH / f"strain_{config.STRAIN}"
    
    for chrom in config.CHROMOSOMES:
        segment_file = (strain_path / chrom / f"{chrom}_distances" / "window100" / 
                       "merged_segments" / 
                       f"{chrom}_th{config.THRESHOLD:.2f}_merged_segments_muZ{config.MU_Z}.csv")
        
        if not segment_file.exists():
            print(f"  Warning: {segment_file} not found")
            continue
        
        try:
            df = pd.read_csv(segment_file)
            df['chromosome'] = chrom
            all_segments.append(df)
            print(f"  Loaded {chrom}: {len(df)} segments")
        except Exception as e:
            print(f"  Error loading {chrom}: {e}")
    
    if not all_segments:
        raise ValueError(f"No segment data found for strain {config.STRAIN}")
    
    combined = pd.concat(all_segments, ignore_index=True)
    print(f"  Total segments: {len(combined)}")
    
    return combined


# ============================================================================
# STEP 3: Aggregate position-level scores to gene level
# ============================================================================

def aggregate_segments_to_genes(gene_df, segments_df):
    """
    For each gene, aggregate overlapping segment mu_z scores.
    
    Uses weighted mean where weight = segment length.
    
    Args:
        gene_df: DataFrame with gene information
        segments_df: DataFrame with segment information
    
    Returns:
        gene_df with added 'aggregated_mu_z' column
    """
    print("\nSTEP 3: Aggregating position-level scores to gene level...")
    
    gene_df = gene_df.copy()
    gene_df['aggregated_mu_z'] = np.nan
    gene_df['n_overlapping_segments'] = 0
    
    for idx, gene in gene_df.iterrows():
        chrom = gene['chromosome']
        # Use central 80% region
        gene_start = gene['central_start']
        gene_end = gene['central_end']
        
        # Find overlapping segments
        chrom_segments = segments_df[segments_df['chromosome'] == chrom]
        overlapping = chrom_segments[
            (chrom_segments['end_index_exclusive'] > gene_start) &
            (chrom_segments['start_index'] < gene_end)
        ]
        
        if len(overlapping) == 0:
            continue
        
        # Calculate weighted mean
        # Weight = overlap length between segment and gene
        weights = []
        scores = []
        
        for _, seg in overlapping.iterrows():
            overlap_start = max(seg['start_index'], gene_start)
            overlap_end = min(seg['end_index_exclusive'], gene_end)
            overlap_length = overlap_end - overlap_start
            
            weights.append(overlap_length)
            scores.append(seg['mu_z_score'])
        
        weighted_mean = np.average(scores, weights=weights)
        gene_df.at[idx, 'aggregated_mu_z'] = weighted_mean
        gene_df.at[idx, 'n_overlapping_segments'] = len(overlapping)
    
    n_with_segments = gene_df['aggregated_mu_z'].notna().sum()
    print(f"  {n_with_segments} genes have overlapping segments")
    
    return gene_df


# ============================================================================
# STEP 4: Analysis & Visualization
# ============================================================================

def plot_correlation_analysis(gene_df, output_dir):
    """
    Create scatter plot comparing fitness score vs aggregated mu_z score.
    """
    print("\nSTEP 4A: Correlation Analysis...")
    
    # Filter to valid genes with both scores
    valid = gene_df[
        gene_df['valid_for_fitness'] &
        gene_df['aggregated_mu_z'].notna()
    ].copy()
    
    print(f"  {len(valid)} genes with both scores")
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(valid['fitness_score'], valid['aggregated_mu_z'])
    spearman_r, spearman_p = spearmanr(valid['fitness_score'], valid['aggregated_mu_z'])
    
    print(f"  Pearson correlation: r={pearson_r:.3f}, p={pearson_p:.2e}")
    print(f"  Spearman correlation: r={spearman_r:.3f}, p={spearman_p:.2e}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot colored by essentiality
    essential = valid[valid['is_essential']]
    non_essential = valid[~valid['is_essential']]
    
    ax.scatter(non_essential['fitness_score'], non_essential['aggregated_mu_z'],
               alpha=0.6, s=40, label=f'Non-essential (n={len(non_essential)})', 
               color=COLORS['light_blue'], edgecolors='black', linewidth=0.3)
    ax.scatter(essential['fitness_score'], essential['aggregated_mu_z'],
               alpha=0.6, s=40, label=f'Essential (n={len(essential)})', 
               color=COLORS['red'], edgecolors='black', linewidth=0.3)
    
    # Add regression line
    z = np.polyfit(valid['fitness_score'], valid['aggregated_mu_z'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['fitness_score'].min(), valid['fitness_score'].max(), 100)
    ax.plot(x_line, p(x_line), color='black', linestyle='--', alpha=0.7, 
            linewidth=2, label='Linear fit')
    
    ax.set_xlabel('Gene-Level Fitness Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Aggregated Position-Level μ Z-Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Fitness Score vs μ Z-Score Comparison\n'
                 f'Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}',
                 fontsize=14, fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'correlation_fitness_vs_mu_z.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'correlation_fitness_vs_mu_z.png'}")
    
    return valid, pearson_r, spearman_r


def plot_gene_level_histograms(gene_df, output_dir, n_bins=15):
    """
    Compare gene-level distributions of fitness scores vs aggregated mu_z scores.
    
    Creates two histograms (stacked vertically) with matching y-axes:
    - Top: Fitness score distribution
    - Bottom: Aggregated mu_z score distribution
    
    Both show essential vs non-essential genes with overlapping transparent histograms.
    Styled to match SATAY_CPD_results/essentiality_enrichment plots.
    """
    print("\nSTEP 4B: Gene-Level Distribution Comparison...")
    
    # Filter to valid genes
    fitness_valid = gene_df[gene_df['valid_for_fitness']].copy()
    mu_z_valid = gene_df[gene_df['aggregated_mu_z'].notna()].copy()
    
    print(f"  Genes with fitness scores: {len(fitness_valid)}")
    print(f"  Genes with mu_z scores: {len(mu_z_valid)}")
    
    # Separate by essentiality
    fitness_essential = fitness_valid[fitness_valid['is_essential']]['fitness_score']
    fitness_non_essential = fitness_valid[~fitness_valid['is_essential']]['fitness_score']
    
    mu_z_essential = mu_z_valid[mu_z_valid['is_essential']]['aggregated_mu_z']
    mu_z_non_essential = mu_z_valid[~mu_z_valid['is_essential']]['aggregated_mu_z']
    
    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Determine bin edges to use for both plots
    # Use the same bins for both essential and non-essential
    fitness_min = min(fitness_essential.min(), fitness_non_essential.min())
    fitness_max = max(fitness_essential.max(), fitness_non_essential.max())
    fitness_bins = np.linspace(fitness_min, fitness_max, n_bins + 1)
    
    mu_z_min = min(mu_z_essential.min(), mu_z_non_essential.min())
    mu_z_max = max(mu_z_essential.max(), mu_z_non_essential.max())
    mu_z_bins = np.linspace(mu_z_min, mu_z_max, n_bins + 1)
    
    # Calculate bin width for x-axis positioning
    fitness_bin_width = (fitness_bins[1] - fitness_bins[0])
    mu_z_bin_width = (mu_z_bins[1] - mu_z_bins[0])
    
    # Get counts for each bin
    fitness_non_counts, _ = np.histogram(fitness_non_essential, bins=fitness_bins)
    fitness_ess_counts, _ = np.histogram(fitness_essential, bins=fitness_bins)
    mu_z_non_counts, _ = np.histogram(mu_z_non_essential, bins=mu_z_bins)
    mu_z_ess_counts, _ = np.histogram(mu_z_essential, bins=mu_z_bins)
    
    # Determine shared y-axis limit
    max_count = max(
        (fitness_non_counts + fitness_ess_counts).max(),
        (mu_z_non_counts + mu_z_ess_counts).max()
    )
    y_limit = max_count * 1.1  # Add 10% padding
    
    # TOP PLOT: Fitness scores (stacked)
    fitness_bin_centers = (fitness_bins[:-1] + fitness_bins[1:]) / 2
    
    axes[0].bar(fitness_bin_centers, fitness_non_counts, width=fitness_bin_width * 0.9,
                label=f'Non-essential (n={len(fitness_non_essential)})', 
                color=COLORS['light_blue'], edgecolor='black', linewidth=0.5)
    axes[0].bar(fitness_bin_centers, fitness_ess_counts, width=fitness_bin_width * 0.9,
                bottom=fitness_non_counts,
                label=f'Essential (n={len(fitness_essential)})', 
                color=COLORS['red'], edgecolor='black', linewidth=0.5)
    
    axes[0].set_xlabel('Fitness Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Gene Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Gene-Level Fitness Score Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', framealpha=0.9)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim(0, y_limit)
    
    # BOTTOM PLOT: Aggregated mu_z scores (stacked)
    mu_z_bin_centers = (mu_z_bins[:-1] + mu_z_bins[1:]) / 2
    
    axes[1].bar(mu_z_bin_centers, mu_z_non_counts, width=mu_z_bin_width * 0.9,
                label=f'Non-essential (n={len(mu_z_non_essential)})', 
                color=COLORS['light_blue'], edgecolor='black', linewidth=0.5)
    axes[1].bar(mu_z_bin_centers, mu_z_ess_counts, width=mu_z_bin_width * 0.9,
                bottom=mu_z_non_counts,
                label=f'Essential (n={len(mu_z_essential)})', 
                color=COLORS['red'], edgecolor='black', linewidth=0.5)
    
    axes[1].set_xlabel('Aggregated μ Z-Score', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Gene Count', fontsize=12, fontweight='bold')
    axes[1].set_title('Position-Level μ Z-Score Distribution (Aggregated to Gene)', 
                      fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', framealpha=0.9)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim(0, y_limit)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_level_score_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'gene_level_score_distributions.png'}")
    
    # Statistical tests
    print("\n  Statistical Comparisons:")
    print("  Fitness Scores (Essential vs Non-essential):")
    u_stat_fit, u_p_fit = mannwhitneyu(fitness_essential, fitness_non_essential, 
                                        alternative='two-sided')
    t_stat_fit, t_p_fit = ttest_ind(fitness_essential, fitness_non_essential)
    print(f"    Mann-Whitney U: U={u_stat_fit:.0f}, p={u_p_fit:.2e}")
    print(f"    t-test: t={t_stat_fit:.3f}, p={t_p_fit:.2e}")
    print(f"    Mean (essential): {fitness_essential.mean():.3f}")
    print(f"    Mean (non-essential): {fitness_non_essential.mean():.3f}")
    
    print("\n  mu_z Scores (Essential vs Non-essential):")
    u_stat_mu, u_p_mu = mannwhitneyu(mu_z_essential, mu_z_non_essential, 
                                      alternative='two-sided')
    t_stat_mu, t_p_mu = ttest_ind(mu_z_essential, mu_z_non_essential)
    print(f"    Mann-Whitney U: U={u_stat_mu:.0f}, p={u_p_mu:.2e}")
    print(f"    t-test: t={t_stat_mu:.3f}, p={t_p_mu:.2e}")
    print(f"    Mean (essential): {mu_z_essential.mean():.3f}")
    print(f"    Mean (non-essential): {mu_z_non_essential.mean():.3f}")


def print_summary_statistics(gene_df):
    """
    Print summary statistics comparing both scoring methods.
    """
    print("\nSTEP 4C: Summary Statistics")
    print("=" * 60)
    
    # Gene-level fitness scores
    essential = gene_df[gene_df['is_essential'] & gene_df['valid_for_fitness']]
    non_essential = gene_df[~gene_df['is_essential'] & gene_df['valid_for_fitness']]
    
    print("\nGene-Level Fitness Scores:")
    print(f"  Essential genes (n={len(essential)}):")
    print(f"    Mean: {essential['fitness_score'].mean():.3f}")
    print(f"    Median: {essential['fitness_score'].median():.3f}")
    print(f"  Non-essential genes (n={len(non_essential)}):")
    print(f"    Mean: {non_essential['fitness_score'].mean():.3f}")
    print(f"    Median: {non_essential['fitness_score'].median():.3f}")
    
    # Aggregated mu_z scores
    essential_with_mu = gene_df[gene_df['is_essential'] & gene_df['aggregated_mu_z'].notna()]
    non_essential_with_mu = gene_df[~gene_df['is_essential'] & gene_df['aggregated_mu_z'].notna()]
    
    print("\nAggregated Position-Level mu_z Scores:")
    print(f"  Essential genes (n={len(essential_with_mu)}):")
    print(f"    Mean: {essential_with_mu['aggregated_mu_z'].mean():.3f}")
    print(f"    Median: {essential_with_mu['aggregated_mu_z'].median():.3f}")
    print(f"  Non-essential genes (n={len(non_essential_with_mu)}):")
    print(f"    Mean: {non_essential_with_mu['aggregated_mu_z'].mean():.3f}")
    print(f"    Median: {non_essential_with_mu['aggregated_mu_z'].median():.3f}")
    
    print("=" * 60)


# ============================================================================
# STEP 5: Save results
# ============================================================================

def save_results(gene_df, config):
    """
    Save combined results to CSV.
    """
    print("\nSTEP 5: Saving results...")
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = config.OUTPUT_DIR / f'combined_essentiality_scores_{config.STRAIN}.csv'
    gene_df.to_csv(output_file, index=False)
    
    print(f"  Saved: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    config = Config()
    
    print("COMPARING ESSENTIALITY SCORING METHODS")
    print("=" * 60)
    
    # Step 1: Load gene-level fitness scores
    gene_df = load_gene_fitness_scores(config)
    
    # Step 2: Load position-level segments
    segments_df = load_position_level_segments(config)
    
    # Step 3: Aggregate to gene level
    gene_df = aggregate_segments_to_genes(gene_df, segments_df)
    
    # Step 4: Analysis & Visualization
    valid_df, pearson_r, spearman_r = plot_correlation_analysis(gene_df, config.OUTPUT_DIR)
    plot_gene_level_histograms(gene_df, config.OUTPUT_DIR, n_bins=15)
    print_summary_statistics(gene_df)
    
    # Step 5: Save results
    save_results(gene_df, config)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
