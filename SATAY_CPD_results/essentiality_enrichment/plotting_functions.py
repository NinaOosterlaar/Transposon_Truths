import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from Utils.plot_config import COLORS


def plot_stacked_percentage(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Gene Class Distribution by Score Bin",
    figsize: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """
    Create stacked bar plot showing percentage distribution within each bin.
    
    Stack order (bottom to top): outside_gene, non_essential_gene, essential_gene
    
    Args:
        summary_df: Summary DataFrame from position_level_analysis
        output_path: Path to save figure (if None, don't save)
        title: Plot title
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get bin centers for x-axis
    x = summary_df['bin_center'].values
    width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 1.0
    
    # Get percentages
    outside = summary_df['outside_gene_percent'].values
    non_essential = summary_df['non_essential_gene_percent'].values
    essential = summary_df['essential_gene_percent'].values
    
    # Create stacked bars (bottom to top)
    ax.bar(x, outside, width, label='Outside Gene', color=COLORS['black'], alpha=0.3, edgecolor='black', linewidth=0.5)
    ax.bar(x, non_essential, width, bottom=outside, label='Non-Essential Gene', 
           color=COLORS['light_blue'], edgecolor='black', linewidth=0.5)
    ax.bar(x, essential, width, bottom=outside + non_essential, label='Essential Gene', 
           color=COLORS['red'], edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('μ Z-Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved stacked percentage plot to: {output_path}")
    
    return fig


def plot_enrichment(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Enrichment by Score Bin",
    use_log2: bool = False,
    figsize: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """
    Create line plot showing enrichment values.
    
    Enrichment = (fraction in bin) / (overall fraction)
    
    Args:
        summary_df: Summary DataFrame from position_level_analysis
        output_path: Path to save figure
        title: Plot title
        use_log2: If True, plot log2(enrichment) instead of raw enrichment
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = summary_df['bin_center'].values
    
    # Choose enrichment column
    suffix = '_log2_enrichment' if use_log2 else '_enrichment'
    
    outside = summary_df[f'outside_gene{suffix}'].values
    non_essential = summary_df[f'non_essential_gene{suffix}'].values
    essential = summary_df[f'essential_gene{suffix}'].values
    
    # Plot lines with markers
    ax.plot(x, outside, marker='s', linewidth=2, markersize=6, 
            label='Outside Gene', color=COLORS['black'], alpha=0.5)
    ax.plot(x, non_essential, marker='o', linewidth=2, markersize=6,
            label='Non-Essential Gene', color=COLORS['light_blue'])
    ax.plot(x, essential, marker='^', linewidth=2, markersize=7,
            label='Essential Gene', color=COLORS['red'])
    
    # Add reference line at 1.0 (or 0 for log2)
    reference = 0 if use_log2 else 1.0
    ax.axhline(y=reference, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label='No Enrichment' if not use_log2 else 'No Enrichment (log2=0)')
    
    # Formatting
    ax.set_xlabel('μ Z-Score', fontsize=12, fontweight='bold')
    ylabel = 'Log₂(Enrichment)' if use_log2 else 'Enrichment (Fold Change)'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enrichment plot to: {output_path}")
    
    return fig


def plot_counts(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Position Counts by Score Bin",
    figsize: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """
    Create stacked bar plot showing raw position counts.
    
    Args:
        summary_df: Summary DataFrame from position_level_analysis
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = summary_df['bin_center'].values
    width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 1.0
    
    # Get counts
    outside = summary_df['outside_gene_count'].values
    non_essential = summary_df['non_essential_gene_count'].values
    essential = summary_df['essential_gene_count'].values
    
    # Create stacked bars
    ax.bar(x, outside, width, label='Outside Gene', color=COLORS['black'], alpha=0.3,
           edgecolor='black', linewidth=0.5)
    ax.bar(x, non_essential, width, bottom=outside, label='Non-Essential Gene',
           color=COLORS['light_blue'], edgecolor='black', linewidth=0.5)
    ax.bar(x, essential, width, bottom=outside + non_essential, label='Essential Gene',
           color=COLORS['red'], edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('μ Z-Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved counts plot to: {output_path}")
    
    return fig


def create_all_plots(
    summary_df: pd.DataFrame,
    output_dir: Path,
    strain: str,
    show_plots: bool = False
):
    """
    Create all three plot types for a strain.
    
    Args:
        summary_df: Summary DataFrame
        output_dir: Directory to save plots
        strain: Strain name for filenames
        show_plots: Whether to display plots interactively
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stacked percentage plot
    plot_stacked_percentage(
        summary_df,
        output_path=output_dir / f'strain_{strain}_stacked_percentage.png',
        title=f'Gene Class Distribution - Strain {strain}'
    )
    
    # Enrichment plot (regular)
    plot_enrichment(
        summary_df,
        output_path=output_dir / f'strain_{strain}_enrichment.png',
        title=f'Enrichment Analysis - Strain {strain}',
        use_log2=False
    )
    
    # Enrichment plot (log2)
    plot_enrichment(
        summary_df,
        output_path=output_dir / f'strain_{strain}_log2_enrichment.png',
        title=f'Log₂ Enrichment Analysis - Strain {strain}',
        use_log2=True
    )
    
    # Counts plot
    plot_counts(
        summary_df,
        output_path=output_dir / f'strain_{strain}_counts.png',
        title=f'Position Counts - Strain {strain}'
    )
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')


def main():
    """Test visualization with sample data."""
    # Create sample data
    n_bins = 10
    bin_centers = np.linspace(0, 10, n_bins)
    
    # Simulate enrichment pattern
    essential_enrichment = 1 + 0.5 * (bin_centers - 5)  # Increases with score
    non_essential_enrichment = 1 - 0.2 * (bin_centers - 5)  # Decreases
    outside_enrichment = np.ones(n_bins)  # Constant
    
    # Create sample DataFrame
    data = {
        'bin_center': bin_centers,
        'outside_gene_count': np.random.randint(1000, 5000, n_bins),
        'non_essential_gene_count': np.random.randint(500, 2000, n_bins),
        'essential_gene_count': np.random.randint(200, 1000, n_bins),
    }
    
    df = pd.DataFrame(data)
    df['total_positions'] = (df['outside_gene_count'] + 
                              df['non_essential_gene_count'] + 
                              df['essential_gene_count'])
    
    # Calculate percentages
    for cls in ['outside_gene', 'non_essential_gene', 'essential_gene']:
        df[f'{cls}_percent'] = 100 * df[f'{cls}_count'] / df['total_positions']
        df[f'{cls}_enrichment'] = [essential_enrichment, non_essential_enrichment, 
                                     outside_enrichment][['essential_gene', 'non_essential_gene', 
                                                          'outside_gene'].index(cls)]
        df[f'{cls}_log2_enrichment'] = np.log2(df[f'{cls}_enrichment'])
    
    # Create plots
    output_dir = Path(__file__).parent / 'test_plots'
    create_all_plots(df, output_dir, 'TEST', show_plots=True)
    
    print(f"\nTest plots created in: {output_dir}")


if __name__ == '__main__':
    main()
