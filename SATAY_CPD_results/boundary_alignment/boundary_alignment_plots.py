import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from Utils.plot_config import setup_plot_style, COLORS

# Setup plotting style
setup_plot_style()


# ================================================================================
# CONFIGURATION
# ================================================================================

# Paths
DATA_DIR = Path(__file__).parent.parent / "results" / "boundary_alignment"
OUTPUT_DIR = DATA_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Boundary type configurations
BOUNDARY_COLORS = {
    'gene_start': COLORS['blue'],
    'gene_end': COLORS['blue'],
    'cds_start': COLORS['orange'],
    'cds_end': COLORS['orange']
}

BOUNDARY_LABELS = {
    'gene_start': 'Gene boundaries',
    'gene_end': 'Gene boundaries',
    'cds_start': 'CDS boundaries',
    'cds_end': 'CDS boundaries'
}

# Strain colors
STRAIN_COLORS = {
    'FD': COLORS['blue'],
    'yEK19': COLORS['orange'],
    'yEK23': COLORS['green']
}

# Essentiality colors
ESSENTIALITY_COLORS = {
    True: COLORS['red'],      # Red for essential
    False: COLORS['blue']     # Blue for non-essential
}

ESSENTIALITY_LABELS = {
    True: 'Essential genes',
    False: 'Non-essential genes'
}


# ================================================================================
# PLOTTING FUNCTIONS
# ================================================================================

def plot_distance_vs_threshold(
    distance_stats_df: pd.DataFrame,
    metric: str = 'mean',
    output_dir: Path = OUTPUT_DIR
):
    """
    Plot distance to nearest boundary vs threshold.
    All strains in one plot: different colors per strain, dashed lines for CDS.
    
    Args:
        distance_stats_df: DataFrame with distance statistics (raw per-CP data)
        metric: 'mean' or 'median'
        output_dir: Directory to save plots
    """
    # First aggregate the raw data
    if 'nearest_distance' in distance_stats_df.columns:
        # Raw per-changepoint data - aggregate it
        agg_func = 'mean' if metric == 'mean' else 'median'
        stats_aggregated = distance_stats_df.groupby(['threshold', 'strain', 'boundary_type'])['nearest_distance'].agg(
            metric_value=agg_func,
            count='count'
        ).reset_index()
        metric_col = 'metric_value'
    else:
        # Already aggregated data
        metric_col = f'{metric}_distance'
        stats_aggregated = distance_stats_df.copy()
    
    # Combine boundary types (start/end are the same boundary)
    stats_aggregated['boundary_category'] = stats_aggregated['boundary_type'].map({
        'gene_start': 'gene',
        'gene_end': 'gene',
        'cds_start': 'cds',
        'cds_end': 'cds'
    })
    
    # Remove PF domain results and combined strain
    stats_aggregated = stats_aggregated[
        (stats_aggregated['boundary_category'].notna()) &
        (stats_aggregated['strain'] != 'combined')
    ]
    
    # Aggregate by category
    stats_agg = stats_aggregated.groupby(['threshold', 'strain', 'boundary_category'])[metric_col].mean().reset_index()
    
    # Create combined plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each strain with different color, gene=solid, cds=dashed
    for strain in ['FD', 'yEK19', 'yEK23']:
        for boundary_cat in ['gene', 'cds']:
            data = stats_agg[
                (stats_agg['strain'] == strain) & 
                (stats_agg['boundary_category'] == boundary_cat)
            ]
            
            if len(data) == 0:
                continue
            
            data = data.sort_values('threshold')
            
            linestyle = '-' if boundary_cat == 'gene' else '--'
            boundary_label = 'Gene' if boundary_cat == 'gene' else 'CDS'
            label = f'{strain}'
            
            ax.plot(
                data['threshold'],
                data[metric_col],
                marker='o',
                linewidth=2,
                markersize=6,
                label=label,
                color=STRAIN_COLORS[strain],
                linestyle=linestyle
            )
    
    ax.set_xlabel('Threshold', fontsize=16)
    ax.set_ylabel(f'{metric.capitalize()} Distance to Nearest Boundary (bp)', fontsize=16)
    ax.set_title(f'{metric.capitalize()} Distance: Change Points to Gene Boundaries', 
                fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    
    ax.legend(frameon=True, loc='best', ncol=2, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f"distance_vs_threshold_{metric}_all_strains.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_boundary_to_cp_distance(
    boundary_to_cp_stats_df: pd.DataFrame,
    metric: str = 'mean',
    output_dir: Path = OUTPUT_DIR
):
    """
    Plot distance from gene boundaries to nearest change point vs threshold.
    REVERSE of plot_distance_vs_threshold (boundary→CP instead of CP→boundary).
    All strains in one plot. Aggregates gene_start/end into 'gene', cds_start/end into 'cds'.
    
    Args:
        boundary_to_cp_stats_df: DataFrame with boundary-to-CP distance statistics
        metric: 'mean' or 'median'
        output_dir: Directory to save plots
    """
    metric_col = f'{metric}_distance'
    
    # Create a copy and filter
    df = boundary_to_cp_stats_df.copy()
    df = df[df['strain'] != 'combined']  # Remove combined if present
    
    # Aggregate boundary types: gene_start/end -> gene, cds_start/end -> cds
    df['boundary_category'] = df['boundary_type'].apply(
        lambda x: 'gene' if 'gene' in x else 'cds' if 'cds' in x else x
    )
    
    # Aggregate by threshold, strain, and boundary_category
    agg_df = df.groupby(['threshold', 'strain', 'boundary_category']).agg({
        metric_col: 'mean',  # Average the mean/median across start/end
        'count': 'sum'  # Sum the counts
    }).reset_index()
    
    # Create combined plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define line styles
    boundary_linestyles = {
        'gene': '-',
        'cds': '--'
    }
    
    # Plot each strain with different color, gene=solid, cds=dashed
    for strain in ['FD', 'yEK19', 'yEK23']:
        for boundary_cat in ['gene', 'cds']:
            data = agg_df[
                (agg_df['strain'] == strain) & 
                (agg_df['boundary_category'] == boundary_cat)
            ]
            
            if len(data) == 0:
                continue
            
            data = data.sort_values('threshold')
            
            label = f'{strain} - {boundary_cat.upper()}'
            
            ax.plot(
                data['threshold'],
                data[metric_col],
                marker='o',
                linewidth=2,
                markersize=6,
                label=label,
                color=STRAIN_COLORS[strain],
                linestyle=boundary_linestyles[boundary_cat]
            )
    
    ax.set_xlabel('Threshold', fontsize=16)
    ax.set_ylabel(f'{metric.capitalize()} Distance to Nearest Change Point (bp)', fontsize=16)
    ax.set_title(f'{metric.capitalize()} Distance: Gene Boundaries to Change Points', 
                fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    
    ax.legend(frameon=True, loc='best', ncol=2, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f"boundary_to_cp_distance_{metric}_all_strains.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


from matplotlib.lines import Line2D

def plot_genes_with_nearby_changepoints(
    genes_with_cp_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR
):
    """
    Plot percentage of genes with nearby change points vs threshold.
    Colors indicate strain, line style indicates essentiality.
    """
    strain_data = genes_with_cp_df[genes_with_cp_df['strain'] != 'combined'].copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    strains = ['FD', 'yEK19', 'yEK23']

    for strain in strains:
        for is_essential in [True, False]:
            data = strain_data[
                (strain_data['strain'] == strain) &
                (strain_data['is_essential'] == is_essential)
            ]

            if len(data) == 0:
                continue

            data = data.sort_values('threshold')

            linestyle = '-' if is_essential else '--'

            ax.plot(
                data['threshold'],
                data['percentage'],
                marker='o',
                linewidth=2.5,
                markersize=6,
                color=STRAIN_COLORS[strain],
                linestyle=linestyle
            )

    ax.set_xlabel('Threshold', fontsize=16)
    ax.set_ylabel('Percentage of Genes (%)', fontsize=16)
    ax.set_title(
        'Genes with Change Point Within ±100 bp',
        fontsize=22,
        fontweight='bold'
    )
    ax.tick_params(axis='both', labelsize=16)

    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Legend 1: strain colors
    strain_handles = [
        Line2D([0], [0], color=STRAIN_COLORS[strain], lw=3, label=strain)
        for strain in strains
    ]
    legend1 = ax.legend(
        handles=strain_handles,
        title="Strain",
        loc='lower left',
        frameon=True,
        fontsize=16,
        title_fontsize=16
    )
    ax.add_artist(legend1)

    # Legend 2: line styles
    style_handles = [
        Line2D([0], [0], color='black', lw=3, linestyle='-', label='Essential'),
        Line2D([0], [0], color='black', lw=3, linestyle='--', label='Non-essential'),
    ]
    ax.legend(
        handles=style_handles,
        title="Gene class",
        loc='lower right',
        frameon=True,
        fontsize=16,
        title_fontsize=16
    )

    plt.tight_layout()

    output_path = output_dir / "genes_with_nearby_changepoints.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_changepoints_with_nearby_boundaries(
    cps_with_boundary_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR
):
    """
    Plot percentage of change points with nearby boundaries vs threshold.
    All strains in one plot: different colors per strain, dashed lines for CDS.
    
    Args:
        cps_with_boundary_df: DataFrame with change points analysis
        output_dir: Directory to save plots
    """
    # Combine boundary types (start/end are the same)
    cps_with_boundary_df = cps_with_boundary_df.copy()
    cps_with_boundary_df['boundary_category'] = cps_with_boundary_df['boundary_type'].map({
        'gene_start': 'gene',
        'gene_end': 'gene',
        'cds_start': 'cds',
        'cds_end': 'cds'
    })
    
    # Remove PF domain results and combined strain
    cps_with_boundary_df = cps_with_boundary_df[
        (cps_with_boundary_df['boundary_category'].notna()) &
        (cps_with_boundary_df['strain'] != 'combined')
    ]
    
    # Aggregate by category (take max percentage since it's the same boundary)
    cps_agg = cps_with_boundary_df.groupby(['threshold', 'strain', 'boundary_category']).agg({
        'percentage': 'max',
        'total_cps': 'first'
    }).reset_index()
    
    # Create combined plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each strain with different color, gene=solid, cds=dashed
    for strain in ['FD', 'yEK19', 'yEK23']:
        for boundary_cat in ['gene', 'cds']:
            data = cps_agg[
                (cps_agg['strain'] == strain) & 
                (cps_agg['boundary_category'] == boundary_cat)
            ]
            
            if len(data) == 0:
                continue
            
            data = data.sort_values('threshold')
            
            linestyle = '-' if boundary_cat == 'gene' else '--'
            boundary_label = 'Gene' if boundary_cat == 'gene' else 'CDS'
            label = f'{strain} '
            
            ax.plot(
                data['threshold'],
                data['percentage'],
                marker='o',
                linewidth=2,
                markersize=6,
                label=label,
                color=STRAIN_COLORS[strain],
                linestyle=linestyle
            )
    
    ax.set_xlabel('Threshold', fontsize=16)
    ax.set_ylabel('Percentage of Change Points (%)', fontsize=16)
    ax.set_title('Change Points with Boundary Within ±100 bp', 
                fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    
    ax.legend(frameon=True, loc='best', ncol=2, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / "changepoints_with_nearby_boundaries_all_strains.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ================================================================================
# MAIN PLOTTING PIPELINE
# ================================================================================

def main():
    """Generate all plots from saved data."""
    
    # Load data
    
    distance_stats_file = DATA_DIR / "distance_stats_per_threshold.csv"
    genes_with_cp_file = DATA_DIR / "genes_with_nearby_cp.csv"
    cps_with_boundary_file = DATA_DIR / "cps_with_nearby_boundary.csv"
    boundary_to_cp_file = DATA_DIR / "boundary_to_cp_distance_stats.csv"
    
    if not distance_stats_file.exists():
        print(f"Distance stats file not found: {distance_stats_file}")
        print("Please run changepoint_boundary_alignment_analysis.py first!")
        return
    
    distance_stats_df = pd.read_csv(distance_stats_file)
    
    if genes_with_cp_file.exists():
        genes_with_cp_df = pd.read_csv(genes_with_cp_file)
    else:
        genes_with_cp_df = None
    
    if cps_with_boundary_file.exists():
        cps_with_boundary_df = pd.read_csv(cps_with_boundary_file)
    else:
        cps_with_boundary_df = None
    
    if boundary_to_cp_file.exists():
        boundary_to_cp_df = pd.read_csv(boundary_to_cp_file)
    else:
        boundary_to_cp_df = None
    
    # Generate plots
    
    # Plot 1: Distance vs threshold (mean) - CP to boundary
    plot_distance_vs_threshold(distance_stats_df, metric='mean', output_dir=OUTPUT_DIR)
    
    # Plot 2: Distance vs threshold (median) - CP to boundary
    plot_distance_vs_threshold(distance_stats_df, metric='median', output_dir=OUTPUT_DIR)
    
    # Plot 3: Distance vs threshold (mean) - Boundary to CP
    if boundary_to_cp_df is not None:
        plot_boundary_to_cp_distance(boundary_to_cp_df, metric='mean', output_dir=OUTPUT_DIR)
    
    # Plot 4: Distance vs threshold (median) - Boundary to CP
    if boundary_to_cp_df is not None:
        plot_boundary_to_cp_distance(boundary_to_cp_df, metric='median', output_dir=OUTPUT_DIR)
    
    # Plot 5: Genes with nearby change points
    if genes_with_cp_df is not None:
        plot_genes_with_nearby_changepoints(genes_with_cp_df, output_dir=OUTPUT_DIR)
    
    # Plot 6: Change points with nearby boundaries
    if cps_with_boundary_df is not None:
        plot_changepoints_with_nearby_boundaries(cps_with_boundary_df, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
