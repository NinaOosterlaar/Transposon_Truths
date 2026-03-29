import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from Utils.plot_config import setup_plot_style

# Setup plotting style
setup_plot_style()

# Configuration
genes = ["PRO3", "CDC28", "SEC18", "KAR2", "RIB3", "POL2", "NOT1", "SUP35", "RNR1", "RNR2", "BEM1", "BEM2", "BEM3", "BEM4"]
protein_domain = "PF"
threshold = 3.0
strains = ["FD", "dnrp", "yEK19", "yEK23", "ylic137", "yTW001", "yWT03a", "yWT04a"]
window_size = 100
overlap = 50
padding_bp = 500  # Base pairs to show before and after gene

# Paths
gene_info_path = Path(__file__).parent.parent / "Utils" / "SGD_API" / "architecture_info" / "yeast_genes_with_info.json"
strains_data_path = Path(__file__).parent.parent / "Signal_processing" / "strains"
count_data_path = Path(__file__).parent.parent / "Data" / "combined_strains"
output_dir = Path(__file__).parent / "genes_overview_plots"
output_dir.mkdir(exist_ok=True)


def load_gene_info():
    """Load gene information from JSON file."""
    with open(gene_info_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping from gene_name to full info
    gene_dict = {}
    for orf, info in data.items():
        if info['gene_name'] in genes:
            gene_dict[info['gene_name']] = {
                'orf': orf,
                'chromosome': info['location']['chromosome'],
                'start': info['location']['start'],
                'end': info['location']['end'],
                'essentiality': info['essentiality'],
                'protein_domains': {k: v for k, v in info['protein_domains'].items() 
                                   if k.startswith(protein_domain)}
            }
    return gene_dict


def load_strain_segments(strain, chromosome, threshold, window_size):
    """Load segment mu data for a specific strain, chromosome, and threshold."""
    # Convert chromosome format
    chr_short = chromosome.replace('Chromosome_', 'Chr')
    
    file_path = (strains_data_path / f"strain_{strain}" / chr_short / 
                 f"{chr_short}_distances" / f"window{window_size}" / "segment_mu" /
                 f"{chr_short}_distances_ws{window_size}_ov{overlap}_th{threshold:.2f}_segment_mu.csv")
    
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    return df


def load_count_data(strain, chromosome):
    """Load raw count data for a specific strain and chromosome."""
    chr_short = chromosome.replace('Chromosome_', 'Chr')
    
    file_path = count_data_path / f"strain_{strain}" / f"{chr_short}_distances.csv"
    
    if not file_path.exists():
        print(f"Warning: Count data not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    return df


def get_overlapping_segments(segments_df, start, end):
    """Filter segments that overlap with the specified region."""
    if segments_df is None:
        return None
    
    # Segment overlaps if: segment_start < region_end AND segment_end > region_start
    mask = (segments_df['start_index'] < end) & (segments_df['end_index_exclusive'] > start)
    return segments_df[mask].copy()


def plot_gene_overview(gene_name, gene_info):
    """Create overview plot for a single gene showing protein domains and strain segments."""
    
    # Calculate display region
    gene_start = gene_info['start']
    gene_end = gene_info['end']
    display_start = gene_start - padding_bp
    display_end = gene_end + padding_bp
    
    # Create figure
    num_rows = len(strains) + 1  # +1 for gene annotation row
    fig, ax = plt.subplots(figsize=(16, num_rows * 0.8 + 2))
    
    # Y-axis positions for each row (more space for count data)
    y_positions = list(range(num_rows, 0, -1))
    y_spacing = 1.0  # Increased spacing to accommodate count data
    
    # Row 0 (top): Gene annotation with protein domains
    gene_y = y_positions[0] * y_spacing
    
    # Draw gene boundaries
    ax.axvline(gene_start, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Gene boundaries')
    ax.axvline(gene_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Draw gene as a horizontal line
    ax.plot([gene_start, gene_end], [gene_y, gene_y], color='black', linewidth=3, label='Gene')
    
    # Draw protein domains
    protein_domains = gene_info['protein_domains']
    domain_colors = {}
    
    # Generate random colors for each domain
    np.random.seed(42)  # For reproducibility
    cmap = plt.cm.get_cmap('tab20')
    
    for idx, (domain_id, domain_data) in enumerate(protein_domains.items()):
        color = cmap(idx % 20)
        domain_colors[domain_id] = color
        
        # Domains are in amino acid coordinates, need to convert to bp
        # Each amino acid is approximately 3 base pairs
        for start_aa, end_aa in zip(domain_data['start'], domain_data['end']):
            # Convert AA to BP (approximate)
            domain_start_bp = gene_start + (start_aa - 1) * 3
            domain_end_bp = gene_start + (end_aa * 3)
            
            # Ensure we don't go beyond gene boundaries
            domain_start_bp = max(gene_start, min(gene_end, domain_start_bp))
            domain_end_bp = max(gene_start, min(gene_end, domain_end_bp))
            
            rect = mpatches.Rectangle((domain_start_bp, gene_y - 0.2), 
                                     domain_end_bp - domain_start_bp, 0.4,
                                     linewidth=2, edgecolor='black', 
                                     facecolor=color, alpha=0.7)
            ax.add_patch(rect)
    
    # Collect all mu_z_scores to determine range for colormap
    all_mu_z_scores = []
    for strain in strains:
        segments_df = load_strain_segments(strain, gene_info['chromosome'], 
                                          threshold, window_size)
        if segments_df is not None:
            overlapping = get_overlapping_segments(segments_df, display_start, display_end)
            if overlapping is not None and len(overlapping) > 0:
                all_mu_z_scores.extend(overlapping['mu_z_score'].values)
    
    if all_mu_z_scores:
        # Create diverging colormap centered at 0 (RdBu: red for negative, blue for positive)
        vmin = min(all_mu_z_scores)
        vmax = max(all_mu_z_scores)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap_div = plt.cm.RdBu  # Red for negative, blue for positive
        
        # Clear and replot everything with proper colors
        ax.clear()
        
        # Redraw gene annotation
        ax.axvline(gene_start, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(gene_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot([gene_start, gene_end], [gene_y, gene_y], color='black', linewidth=3)
        
        # Redraw protein domains
        for idx, (domain_id, domain_data) in enumerate(protein_domains.items()):
            color = domain_colors[domain_id]
            for start_aa, end_aa in zip(domain_data['start'], domain_data['end']):
                domain_start_bp = gene_start + (start_aa - 1) * 3
                domain_end_bp = gene_start + (end_aa * 3)
                domain_start_bp = max(gene_start, min(gene_end, domain_start_bp))
                domain_end_bp = max(gene_start, min(gene_end, domain_end_bp))
                
                rect = mpatches.Rectangle((domain_start_bp, gene_y - 0.2), 
                                         domain_end_bp - domain_start_bp, 0.4,
                                         linewidth=2, edgecolor='black', 
                                         facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Add domain name as text on the domain
                domain_center_bp = (domain_start_bp + domain_end_bp) / 2
                domain_label = f"{domain_id}"
                ax.text(domain_center_bp, gene_y, domain_label, 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='none', alpha=0.7))
        
        # Redraw strain segments with color and count data
        for idx, strain in enumerate(strains):
            strain_y = y_positions[idx + 1] * y_spacing
            
            segments_df = load_strain_segments(strain, gene_info['chromosome'], 
                                              threshold, window_size)
            
            if segments_df is not None:
                overlapping = get_overlapping_segments(segments_df, display_start, display_end)
                
                if overlapping is not None and len(overlapping) > 0:
                    for _, seg in overlapping.iterrows():
                        seg_start = max(display_start, seg['start_index'])
                        seg_end = min(display_end, seg['end_index_exclusive'])
                        mu_z = seg['mu_z_score']
                        
                        color = cmap_div(norm(mu_z))
                        rect = mpatches.Rectangle((seg_start, strain_y - 0.3), 
                                                 seg_end - seg_start, 0.6,
                                                 linewidth=2, edgecolor='black',
                                                 facecolor=color, alpha=0.9)
                        ax.add_patch(rect)
                        
                        # Add z-score text on the bar if the segment is wide enough
                        seg_width = seg_end - seg_start
                        if seg_width > (display_end - display_start) * 0.05:  # Only show if >5% of display width
                            seg_center = (seg_start + seg_end) / 2
                            ax.text(seg_center, strain_y, f"{mu_z:.1f}", 
                                   ha='center', va='center', fontsize=7,
                                   color='white' if abs(norm(mu_z) - 0.5) > 0.3 else 'black',
                                   fontweight='bold')
            
            # Add count data above the segments
            count_data = load_count_data(strain, gene_info['chromosome'])
            if count_data is not None:
                # Filter to display region
                region_data = count_data[
                    (count_data['Position'] >= display_start) & 
                    (count_data['Position'] <= display_end)
                ]
                
                if len(region_data) > 0:
                    # Normalize count data for display (small height)
                    max_count = region_data['Value'].max()
                    if max_count > 0:
                        normalized_counts = region_data['Value'] / max_count * 0.25  # Scale to 0.25 units height
                        
                        # Plot as a line above the segments
                        ax.plot(region_data['Position'], 
                               strain_y + 0.35 + normalized_counts,
                               color='darkgray', linewidth=0.8, alpha=0.7, zorder=1)
    
    # Set axis properties
    ax.set_xlim(display_start, display_end)
    ax.set_ylim(0.5 * y_spacing, (num_rows + 0.5) * y_spacing)
    ax.set_xlabel('Genomic Position (bp)')
    ax.set_yticks([y * y_spacing for y in y_positions])
    ax.set_yticklabels(['Gene\nAnnotation'] + strains)
    
    # Title
    essential_status = "Essential" if gene_info['essentiality'] else "Non-essential"
    title = (f"{gene_name} ({gene_info['orf']}) - {gene_info['chromosome']}\n"
             f"{essential_status}, Position: {gene_start:,}-{gene_end:,} bp, "
             f"Threshold: {threshold}")
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add colorbar for mu_z_score
    if all_mu_z_scores:
        sm = plt.cm.ScalarMappable(cmap=cmap_div, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('μ z-score', rotation=270, labelpad=20)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axhline(y_positions[1] * y_spacing - 0.5 * y_spacing, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{gene_name}_{gene_info['orf']}_overview.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.close()


def main():
    """Main function to generate all gene overview plots."""
    print("Loading gene information...")
    gene_dict = load_gene_info()
    
    print(f"\nGenerating overview plots for {len(gene_dict)} genes...")
    print(f"Threshold: {threshold}")
    print(f"Strains: {', '.join(strains)}")
    print(f"Padding: ±{padding_bp} bp")
    print(f"Output directory: {output_dir}\n")
    
    for gene_name in genes:
        if gene_name in gene_dict:
            print(f"Processing {gene_name}...")
            plot_gene_overview(gene_name, gene_dict[gene_name])
        else:
            print(f"Warning: {gene_name} not found in gene info file")
    
    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
