import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.SGD_API.yeast_genes import SGD_Genes
from Utils.plot_config import setup_plot_style

setup_plot_style()

# Chromosome list (S288C reference)
CHROMOSOMES = [
    "ChrI", "ChrII", "ChrIII", "ChrIV", "ChrV", "ChrVI", "ChrVII", 
    "ChrVIII", "ChrIX", "ChrX", "ChrXI", "ChrXII", "ChrXIII", "ChrXIV", 
    "ChrXV", "ChrXVI"
]

# Mu offset values
MU_OFFSETS = ["muoff0", "muoff1", "muoff2", "muoff4", "muoff5"]

# Splits to analyze
SPLITS = ["train", "val", "test"]

# Window parameters for position mapping
WINDOW_SIZE = 19
STEP_SIZE = 1


def normalize_chromosome_name(chr_name: Optional[str]) -> Optional[str]:
    """Normalize chromosome labels from SGD metadata to reconstruction labels."""
    if not chr_name:
        return None

    if chr_name in CHROMOSOMES:
        return chr_name

    if chr_name.startswith("Chromosome_"):
        return f"Chr{chr_name.split('_', 1)[1]}"

    if chr_name.startswith("chr"):
        suffix = chr_name[3:]
        normalized = f"Chr{suffix}"
        if normalized in CHROMOSOMES:
            return normalized

    return chr_name


def concatenate_nonempty(arrays: List[np.ndarray]) -> np.ndarray:
    """Concatenate non-empty arrays and return an empty array when none exist."""
    nonempty_arrays = [np.asarray(array, dtype=float) for array in arrays if len(array) > 0]
    if not nonempty_arrays:
        return np.array([], dtype=float)
    return np.concatenate(nonempty_arrays)


def load_gene_data(mu_offset_root_dir: str) -> Dict:
    """
    Load gene data from SGD_Genes class and filter to genes in reconstruction.
    
    Args:
        mu_offset_root_dir: Root directory containing mu_offset folders
        
    Returns:
        Dictionary mapping gene_location strings to gene info
        {chr_name -> {start -> {end -> {essentiality, gene_name}}}}
    """
    # Try to find existing gene JSON file
    gene_files = [
        "Utils/SGD_API/architecture_info/yeast_genes_with_info.json",
        "Utils/SGD_API/S288C/genes_info.json",
        "genes_info.json",
    ]
    
    genes_dict = {}
    for gene_file in gene_files:
        if os.path.exists(gene_file):
            print(f"Loading genes from {gene_file}")
            with open(gene_file, 'r') as f:
                genes_raw = json.load(f)
                genes_dict = genes_raw
                break
    
    if not genes_dict:
        print("No pre-cached gene file found. Attempting to load from SGD API...")
        try:
            sgd = SGD_Genes(gene_list_with_info="Utils/SGD_API/architecture_info/yeast_genes_with_info.json")
            genes_dict = sgd.list_all_gene_info()
        except Exception as e:
            print(f"Warning: Could not load from SGD: {e}")
            genes_dict = {}
    
    # Organize genes by chromosome and position for efficient overlap queries
    genes_by_chr = defaultdict(list)
    for gene_name, info in genes_dict.items():
        if "location" in info:
            loc = info["location"]
            chr_name = normalize_chromosome_name(loc.get("chromosome"))
            start = loc.get("start")
            end = loc.get("end")
            if chr_name and start is not None and end is not None:
                genes_by_chr[chr_name].append({
                    "gene": gene_name,
                    "start": int(start),
                    "end": int(end),
                    "essentiality": info.get("essentiality", None)
                })
    
    # Sort by start position for efficient overlap detection
    for chr_name in genes_by_chr:
        genes_by_chr[chr_name].sort(key=lambda x: x["start"])
    
    return genes_by_chr


def get_genes_overlapping_position(
    genes_by_chr: Dict,
    chr_name: str,
    pos_index: int,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE
) -> List[Dict]:
    """
    Find all genes overlapping with a position's window.
    
    Position index i maps to genomic range [i*step_size, i*step_size + window_size - 1]
    
    Args:
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name (e.g., "ChrIII")
        pos_index: Position index in CSV
        window_size: Window size (19)
        step_size: Step size (1)
        
    Returns:
        List of genes overlapping this position
    """
    if chr_name not in genes_by_chr:
        return []
    
    # SGD coordinates are 1-based; reconstruction positions are zero-based indices.
    window_start = pos_index * step_size + 1
    window_end = window_start + window_size - 1
    
    # Binary search to find genes in range (chromosomes sorted by start)
    genes = genes_by_chr[chr_name]
    overlapping = []
    
    for gene in genes:
        gene_start = gene["start"]
        gene_end = gene["end"]
        
        # Check overlap: [window_start, window_end] intersects [gene_start, gene_end]
        if gene_start <= window_end and gene_end >= window_start:
            overlapping.append(gene)
    
    return overlapping


def assign_pi_to_genes(
    csv_file: str,
    genes_by_chr: Dict,
    chr_name: str
) -> Dict[str, List[float]]:
    """
    Parse CSV and assign pi values to genes based on position-gene overlap.
    
    Args:
        csv_file: Path to CSV file with columns [position, reconstruction, mu, pi, theta]
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name
        
    Returns:
        Dictionary mapping gene_name -> [list of pi values]
    """
    raw_df = pd.read_csv(
        csv_file,
        usecols=["position", "pi"],
        low_memory=False,
        dtype={"position": "string", "pi": "string"}
    )
    df = raw_df.copy()
    df["position"] = pd.to_numeric(raw_df["position"], errors="coerce")
    df["pi"] = pd.to_numeric(raw_df["pi"], errors="coerce")

    invalid_mask = df["position"].isna() | df["pi"].isna()
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        example_messages = []
        for row_index in df.index[invalid_mask][:3]:
            line_number = int(row_index) + 2
            reasons = []

            raw_position = raw_df.at[row_index, "position"]
            raw_pi = raw_df.at[row_index, "pi"]

            if pd.isna(raw_position):
                reasons.append("missing position")
            elif pd.isna(df.at[row_index, "position"]):
                reasons.append(f"non-numeric position '{raw_position}'")

            if pd.isna(raw_pi):
                reasons.append("missing pi")
            elif pd.isna(df.at[row_index, "pi"]):
                reasons.append(f"non-numeric pi '{raw_pi}'")

            example_messages.append(f"line {line_number} ({', '.join(reasons)})")

        examples = "; ".join(example_messages)
        print(f"    Skipping {invalid_count} malformed rows in {csv_file}: {examples}")

    df = df.loc[~invalid_mask, ["position", "pi"]]
    if df.empty:
        return defaultdict(list)
    
    genes_pi = defaultdict(list)
    
    for pos_index, pi_value in df.itertuples(index=False, name=None):
        pos_index = int(pos_index)
        pi_value = float(pi_value)
        
        # Find overlapping genes
        overlapping_genes = get_genes_overlapping_position(
            genes_by_chr, chr_name, pos_index
        )
        
        # Assign pi value to each overlapping gene
        for gene in overlapping_genes:
            genes_pi[gene['gene']].append(pi_value)
    
    return genes_pi


def aggregate_pi_by_essentiality(
    genes_pi: Dict[str, List[float]],
    genes_by_chr: Dict,
    chr_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate all pi values by essentiality status.
    
    Args:
        genes_pi: Dictionary mapping gene name to list of pi values
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name
        
    Returns:
        Tuple of (essential_pi_values, nonessential_pi_values) as numpy arrays
    """
    # Create lookup for essentiality
    essentiality_lookup = {}
    for gene in genes_by_chr.get(chr_name, []):
        essentiality_lookup[gene['gene']] = gene['essentiality']
    
    essential_pi = []
    nonessential_pi = []
    
    for gene_name, pi_values in genes_pi.items():
        essentiality = essentiality_lookup.get(gene_name)
        
        if essentiality is None:
            continue
        
        if essentiality:  # Essential
            essential_pi.extend(pi_values)
        else:  # Non-essential
            nonessential_pi.extend(pi_values)
    
    return np.array(essential_pi), np.array(nonessential_pi)


def process_single_chromosomal_split(
    mu_offset_dir: str,
    split: str,
    genes_by_chr: Dict,
    strains: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Process all chromosomes for a single mu_offset and split (e.g., train).
    Collect results separately per strain and combined.
    
    Args:
        mu_offset_dir: Path to specific muoff* directory
        split: "train", "val", or "test"
        genes_by_chr: Gene data
        strains: List of strains to process, or None for all found
        
    Returns:
        Dictionary with structure:
        {
            'per_strain': {strain_name: {chr_name: {'essential': [...], 'nonessential': [...]}}},
            'combined': {chr_name: {'essential': [...], 'nonessential': [...]}}
        }
    """
    split_dir = os.path.join(mu_offset_dir, split)
    
    if not os.path.exists(split_dir):
        return {'per_strain': {}, 'combined': {}}
    
    # Find all strain directories
    if strains is None:
        strains = sorted([d for d in os.listdir(split_dir) 
                         if os.path.isdir(os.path.join(split_dir, d)) and d.startswith('strain_')])
    
    results = {
        'per_strain': defaultdict(lambda: defaultdict(lambda: {'essential': [], 'nonessential': []})),
        'combined': defaultdict(lambda: {'essential': [], 'nonessential': []})
    }
    
    # Process each strain
    for strain in strains:
        strain_dir = os.path.join(split_dir, strain)
        
        for chr_name in CHROMOSOMES:
            csv_file = os.path.join(strain_dir, f"{chr_name}.csv")
            
            if not os.path.exists(csv_file):
                continue
            
            # Assign pi to genes
            genes_pi = assign_pi_to_genes(csv_file, genes_by_chr, chr_name)
            
            # Aggregate by essentiality
            essential, nonessential = aggregate_pi_by_essentiality(
                genes_pi, genes_by_chr, chr_name
            )
            
            # Store per-strain results
            results['per_strain'][strain][chr_name]['essential'] = essential
            results['per_strain'][strain][chr_name]['nonessential'] = nonessential
            
            # Accumulate for combined results
            results['combined'][chr_name]['essential'].extend(essential)
            results['combined'][chr_name]['nonessential'].extend(nonessential)
    
    # Convert combined lists to arrays
    for chr_name in results['combined']:
        results['combined'][chr_name]['essential'] = np.array(results['combined'][chr_name]['essential'])
        results['combined'][chr_name]['nonessential'] = np.array(results['combined'][chr_name]['nonessential'])
    
    return results


def create_comparison_boxplot(
    ax,
    essential_pi: np.ndarray,
    nonessential_pi: np.ndarray,
    title: str
) -> None:
    """
    Create a boxplot comparing essential vs non-essential gene pi values.
    
    Args:
        ax: Matplotlib axis to plot on
        essential_pi: Array of pi values for essential genes
        nonessential_pi: Array of pi values for non-essential genes
        title: Title for the subplot
    """
    data_to_plot = [essential_pi, nonessential_pi]
    
    bp = ax.boxplot(
        data_to_plot,
        labels=['Essential', 'Non-Essential'],
        patch_artist=True,
        widths=0.6
    )
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('π value', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size info
    n_ess = len(essential_pi)
    n_noness = len(nonessential_pi)
    ax.text(0.98, 0.97, f'n={n_ess}/{n_noness}', 
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_split_figure_combined(
    results_by_muoffset: Dict[str, Dict],
    split: str,
    figsize: Tuple = (16, 10)
) -> plt.Figure:
    """
    Create one figure with subplots for all mu_offsets.
    Each subplot compares essential vs non-essential across all strains (combined).
    
    Args:
        results_by_muoffset: {muoff_name: results_dict}
        split: "train", "val", or "test"
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, muoff_name in enumerate(MU_OFFSETS):
        ax = axes[idx]
        
        if muoff_name not in results_by_muoffset:
            ax.text(0.5, 0.5, f'{muoff_name}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        results = results_by_muoffset[muoff_name]
        combined = results.get('combined', {})
        
        # Aggregate across all chromosomes
        all_essential = concatenate_nonempty([
            v['essential'] for v in combined.values()
        ])
        all_nonessential = concatenate_nonempty([
            v['nonessential'] for v in combined.values()
        ])
        
        if len(all_essential) > 0 and len(all_nonessential) > 0:
            create_comparison_boxplot(
                ax, all_essential, all_nonessential,
                f'{muoff_name} ({split})'
            )
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide last subplot if odd number
    if len(MU_OFFSETS) < len(axes):
        axes[len(MU_OFFSETS)].axis('off')
    
    fig.suptitle(f'Essential vs Non-Essential Genes - {split.upper()}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def create_split_figure_per_strain(
    results_by_muoffset: Dict[str, Dict],
    strain: str,
    split: str,
    figsize: Tuple = (16, 10)
) -> plt.Figure:
    """
    Create one figure with subplots for all mu_offsets for a specific strain.
    
    Args:
        results_by_muoffset: {muoff_name: results_dict}
        strain: Strain name
        split: "train", "val", or "test"
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, muoff_name in enumerate(MU_OFFSETS):
        ax = axes[idx]
        
        if muoff_name not in results_by_muoffset:
            ax.text(0.5, 0.5, f'{muoff_name}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        results = results_by_muoffset[muoff_name]
        per_strain = results.get('per_strain', {})
        
        if strain not in per_strain:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        strain_results = per_strain[strain]
        
        # Aggregate across all chromosomes for this strain
        all_essential = concatenate_nonempty([
            v['essential'] for v in strain_results.values()
        ])
        all_nonessential = concatenate_nonempty([
            v['nonessential'] for v in strain_results.values()
        ])
        
        if len(all_essential) > 0 and len(all_nonessential) > 0:
            create_comparison_boxplot(
                ax, all_essential, all_nonessential,
                f'{muoff_name} ({split})'
            )
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide last subplot
    axes[len(MU_OFFSETS)].axis('off')
    
    fig.suptitle(f'{strain} - Essential vs Non-Essential Genes - {split.upper()}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def run_full_analysis(
    mu_offset_root_dir: str = "Data/reconstruction/mu_offset",
    output_dir: str = "AE/results/pi_analysis"
) -> None:
    """
    Run complete analysis: load genes, process all splits and mu_offsets,
    generate visualizations for combined and per-strain analyses.
    
    Args:
        mu_offset_root_dir: Root directory containing mu_offset folders
        output_dir: Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading gene data...")
    genes_by_chr = load_gene_data(mu_offset_root_dir)
    
    # Find all strains from first mu_offset/train
    sample_dir = os.path.join(mu_offset_root_dir, "muoff0", "train")
    all_strains = sorted([d for d in os.listdir(sample_dir) 
                         if os.path.isdir(os.path.join(sample_dir, d)) and d.startswith('strain_')])
    print(f"Found strains: {all_strains}")
    
    # Process each split
    for split in SPLITS:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        
        # Results organized by mu_offset
        results_by_muoffset = {}
        
        for muoff_name in MU_OFFSETS:
            mu_offset_dir = os.path.join(mu_offset_root_dir, muoff_name)
            
            if not os.path.exists(mu_offset_dir):
                print(f"  Skipping {muoff_name} (not found)")
                continue
            
            print(f"  Processing {muoff_name}...")
            results = process_single_chromosomal_split(
                mu_offset_dir, split, genes_by_chr, all_strains
            )
            results_by_muoffset[muoff_name] = results
        
        # Generate figures for combined analysis
        print(f"  Creating combined analysis figure...")
        fig_combined = create_split_figure_combined(results_by_muoffset, split)
        fig_path_combined = os.path.join(output_dir, f"{split}_combined.png")
        fig_combined.savefig(fig_path_combined, dpi=150, bbox_inches='tight')
        print(f"    Saved: {fig_path_combined}")
        plt.close(fig_combined)
        
        # Generate figures for each strain
        print(f"  Creating per-strain analysis figures...")
        for strain in all_strains:
            fig_strain = create_split_figure_per_strain(results_by_muoffset, strain, split)
            fig_path_strain = os.path.join(output_dir, f"{split}_{strain}.png")
            fig_strain.savefig(fig_path_strain, dpi=150, bbox_inches='tight')
            print(f"    Saved: {fig_path_strain}")
            plt.close(fig_strain)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_full_analysis()
