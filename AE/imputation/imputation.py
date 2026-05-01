"""
Simplified pi imputation analysis using shared utilities.
Analyzes pi values from reconstruction and associates them with genes.
"""

import csv
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for plotting module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imputation_utils import (
    CHROMOSOMES, SPLITS, WINDOW_SIZE, STEP_SIZE,
    concatenate_nonempty, load_gene_data, load_zero_positions_from_original_data,
    assign_values_to_genes, aggregate_by_essentiality, resolve_window_size_for_split
)

from plotting.plot_imputation import create_boxplot

from Utils.plot_config import setup_plot_style
setup_plot_style()


def collect_arrays(results: Dict, strain: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate essential/non-essential arrays across chromosomes."""
    source = results.get('combined', {}) if strain is None else results.get('per_strain', {}).get(strain, {})
    essential = concatenate_nonempty([v['essential'] for v in source.values()])
    nonessential = concatenate_nonempty([v['nonessential'] for v in source.values()])
    return essential, nonessential


def save_plot_data(output_dir: str, file_tag: str, split_label: str, 
                   results: Dict, strains: Optional[List[str]] = None) -> None:
    """Save raw pi values to CSV for later analysis."""
    data_dir = os.path.join(output_dir, "plot_data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{file_tag}_pi_values.csv")
    
    total_rows = 0
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "strain", "essentiality", "pi"])
        
        # Combined data
        essential, nonessential = collect_arrays(results)
        for val in essential:
            writer.writerow([split_label, "combined", "essential", float(val)])
            total_rows += 1
        for val in nonessential:
            writer.writerow([split_label, "combined", "nonessential", float(val)])
            total_rows += 1
        
        # Per-strain data
        if strains:
            for strain in strains:
                strain_ess, strain_non = collect_arrays(results, strain)
                for val in strain_ess:
                    writer.writerow([split_label, strain, "essential", float(val)])
                    total_rows += 1
                for val in strain_non:
                    writer.writerow([split_label, strain, "nonessential", float(val)])
                    total_rows += 1
    
    print(f"    Saved plot data: {file_path} ({total_rows} rows)")


def compute_summary(results: Dict, split_label: str) -> pd.DataFrame:
    """Summarize mean essential vs non-essential pi differences."""
    essential, nonessential = collect_arrays(results)
    
    row = {
        "split": split_label,
        "n_essential": len(essential),
        "n_nonessential": len(nonessential),
        "mean_essential": np.mean(essential) if len(essential) > 0 else np.nan,
        "mean_nonessential": np.mean(nonessential) if len(nonessential) > 0 else np.nan,
        "difference": (np.mean(essential) - np.mean(nonessential)) if len(essential) > 0 and len(nonessential) > 0 else np.nan,
    }
    
    return pd.DataFrame([row])


def process_split(reconstruction_dir: str, split: str, genes_by_chr: Dict, 
                 strains: Optional[List[str]], filter_zeros: bool,
                 zero_cache: Dict, data_root: str) -> Dict:
    """Process all chromosomes for a single split."""
    split_dir = os.path.join(reconstruction_dir, split)
    
    if not os.path.exists(split_dir):
        return {'per_strain': {}, 'combined': {}}
    
    if strains is None:
        strains = sorted([d for d in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, d)) and d.startswith('strain_')])
    
    window_size = resolve_window_size_for_split(split_dir)
    
    results = {
        'per_strain': defaultdict(lambda: defaultdict(lambda: {'essential': [], 'nonessential': []})),
        'combined': defaultdict(lambda: {'essential': [], 'nonessential': []})
    }
    
    for strain in strains:
        strain_dir = os.path.join(split_dir, strain)
        
        for chr_name in CHROMOSOMES:
            csv_file = os.path.join(strain_dir, f"{chr_name}.csv")
            if not os.path.exists(csv_file):
                continue
            
            # Load zero positions if filtering
            zero_positions = None
            if filter_zeros:
                cache_key = (strain, chr_name)
                if cache_key not in zero_cache:
                    orig_file = os.path.join(data_root, strain, f"{chr_name}_distances.csv")
                    zero_cache[cache_key] = load_zero_positions_from_original_data(orig_file)
                zero_positions = zero_cache[cache_key]
            
            # Assign pi values to genes
            genes_values = assign_values_to_genes(
                csv_file, genes_by_chr, chr_name, "pi",
                filter_zeros, zero_positions, window_size, STEP_SIZE
            )
            
            # Aggregate by essentiality
            essential, nonessential = aggregate_by_essentiality(genes_values, genes_by_chr, chr_name)
            
            results['per_strain'][strain][chr_name]['essential'] = essential
            results['per_strain'][strain][chr_name]['nonessential'] = nonessential
            results['combined'][chr_name]['essential'].extend(essential)
            results['combined'][chr_name]['nonessential'].extend(nonessential)
    
    # Convert combined lists to arrays
    for chr_name in results['combined']:
        results['combined'][chr_name]['essential'] = np.array(results['combined'][chr_name]['essential'])
        results['combined'][chr_name]['nonessential'] = np.array(results['combined'][chr_name]['nonessential'])
    
    return results


def run_analysis(reconstruction_dir: str = "Data/reconstruction",
                output_dir: str = "AE/results/main_results/pi_analysis",
                filter_zeros: bool = True,
                data_root: str = "Data/combined_strains") -> None:
    """Run complete pi imputation analysis."""
    
    if filter_zeros and not output_dir.endswith("_original_zero_filter"):
        output_dir = f"{output_dir}_original_zero_filter"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading gene data...")
    genes_by_chr = load_gene_data(reconstruction_dir)
    
    if filter_zeros:
        print(f"Filtering pi values to original zeros from {data_root}")
        print(f"Using filtered output directory: {output_dir}")
    
    # Find strains
    sample_dir = os.path.join(reconstruction_dir, "train")
    all_strains = sorted([d for d in os.listdir(sample_dir)
                         if os.path.isdir(os.path.join(sample_dir, d)) and d.startswith('strain_')])
    print(f"Found strains: {all_strains}")
    
    overall_essential = []
    overall_nonessential = []
    overall_by_strain = defaultdict(lambda: {'essential': [], 'nonessential': []})
    zero_cache = {}
    
    # Process each split
    for split in SPLITS:
        print(f"\n{'='*60}\nProcessing {split.upper()} split\n{'='*60}")
        
        results = process_split(reconstruction_dir, split, genes_by_chr, all_strains,
                               filter_zeros, zero_cache, data_root)
        
        # Accumulate for overall figures
        for chr_values in results.get('combined', {}).values():
            overall_essential.append(chr_values['essential'])
            overall_nonessential.append(chr_values['nonessential'])
        
        for strain_name, strain_results in results.get('per_strain', {}).items():
            for chr_values in strain_results.values():
                overall_by_strain[strain_name]['essential'].append(chr_values['essential'])
                overall_by_strain[strain_name]['nonessential'].append(chr_values['nonessential'])
        
        # Save data and summary
        print("  Saving plot data...")
        save_plot_data(output_dir, split, split, results, all_strains)
        
        summary_df = compute_summary(results, split)
        if not summary_df.empty:
            summary_path = os.path.join(output_dir, "plot_data", f"{split}_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"    Saved summary data: {summary_path}")
        
        # Create figures
        print("  Creating figures...")
        essential, nonessential = collect_arrays(results)
        fig, ax = plt.subplots(figsize=(8, 6))
        create_boxplot(ax, essential, nonessential, f'Essential vs Non-Essential - {split.upper()}')
        fig.savefig(os.path.join(output_dir, f"{split}_combined.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        for strain in all_strains:
            essential, nonessential = collect_arrays(results, strain)
            fig, ax = plt.subplots(figsize=(8, 6))
            create_boxplot(ax, essential, nonessential, f'{strain} - Essential vs Non-Essential - {split.upper()}')
            fig.savefig(os.path.join(output_dir, f"{split}_{strain}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # Overall figures (all splits combined)
    print(f"\n{'='*60}\nCreating overall figures\n{'='*60}")
    
    essential_all = concatenate_nonempty(overall_essential)
    nonessential_all = concatenate_nonempty(overall_nonessential)
    
    all_sets_results = {
        'combined': {'all_splits': {'essential': essential_all, 'nonessential': nonessential_all}}
    }
    
    save_plot_data(output_dir, "all_sets_combined", "all_sets", all_sets_results)
    
    summary_df = compute_summary(all_sets_results, "all_sets")
    if not summary_df.empty:
        summary_path = os.path.join(output_dir, "plot_data", "all_sets_summary.csv")
        summary_df.to_csv(summary_path, index=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    create_boxplot(ax, essential_all, nonessential_all, 'Essential vs Non-Essential - ALL SETS')
    fig.savefig(os.path.join(output_dir, "all_sets_all_strains_combined.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Per-strain overall figures
    for strain in all_strains:
        essential_strain = concatenate_nonempty(overall_by_strain[strain]['essential'])
        nonessential_strain = concatenate_nonempty(overall_by_strain[strain]['nonessential'])
        
        strain_results = {
            'per_strain': {strain: {'all_splits': {'essential': essential_strain, 'nonessential': nonessential_strain}}}
        }
        
        save_plot_data(output_dir, f"all_sets_{strain}", "all_sets", strain_results, [strain])
        fig, ax = plt.subplots(figsize=(8, 6))
        create_boxplot(ax, essential_strain, nonessential_strain, f'{strain} - Essential vs Non-Essential - ALL SETS')
        fig.savefig(os.path.join(output_dir, f"all_sets_{strain}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n{'='*60}\nAnalysis complete! Figures saved to: {output_dir}\n{'='*60}")


if __name__ == "__main__":
    run_analysis(
        reconstruction_dir="Data/reconstruction/ZINBAE_layers752_ep93_noise0.150_muoff0.000",
        output_dir="AE/results/main_results/pi_analysis_ZINBAE_layers752",
    )
