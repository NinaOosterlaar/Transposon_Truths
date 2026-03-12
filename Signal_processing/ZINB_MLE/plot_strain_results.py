"""
Script to create plots from ZINB estimation results saved in CSV format.
Can be run independently after estimation is complete.

Usage:
    python plot_strain_results.py
    
Or specify custom paths:
    python plot_strain_results.py --csv path/to/results.csv --output path/to/output/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import gc

# Add project root to path for Utils
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from Utils.colors import COLORBLIND_COLORS


def plot_zinb_results(csv_file, output_dir=None):
    """
    Create plots from ZINB estimation results for all strains combined and separately.
    
    Parameters:
    -----------
    csv_file : str or Path
        Path to the CSV file with ZINB estimates
    output_dir : str or Path, optional
        Directory to save plots. If None, uses same directory as csv_file
    """
    # Read the CSV file
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} samples from {csv_file}")
    
    # Set output directory - create plots subfolder
    if output_dir is None:
        output_dir = csv_file.parent / 'plots'
    else:
        output_dir = Path(output_dir) / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out theta >= 1000
    total_samples = len(df)
    df_filtered = df[df['theta'] < 1000].copy()
    excluded_samples = total_samples - len(df_filtered)
    
    print(f"\nCreating plots...")
    print(f"Total samples: {total_samples}")
    print(f"Excluded samples (theta >= 1000): {excluded_samples}")
    print(f"Samples used for plotting: {len(df_filtered)}")
    
    # ============= Plot 1: Histogram of pi =============
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(0, 1.1, 0.1)  # 0-0.1, 0.1-0.2, ..., 0.9-1.0
    counts, edges, patches = ax1.hist(df_filtered['pi'], bins=bins, 
                                      color=COLORBLIND_COLORS['blue'], 
                                      edgecolor='black', alpha=0.7)
    
    ax1.set_xlabel('π (Zero-inflation parameter)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Distribution of π estimates\n({excluded_samples} samples excluded from {total_samples} total)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(bins)
    
    # Add count labels on top of bars
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        if height > 0:
            ax1.text(patch.get_x() + patch.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9)
    
    fig1_path = output_dir / 'pi_histogram.png'
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig1_path}")
    plt.close(fig1)
    del fig1, ax1
    gc.collect()
    
    # ============= Plot 2: Boxplots of pi, mu, theta =============
    fig2, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig2.suptitle(f'Distribution of ZINB Parameters\n({excluded_samples} samples excluded from {total_samples} total)', 
                  fontsize=14, fontweight='bold')
    
    # Boxplot for pi
    bp1 = axes[0].boxplot([df_filtered['pi']], patch_artist=True, widths=0.5)
    for patch in bp1['boxes']:
        patch.set_facecolor(COLORBLIND_COLORS['blue'])
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color=COLORBLIND_COLORS['black'])
    axes[0].set_ylabel('π (Zero-inflation)', fontsize=12)
    axes[0].set_title('π Distribution', fontsize=12)
    axes[0].set_xticks([])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Boxplot for mu
    bp2 = axes[1].boxplot([df_filtered['mu']], patch_artist=True, widths=0.5)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORBLIND_COLORS['orange'])
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp2[element], color=COLORBLIND_COLORS['black'])
    axes[1].set_ylabel('μ (Mean)', fontsize=12)
    axes[1].set_title('μ Distribution', fontsize=12)
    axes[1].set_xticks([])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Boxplot for theta
    bp3 = axes[2].boxplot([df_filtered['theta']], patch_artist=True, widths=0.5)
    for patch in bp3['boxes']:
        patch.set_facecolor(COLORBLIND_COLORS['green'])
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp3[element], color=COLORBLIND_COLORS['black'])
    axes[2].set_ylabel('θ (Dispersion)', fontsize=12)
    axes[2].set_title('θ Distribution', fontsize=12)
    axes[2].set_xticks([])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    fig2_path = output_dir / 'parameter_boxplots.png'
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig2_path}")
    plt.close(fig2)
    del fig2, axes
    gc.collect()
    
    print("\n" + "="*60)
    print("Creating separate plots for each strain...")
    print("="*60)
    
    # ============= Create plots for each strain separately =============
    strains = df_filtered['strain'].unique()
    
    for strain in strains:
        print(f"\nProcessing strain: {strain}")
        df_strain = df_filtered[df_filtered['strain'] == strain].copy()
        
        # Count excluded samples for this strain
        df_strain_all = df[df['strain'] == strain]
        excluded_strain = len(df_strain_all) - len(df_strain)
        
        print(f"  Samples: {len(df_strain)} (excluded: {excluded_strain})")
        
        # Create strain-specific subfolder
        strain_dir = output_dir / strain.replace('strain_', '')
        strain_dir.mkdir(parents=True, exist_ok=True)
        
        # -------- Strain-specific histogram --------
        fig_s1, ax_s1 = plt.subplots(figsize=(10, 6))
        
        bins = np.arange(0, 1.1, 0.1)
        counts_s, edges_s, patches_s = ax_s1.hist(df_strain['pi'], bins=bins, 
                                          color=COLORBLIND_COLORS['blue'], 
                                          edgecolor='black', alpha=0.7)
        
        ax_s1.set_xlabel('π (Zero-inflation parameter)', fontsize=12)
        ax_s1.set_ylabel('Frequency', fontsize=12)
        ax_s1.set_title(f'Distribution of π estimates - {strain}\n' + 
                       f'({excluded_strain} samples excluded from {len(df_strain_all)} total)', 
                       fontsize=14, fontweight='bold')
        ax_s1.grid(True, alpha=0.3, axis='y')
        ax_s1.set_xticks(bins)
        
        # Add count labels
        for count, patch in zip(counts_s, patches_s):
            height = patch.get_height()
            if height > 0:
                ax_s1.text(patch.get_x() + patch.get_width()/2., height,
                          f'{int(count)}',
                          ha='center', va='bottom', fontsize=9)
        
        fig_s1_path = strain_dir / 'pi_histogram.png'
        fig_s1.savefig(fig_s1_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fig_s1_path}")
        plt.close(fig_s1)
        del fig_s1, ax_s1
        gc.collect()
        
        # -------- Strain-specific boxplots --------
        fig_s2, axes_s = plt.subplots(1, 3, figsize=(15, 6))
        fig_s2.suptitle(f'Distribution of ZINB Parameters - {strain}\n' +
                       f'({excluded_strain} samples excluded from {len(df_strain_all)} total)', 
                       fontsize=14, fontweight='bold')
        
        # Boxplot for pi
        bp_s1 = axes_s[0].boxplot([df_strain['pi']], patch_artist=True, widths=0.5)
        for patch in bp_s1['boxes']:
            patch.set_facecolor(COLORBLIND_COLORS['blue'])
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp_s1[element], color=COLORBLIND_COLORS['black'])
        axes_s[0].set_ylabel('π (Zero-inflation)', fontsize=12)
        axes_s[0].set_title('π Distribution', fontsize=12)
        axes_s[0].set_xticks([])
        axes_s[0].grid(True, alpha=0.3, axis='y')
        
        # Boxplot for mu
        bp_s2 = axes_s[1].boxplot([df_strain['mu']], patch_artist=True, widths=0.5)
        for patch in bp_s2['boxes']:
            patch.set_facecolor(COLORBLIND_COLORS['orange'])
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp_s2[element], color=COLORBLIND_COLORS['black'])
        axes_s[1].set_ylabel('μ (Mean)', fontsize=12)
        axes_s[1].set_title('μ Distribution', fontsize=12)
        axes_s[1].set_xticks([])
        axes_s[1].grid(True, alpha=0.3, axis='y')
        
        # Boxplot for theta
        bp_s3 = axes_s[2].boxplot([df_strain['theta']], patch_artist=True, widths=0.5)
        for patch in bp_s3['boxes']:
            patch.set_facecolor(COLORBLIND_COLORS['green'])
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp_s3[element], color=COLORBLIND_COLORS['black'])
        axes_s[2].set_ylabel('θ (Dispersion)', fontsize=12)
        axes_s[2].set_title('θ Distribution', fontsize=12)
        axes_s[2].set_xticks([])
        axes_s[2].grid(True, alpha=0.3, axis='y')
        
        fig_s2_path = strain_dir / 'parameter_boxplots.png'
        fig_s2.savefig(fig_s2_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fig_s2_path}")
        plt.close(fig_s2)
        del fig_s2, axes_s, df_strain, df_strain_all
        gc.collect()
    
    print("\n" + "="*60)
    print("All plotting complete!")
    print("="*60)
    
    # Print summary statistics
    print("\nFiltered data summary:")
    print(df_filtered[['pi', 'mu', 'theta']].describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ZINB estimation results from CSV file')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV file with ZINB estimates')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Default path if not specified
    if args.csv is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        csv_file = project_root / "Signal_processing" / "results" / "ZINB_estimates" / "zinb_estimates_windows_size2000.csv"
    else:
        csv_file = Path(args.csv)
    
    # Use output directory from args or default to CSV directory
    output_dir = Path(args.output) if args.output else None
    
    plot_zinb_results(csv_file, output_dir)
