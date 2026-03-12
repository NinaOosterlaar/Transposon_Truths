import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import gc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ZINB_MLE.estimate_ZINB import estimate_zinb

# Add project root to path for Utils
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from Utils.colors import COLORBLIND_COLORS


def estimate_whole_dataset():
    """
    Estimate ZINB parameters (pi, mu, theta) for the entire dataset of each strain.
    All chromosomes are combined for each strain to produce a single set of estimates.
    
    Returns:
    --------
    dict : Dictionary where keys are strain names and values are dictionaries containing:
        - 'pi': zero-inflation parameter estimate
        - 'mu': mean parameter estimate
        - 'theta': dispersion parameter estimate
        - 'iterations': number of EM iterations
        - 'converged': boolean indicating convergence
        - 'log_likelihood': final log-likelihood
        - 'n_observations': total number of data points used
        - 'zero_fraction': fraction of zeros in the data
    """
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "Data" / "combined_strains"
    
    # Get all strain folders
    strain_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    
    print(f"Estimating ZINB parameters for whole dataset")
    print(f"Found {len(strain_folders)} strain folders\n")
    
    # Dictionary to store results for each strain
    results = {}
    
    # Process each strain
    for strain_folder in strain_folders:
        strain_name = strain_folder.name
        print(f"Processing {strain_name}...")
        
        # Get all CSV files in this strain folder
        csv_files = sorted(strain_folder.glob("*.csv"))
        
        # Collect all data from all chromosomes
        all_data = []
        
        for csv_file in csv_files:
            chromosome = csv_file.stem.replace("_distances", "")
            
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Extract Value column and round to integers
                values = df['Value'].values
                rounded_values = np.round(values).astype(int)
                
                # Append to combined dataset
                all_data.append(rounded_values)
                
                print(f"  Added {chromosome}: {len(rounded_values)} data points")
                
            except Exception as e:
                print(f"  ERROR reading {csv_file.name}: {e}")
                continue
        
        # Combine all chromosome data
        if len(all_data) == 0:
            print(f"  WARNING: No data found for {strain_name}, skipping.\n")
            continue
        
        combined_data = np.concatenate(all_data)
        # Remove 95 percentile to filter out extreme outliers
        threshold = np.percentile(combined_data, 95)
        combined_data = combined_data[combined_data <= threshold]
        n_obs = len(combined_data)
        zero_fraction = np.sum(combined_data == 0) / n_obs
        
        print(f"  Total observations: {n_obs}")
        print(f"  Zero fraction: {zero_fraction:.4f}")
        
        # Estimate ZINB parameters for the whole dataset
        print(f"  Estimating ZINB parameters...")
        try:
            estimates = estimate_zinb(combined_data, max_iter=1000)
            
            # Store results with additional metadata
            results[strain_name] = {
                'pi': estimates['pi'],
                'mu': estimates['mu'],
                'theta': estimates['theta'],
                'iterations': estimates['iterations'],
                'converged': estimates['converged'],
                'log_likelihood': estimates['log_likelihood'],
                'n_observations': n_obs,
                'zero_fraction': zero_fraction
            }
            
            print(f"  SUCCESS: pi={estimates['pi']:.4f}, mu={estimates['mu']:.4f}, "
                  f"theta={estimates['theta']:.4f}\n")
            
        except Exception as e:
            print(f"  ERROR during estimation: {e}\n")
            results[strain_name] = {
                'pi': None,
                'mu': None,
                'theta': None,
                'iterations': None,
                'converged': False,
                'log_likelihood': None,
                'n_observations': n_obs,
                'zero_fraction': zero_fraction,
                'error': str(e)
            }
    
    print("="*60)
    print("Estimation complete!")
    print("="*60)
    
    return results


def process_all_strains(window_size=2000):
    """
    Process all CSV files in Data/combined_strains/, split into windows,
    estimate ZINB parameters for each window, and save results with window locations.
    
    Parameters:
    -----------
    window_size : int
        Size of windows to split each chromosome into (default: 2000)
    """
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "Data" / "combined_strains"
    output_dir = base_dir / "Signal_processing" / "results" / "ZINB_estimates" / f"window{window_size}"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results list
    results = []
    
    # Get all strain folders
    strain_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    
    print(f"Found {len(strain_folders)} strain folders")
    print(f"Window size: {window_size}")
    
    # Process each strain
    for strain_folder in strain_folders:
        strain_name = strain_folder.name
        print(f"\nProcessing {strain_name}...")
        
        # Get all CSV files in this strain folder
        csv_files = sorted(strain_folder.glob("*.csv"))
        
        total_windows = 0
        
        for csv_file in csv_files:
            chromosome = csv_file.stem.replace("_distances", "")
            print(f"  Processing {chromosome}...")
            
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Extract Position and Value columns
                positions = df['Position'].values
                values = df['Value'].values
                rounded_values = np.round(values).astype(int)
                
                # Split into windows
                n_positions = len(rounded_values)
                n_windows = int(np.ceil(n_positions / window_size))
                
                for i in range(n_windows):
                    start_idx = i * window_size
                    end_idx = min((i + 1) * window_size, n_positions)
                    window_data = rounded_values[start_idx:end_idx]
                    window_positions = positions[start_idx:end_idx]
                    
                    # Get actual start and end positions
                    start_pos = int(window_positions[0])
                    end_pos = int(window_positions[-1])
                    
                    # Check if more than 95% are zeros
                    # zero_fraction = np.sum(window_data == 0) / len(window_data)
                    # if zero_fraction > 0.95:
                    #     print(f"    Window {i+1}: pos {start_pos}-{end_pos} | "
                    #           f"SKIPPED (>95% zeros: {zero_fraction:.2%})")
                    #     continue
                    
                    # Filter out the 95 percentile of the data to remove extreme outliers
                    threshold = np.percentile(window_data, 95)
                    window_data = window_data[window_data <= threshold]
                
                    
                    # Estimate ZINB parameters for this window
                    try:
                        estimates = estimate_zinb(window_data, max_iter=1000)
                        
                        # Store results
                        result = {
                            'strain': strain_name,
                            'chromosome': chromosome,
                            'window_id': i + 1,
                            'start_position': start_pos,
                            'end_position': end_pos,
                            'pi': estimates['pi'],
                            'mu': estimates['mu'],
                            'theta': estimates['theta'],
                            'log_likelihood': estimates['log_likelihood'],
                            'iterations': estimates['iterations'],
                            'converged': estimates['converged'],
                            'n_observations': len(window_data),
                        }
                        results.append(result)
                        total_windows += 1
                        
                        # Print window location and results
                        print(f"    Window {i+1}: pos {start_pos}-{end_pos} | "
                              f"pi={estimates['pi']:.4f}, mu={estimates['mu']:.4f}, "
                              f"theta={estimates['theta']:.4f}, converged={estimates['converged']}")
                        
                    except Exception as e:
                        print(f"    ERROR estimating window {i+1}: {str(e)}")
                        continue
                
                print(f"    Processed {n_windows} windows")
                
            except Exception as e:
                print(f"    ERROR reading {chromosome}: {str(e)}")
                continue
            finally:
                # Clean up memory after each chromosome
                del df, positions, values, rounded_values
                gc.collect()
        
        print(f"  Total windows processed for {strain_name}: {total_windows}")
        gc.collect()  # Clean up after each strain
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = output_dir / f"zinb_estimates_windows_size{window_size}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total windows processed: {len(results_df)}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(results_df[['pi', 'mu', 'theta']].describe())
    
    return results_df


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
    df = pd.read_csv(csv_file)
    
    # Set output directory - create plots subfolder
    if output_dir is None:
        output_dir = Path(csv_file).parent / 'plots'
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


if __name__ == "__main__":
    window_size = 100
    results = process_all_strains(window_size=window_size)
    
    # Generate plots
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "Signal_processing" / "results" / "ZINB_estimates" / f"window{window_size}"
    csv_file = output_dir / "zinb_estimates_windows_size2000.csv"
    
    if csv_file.exists():
        plot_zinb_results(csv_file, output_dir)
    # estimate_whole_dataset()
