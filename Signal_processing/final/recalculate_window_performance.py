"""
Recalculate window performance metrics with corrected one-to-one matching.

This script reads already-detected change points from the results folders
and recalculates precision/recall using a one-to-one matching algorithm
to avoid bias from multiple change points within a window.
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS
from sklearn.metrics import auc

# Set up standardized plot style
setup_plot_style()


def precision_recall_one_to_one(detected_cps, true_cps, tol):
    """
    Calculate precision and recall with one-to-one matching.
    
    Uses greedy nearest-neighbor matching: each detected change point
    is matched to its nearest true change point (if within tolerance),
    and each true/detected change point can only be matched once.
    
    Args:
        detected_cps: List of detected change point positions
        true_cps: List of true change point positions
        tol: Tolerance (maximum distance for a match)
    
    Returns:
        (precision, recall) tuple
    """
    detected_cps = np.array(detected_cps)
    true_cps = np.array(true_cps)
    
    if len(detected_cps) == 0:
        return 0.0, 0.0
    if len(true_cps) == 0:
        return 0.0, 0.0
    
    # Track which change points have been matched
    matched_true = set()
    matched_detected = set()
    
    # Create pairs of (detected_idx, true_idx, distance) for all pairs within tolerance
    pairs = []
    for i, det_cp in enumerate(detected_cps):
        for j, true_cp in enumerate(true_cps):
            dist = abs(det_cp - true_cp)
            if dist <= tol:
                pairs.append((i, j, dist))
    
    # Sort by distance (greedy nearest-neighbor matching)
    pairs.sort(key=lambda x: x[2])
    
    # Match greedily: closest pairs first
    true_positives = 0
    for det_idx, true_idx, dist in pairs:
        if det_idx not in matched_detected and true_idx not in matched_true:
            matched_detected.add(det_idx)
            matched_true.add(true_idx)
            true_positives += 1
    
    # Calculate precision and recall
    precision = true_positives / len(detected_cps) if len(detected_cps) > 0 else 0.0
    recall = true_positives / len(true_cps) if len(true_cps) > 0 else 0.0
    
    return precision, recall


def read_detected_change_points(result_file):
    """
    Read detected change points from a result file.
    
    Args:
        result_file: Path to .txt file with detected change points
    
    Returns:
        List of change point positions
    """
    change_points = []
    
    with open(result_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip lines that contain metadata
            if line.startswith('scores:') or line.startswith('theta_global:') or \
               line.startswith('window_size:') or not line:
                continue
            # Read change point position
            try:
                cp = int(float(line))
                change_points.append(cp)
            except ValueError:
                continue
    
    return change_points


def read_true_change_points(param_file):
    """Read true change points from parameter file."""
    df = pd.read_csv(param_file)
    # Change points are at region_start positions (excluding the first one which is 0)
    change_points = df['region_start'].values[1:]
    return change_points.tolist()


def extract_params_from_filename(filename):
    """
    Extract window size, overlap, and threshold from filename.
    
    Expected format: dataset_X_wsYYY_ovZZ_thW.WW.txt
    """
    pattern = r'dataset_(\d+)_ws(\d+)_ov(\d+)_th([\d.]+)\.txt'
    match = re.match(pattern, filename)
    if match:
        dataset_id = int(match.group(1))
        window_size = int(match.group(2))
        overlap = int(match.group(3)) / 100.0
        threshold = float(match.group(4))
        return dataset_id, window_size, overlap, threshold
    return None, None, None, None


def process_dataset_results(dataset_id, base_results_folder, param_file):
    """
    Process all results for a single dataset.
    
    Args:
        dataset_id: ID of the dataset (1-10)
        base_results_folder: Base folder containing window results
        param_file: Path to true change points parameter file
    
    Returns:
        DataFrame with recalculated metrics
    """
    dataset_folder = os.path.join(base_results_folder, f"dataset_{dataset_id}")
    
    if not os.path.exists(dataset_folder):
        print(f"Warning: Dataset folder not found: {dataset_folder}")
        return pd.DataFrame()
    
    # Read true change points
    true_cps = read_true_change_points(param_file)
    
    results = []
    
    # Process each window size folder
    window_folders = [f for f in os.listdir(dataset_folder) if f.startswith('window')]
    
    for window_folder in window_folders:
        window_path = os.path.join(dataset_folder, window_folder)
        
        # Extract window size from folder name
        ws = int(window_folder.replace('window', ''))
        
        # Process each result file in the window folder
        if os.path.isdir(window_path):
            for result_file in os.listdir(window_path):
                if result_file.endswith('.txt'):
                    result_path = os.path.join(window_path, result_file)
                    
                    # Extract parameters from filename
                    _, _, overlap, threshold = extract_params_from_filename(result_file)
                    
                    if threshold is None:
                        continue
                    
                    # Read detected change points
                    detected_cps = read_detected_change_points(result_path)
                    
                    # Calculate precision and recall with one-to-one matching
                    # Only use full window tolerance
                    tol = ws
                    prec, rec = precision_recall_one_to_one(detected_cps, true_cps, tol)
                    
                    results.append({
                        'dataset_id': dataset_id,
                        'window_size': ws,
                        'threshold': threshold,
                        'tolerance_type': 'full_window',
                        'tolerance_value': tol,
                        'precision': prec,
                        'recall': rec,
                        'num_detected': len(detected_cps),
                        'num_true': len(true_cps),
                    })
    
    return pd.DataFrame(results)


def aggregate_results(all_results_df, output_folder):
    """
    Aggregate results across all datasets.
    
    Args:
        all_results_df: DataFrame containing results from all datasets
        output_folder: Where to save aggregated results
    
    Returns:
        DataFrame with mean and std of precision/recall for each window/threshold
    """
    # Group by window_size and threshold
    grouped = all_results_df.groupby(['window_size', 'threshold'])
    
    # Calculate mean and std for precision and recall
    agg_results = grouped.agg({
        'precision': ['mean', 'std', 'count'],
        'recall': ['mean', 'std', 'count'],
        'tolerance_value': 'first',
        'num_detected': 'mean',
        'num_true': 'mean'
    }).reset_index()
    
    # Flatten column names
    agg_results.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in agg_results.columns.values]
    
    # Save aggregated results
    agg_output_csv = os.path.join(output_folder, "aggregated_performance_metrics_corrected.csv")
    agg_results.to_csv(agg_output_csv, index=False)
    print(f"Aggregated results saved to {agg_output_csv}")
    
    return agg_results


def plot_precision_recall(agg_results_df, output_folder, dataset_name="SATAY_synthetic"):
    """
    Plot precision-recall curves for each window size (without error bars and AUC in legend).
    
    Args:
        agg_results_df: DataFrame with aggregated results (mean and std)
        output_folder: Where to save plots
        dataset_name: Name of dataset for plot titles
    """
    # Create output folder for plots
    plots_folder = os.path.join(output_folder, "corrected_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # Get unique window sizes
    window_sizes = sorted(agg_results_df['window_size'].unique())
    
    print(f"\nCreating precision-recall curve (corrected metrics)")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for ws in window_sizes:
        # Filter data for this window size
        window_data = agg_results_df[agg_results_df['window_size'] == ws].sort_values('threshold')
        
        if len(window_data) == 0:
            continue
        
        recalls = window_data['recall_mean'].values
        precisions = window_data['precision_mean'].values
        
        # Plot without error bars
        ax.plot(recalls, precisions, 
               marker='o', markersize=4, linewidth=2,
               label=f'Window {ws}',  # No AUC in legend
               alpha=0.8)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - Averaged over 10 Datasets\n(Full Window Tolerance, Corrected One-to-One Matching)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    output_path = os.path.join(plots_folder, 'precision_recall_corrected.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main execution function."""
    # Configuration
    base_folder = "Signal_processing/final/SATAY_synthetic"
    results_folder = "Signal_processing/final/results/windows"
    output_folder = "Signal_processing/final/results/windows"
    
    dataset_ids = range(1, 11)  # Datasets 1 through 10
    
    print(f"{'='*60}")
    print("Recalculating Window Performance with Corrected Metrics")
    print(f"{'='*60}")
    print(f"  Results folder: {results_folder}")
    print(f"  Datasets: {list(dataset_ids)}")
    print(f"  Using one-to-one matching for precision and recall")
    print(f"  Using only full_window tolerance")
    
    # Process all datasets
    all_results = []
    
    for dataset_id in dataset_ids:
        print(f"\nProcessing Dataset {dataset_id}...")
        
        param_file = os.path.join(base_folder, str(dataset_id), "SATAY_without_pi_params.csv")
        
        if not os.path.exists(param_file):
            print(f"  Warning: Parameter file not found: {param_file}")
            continue
        
        results_df = process_dataset_results(dataset_id, results_folder, param_file)
        
        if not results_df.empty:
            all_results.append(results_df)
            print(f"  Processed {len(results_df)} result files")
        else:
            print(f"  No results found for dataset {dataset_id}")
    
    if not all_results:
        print("\nNo datasets were successfully processed!")
        return
    
    # Combine all results
    print(f"\n{'='*60}")
    print("Aggregating Results")
    print(f"{'='*60}")
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_output_csv = os.path.join(output_folder, "all_datasets_performance_corrected.csv")
    all_results_df.to_csv(combined_output_csv, index=False)
    print(f"Combined results saved to {combined_output_csv}")
    
    # Aggregate results (mean and std across datasets)
    agg_results_df = aggregate_results(all_results_df, output_folder)
    
    # Create plots
    print(f"\n{'='*60}")
    print("Creating Plots")
    print(f"{'='*60}")
    
    plot_precision_recall(agg_results_df, output_folder)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"\nNumber of datasets processed: {len(all_results)}")
    print(f"Total result entries: {len(all_results_df)}")
    print(f"\nWindow sizes: {sorted(all_results_df['window_size'].unique())}")
    print(f"Thresholds range: {all_results_df['threshold'].min():.1f} to {all_results_df['threshold'].max():.1f}")
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"All results saved to: {output_folder}")


if __name__ == "__main__":
    main()
