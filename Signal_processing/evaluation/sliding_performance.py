import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import auc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Signal_processing.evaluation.evaluation import (precision, recall, F1_score, annotation_error, 
                       hausdorff_distance, rand_index, adjusted_rand_index,
                       precision_recall_curve, plot_precision_recall_curves,
                       roc_curve_from_cps_by_threshold, mean_absolute_error,
                       match_cps_one_to_one)

# Set up standardized plot style
setup_plot_style()
    

# Precision/recall curve or ROC curve
# Localization error (absolute error, median error), how sharp are the detected change points
# Overlay plots of detected change points on the signal


def read_change_points(file_path):
    """Read change points from a result file.
    
    Works with both sliding_mean_CPD (stops at 'means') and 
    sliding_NB_CPD (stops at 'scores:') output formats.
    """
    change_points = []
    with open(file_path, 'r') as f:
        for line in f:
            # Stop at metadata lines
            if "means" in line or "scores:" in line:
                break
            change_points.append(int(line.strip()))
    return change_points

def read_true_params(file_path, other_file=False):
    """Read true change point parameters from a CSV file with region parameters.
    
    Args:
        file_path: Path to the CSV file
        other_file: If True, reads SATAY_synthetic format (region_start, region_end, region_mean)
                   If False, reads standard format (region_lengths, region_means, etc.)
    
    Returns:
        List of change point positions
    """
    
    # Read the CSV file with region parameters
    df = pd.read_csv(file_path)
    
    if other_file:
        # SATAY_synthetic format: has region_start, region_end columns
        if 'region_start' not in df.columns:
            raise ValueError(f"Column 'region_start' not found in {file_path}. Available columns: {df.columns.tolist()}")
        
        # Change points are at region_start positions (excluding the first one which is 0)
        change_points = df['region_start'].values[1:]  # Skip first start position (0)
        return change_points.tolist()
    else:
        # Standard format: has region_lengths column
        if 'region_lengths' not in df.columns:
            raise ValueError(f"Column 'region_lengths' not found in {file_path}. Available columns: {df.columns.tolist()}")
        
        # Get region lengths and round them to integers
        region_lengths = df['region_lengths'].values
        region_lengths = np.rint(region_lengths).astype(int)
        
        # Calculate cumulative positions (change points are at region boundaries)
        cumsum_lengths = np.cumsum(region_lengths)
        
        # Change points are at the end of each region (subtract 1 for 0-based indexing)
        change_points = cumsum_lengths - 1
        
        # Return as a list (excluding the last point which is the end of the data)
        return change_points[:-1].tolist()

def read_data(data_file):
    """Read the signal data from CSV file."""
    with open(data_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = np.array([int(line.strip().split(",")[1]) for line in lines])
    return data

def plot_change_points_overlay(data, detected_cps, true_cps, start_pos, end_pos, 
                                window_size, threshold, output_path):
    """Plot a section of data with detected and true change points overlaid."""
    plt.figure(figsize=(15, 6))
    
    # Plot the data
    positions = np.arange(start_pos, end_pos)
    data_section = data[start_pos:end_pos]
    plt.plot(positions, data_section, 'b-', linewidth=0.8, label='Data', alpha=0.7)
    
    # Plot true change points in the range
    true_in_range = [cp for cp in true_cps if start_pos <= cp <= end_pos]
    for cp in true_in_range:
        plt.axvline(x=cp, color=COLORS['green'], linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot detected change points in the range
    detected_in_range = [cp for cp in detected_cps if start_pos <= cp <= end_pos]
    for cp in detected_in_range:
        plt.axvline(x=cp, color=COLORS['red'], linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['blue'], linewidth=1, label='Signal'),
        Line2D([0], [0], color=COLORS['green'], linestyle='--', linewidth=2, label='True change points'),
        Line2D([0], [0], color=COLORS['red'], linestyle='-', linewidth=1.5, label='Detected change points')
    ]
    
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.title(f'Change Point Detection (Window={window_size}, Threshold={threshold:.2f})')
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_overlay_plots(dataset_name='pretty_data', num_plots=5, section_length=5000, 
                         base_results_folder=None, other_file=False):
    """Create overlay plots showing detected vs true change points on random sections of data."""
    
    # Setup paths
    data_file = f"Signal_processing/sample_data/{dataset_name}.csv"
    param_file = f"Signal_processing/sample_data/{dataset_name}_params.csv"
    
    # If base_results_folder not provided, try dataset-specific folder first, then fall back to general folder
    if base_results_folder is None:
        dataset_specific_folder = f"Signal_processing/results/sliding_mean_CPD/{dataset_name}"
        general_folder = "Signal_processing/results/sliding_mean_CPD"
        
        if os.path.exists(dataset_specific_folder):
            base_results_folder = dataset_specific_folder
        elif os.path.exists(general_folder):
            base_results_folder = general_folder
        else:
            print(f"Error: Could not find results folder for {dataset_name}")
            print(f"Tried: {dataset_specific_folder} and {general_folder}")
            return
    
    output_plots_folder = f"Signal_processing/results/overlay_plots/{dataset_name}"
    
    os.makedirs(output_plots_folder, exist_ok=True)
    
    # Read data and true change points
    data = read_data(data_file)
    true_cps = read_true_params(param_file, other_file=other_file)
    data_length = len(data)
    
    print(f"Data length: {data_length}")
    print(f"Number of true change points: {len(true_cps)}")
    
    # Get all window folders
    window_folders = [f for f in os.listdir(base_results_folder) 
                     if os.path.isdir(os.path.join(base_results_folder, f)) and f.startswith("window")]
    window_folders.sort()
    
    # For each window, select a few representative thresholds
    for window_folder in window_folders:
        window_size = int(window_folder.replace("window", ""))
        window_path = os.path.join(base_results_folder, window_folder)
        
        # Get all result files and select a few thresholds
        result_files = sorted([f for f in os.listdir(window_path) if f.endswith('.txt')])
        
        if len(result_files) == 0:
            continue
        
        # Select low, medium, high thresholds
        indices = [0, len(result_files)//2, -1]
        selected_files = [result_files[i] for i in indices if i < len(result_files)]
        
        for result_file in selected_files:
            # Extract threshold
            match = re.search(r'th(\d+\.\d+)', result_file)
            if not match:
                continue
            threshold = float(match.group(1))
            
            # Read detected change points
            file_path = os.path.join(window_path, result_file)
            detected_cps = read_change_points(file_path)
            
            print(f"Window {window_size}, Threshold {threshold:.2f}: {len(detected_cps)} detected CPs")
            
            # Generate random sections for overlay plots
            for plot_idx in range(num_plots):
                # Select a random starting position
                max_start = max(0, data_length - section_length)
                start_pos = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                end_pos = min(start_pos + section_length, data_length)
                
                # Create output filename
                output_filename = f"overlay_ws{window_size}_th{threshold:.2f}_section{plot_idx+1}.png"
                output_path = os.path.join(output_plots_folder, output_filename)
                
                # Create the plot
                plot_change_points_overlay(data, detected_cps, true_cps, start_pos, end_pos,
                                          window_size, threshold, output_path)
            
            print(f"  Created {num_plots} overlay plots for window={window_size}, threshold={threshold:.2f}")
    
    print(f"\nOverlay plots saved to {output_plots_folder}")


def plot_metric_vs_threshold(results_df, metric_name, output_folder, dataset_name, 
                            metric_label=None, use_log_scale=False, exclude_inf=True):
    """Plot a tolerance-independent metric vs threshold for each window size.
    
    Args:
        results_df: DataFrame with results
        metric_name: Name of the metric column to plot
        output_folder: Where to save the plot
        dataset_name: Name of the dataset
        metric_label: Label for y-axis (defaults to metric_name)
        use_log_scale: Whether to use log scale for y-axis
        exclude_inf: Whether to exclude infinite values from the plot
    """
    if metric_label is None:
        metric_label = metric_name.replace('_', ' ').title()
    
    # Get unique values (one per window/threshold combo)
    unique_results = results_df.drop_duplicates(subset=['window_size', 'threshold'])
    window_sizes = sorted(unique_results['window_size'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for window_size in window_sizes:
        window_data = unique_results[unique_results['window_size'] == window_size].sort_values('threshold')
        
        thresholds = window_data['threshold'].values
        metric_values = window_data[metric_name].values
        
        # Filter out inf values if requested
        if exclude_inf:
            mask = np.isfinite(metric_values)
            thresholds = thresholds[mask]
            metric_values = metric_values[mask]
        
        if len(thresholds) > 0:
            ax.plot(thresholds, metric_values, marker='o', label=f'Window {window_size}', 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} vs Threshold ({dataset_name})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if use_log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{metric_name}_vs_threshold.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_all_metrics_comparison(results_df, output_folder, dataset_name):
    """Create a multi-panel figure comparing all tolerance-independent metrics.
    
    Args:
        results_df: DataFrame with results
        output_folder: Where to save the plot
        dataset_name: Name of the dataset
    """
    unique_results = results_df.drop_duplicates(subset=['window_size', 'threshold'])
    window_sizes = sorted(unique_results['window_size'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    metrics = [
        ('annotation_error', 'Annotation Error', False, False),
        ('hausdorff_distance', 'Hausdorff Distance', True, True),  # log scale, exclude inf
        ('mae_localization', 'MAE Localization Error', True, True),  # log scale, exclude inf
        ('rand_index', 'Rand Index', False, False),
        ('adjusted_rand_index', 'Adjusted Rand Index', False, False)
    ]
    
    for ax, (metric_name, metric_label, use_log, exclude_inf) in zip(axes.flat, metrics):
        for window_size in window_sizes:
            window_data = unique_results[unique_results['window_size'] == window_size].sort_values('threshold')
            
            thresholds = window_data['threshold'].values
            metric_values = window_data[metric_name].values
            
            # Filter out inf values if requested
            if exclude_inf:
                mask = np.isfinite(metric_values)
                thresholds = thresholds[mask]
                metric_values = metric_values[mask]
            
            if len(thresholds) > 0:
                ax.plot(thresholds, metric_values, marker='o', label=f'Window {window_size}', 
                       linewidth=2, markersize=4)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if use_log:
            ax.set_yscale('log')
    
    # Hide the unused subplot
    axes[-1].axis('off')
    
    plt.suptitle(f'Tolerance-Independent Metrics vs Threshold ({dataset_name})', fontsize=16, y=0.995)
    plt.tight_layout()
    output_path = os.path.join(output_folder, 'all_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def create_all_datasets_metric_plots(combined_df, output_folder):
    """Create aggregate metric plots across all datasets.

    Aggregates tolerance-independent metrics by taking the mean across datasets
    for each (window_size, threshold) pair.
    """
    plots_folder = os.path.join(output_folder, "performance_plots", "all_datasets")
    os.makedirs(plots_folder, exist_ok=True)

    # Keep one row per dataset/window/threshold because these metrics do not depend on tolerance.
    unique_rows = combined_df.drop_duplicates(subset=['dataset_id', 'window_size', 'threshold'])

    def mean_without_inf(series):
        cleaned = series.replace([np.inf, -np.inf], np.nan)
        return cleaned.mean()

    aggregated = (
        unique_rows
        .groupby(['window_size', 'threshold'], as_index=False)
        .agg({
            'annotation_error': 'mean',
            'hausdorff_distance': mean_without_inf,
            'mae_localization': mean_without_inf,
            'rand_index': 'mean',
            'adjusted_rand_index': 'mean',
        })
    )

    if aggregated.empty:
        print("No combined rows available for all-datasets metric plots.")
        return

    dataset_name = "all_datasets_mean"
    plot_all_metrics_comparison(aggregated, plots_folder, dataset_name)
    plot_metric_vs_threshold(
        aggregated,
        'annotation_error',
        plots_folder,
        dataset_name,
        metric_label='Annotation Error (|#predicted - #true|)',
        use_log_scale=False,
    )
    plot_metric_vs_threshold(
        aggregated,
        'hausdorff_distance',
        plots_folder,
        dataset_name,
        metric_label='Hausdorff Distance',
        use_log_scale=True,
        exclude_inf=True,
    )
    plot_metric_vs_threshold(
        aggregated,
        'mae_localization',
        plots_folder,
        dataset_name,
        metric_label='MAE Localization Error',
        use_log_scale=True,
        exclude_inf=True,
    )
    plot_metric_vs_threshold(
        aggregated,
        'rand_index',
        plots_folder,
        dataset_name,
        metric_label='Rand Index',
        use_log_scale=False,
    )
    plot_metric_vs_threshold(
        aggregated,
        'adjusted_rand_index',
        plots_folder,
        dataset_name,
        metric_label='Adjusted Rand Index',
        use_log_scale=False,
    )
    print(f"All-datasets metric plots saved to {plots_folder}")


def create_all_datasets_precision_recall_curves(combined_df, output_folder):
    """Create combined precision-recall curves across all datasets.

    For each tolerance and window size, precision/recall are averaged across
    datasets at each threshold. Cases with no detected change points keep
    precision as NaN and recall as 0 in the aggregated table.
    """
    plots_folder = os.path.join(output_folder, "performance_plots", "all_datasets")
    os.makedirs(plots_folder, exist_ok=True)

    tolerance_types = ["full window", "half window", "quarter window"]
    aggregate_rows = []

    for tol_name in tolerance_types:
        tol_df = combined_df[combined_df['tolerance_type'] == tol_name].copy()
        if tol_df.empty:
            continue

        pr_curves_data = []

        for window_size in sorted(tol_df['window_size'].unique()):
            window_df = tol_df[tol_df['window_size'] == window_size].copy()

            aggregated = (
                window_df
                .groupby('threshold', as_index=False)
                .agg({
                    'precision': 'mean',  # keeps NaN if all datasets are NaN
                    'recall': 'mean',
                    'dataset_id': 'nunique',
                })
                .rename(columns={'dataset_id': 'num_datasets'})
                .sort_values('threshold')
            )

            for _, row in aggregated.iterrows():
                aggregate_rows.append({
                    'tolerance_type': tol_name,
                    'window_size': window_size,
                    'threshold': row['threshold'],
                    'mean_precision': row['precision'],
                    'mean_recall': row['recall'],
                    'num_datasets': int(row['num_datasets']),
                })

            # PR plotting and AUC require finite precision and recall values.
            valid_mask = np.isfinite(aggregated['precision'].values) & np.isfinite(aggregated['recall'].values)
            if np.sum(valid_mask) == 0:
                continue

            recalls = aggregated.loc[valid_mask, 'recall'].values
            precisions = aggregated.loc[valid_mask, 'precision'].values
            thresholds = aggregated.loc[valid_mask, 'threshold'].values

            pr_data = precision_recall_curve(
                detected_cps_list=None,
                precisions=precisions,
                recalls=recalls,
                thresholds=thresholds,
                label=f'Window {window_size}'
            )

            recalls_curve = np.asarray(pr_data['recalls'])
            precisions_curve = np.asarray(pr_data['precisions'])
            finite_curve = np.isfinite(recalls_curve) & np.isfinite(precisions_curve)
            recalls_curve = recalls_curve[finite_curve]
            precisions_curve = precisions_curve[finite_curve]

            if len(recalls_curve) == 0:
                continue

            pr_curves_data.append({
                'recalls': recalls_curve,
                'precisions': precisions_curve,
                'label': f'Window {window_size}',
                'color': 'black',
                'marker': 'o',
                'markersize': 4,
                'linewidth': 2,
            })

        if pr_curves_data:
            tol_slug = tol_name.replace(' ', '_')
            title = f'Combined Precision-Recall Curve (Tolerance: {tol_name})'

            plot_path = os.path.join(plots_folder, f'precision_recall_combined_{tol_slug}.png')
            plot_precision_recall_curves(
                pr_curves_data,
                output_path=plot_path,
                title=title,
                figsize=(10, 8),
            )
            print(f"Saved combined PR curve: {plot_path}")

    if aggregate_rows:
        aggregate_df = pd.DataFrame(aggregate_rows)
        aggregate_csv = os.path.join(output_folder, "precision_recall_aggregated_all_datasets.csv")
        aggregate_df.to_csv(aggregate_csv, index=False)
        print(f"Saved combined PR table: {aggregate_csv}")


def plot_roc_curves(roc_curves_data, output_folder, tol_name, dataset_name):
    """Plot ROC curves for different window sizes.
    
    Args:
        roc_curves_data: List of dictionaries with 'fprs', 'tprs', and 'label' keys
        output_folder: Where to save the plot
        tol_name: Name of the tolerance type
        dataset_name: Name of the dataset
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    auc_values = []
    for curve_data in roc_curves_data:
        fprs = curve_data['fprs']
        tprs = curve_data['tprs']
        label = curve_data.get('label', None)
        marker = curve_data.get('marker', 'o')
        markersize = curve_data.get('markersize', 4)
        linewidth = curve_data.get('linewidth', 2)
        
        # Calculate AUC using sklearn
        sort_idx = np.argsort(fprs)
        roc_auc = auc(fprs[sort_idx], tprs[sort_idx])
        auc_values.append(roc_auc)
        
        # Add AUC to label if label exists
        if label is not None:
            label = f"{label} (AUC={roc_auc:.3f})"
        
        ax.plot(fprs, tprs, marker=marker, markersize=markersize, 
               linewidth=linewidth, label=label)
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Random')
    
    # Create title with average AUC if we have data
    if auc_values:
        avg_auc = np.mean(auc_values)
        title = f'ROC Curve (Tolerance: {tol_name}, Avg AUC={avg_auc:.3f})'
    else:
        title = f'ROC Curve (Tolerance: {tol_name})'
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'roc_curve_{tol_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve: {output_path}")


def evaluate_all_windows_and_thresholds(data_file, input_folder, output_folder, dataset_name, other_file=False):
    """Evaluate precision, recall, and F1 for all window sizes and thresholds.
    
    Args:
        data_file: Path to the input data file (e.g., 'Signal_processing/sample_data/pretty_data.csv')
        input_folder: Path to the folder containing results
        output_folder: Path to the output folder for performance metrics and plots
        dataset_name: Name of the dataset
        other_file: If True, reads SATAY_synthetic params format
    """

    # Setup paths
    data_file = data_file
    true_param_file = data_file.replace('.csv', '_params.csv')
    base_results_folder = input_folder
    output_csv = f"{output_folder}/performance_metrics_{dataset_name}.csv"
    output_plots_folder = f"{output_folder}/performance_plots/{dataset_name}"
    
    # Create output folder for plots
    os.makedirs(output_plots_folder, exist_ok=True)
    
    # Read data and true change points
    data = read_data(data_file)
    n_points = len(data)
    true_cps = read_true_params(true_param_file, other_file=other_file)
    print(f"Total data points: {n_points}")
    print(f"Total true change points: {len(true_cps)}")
    
    # Collect all results
    results = []
    
    # Store detected CPs by window/threshold for ROC curve calculation
    window_threshold_cps = {}  # {window_size: [(threshold, detected_cps), ...]}
    
    # Get all window folders
    window_folders = [f for f in os.listdir(base_results_folder) 
                     if os.path.isdir(os.path.join(base_results_folder, f)) and f.startswith("window")]
    window_folders.sort()
    
    print(f"Found window folders: {window_folders}")
    
    for window_folder in window_folders:
        # Extract window size from folder name
        window_size = int(window_folder.replace("window", ""))
        print(f"\nProcessing {window_folder} (window size: {window_size})...")
        
        # Define tolerances based on window size
        tolerances = {
            "full window": window_size,
            "half window": window_size // 2,
            "quarter window": window_size // 4
        }
        
        window_path = os.path.join(base_results_folder, window_folder)
        
        # Get all result files in this window folder
        result_files = [f for f in os.listdir(window_path) if f.endswith('.txt')]
        result_files.sort()
        
        print(f"  Found {len(result_files)} result files")
        
        for result_file in result_files:
            # Extract threshold from filename using regex
            match = re.search(r'th(\d+\.\d+)', result_file)
            if not match:
                continue
            threshold = float(match.group(1))
            
            # Read detected change points
            file_path = os.path.join(window_path, result_file)
            detected_cps = read_change_points(file_path)
            
            # Store for ROC curve calculation
            if window_size not in window_threshold_cps:
                window_threshold_cps[window_size] = []
            window_threshold_cps[window_size].append((threshold, detected_cps))
            
            # Calculate annotation error (independent of tolerance)
            ann_error = annotation_error(detected_cps, true_cps)
            
            # Calculate Hausdorff distance
            hausdorff = hausdorff_distance(true_cps, detected_cps)
            
            # Calculate Rand Index and Adjusted Rand Index
            rand_idx = rand_index(true_cps, detected_cps, n_points)
            adj_rand_idx = adjusted_rand_index(true_cps, detected_cps, n_points)
            
            # Calculate MAE localization error (mean distance of matched CPs)
            # Use window_size as tolerance for matching
            matches, _, _ = match_cps_one_to_one(true_cps, detected_cps, window_size)
            if matches:
                mae_loc = np.mean([abs(pred - true) for pred, true in matches])
            else:
                mae_loc = np.inf  # No matches found
            
            # Calculate metrics for each tolerance
            for tol_name, tol_value in tolerances.items():
                # Handle case with no detected change points: precision is undefined, recall is 0
                if len(detected_cps) == 0:
                    prec = np.nan  # undefined when no predictions
                    rec = 0.0      # 0 out of N true points found
                    f1 = np.nan
                else:
                    prec = precision(detected_cps, true_cps, tol_value)
                    rec = recall(detected_cps, true_cps, tol_value)
                    f1 = F1_score(prec, rec)
                
                results.append({
                    'window_size': window_size,
                    'threshold': threshold,
                    'tolerance_type': tol_name,
                    'tolerance_value': tol_value,
                    'precision': prec,
                    'mae_localization': mae_loc,
                    'recall': rec,
                    'F1': f1,
                    'num_detected': len(detected_cps),
                    'num_true': len(true_cps),
                    'annotation_error': ann_error,
                    'hausdorff_distance': hausdorff,
                    'rand_index': rand_idx,
                    'adjusted_rand_index': adj_rand_idx,
                })
        
        print(f"  Processed {len(result_files)} files for {window_folder}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Create precision-recall curves for each window and tolerance using evaluation.py functions
    for tol_name in ["full window", "half window", "quarter window"]:
        pr_curves_data = []
        
        for window_folder in window_folders:
            window_size = int(window_folder.replace("window", ""))
            
            # Filter data for this window and tolerance
            window_data = results_df[
                (results_df['window_size'] == window_size) & 
                (results_df['tolerance_type'] == tol_name)
            ].sort_values('threshold')
            
            if len(window_data) > 0:
                # Use precision_recall_curve to prepare the data
                pr_data = precision_recall_curve(
                    detected_cps_list=None,  # Not needed since we already have precision/recall
                    precisions=window_data['precision'].values,
                    recalls=window_data['recall'].values,
                    thresholds=window_data['threshold'].values,
                    label=f'Window {window_size}'
                )
                
                pr_curves_data.append({
                    'recalls': pr_data['recalls'],
                    'precisions': pr_data['precisions'],
                    'label': f'Window {window_size}',
                    'color': 'black',
                    'marker': 'o',
                    'markersize': 4,
                    'linewidth': 2
                })
        
        # No AUC text in title.
        title = f'Precision-Recall Curve (Tolerance: {tol_name})'
        
        # Use the evaluation.py function to create the plot
        plot_path = os.path.join(output_plots_folder, f'precision_recall_{tol_name}.png')
        plot_precision_recall_curves(
            pr_curves_data,
            output_path=plot_path,
            title=title,
            figsize=(10, 8)
        )
        print(f"Saved plot: {plot_path}")
    
    # Create ROC curves for each window and tolerance using evaluation.py function
    print("\nCreating ROC curves...")
    for tol_name in ["full_window", "half_window", "quarter_window"]:
        roc_curves_data = []
        
        # Get tolerance value
        tol_value = None
        for _, row in results_df[results_df['tolerance_type'] == tol_name].iloc[:1].iterrows():
            tol_value = int(row['tolerance_value'])
            break
        
        if tol_value is None:
            continue
        
        for window_size in sorted(window_threshold_cps.keys()):
            # Get the list of (threshold, detected_cps) for this window
            threshold_cps_list = window_threshold_cps[window_size]
            
            # Use the evaluation.py function to calculate ROC curve
            fpr, tpr, thresholds = roc_curve_from_cps_by_threshold(
                threshold_cps_list, true_cps, n_points, tol_value
            )
            
            if len(fpr) > 0:
                roc_curves_data.append({
                    'fprs': fpr,
                    'tprs': tpr,
                    'label': f'Window {window_size}',
                    'marker': 'o',
                    'markersize': 4,
                    'linewidth': 2
                })
        
        # Create ROC curve plot
        plot_roc_curves(
            roc_curves_data,
            output_plots_folder,
            tol_name,
            dataset_name
        )
    
    # Create plots for tolerance-independent metrics vs threshold
    print("\nCreating tolerance-independent metric plots...")
    plot_all_metrics_comparison(results_df, output_plots_folder, dataset_name)
    
    # Individual metric plots
    plot_metric_vs_threshold(results_df, 'annotation_error', output_plots_folder, dataset_name,
                            metric_label='Annotation Error (|#predicted - #true|)', use_log_scale=False)
    plot_metric_vs_threshold(results_df, 'hausdorff_distance', output_plots_folder, dataset_name,
                            metric_label='Hausdorff Distance', use_log_scale=True, exclude_inf=True)
    plot_metric_vs_threshold(results_df, 'mae_localization', output_plots_folder, dataset_name,
                            metric_label='MAE Localization Error', use_log_scale=True, exclude_inf=True)
    plot_metric_vs_threshold(results_df, 'rand_index', output_plots_folder, dataset_name,
                            metric_label='Rand Index', use_log_scale=False)
    plot_metric_vs_threshold(results_df, 'adjusted_rand_index', output_plots_folder, dataset_name,
                            metric_label='Adjusted Rand Index', use_log_scale=False)
    # Print succesfully saved to output folder
    print(f"\nAll performance plots saved to {output_plots_folder}")
    
    return results_df


if __name__ == "__main__":
    # Compare version4 results against SATAY_synthetic ground truth files.
    results_root = "Signal_processing/results/version4"
    truth_root = "Data/SATAY_synthetic"
    output_folder = "Signal_processing/results/sliding_cpd_performance/version4_vs_satay_synthetic"
    other_file = True

    os.makedirs(output_folder, exist_ok=True)

    # Only evaluate dataset IDs that exist in both folders.
    result_ids = {d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))}
    truth_ids = {d for d in os.listdir(truth_root) if os.path.isdir(os.path.join(truth_root, d))}
    dataset_ids = sorted(result_ids.intersection(truth_ids), key=lambda x: int(x))

    if not dataset_ids:
        raise ValueError(
            f"No matching dataset folders found between {results_root} and {truth_root}."
        )

    all_results = []

    for dataset_id in dataset_ids:
        dataset_name = f"SATAY_synthetic_{dataset_id}"
        input_folder = os.path.join(results_root, dataset_id)
        data_file = os.path.join(truth_root, dataset_id, "SATAY_without_pi.csv")

        if not os.path.exists(data_file):
            print(f"Skipping dataset {dataset_id}: missing data file {data_file}")
            continue

        print(f"\nAnalyzing dataset: {dataset_name}")
        print("=" * 50)

        results_df = evaluate_all_windows_and_thresholds(
            data_file,
            input_folder,
            output_folder,
            dataset_name,
            other_file=other_file,
        )
        results_df['dataset_id'] = int(dataset_id)
        all_results.append(results_df)

        print("\nSummary statistics by window size and tolerance:")
        print(results_df.groupby(['window_size', 'tolerance_type'])[['precision', 'recall', 'F1']].mean())

    if not all_results:
        raise ValueError("No datasets were evaluated. Check input folders and data files.")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_csv = os.path.join(output_folder, "performance_metrics_version4_all_datasets.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nCombined results saved to {combined_csv}")

    print("\nCombined summary by window size and tolerance:")
    print(combined_df.groupby(['window_size', 'tolerance_type'])[['precision', 'recall', 'F1']].mean())

    print("\nCombined tolerance-independent summary by window size:")
    unique_combined = combined_df.drop_duplicates(subset=['dataset_id', 'window_size', 'threshold'])
    for window_size in sorted(unique_combined['window_size'].unique()):
        window_data = unique_combined[unique_combined['window_size'] == window_size]
        print(f"\nWindow size: {window_size}")
        print(f"  Mean annotation error: {window_data['annotation_error'].mean():.2f}")
        print(f"  Max annotation error: {window_data['annotation_error'].max()}")
        print(f"  Mean Hausdorff distance: {window_data['hausdorff_distance'].replace(np.inf, np.nan).mean():.2f}")
        print(f"  Max Hausdorff distance: {window_data['hausdorff_distance'].replace(np.inf, np.nan).max()}")
        print(f"  Mean MAE localization error: {window_data['mae_localization'].replace(np.inf, np.nan).mean():.2f}")
        print(f"  Max MAE localization error: {window_data['mae_localization'].replace(np.inf, np.nan).max()}")

    # Create all-datasets aggregate performance metric plots.
    create_all_datasets_metric_plots(combined_df, output_folder)
    create_all_datasets_precision_recall_curves(combined_df, output_folder)
    