import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Signal_processing.sliding_mean.sliding_ZINB_CPD_ref import sliding_ZINB_CPD_ref
from Signal_processing.evaluation.evaluation import precision, recall
from Utils.plot_config import setup_plot_style, COLORS
from sklearn.metrics import auc

# Set up standardized plot style
setup_plot_style()


def initialize_theta_global(data, eps=1e-10, theta_max=1000):
    """Estimate global theta parameter from data."""
    results = estimate_zinb(data, eps=eps)
    theta_global = results['theta']
    print(f"  Estimated global theta: {theta_global:.4f}")
    print(f"  (Estimated global pi: {results['pi']:.4f}, mu: {results['mu']:.4f})")
    if theta_global >= theta_max:
        raise ValueError("Estimated global theta is very large, indicating a failure in estimation.")
    return theta_global


def save_results(output_folder, dataset_name, change_points, scores, theta_global, window_size, overlap, threshold):
    """Save detected change points and metadata to file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f"{dataset_name}_ws{window_size}_ov{int(overlap*100)}_th{threshold:.2f}.txt")
    with open(output_file, "w") as f:
        for cp in change_points:
            f.write(f"{cp} \n")
        f.write(f"scores: {scores}\n")
        f.write(f"theta_global: {theta_global}\n")
        f.write(f"window_size: {window_size}, overlap: {overlap}, threshold: {threshold}\n")


def read_data(data_file):
    """Read the signal data from CSV file."""
    with open(data_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = np.array([int(float(line.strip().split(",")[1])) for line in lines])
    return data


def read_true_change_points(param_file):
    """Read true change points from parameter file.
    
    Args:
        param_file: Path to CSV file with region_start, region_end, region_mean columns
    
    Returns:
        List of change point positions
    """
    df = pd.read_csv(param_file)
    # Change points are at region_start positions (excluding the first one which is 0)
    change_points = df['region_start'].values[1:]
    return change_points.tolist()


def process_window_size_for_dataset(ws, dataset_id, data, true_cps, overlap, thresholds, 
                                     theta_global, pi_file, output_folder):
    """Process a single window size for a dataset (to be run in parallel).
    
    Args:
        ws: Window size to process
        dataset_id: ID of the dataset
        data: The data array
        true_cps: True change points
        overlap: Overlap fraction (0-1)
        thresholds: Array of thresholds to test
        theta_global: Global theta parameter
        pi_file: Path to pi_values.csv file
        output_folder: Output folder for results
    
    Returns:
        List of result dictionaries for this window size
    """
    print(f"  [Dataset {dataset_id}] Processing window size: {ws}")
    window_output_folder = os.path.join(output_folder, f"dataset_{dataset_id}", f"window{ws}")
    
    results = []
    
    for threshold in thresholds:
        print(f"    Threshold: {threshold:.2f}")
        # Run ZINB_ref change point detection
        change_points, scores = sliding_ZINB_CPD_ref(
            data, ws, overlap, threshold, 
            theta_global=theta_global,
            pi_file=pi_file
        )
        
        # Save results
        save_results(window_output_folder, f"dataset_{dataset_id}", 
                    change_points, scores, theta_global, ws, overlap, threshold)
        
        # Calculate precision and recall for different tolerances
        tolerances = {
            "full_window": ws,
            "half_window": ws // 2,
            "quarter_window": ws // 4
        }
        
        for tol_name, tol_value in tolerances.items():
            prec = precision(change_points, true_cps, tol_value)
            rec = recall(change_points, true_cps, tol_value)
            
            results.append({
                'dataset_id': dataset_id,
                'window_size': ws,
                'threshold': threshold,
                'tolerance_type': tol_name,
                'tolerance_value': tol_value,
                'precision': prec,
                'recall': rec,
                'num_detected': len(change_points),
                'num_true': len(true_cps),
            })
    
    print(f"  [Dataset {dataset_id}] Completed window size: {ws}")
    return results


def process_single_dataset(dataset_id, window_sizes, overlap, thresholds, base_folder, output_folder, n_workers=1):
    """Process a single synthetic dataset with all window sizes and thresholds.
    
    Args:
        dataset_id: ID of the dataset (1-10)
        window_sizes: List of window sizes to test
        overlap: Overlap fraction (0-1)
        thresholds: Array of thresholds to test
        base_folder: Base folder containing SATAY_synthetic datasets
        output_folder: Output folder for results
        n_workers: Number of parallel workers for window sizes
    
    Returns:
        DataFrame with results for this dataset
    """
    dataset_folder = os.path.join(base_folder, str(dataset_id))
    data_file = os.path.join(dataset_folder, "SATAY_with_pi.csv")
    param_file = os.path.join(dataset_folder, "SATAY_without_pi_params.csv")
    pi_file = os.path.join(dataset_folder, "pi_values.csv")
    
    print(f"\n{'='*60}")
    print(f"Processing Dataset {dataset_id}")
    print(f"{'='*60}")
    
    # Read data and true change points
    data = read_data(data_file)
    true_cps = read_true_change_points(param_file)
    n_points = len(data)
    
    print(f"  Data points: {n_points}")
    print(f"  True change points: {len(true_cps)}")
    
    # Estimate global theta
    theta_global = initialize_theta_global(data)
    
    # Process all window sizes and thresholds (parallelized by window size)
    results = []
    
    # Determine number of workers for this dataset
    actual_workers = min(n_workers, len(window_sizes))
    
    if actual_workers > 1:
        print(f"  Using {actual_workers} parallel workers for window sizes")
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all window sizes for parallel processing
            futures = [
                executor.submit(
                    process_window_size_for_dataset,
                    ws, dataset_id, data, true_cps, overlap, thresholds,
                    theta_global, pi_file, output_folder
                )
                for ws in window_sizes
            ]
            
            # Collect results as they complete
            for future in futures:
                window_results = future.result()
                results.extend(window_results)
    else:
        # Sequential processing
        for ws in window_sizes:
            window_results = process_window_size_for_dataset(
                ws, dataset_id, data, true_cps, overlap, thresholds,
                theta_global, pi_file, output_folder
            )
            results.extend(window_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save individual dataset results
    dataset_output_csv = os.path.join(output_folder, f"dataset_{dataset_id}", "performance_metrics.csv")
    os.makedirs(os.path.dirname(dataset_output_csv), exist_ok=True)
    results_df.to_csv(dataset_output_csv, index=False)
    print(f"\nDataset {dataset_id} results saved to {dataset_output_csv}")
    
    # Calculate and save AUC for this dataset
    auc_df = calculate_individual_auc(results_df, dataset_id)
    auc_output_csv = os.path.join(output_folder, f"dataset_{dataset_id}", "auc_values.csv")
    auc_df.to_csv(auc_output_csv, index=False)
    print(f"Dataset {dataset_id} AUC values saved to {auc_output_csv}")
    
    return results_df, auc_df


def calculate_individual_auc(results_df, dataset_id):
    """Calculate AUC for precision-recall curves for a single dataset.
    
    Args:
        results_df: DataFrame with precision/recall results for this dataset
        dataset_id: ID of the dataset
    
    Returns:
        DataFrame with AUC values for each window size and tolerance type
    """
    window_sizes = sorted(results_df['window_size'].unique())
    tolerance_types = results_df['tolerance_type'].unique()
    
    auc_data = []
    
    for tol_type in tolerance_types:
        for ws in window_sizes:
            # Filter data for this window size and tolerance
            mask = (results_df['window_size'] == ws) & \
                   (results_df['tolerance_type'] == tol_type)
            window_data = results_df[mask].sort_values('threshold')
            
            if len(window_data) == 0:
                continue
            
            recalls = window_data['recall'].values
            precisions = window_data['precision'].values
            
            # Calculate AUC (sort by recall first for proper calculation)
            sort_idx = np.argsort(recalls)
            try:
                pr_auc = auc(recalls[sort_idx], precisions[sort_idx])
            except:
                pr_auc = np.nan
            
            auc_data.append({
                'dataset_id': dataset_id,
                'window_size': ws,
                'tolerance_type': tol_type,
                'tolerance_value': window_data['tolerance_value'].iloc[0],
                'auc': pr_auc
            })
    
    return pd.DataFrame(auc_data)


def aggregate_auc_results(all_auc_df, output_folder):
    """Aggregate AUC values across all datasets.
    
    Args:
        all_auc_df: DataFrame containing AUC values from all datasets
        output_folder: Where to save aggregated AUC results
    
    Returns:
        DataFrame with mean and std of AUC for each window/tolerance combination
    """
    # Group by window_size and tolerance_type
    grouped = all_auc_df.groupby(['window_size', 'tolerance_type'])
    
    # Calculate mean and std for AUC
    agg_auc = grouped.agg({
        'auc': ['mean', 'std', 'count'],
        'tolerance_value': 'first'
    }).reset_index()
    
    # Flatten column names
    agg_auc.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in agg_auc.columns.values]
    
    # Save aggregated AUC results
    agg_auc_csv = os.path.join(output_folder, "aggregated_auc_values.csv")
    agg_auc.to_csv(agg_auc_csv, index=False)
    print(f"\nAggregated AUC values saved to {agg_auc_csv}")
    
    return agg_auc


def aggregate_results(all_results_df, output_folder):
    """Aggregate results across all datasets.
    
    Args:
        all_results_df: DataFrame containing results from all datasets
        output_folder: Where to save aggregated results
    
    Returns:
        DataFrame with mean and std of precision/recall for each window/threshold/tolerance
    """
    # Group by window_size, threshold, and tolerance_type
    grouped = all_results_df.groupby(['window_size', 'threshold', 'tolerance_type'])
    
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
    agg_output_csv = os.path.join(output_folder, "aggregated_performance_metrics.csv")
    agg_results.to_csv(agg_output_csv, index=False)
    print(f"\nAggregated results saved to {agg_output_csv}")
    
    return agg_results


def plot_precision_recall_with_errorbars(agg_results_df, output_folder, dataset_name="SATAY_synthetic"):
    """Plot precision-recall curves with error bars for each window size and tolerance.
    
    Args:
        agg_results_df: DataFrame with aggregated results (mean and std)
        output_folder: Where to save plots
        dataset_name: Name of dataset for plot titles
    """
    # Create output folder for plots
    plots_folder = os.path.join(output_folder, "averaged_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # Get unique window sizes
    window_sizes = sorted(agg_results_df['window_size'].unique())
    
    # Create precision-recall curves for each tolerance type
    tolerance_types = ['full_window', 'half_window', 'quarter_window']
    
    for tol_type in tolerance_types:
        print(f"\nCreating precision-recall curve for tolerance: {tol_type}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        auc_values = []
        
        for ws in window_sizes:
            # Filter data for this window size and tolerance
            mask = (agg_results_df['window_size'] == ws) & \
                   (agg_results_df['tolerance_type'] == tol_type)
            window_data = agg_results_df[mask].sort_values('threshold')
            
            if len(window_data) == 0:
                continue
            
            recalls = window_data['recall_mean'].values
            precisions = window_data['precision_mean'].values
            recall_stds = window_data['recall_std'].values
            precision_stds = window_data['precision_std'].values
            
            # Replace NaN stds with 0 (happens when all datasets give same value)
            recall_stds = np.nan_to_num(recall_stds, nan=0.0)
            precision_stds = np.nan_to_num(precision_stds, nan=0.0)
            
            # Calculate AUC (sort by recall first)
            sort_idx = np.argsort(recalls)
            try:
                pr_auc = auc(recalls[sort_idx], precisions[sort_idx])
                auc_values.append(pr_auc)
            except:
                pr_auc = 0.0
            
            # Plot with error bars
            ax.errorbar(recalls, precisions, 
                       xerr=recall_stds, yerr=precision_stds,
                       marker='o', markersize=4, linewidth=2,
                       label=f'Window {ws} (AUC={pr_auc:.3f})',
                       capsize=3, capthick=1, alpha=0.8)
        
        # Calculate average AUC
        if auc_values:
            avg_auc = np.mean(auc_values)
            title = f'Precision-Recall Curve - Averaged over 10 Datasets\n(Tolerance: {tol_type}, Avg AUC={avg_auc:.3f})'
        else:
            title = f'Precision-Recall Curve - Averaged over 10 Datasets\n(Tolerance: {tol_type})'
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        output_path = os.path.join(plots_folder, f'precision_recall_{tol_type}_averaged.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    print(f"\nAll averaged plots saved to {plots_folder}")


def plot_auc_comparison(agg_results_df, output_folder):
    """Plot AUC values for different window sizes and tolerances.
    
    Args:
        agg_results_df: DataFrame with aggregated results
        output_folder: Where to save plots
    """
    plots_folder = os.path.join(output_folder, "averaged_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    window_sizes = sorted(agg_results_df['window_size'].unique())
    tolerance_types = ['full_window', 'half_window', 'quarter_window']
    
    # Calculate AUC for each window size and tolerance
    auc_data = []
    
    for tol_type in tolerance_types:
        for ws in window_sizes:
            mask = (agg_results_df['window_size'] == ws) & \
                   (agg_results_df['tolerance_type'] == tol_type)
            window_data = agg_results_df[mask].sort_values('threshold')
            
            if len(window_data) == 0:
                continue
            
            recalls = window_data['recall_mean'].values
            precisions = window_data['precision_mean'].values
            
            # Calculate AUC
            sort_idx = np.argsort(recalls)
            try:
                pr_auc = auc(recalls[sort_idx], precisions[sort_idx])
            except:
                pr_auc = 0.0
            
            auc_data.append({
                'window_size': ws,
                'tolerance_type': tol_type,
                'auc': pr_auc
            })
    
    auc_df = pd.DataFrame(auc_data)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(window_sizes))
    width = 0.25
    
    for i, tol_type in enumerate(tolerance_types):
        tol_data = auc_df[auc_df['tolerance_type'] == tol_type]
        aucs = [tol_data[tol_data['window_size'] == ws]['auc'].values[0] 
                if len(tol_data[tol_data['window_size'] == ws]) > 0 else 0
                for ws in window_sizes]
        ax.bar(x + i * width, aucs, width, label=tol_type)
    
    ax.set_xlabel('Window Size', fontsize=12)
    ax.set_ylabel('AUC (Precision-Recall)', fontsize=12)
    ax.set_title('AUC Comparison Across Window Sizes and Tolerances', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(window_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(plots_folder, 'auc_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC comparison plot saved to {output_path}")


def plot_auc_with_errorbars(agg_auc_df, output_folder):
    """Plot AUC values with error bars (mean +/- std) for each window size and tolerance.
    
    Args:
        agg_auc_df: DataFrame with aggregated AUC values (mean and std)
        output_folder: Where to save plots
    """
    plots_folder = os.path.join(output_folder, "averaged_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    window_sizes = sorted(agg_auc_df['window_size'].unique())
    tolerance_types = ['full_window', 'half_window', 'quarter_window']
    
    # Create bar plot with error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(window_sizes))
    width = 0.25
    
    for i, tol_type in enumerate(tolerance_types):
        tol_data = agg_auc_df[agg_auc_df['tolerance_type'] == tol_type].sort_values('window_size')
        
        aucs_mean = []
        aucs_std = []
        
        for ws in window_sizes:
            ws_data = tol_data[tol_data['window_size'] == ws]
            if len(ws_data) > 0:
                aucs_mean.append(ws_data['auc_mean'].values[0])
                # Replace NaN std with 0
                std_val = ws_data['auc_std'].values[0]
                aucs_std.append(0.0 if np.isnan(std_val) else std_val)
            else:
                aucs_mean.append(0.0)
                aucs_std.append(0.0)
        
        ax.bar(x + i * width, aucs_mean, width, yerr=aucs_std, 
               label=tol_type, capsize=5, error_kw={'linewidth': 1.5})
    
    ax.set_xlabel('Window Size', fontsize=12)
    ax.set_ylabel('AUC (Precision-Recall)', fontsize=12)
    ax.set_title('AUC Values with Error Bars (Mean ± Std over 10 Datasets)', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(window_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_path = os.path.join(plots_folder, 'auc_with_errorbars.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC plot with error bars saved to {output_path}")


def main():
    """Main execution function."""
    # Configuration
    base_folder = "Signal_processing/final/SATAY_synthetic"
    output_folder = "Signal_processing/final/results/windows"
    
    window_sizes = [10, 20, 30, 50, 80, 100, 150, 200]
    overlap = 0.5
    thresholds = np.linspace(0, 40, 41)
    dataset_ids = range(1, 11)  # Datasets 1 through 10
    
    # Determine number of workers (leave some CPUs free for system)
    n_workers = max(1, multiprocessing.cpu_count() - 2)
    
    print(f"Configuration:")
    print(f"  Window sizes: {window_sizes}")
    print(f"  Overlap: {overlap}")
    print(f"  Thresholds: {len(thresholds)} values from {thresholds[0]:.1f} to {thresholds[-1]:.1f}")
    print(f"  Datasets: {list(dataset_ids)}")
    print(f"  Output folder: {output_folder}")
    
    # Process all datasets
    all_results = []
    all_auc_results = []
    
    for dataset_id in dataset_ids:
        try:
            results_df, auc_df = process_single_dataset(
                dataset_id, window_sizes, overlap, thresholds,
                base_folder, output_folder, n_workers=n_workers
            )
            all_results.append(results_df)
            all_auc_results.append(auc_df)
        except Exception as e:
            print(f"\nError processing dataset {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("\nNo datasets were successfully processed!")
        return
    
    # Combine all results
    print(f"\n{'='*60}")
    print("Aggregating Results")
    print(f"{'='*60}")
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_auc_df = pd.concat(all_auc_results, ignore_index=True)
    
    # Save combined results
    combined_output_csv = os.path.join(output_folder, "all_datasets_performance.csv")
    all_results_df.to_csv(combined_output_csv, index=False)
    print(f"Combined results saved to {combined_output_csv}")
    
    # Save combined AUC results
    combined_auc_csv = os.path.join(output_folder, "all_datasets_auc.csv")
    all_auc_df.to_csv(combined_auc_csv, index=False)
    print(f"Combined AUC values saved to {combined_auc_csv}")
    
    # Aggregate results (mean and std across datasets)
    agg_results_df = aggregate_results(all_results_df, output_folder)
    
    # Aggregate AUC values (mean and std across datasets)
    agg_auc_df = aggregate_auc_results(all_auc_df, output_folder)
    
    # Create plots
    print(f"\n{'='*60}")
    print("Creating Plots")
    print(f"{'='*60}")
    
    plot_precision_recall_with_errorbars(agg_results_df, output_folder)
    plot_auc_comparison(agg_results_df, output_folder)
    plot_auc_with_errorbars(agg_auc_df, output_folder)
    
    # Print summary of AUC results
    print(f"\n{'='*60}")
    print("AUC Summary")
    print(f"{'='*60}")
    print(agg_auc_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"All results saved to: {output_folder}")


if __name__ == "__main__":
    main()
