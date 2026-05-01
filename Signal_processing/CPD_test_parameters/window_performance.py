import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Signal_processing.CPD_algorithms.sliding_ZINB.sliding_ZINB_CPD_ref import sliding_ZINB_CPD_ref
from Signal_processing.CPD_evaluation.evaluation_util import match_cps_one_to_one
from Utils.plot_config import setup_plot_style, COLORS

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
    df = pd.read_csv(data_file)
    data = df['Value'].values.astype(int)
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


def apply_threshold_to_scores(scores, threshold, window_size, overlap, step_size):
    """Apply threshold filtering to LRT scores to extract change points.
    
    This replicates the threshold logic from the detector functions without re-running the likelihood calculation.
    
    Args:
        scores: Array of LRT scores from the detector
        threshold: Threshold value to apply
        window_size: Window size used
        overlap: Overlap fraction (0-1)
        step_size: Step size between windows
    
    Returns:
        List of detected change point positions
    """
    change_points = []
    last_cp, last_score = -np.inf, 0.0
    
    for idx, score in enumerate(scores):
        if score > threshold:
            # Position calculation: idx corresponds to window index
            # Change point is at start + window_size
            cp = idx * step_size + window_size
            
            if (cp - last_cp) >= window_size:
                change_points.append(cp)
                last_cp, last_score = cp, score
            elif score > last_score:
                change_points[-1] = cp
                last_cp, last_score = cp, score
    
    return change_points


def process_window_size_for_dataset(ws, dataset_id, data, pi_file, true_cps, overlap, thresholds, 
                                     theta_global, output_folder):
    """Process a single window size for a dataset (to be run in parallel).
    
    Args:
        ws: Window size to process
        dataset_id: ID of the dataset
        data: The data array
        pi_file: Path to pi_values.csv file
        true_cps: True change points
        overlap: Overlap fraction (0-1)
        thresholds: Array of thresholds to test
        theta_global: Global theta parameter
        output_folder: Output folder for results
    
    Returns:
        List of result dictionaries for this window size
    """
    print(f"  [Dataset {dataset_id}] Processing window size: {ws}")
    window_output_folder = os.path.join(output_folder, f"dataset_{dataset_id}", f"window{ws}")
    
    # Run detector once with threshold=0 to get all LRT scores
    print(f"    Running CPD detection (calculating LRT scores)...")
    _, scores = sliding_ZINB_CPD_ref(
        data,
        window_size=ws,
        overlap=overlap,
        threshold=0.0,  # Use 0 to capture all scores
        theta_global=theta_global,
        pi_file=pi_file
    )
    
    # Calculate step size for position mapping
    step_size = max(1, int(ws * (1 - overlap)))
    
    results = []
    
    # Now apply each threshold to the saved scores
    for threshold in thresholds:
        print(f"    Applying threshold: {threshold:.2f}")
        
        # Apply threshold to pre-computed scores
        change_points = apply_threshold_to_scores(scores, threshold, ws, overlap, step_size)
        
        # Save results
        save_results(window_output_folder, f"dataset_{dataset_id}", 
                    change_points, scores, theta_global, ws, overlap, threshold)
        
        # Calculate TP/FP/FN for different tolerances
        tolerances = {
            "full_window": ws,
            "half_window": ws // 2,
            "quarter_window": ws // 4
        }
        
        for tol_name, tol_value in tolerances.items():
            # Use match_cps_one_to_one to get TP, FP, FN
            matches, unmatched_pred, unmatched_true = match_cps_one_to_one(
                true_cps, change_points, tol_value
            )
            
            tp = len(matches)
            fp = len(unmatched_pred)
            fn = len(unmatched_true)
            
            results.append({
                'dataset_id': dataset_id,
                'window_size': ws,
                'threshold': threshold,
                'tolerance_type': tol_name,
                'tolerance_value': tol_value,
                'TP': tp,
                'FP': fp,
                'FN': fn,
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
    theta_global = 0.5
    
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
                    ws, dataset_id, data, pi_file,
                    true_cps, overlap, thresholds, theta_global, output_folder
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
                ws, dataset_id, data, pi_file,
                true_cps, overlap, thresholds, theta_global, output_folder
            )
            results.extend(window_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save individual dataset results
    dataset_output_csv = os.path.join(output_folder, f"dataset_{dataset_id}", "performance_metrics.csv")
    os.makedirs(os.path.dirname(dataset_output_csv), exist_ok=True)
    results_df.to_csv(dataset_output_csv, index=False)
    print(f"\nDataset {dataset_id} results saved to {dataset_output_csv}")
    
    return results_df


def aggregate_results(all_results_df, output_folder):
    """Aggregate results across all datasets by summing TP/FP/FN counts.
    
    Args:
        all_results_df: DataFrame containing results from all datasets
        output_folder: Where to save aggregated results
    
    Returns:
        DataFrame with precision/recall calculated from aggregated counts
    """
    # Group by window_size, threshold, and tolerance_type
    grouped = all_results_df.groupby(['window_size', 'threshold', 'tolerance_type'])
    
    # Sum TP, FP, FN across all datasets
    agg_results = grouped.agg({
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum',
        'tolerance_value': 'first',
        'num_detected': 'sum',
        'num_true': 'sum'
    }).reset_index()
    
    # Calculate precision and recall from aggregated counts
    agg_results['precision'] = agg_results['TP'] / (agg_results['TP'] + agg_results['FP'])
    agg_results['recall'] = agg_results['TP'] / (agg_results['TP'] + agg_results['FN'])
    
    # Handle division by zero
    agg_results['precision'] = agg_results['precision'].replace([np.inf, -np.inf], np.nan)
    agg_results['recall'] = agg_results['recall'].fillna(0.0)
    
    # Save aggregated results
    agg_output_csv = os.path.join(output_folder, "aggregated_performance_metrics.csv")
    agg_results.to_csv(agg_output_csv, index=False)
    print(f"\nAggregated results saved to {agg_output_csv}")
    
    return agg_results


def plot_precision_recall_aggregated(agg_results_df, output_folder, dataset_name="SATAY_synthetic"):
    """Plot precision-recall curves from aggregated TP/FP/FN counts (no error bars).
    
    Args:
        agg_results_df: DataFrame with aggregated results (precision/recall from summed counts)
        output_folder: Where to save plots
        dataset_name: Name of dataset for plot titles
    """
    # Create output folder for plots
    plots_folder = os.path.join(output_folder, "aggregated_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # Get unique window sizes
    window_sizes = sorted(agg_results_df['window_size'].unique())
    
    # Define colors for window sizes
    color_list = [COLORS['blue'], COLORS['orange'], COLORS['green'], 
                 COLORS['red'], COLORS['pink'], COLORS['light_blue'], 
                 COLORS['yellow'], COLORS['black']]
    
    # Create precision-recall curves for each tolerance type
    tolerance_types = ['full_window', 'half_window', 'quarter_window']
    
    for tol_type in tolerance_types:
        print(f"\nCreating precision-recall curve for tolerance: {tol_type}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for idx, ws in enumerate(window_sizes):
            # Filter data for this window size and tolerance
            mask = (agg_results_df['window_size'] == ws) & \
                   (agg_results_df['tolerance_type'] == tol_type)
            window_data = agg_results_df[mask].sort_values('threshold')
            
            if len(window_data) == 0:
                continue
            
            recalls = window_data['recall'].values
            precisions = window_data['precision'].values
            
            # Plot without error bars
            ax.plot(recalls, precisions,
                   marker='o', markersize=4, linewidth=2,
                   label=f'Window {ws}', color=color_list[idx % len(color_list)])
        
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title(f'Precision-Recall Curve - Aggregated over 10 Datasets\n(Tolerance: {tol_type})', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        output_path = os.path.join(plots_folder, f'precision_recall_{tol_type}_aggregated.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    print(f"\nAll aggregated plots saved to {plots_folder}")



def main():
    """Main execution function."""
    # Configuration
    base_folder = "Data/SATAY_synthetic"
    output_folder = "Signal_processing/results_new/compare_window_performance"
    
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
    
    for dataset_id in dataset_ids:
        try:
            results_df = process_single_dataset(
                dataset_id, window_sizes, overlap, thresholds,
                base_folder, output_folder, n_workers
            )
            all_results.append(results_df)
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
    
    # Save combined results
    combined_output_csv = os.path.join(output_folder, "all_datasets_performance.csv")
    all_results_df.to_csv(combined_output_csv, index=False)
    print(f"Combined results saved to {combined_output_csv}")
    
    # Aggregate results (sum TP/FP/FN across datasets)
    agg_results_df = aggregate_results(all_results_df, output_folder)
    
    # Create plots
    print(f"\n{'='*60}")
    print("Creating Plots")
    print(f"{'='*60}")
    
    plot_precision_recall_aggregated(agg_results_df, output_folder)
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"All results saved to: {output_folder}")


if __name__ == "__main__":
    main()
