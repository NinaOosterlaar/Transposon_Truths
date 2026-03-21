import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sliding_performance import (read_change_points, plot_metric_vs_threshold, 
                                 plot_all_metrics_comparison, plot_roc_curves)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Signal_processing.evaluation.evaluation import (precision, recall, F1_score, annotation_error, 
                       hausdorff_distance, rand_index, adjusted_rand_index,
                       precision_recall_curve, plot_precision_recall_curves,
                       roc_curve_from_cps_by_threshold, mean_absolute_error,
                       match_cps_one_to_one)

setup_plot_style()

CHROMS = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI"]
CPD = {
    "I": [-910, -780, -500, -350, -200, -70, 80, 760],
    "II": [-740, -580, -70, 65, 800],
    "III": [-900, -200, -80, 80, 750],
    "IV": [-900, -800, -740, -500, -380, -60, 60 ],
    "V": [-620, -280, -70, 90, 200, 480],
    "VI": [-960, -860, 460],    
    "VII": [-440, -110, 70, 260, 320, 730],
    "VIII": [-860, -510, -80, 80, 200, 370, 580, ],
    "IX": [-400, -60, 80, 200, 300, 760],
    "X": [-920, -480, -80, 60, 200 ],
    "XI": [-660, -380 ,-70, 60, 180, 280, 470],
    "XII": [-180, -75, 70, 460 ],
    "XIII": [-770, -420, -300, -140, -65, 80, 780],
    "XIV": [-920, -140, -65, 80, ],
    "XV": [-740, -120, 60, 380, 930],
    "XVI": [-580, -170, -70, 80,],
}



def get_available_chromosomes(input_folder, data_folder, cpd_dict):
    """Find chromosomes that have both result files and non-empty true change points.
    
    Args:
        input_folder: Folder containing chromosome folders with window subfolders
        data_folder: Folder containing chromosome data files
        cpd_dict: Dictionary with true change points per chromosome
    
    Returns:
        List of chromosome names (e.g., ["I", "II", "III"])
    """
    available_chroms = []
    
    for chrom in CHROMS:
        # Check if chromosome has non-empty true change points
        if chrom not in cpd_dict or len(cpd_dict[chrom]) == 0:
            continue
        
        # Check if data file exists
        data_file = os.path.join(data_folder, f"Chr{chrom}_centromere_window.csv")
        if not os.path.exists(data_file):
            continue
        
        # Check if chromosome folder exists
        chrom_folder = os.path.join(input_folder, f"Chr{chrom}")
        if not os.path.exists(chrom_folder) or not os.path.isdir(chrom_folder):
            continue
        
        # Check if there are window folders inside
        window_folders = [f for f in os.listdir(chrom_folder) 
                         if os.path.isdir(os.path.join(chrom_folder, f)) and f.startswith("window")]
        
        if len(window_folders) == 0:
            continue
        
        # Count result files in the first window folder
        first_window = os.path.join(chrom_folder, window_folders[0])
        result_files = [f for f in os.listdir(first_window) if f.endswith('.txt')]
        
        available_chroms.append(chrom)
        print(f"Chromosome {chrom}: Found {len(window_folders)} window sizes, {len(result_files)} result files per window")
    
    return available_chroms


def read_chromosome_data(data_file):
    """Read chromosome data and return position to centromere distance mapping.
    
    Args:
        data_file: Path to chromosome CSV file
    
    Returns:
        DataFrame with Position, Value, and Centromere_Distance columns
    """
    df = pd.read_csv(data_file)
    return df



def evaluate_chromosome(chrom, input_folder, data_folder, cpd_dict):
    """Evaluate performance for a single chromosome across all windows and thresholds.
    
    Args:
        chrom: Chromosome name (e.g., "I")
        input_folder: Folder containing result files
        data_folder: Folder containing chromosome data files
        cpd_dict: Dictionary with true change points
    
    Returns:
        List of result dictionaries, dict of window_threshold_cps for ROC curves
    """
    # Read chromosome data
    data_file = os.path.join(data_folder, f"Chr{chrom}_centromere_window.csv")
    chrom_data = read_chromosome_data(data_file)
    n_points = len(chrom_data)
    
    # True change points in centromere coordinates
    true_cps = cpd_dict[chrom] 

    
    print(f"\nChromosome {chrom}:")
    print(f"  Data points: {n_points}")
    print(f"  True change points: {len(true_cps)}")
    
    results = []
    window_threshold_cps = {}  # For ROC curves
    
    # Get all window folders from the chromosome-specific folder
    chrom_folder = os.path.join(input_folder, f"Chr{chrom}")
    window_folders = [f for f in os.listdir(chrom_folder) 
                     if os.path.isdir(os.path.join(chrom_folder, f)) and f.startswith("window")]
    window_folders.sort()
    
    for window_folder in window_folders:
        window_size = int(window_folder.replace("window", ""))
        window_path = os.path.join(chrom_folder, window_folder)
        if window_size != 100:
            continue
        
        # Define tolerances based on window size
        tolerances = {
            "full_window": window_size,
            "half_window": window_size // 2,
            "quarter_window": window_size // 4
        }
        
        # Get result files for this chromosome
        result_files = [f for f in os.listdir(window_path) 
                       if f.startswith(f"Chr{chrom}_") and f.endswith('.txt')]
        result_files.sort()
        
        print(f"  Window {window_size}: {len(result_files)} result files")
        
        for result_file in result_files:
            # Extract threshold from filename
            match = re.search(r'th(\d+\.\d+)', result_file)
            if not match:
                continue
            threshold = float(match.group(1))
            
            # Read detected change points (in position coordinates)
            file_path = os.path.join(window_path, result_file)
            detected_positions = read_change_points(file_path)
            
            # Convert to centromere coordinates
            detected_cps = list(np.array(detected_positions) - 1000)  # Assuming positions are sorted and start from the first position in the data
            
            # Store for ROC curves
            if window_size not in window_threshold_cps:
                window_threshold_cps[window_size] = []
            window_threshold_cps[window_size].append((threshold, detected_cps))
            
            # Calculate tolerance-independent metrics
            ann_error = annotation_error(detected_cps, true_cps)
            hausdorff = hausdorff_distance(true_cps, detected_cps)
            rand_idx = rand_index(true_cps, detected_cps, n_points)
            adj_rand_idx = adjusted_rand_index(true_cps, detected_cps, n_points)
            
            # Calculate MAE localization error
            matches, _, _ = match_cps_one_to_one(true_cps, detected_cps, window_size)
            if matches:
                mae_loc = np.mean([abs(pred - true) for pred, true in matches])
            else:
                mae_loc = np.inf
            
            # Calculate metrics for each tolerance
            for tol_name, tol_value in tolerances.items():
                prec = precision(detected_cps, true_cps, tol_value)
                rec = recall(detected_cps, true_cps, tol_value)
                f1 = F1_score(prec, rec)
                
                results.append({
                    'chromosome': chrom,
                    'window_size': window_size,
                    'threshold': threshold,
                    'tolerance_type': tol_name,
                    'tolerance_value': tol_value,
                    'precision': prec,
                    'recall': rec,
                    'F1': f1,
                    'num_detected': len(detected_cps),
                    'num_true': len(true_cps),
                    'annotation_error': ann_error,
                    'hausdorff_distance': hausdorff,
                    'rand_index': rand_idx,
                    'adjusted_rand_index': adj_rand_idx,
                    'mae_localization': mae_loc,
                })
    
    return results, window_threshold_cps


def evaluate_all_chromosomes(input_folder, data_folder, output_folder, cpd_dict):
    """Evaluate performance for all available chromosomes.
    
    Args:
        input_folder: Folder containing result files
        data_folder: Folder containing chromosome data files
        output_folder: Folder for output files
        cpd_dict: Dictionary with true change points
    
    Returns:
        Combined DataFrame with all results
    """
    # Find available chromosomes
    available_chroms = get_available_chromosomes(input_folder, data_folder, cpd_dict)
    
    if not available_chroms:
        print("No chromosomes with both result files and true change points found!")
        return None
    
    print(f"\nAvailable chromosomes: {available_chroms}")
    print("="*60)
    
    all_results = []
    all_window_threshold_cps = {}  # {chrom: {window_size: [(threshold, cps), ...]}}
    
    # Evaluate each chromosome
    for chrom in available_chroms:
        chrom_results, chrom_window_threshold_cps = evaluate_chromosome(
            chrom, input_folder, data_folder, cpd_dict
        )
        all_results.extend(chrom_results)
        all_window_threshold_cps[chrom] = chrom_window_threshold_cps
        
        # Save individual chromosome results
        chrom_df = pd.DataFrame(chrom_results)
        chrom_output_csv = os.path.join(output_folder, f"performance_Chr{chrom}.csv")
        chrom_df.to_csv(chrom_output_csv, index=False)
        print(f"  Saved individual results to {chrom_output_csv}")
    
    # Combine all results
    combined_df = pd.DataFrame(all_results)
    combined_output_csv = os.path.join(output_folder, "performance_all_chromosomes.csv")
    combined_df.to_csv(combined_output_csv, index=False)
    print(f"\nSaved combined results to {combined_output_csv}")
    
    # Create plots
    print("\n" + "="*60)
    print("Creating plots...")
    print("="*60)
    
    output_plots_folder = os.path.join(output_folder, "performance_plots")
    os.makedirs(output_plots_folder, exist_ok=True)
    
    # Tolerance-independent metric plots (combined across all chromosomes)
    plot_all_metrics_comparison(combined_df, output_plots_folder, "All_Chromosomes")
    
    plot_metric_vs_threshold(combined_df, 'annotation_error', output_plots_folder, "All_Chromosomes",
                            metric_label='Annotation Error (|#predicted - #true|)', use_log_scale=False)
    plot_metric_vs_threshold(combined_df, 'hausdorff_distance', output_plots_folder, "All_Chromosomes",
                            metric_label='Hausdorff Distance', use_log_scale=True, exclude_inf=True)
    plot_metric_vs_threshold(combined_df, 'mae_localization', output_plots_folder, "All_Chromosomes",
                            metric_label='MAE Localization Error', use_log_scale=True, exclude_inf=True)
    plot_metric_vs_threshold(combined_df, 'rand_index', output_plots_folder, "All_Chromosomes",
                            metric_label='Rand Index', use_log_scale=False)
    plot_metric_vs_threshold(combined_df, 'adjusted_rand_index', output_plots_folder, "All_Chromosomes",
                            metric_label='Adjusted Rand Index', use_log_scale=False)
    
    # Precision-Recall and ROC curves for each tolerance
    for tol_name in ["full_window", "half_window", "quarter_window"]:
        print(f"\nCreating precision-recall curves for {tol_name}...")
        pr_curves_data = []
        
        # Get unique window sizes
        window_sizes = sorted(combined_df['window_size'].unique())
        
        for window_size in window_sizes:
            # Combine data across all chromosomes for this window
            window_data = combined_df[
                (combined_df['window_size'] == window_size) & 
                (combined_df['tolerance_type'] == tol_name)
            ].sort_values('threshold')
            
            if len(window_data) > 0:
                # Average precision and recall across chromosomes for each threshold
                threshold_data = window_data.groupby('threshold').agg({
                    'precision': 'mean',
                    'recall': 'mean'
                }).reset_index()
                
                pr_data = precision_recall_curve(
                    detected_cps_list=None,
                    precisions=threshold_data['precision'].values,
                    recalls=threshold_data['recall'].values,
                    thresholds=threshold_data['threshold'].values,
                    label=f'Window {window_size}'
                )
                
                pr_curves_data.append({
                    'recalls': pr_data['recalls'],
                    'precisions': pr_data['precisions'],
                    'label': f'Window {window_size}',
                    'marker': 'o',
                    'markersize': 4,
                    'linewidth': 2
                })
        
        plot_path = os.path.join(output_plots_folder, f'precision_recall_{tol_name}.png')
        plot_precision_recall_curves(
            pr_curves_data,
            output_path=plot_path,
            title=f'Precision-Recall Curve (Tolerance: {tol_name})',
            figsize=(10, 8)
        )
    
    # ROC curves
    print("\nCreating ROC curves...")
    for tol_name in ["full_window", "half_window", "quarter_window"]:
        roc_curves_data = []
        
        # Get tolerance value
        tol_value = None
        for _, row in combined_df[combined_df['tolerance_type'] == tol_name].iloc[:1].iterrows():
            tol_value = int(row['tolerance_value'])
            break
        
        if tol_value is None:
            continue
        
        for window_size in window_sizes:
            # Combine detected CPs and true CPs from all chromosomes
            combined_threshold_cps_list = []
            combined_true_cps = []
            total_n_points = 0
            
            for chrom in available_chroms:
                if window_size in all_window_threshold_cps[chrom]:
                    threshold_cps_list = all_window_threshold_cps[chrom][window_size]
                    
                    # Get true CPs and n_points for this chromosome
                    data_file = os.path.join(data_folder, f"Chr{chrom}_centromere_window.csv")
                    chrom_data = read_chromosome_data(data_file)
                    n_points = len(chrom_data)
                    true_cps = cpd_dict[chrom]
                    
                    # Store for aggregation
                    if not combined_threshold_cps_list:
                        # Initialize with thresholds from first chromosome
                        combined_threshold_cps_list = [(th, []) for th, _ in threshold_cps_list]
                    
                    # Aggregate detected CPs for each threshold
                    for i, (th, detected_cps) in enumerate(threshold_cps_list):
                        combined_threshold_cps_list[i] = (th, combined_threshold_cps_list[i][1] + detected_cps)
                    
                    combined_true_cps.extend(true_cps)
                    total_n_points += n_points
            
            if combined_threshold_cps_list and total_n_points > 0:
                fpr, tpr, thresholds = roc_curve_from_cps_by_threshold(
                    combined_threshold_cps_list, combined_true_cps, total_n_points, tol_value
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
        
        plot_roc_curves(
            roc_curves_data,
            output_plots_folder,
            tol_name,
            "All_Chromosomes"
        )
    
    print("\n" + "="*60)
    print("Performance evaluation completed!")
    print("="*60)
    
    return combined_df


def evaluate_saturation_level(saturation_level, base_results_folder, window_folder, cpd_dict, 
                             window_size=100):
    """Evaluate performance for a single saturation level across all datasets and chromosomes.
    
    Args:
        saturation_level: Saturation level (1-6)
        base_results_folder: Base folder containing saturation-level subfolders
        window_folder: Folder containing centromere window files
        cpd_dict: Dictionary with true change points
        window_size: Window size to evaluate
    
    Returns:
        DataFrame with averaged results per threshold
    """
    saturation_folder = os.path.join(base_results_folder, str(saturation_level))
    
    if not os.path.exists(saturation_folder):
        print(f"Saturation level {saturation_level} folder not found!")
        return None
    
    # Get all datasets for this saturation level
    datasets = [d for d in os.listdir(saturation_folder) 
                if os.path.isdir(os.path.join(saturation_folder, d))]
    
    print(f"\nSaturation Level {saturation_level}: Found {len(datasets)} datasets")
    
    all_results = []
    
    # Process each dataset
    for dataset in datasets:
        dataset_folder = os.path.join(saturation_folder, dataset)
        data_folder = os.path.join(window_folder, str(saturation_level), dataset)
        
        if not os.path.exists(data_folder):
            print(f"  Warning: Data folder not found for {dataset}")
            continue
        
        print(f"  Processing {dataset}...")
        
        # Process each chromosome
        for chrom in CHROMS:
            if chrom not in cpd_dict or len(cpd_dict[chrom]) == 0:
                continue
            
            # Check if results exist for this chromosome
            chrom_results_path = os.path.join(
                dataset_folder, f"Chr{chrom}", f"Chr{chrom}_centromere_window", 
                f"window{window_size}"
            )
            
            if not os.path.exists(chrom_results_path):
                continue
            
            # Read chromosome data
            data_file = os.path.join(data_folder, f"Chr{chrom}_centromere_window.csv")
            if not os.path.exists(data_file):
                continue
            
            chrom_data = read_chromosome_data(data_file)
            n_points = len(chrom_data)
            true_cps = cpd_dict[chrom]
            
            # Define tolerances
            tolerances = {
                "full_window": window_size,
                "half_window": window_size // 2,
                "quarter_window": window_size // 4
            }
            
            # Get result files
            result_files = [f for f in os.listdir(chrom_results_path) 
                           if f.endswith('.txt')]
            
            for result_file in result_files:
                # Extract threshold from filename
                match = re.search(r'th(\d+\.\d+)', result_file)
                if not match:
                    continue
                threshold = float(match.group(1))
                
                # Read detected change points
                file_path = os.path.join(chrom_results_path, result_file)
                detected_positions = read_change_points(file_path)
                
                # Convert to centromere coordinates (assuming position 0 is at index 1000)
                detected_cps = [pos - 1000 for pos in detected_positions]
                
                # Calculate metrics for each tolerance
                for tol_name, tol_value in tolerances.items():
                    prec = precision(detected_cps, true_cps, tol_value)
                    rec = recall(detected_cps, true_cps, tol_value)
                    f1 = F1_score(prec, rec)
                    
                    all_results.append({
                        'saturation_level': saturation_level,
                        'dataset': dataset,
                        'chromosome': chrom,
                        'threshold': threshold,
                        'tolerance_type': tol_name,
                        'tolerance_value': tol_value,
                        'precision': prec,
                        'recall': rec,
                        'F1': f1,
                        'num_detected': len(detected_cps),
                        'num_true': len(true_cps),
                    })
    
    if not all_results:
        print(f"  No results found for saturation level {saturation_level}")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Average across datasets and chromosomes for each threshold and tolerance
    averaged_df = results_df.groupby(['threshold', 'tolerance_type', 'tolerance_value']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'F1': 'mean',
        'num_detected': 'mean',
        'num_true': 'mean',
    }).reset_index()
    
    averaged_df['saturation_level'] = saturation_level
    
    print(f"  Processed {len(results_df)} individual results")
    print(f"  Averaged to {len(averaged_df)} threshold/tolerance combinations")
    
    return averaged_df


def calculate_saturation_statistics(window_folder, saturation_level):
    """Calculate saturation statistics (non-zero positions) for a saturation level.
    
    Args:
        window_folder: Folder containing centromere window files
        saturation_level: Saturation level (1-6)
    
    Returns:
        Tuple of (num_nonzero, total_positions, percentage)
    """
    saturation_folder = os.path.join(window_folder, str(saturation_level))
    
    if not os.path.exists(saturation_folder):
        return None, None, None
    
    total_nonzero = 0
    total_positions = 0
    
    # Get all datasets for this saturation level
    datasets = [d for d in os.listdir(saturation_folder) 
                if os.path.isdir(os.path.join(saturation_folder, d))]
    
    for dataset in datasets:
        dataset_folder = os.path.join(saturation_folder, dataset)
        
        # Read all chromosome window files
        for chrom in CHROMS:
            window_file = os.path.join(dataset_folder, f"Chr{chrom}_centromere_window.csv")
            
            if os.path.exists(window_file):
                df = pd.read_csv(window_file)
                # Count non-zero positions
                total_nonzero += (df['value'] > 0).sum()
                total_positions += len(df)
    
    if total_positions == 0:
        return None, None, None
    
    percentage = (total_nonzero / total_positions) * 100
    
    return total_nonzero, total_positions, percentage


def compare_saturation_levels(base_results_folder, window_folder, cpd_dict, 
                              output_folder, saturation_levels=[1,2,3,4,5,6]):
    """Compare performance across different saturation levels.
    
    Args:
        base_results_folder: Base folder containing saturation-level subfolders
        window_folder: Folder containing centromere window files
        cpd_dict: Dictionary with true change points
        output_folder: Output folder for results and plots
        saturation_levels: List of saturation levels to compare
    """
    print("="*60)
    print("Comparing Performance Across Saturation Levels")
    print("="*60)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate saturation statistics
    print("\nCalculating saturation levels (non-zero positions)...")
    saturation_stats = {}
    for sat_level in saturation_levels:
        num_nonzero, total_pos, percentage = calculate_saturation_statistics(window_folder, sat_level)
        if num_nonzero is not None:
            saturation_stats[sat_level] = (num_nonzero, total_pos, percentage)
            print(f"  Level {sat_level}: {num_nonzero}/{total_pos} non-zero positions ({percentage:.1f}%)")
        else:
            saturation_stats[sat_level] = (None, None, None)
            print(f"  Level {sat_level}: No data found")
    
    all_saturation_results = []
    
    # Evaluate each saturation level
    for sat_level in saturation_levels:
        sat_results = evaluate_saturation_level(
            sat_level, base_results_folder, window_folder, cpd_dict
        )
        if sat_results is not None:
            all_saturation_results.append(sat_results)
    
    if not all_saturation_results:
        print("No results found!")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_saturation_results, ignore_index=True)
    
    # Save combined results
    output_csv = os.path.join(output_folder, "saturation_comparison_results.csv")
    combined_df.to_csv(output_csv, index=False)
    print(f"\nSaved combined results to {output_csv}")
    
    # Save saturation statistics
    saturation_stats_df = pd.DataFrame([
        {'saturation_level': level, 'num_nonzero': num_nz, 'total_positions': total, 'percentage_nonzero': pct}
        for level, (num_nz, total, pct) in saturation_stats.items()
    ])
    stats_csv = os.path.join(output_folder, "saturation_statistics.csv")
    saturation_stats_df.to_csv(stats_csv, index=False)
    print(f"Saved saturation statistics to {stats_csv}")
    
    # Create precision-recall curves for each tolerance
    print("\n" + "="*60)
    print("Creating Precision-Recall Curves")
    print("="*60)
    
    for tol_name in ["full_window", "half_window", "quarter_window"]:
        print(f"\nTolerance: {tol_name}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define colors for saturation levels
        color_list = [COLORS['blue'], COLORS['orange'], COLORS['green'], 
                     COLORS['red'], COLORS['pink'], COLORS['light_blue']]
        
        for sat_level in saturation_levels:
            sat_data = combined_df[
                (combined_df['saturation_level'] == sat_level) & 
                (combined_df['tolerance_type'] == tol_name)
            ].sort_values('threshold')
            
            if len(sat_data) == 0:
                continue
            
            # Create label with saturation percentage
            num_nonzero, total_pos, percentage = saturation_stats.get(sat_level, (None, None, None))
            if percentage is not None:
                label = f'Saturation: {percentage:.1f}%'
            else:
                label = f'Level {sat_level}'
            
            # Plot PR curve for this saturation level
            ax.plot(sat_data['recall'], sat_data['precision'], 
                   marker='o', linewidth=2, markersize=4,
                   label=label, color=color_list[sat_level-1])
        
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title(f'Precision-Recall Curves by Saturation Level\n(Tolerance: {tol_name})', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        
        plot_path = os.path.join(output_folder, f'pr_curve_saturation_comparison_{tol_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_path}")
    
    # Create summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    print("\nSaturation levels (% non-zero positions):")
    for level, (num_nz, total, pct) in saturation_stats.items():
        if pct is not None:
            print(f"  Level {level}: {num_nz}/{total} ({pct:.1f}%)")
    
    print("\nMean metrics by saturation level and tolerance:")
    summary = combined_df.groupby(['saturation_level', 'tolerance_type'])[
        ['precision', 'recall', 'F1']
    ].mean()
    print(summary)
    
    # Save summary
    summary_csv = os.path.join(output_folder, "saturation_summary.csv")
    summary.to_csv(summary_csv)
    print(f"\nSaved summary to {summary_csv}")
    
    return combined_df


if __name__ == "__main__":
    print("="*60)
    print("SATAY Change Point Detection Performance Evaluation")
    print("="*60)
    
    # Compare saturation levels (test_CPD data)
    base_results_folder = "Signal_processing/results/centromere_cpd_results"
    window_folder = "Data/test_CPD/centromere_windows"
    output_folder = "Signal_processing/results/saturation_comparison"
    
    results_df = compare_saturation_levels(
        base_results_folder, window_folder, CPD, output_folder,
        saturation_levels=[1, 2, 3, 4, 5, 6]
    )
    
    if results_df is not None:
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)

