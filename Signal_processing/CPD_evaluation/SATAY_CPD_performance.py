#!/usr/bin/env python3
"""
Evaluate SATAY Change Point Detection Performance

Creates aggregated precision-recall curves comparing different saturation levels
for various CPD methods (Gaussian AE, Moving Average, ZINB, etc.).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Signal_processing.CPD_evaluation.evaluation_util import match_cps_one_to_one
from Signal_processing.CPD_evaluation.sliding_performance import read_change_points

setup_plot_style()

CHROMS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]

# True change points for each chromosome (centromere coordinates)
CPD = {
    "I": [-910, -780, -500, -350, -200, -70, 80, 760],
    "II": [-740, -580, -70, 65, 800],
    "III": [-900, -200, -80, 80, 750],
    "IV": [-900, -800, -740, -500, -380, -60, 60],
    "V": [-620, -280, -70, 90, 200, 480],
    "VI": [-960, -860, 460],
    "VII": [-440, -110, 70, 260, 320, 730],
    "VIII": [-860, -510, -80, 80, 200, 370, 580],
    "IX": [-400, -60, 80, 200, 300, 760],
    "X": [-920, -480, -80, 60, 200],
    "XI": [-660, -380, -70, 60, 180, 280, 470],
    "XII": [-180, -75, 70, 460],
    "XIII": [-770, -420, -300, -140, -65, 80, 780],
    "XIV": [-920, -140, -65, 80],
    "XV": [-740, -120, 60, 380, 930],
    "XVI": [-580, -170, -70, 80],
}

# Saturation percentages for each level
SATURATION = [3.2, 6.6, 13.1, 20.5, 28.8, 34.4, 39.2, 43.1]

# Method-specific default paths
METHOD_DEFAULTS = {
    'gaussian_ae': {
        'results': 'Signal_processing/results_new/Gaussian_AE_CPD',
        'output': 'Signal_processing/results_new/Gaussian_AE_CPD/evaluation',
    },
    'gaussian_ma': {
        'results': 'Signal_processing/results_new/Gaussian_MA_CPD',
        'output': 'Signal_processing/results_new/Gaussian_MA_CPD/evaluation',
    },
    'zinb': {
        'results': 'Signal_processing/results_new/CPD_SATAY_v3_window',
        'output': 'Signal_processing/results_new/CPD_SATAY_v3_window/evaluation',
    },
}


def evaluate_saturation_level(saturation_level, results_base, data_base, cpd_dict, window_size=100):
    """Evaluate one saturation level across all strains and chromosomes.
    
    Args:
        saturation_level: Saturation level (0-7)
        results_base: Base folder containing results
        data_base: Base folder containing data files
        cpd_dict: Dictionary with true change points per chromosome
        window_size: Window size used for CPD
    
    Returns:
        DataFrame with aggregated results per threshold
    """
    results_folder = os.path.join(results_base, str(saturation_level))
    data_folder = os.path.join(data_base, str(saturation_level))
    
    if not os.path.exists(results_folder):
        print(f"Saturation level {saturation_level} results folder not found!")
        return None
    
    if not os.path.exists(data_folder):
        print(f"Saturation level {saturation_level} data folder not found!")
        return None
    
    # Get all strain folders
    strain_folders = [
        name for name in os.listdir(results_folder)
        if os.path.isdir(os.path.join(results_folder, name)) and name.startswith("yEK23_")
    ]
    
    print(f"\nSaturation level {saturation_level} ({SATURATION[saturation_level]:.1f}%): Found {len(strain_folders)} strains")
    
    all_results = []
    thresholds = np.arange(0, 21, 1, dtype=float)
    tol_name = "full_window"
    tol_value = window_size
    
    for threshold in thresholds:
        tp_total = 0
        fp_total = 0
        fn_total = 0
        num_detected_total = 0
        num_true_total = 0
        has_points = False
        
        for strain_name in strain_folders:
            strain_results_folder = os.path.join(results_folder, strain_name)
            strain_data_folder = os.path.join(data_folder, strain_name)
            
            if not os.path.exists(strain_data_folder):
                continue
            
            for chrom in CHROMS:
                if chrom not in cpd_dict or len(cpd_dict[chrom]) == 0:
                    continue
                
                # Check for results path - try nested structure first
                chrom_results_path_nested = os.path.join(
                    strain_results_folder,
                    f"Chr{chrom}",
                    f"Chr{chrom}_centromere_window",
                    f"window{window_size}"
                )
                chrom_results_path_flat = os.path.join(
                    strain_results_folder,
                    f"Chr{chrom}",
                    f"window{window_size}"
                )
                
                if os.path.exists(chrom_results_path_nested):
                    chrom_results_path = chrom_results_path_nested
                elif os.path.exists(chrom_results_path_flat):
                    chrom_results_path = chrom_results_path_flat
                else:
                    continue
                
                # Check for data file
                data_file = os.path.join(strain_data_folder, f"Chr{chrom}_centromere_window.csv")
                if not os.path.exists(data_file):
                    continue
                
                # Find result file for this threshold
                result_files = [
                    filename for filename in os.listdir(chrom_results_path)
                    if filename.endswith('.txt')
                ]
                
                threshold_file = None
                for result_file in result_files:
                    match = re.search(r'th(\d+\.\d+)', result_file)
                    if not match:
                        continue
                    file_threshold = float(match.group(1))
                    if np.isclose(file_threshold, threshold):
                        threshold_file = result_file
                        break
                
                if threshold_file is None:
                    continue
                
                # Get true change points
                true_cps = cpd_dict[chrom]
                
                # Read detected change points
                file_path = os.path.join(chrom_results_path, threshold_file)
                detected_positions = read_change_points(file_path)
                
                # Convert to centromere coordinates (position 0 in CSV is at -1000 in centromere coords)
                detected_cps = [pos - 1000 for pos in detected_positions]
                
                # Match change points
                matches, unmatched_pred, unmatched_true = match_cps_one_to_one(
                    true_cps, detected_cps, tol_value
                )
                
                tp_total += len(matches)
                fp_total += len(unmatched_pred)
                fn_total += len(unmatched_true)
                num_detected_total += len(np.unique(np.asarray(detected_cps, dtype=int)))
                num_true_total += len(np.unique(np.asarray(true_cps, dtype=int)))
                has_points = True
        
        if not has_points:
            continue
        
        # Calculate precision and recall
        precision_value = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else np.nan
        recall_value = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        
        all_results.append({
            'saturation_level': saturation_level,
            'saturation_pct': SATURATION[saturation_level],
            'threshold': threshold,
            'tolerance_type': tol_name,
            'tolerance_value': tol_value,
            'TP': tp_total,
            'FP': fp_total,
            'FN': fn_total,
            'precision': precision_value,
            'recall': recall_value,
            'num_detected': num_detected_total,
            'num_true': num_true_total,
        })
    
    if not all_results:
        print(f"  No results found for saturation level {saturation_level}")
        return None
    
    results_df = pd.DataFrame(all_results).sort_values('threshold').reset_index(drop=True)
    print(f"  Aggregated to {len(results_df)} threshold points")
    
    return results_df


def compare_saturation_levels(results_base, data_base, cpd_dict, output_folder,
                              saturation_levels=None, window_size=100):
    """Compare performance across saturation levels.
    
    Args:
        results_base: Base folder containing results
        data_base: Base folder containing data files
        cpd_dict: Dictionary with true change points
        output_folder: Output folder for results and plots
        saturation_levels: List of saturation levels to evaluate
        window_size: Window size used for CPD
    
    Returns:
        DataFrame with combined results
    """
    print("=" * 80)
    print("Evaluating Centromere CPD Performance")
    print("=" * 80)
    
    os.makedirs(output_folder, exist_ok=True)
    
    if saturation_levels is None:
        saturation_levels = list(range(8))
    
    all_level_results = []
    
    # Evaluate each saturation level
    for sat_level in saturation_levels:
        sat_results = evaluate_saturation_level(
            sat_level, results_base, data_base, cpd_dict, window_size
        )
        if sat_results is not None:
            all_level_results.append(sat_results)
    
    if not all_level_results:
        print("No results found!")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_level_results, ignore_index=True)
    
    # Save combined results
    output_csv = os.path.join(output_folder, "centromere_cpd_results.csv")
    combined_df.to_csv(output_csv, index=False)
    print(f"\nSaved combined results to {output_csv}")
    
    # Create precision-recall curves
    print("\n" + "=" * 80)
    print("Creating Precision-Recall Curves")
    print("=" * 80)
    
    tol_name = "full_window"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for saturation levels
    color_list = [
        COLORS['black'], COLORS['blue'], COLORS['orange'], COLORS['green'],
        COLORS['red'], COLORS['pink'], COLORS['light_blue'], COLORS['yellow'],
    ]
    
    for idx, sat_level in enumerate(saturation_levels):
        sat_data = combined_df[
            (combined_df['saturation_level'] == sat_level) &
            (combined_df['tolerance_type'] == tol_name)
        ].sort_values('threshold')
        
        if len(sat_data) == 0:
            continue
        
        # Filter out NaN precision values for plotting
        plot_data = sat_data[~sat_data['precision'].isna()]
        
        if len(plot_data) == 0:
            continue
        
        # Get saturation percentage
        sat_pct = SATURATION[sat_level] if sat_level < len(SATURATION) else None
        label = f'Saturation: {sat_pct:.1f}%' if sat_pct is not None else f'Level {sat_level}'
        
        ax.plot(
            plot_data['recall'],
            plot_data['precision'],
            marker='o',
            linewidth=2,
            markersize=4,
            label=label,
            color=color_list[idx % len(color_list)],
        )
    
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(
        f'Precision-Recall Curves - Centromere CPD\n(Tolerance: {tol_name}, Window: {window_size}bp)',
        fontsize=16,
        fontweight='bold',
    )
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    plot_path = os.path.join(output_folder, f'pr_curve_centromere_cpd_{tol_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    
    return combined_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate centromere CPD performance across saturation levels."
    )
    parser.add_argument(
        "--method",
        choices=['gaussian_ae', 'gaussian_ma', 'zinb', 'custom'],
        default='zinb',
        help="CPD method to evaluate (sets default paths). Use 'custom' to specify paths manually.",
    )
    parser.add_argument(
        "--results_base",
        default=None,
        help="Base folder containing CPD results (overrides method default)",
    )
    parser.add_argument(
        "--data_base",
        default="Data/test_CPD/centromere_windows",
        help="Base folder containing centromere window data files",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        help="Output folder for evaluation results and plots (overrides method default)",
    )
    parser.add_argument(
        "--saturation_levels",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Saturation levels to evaluate",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size used in CPD",
    )
    
    args = parser.parse_args()
    
    # Set defaults based on method if not explicitly provided
    if args.method != 'custom':
        if args.method in METHOD_DEFAULTS:
            if args.results_base is None:
                args.results_base = METHOD_DEFAULTS[args.method]['results']
            if args.output_folder is None:
                args.output_folder = METHOD_DEFAULTS[args.method]['output']
        else:
            print(f"Warning: Unknown method '{args.method}', using custom mode")
    
    # Ensure paths are set
    if args.results_base is None:
        parser.error("--results_base must be specified when using --method=custom")
    if args.output_folder is None:
        parser.error("--output_folder must be specified when using --method=custom")
    
    return args


def main():
    """Main execution function."""
    args = parse_args()
    
    print(f"Method: {args.method}")
    print(f"Results base: {args.results_base}")
    print(f"Data base: {args.data_base}")
    print(f"Output folder: {args.output_folder}")
    print(f"Saturation levels: {args.saturation_levels}")
    print(f"Window size: {args.window_size}")
    
    results_df = compare_saturation_levels(
        args.results_base,
        args.data_base,
        CPD,
        args.output_folder,
        saturation_levels=args.saturation_levels,
        window_size=args.window_size,
    )
    
    if results_df is not None:
        print("\n" + "=" * 80)
        print("SUCCESS: Evaluation Complete!")
        print(f"Results saved to: {args.output_folder}")
        print("=" * 80)
    else:
        print("\nNo evaluation results generated!")
        sys.exit(1)


if __name__ == "__main__":
    main()
