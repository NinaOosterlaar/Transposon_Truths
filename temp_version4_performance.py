"""
Temporary script to compute performance metrics for version4 results on dataset 1.
Reads pre-computed changepoint detection results and evaluates them against ground truth.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import auc
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from Utils.plot_config import COLORS, setup_plot_style

setup_plot_style()

# Paths
DATA_FOLDER = "Signal_processing/final/SATAY_synthetic/1"
RESULTS_FOLDER = "Signal_processing/results/version4/1/window100"
PARAM_FILE = os.path.join(DATA_FOLDER, "SATAY_without_pi_params.csv")

# Parameters
WINDOW_SIZE = 100  # Used for tolerance in matching


def precision_recall_one_to_one(detected_cps, true_cps, tol):
    """Calculate precision and recall with one-to-one greedy matching."""
    detected_cps = np.array(detected_cps)
    true_cps = np.array(true_cps)

    if len(detected_cps) == 0:
        return 0.0, 0.0
    if len(true_cps) == 0:
        return 0.0, 0.0

    matched_true = set()
    matched_detected = set()

    # Find all possible matches within tolerance
    pairs = []
    for i, det_cp in enumerate(detected_cps):
        for j, true_cp in enumerate(true_cps):
            dist = abs(det_cp - true_cp)
            if dist <= tol:
                pairs.append((i, j, dist))

    # Greedy matching: prefer closer matches
    pairs.sort(key=lambda x: x[2])

    true_positives = 0
    for det_idx, true_idx, _ in pairs:
        if det_idx not in matched_detected and true_idx not in matched_true:
            matched_detected.add(det_idx)
            matched_true.add(true_idx)
            true_positives += 1

    precision = true_positives / len(detected_cps) if len(detected_cps) > 0 else 0.0
    recall = true_positives / len(true_cps) if len(true_cps) > 0 else 0.0
    return precision, recall


def read_true_change_points(param_file):
    """Read true change points from SATAY_without_pi_params.csv."""
    df = pd.read_csv(param_file)
    if "region_start" not in df.columns:
        raise ValueError(f"Missing 'region_start' column in {param_file}")
    # First row is the first region, so we skip it for change points
    return df["region_start"].values[1:].astype(int).tolist()


def read_detected_change_points(result_file):
    """Read detected change points from a result file."""
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # Each line contains one detected change point, until we hit metadata lines
    change_points = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Stop when we hit metadata lines (scores:, theta_global:, window_size:, etc.)
        if ':' in line:
            break
        try:
            change_points.append(int(float(line)))
        except ValueError:
            # Skip any lines that can't be parsed as numbers
            continue
    return change_points


def extract_threshold_from_filename(filename):
    """Extract threshold value from filename like 'dataset_1_ws100_ov50_th5.00.txt'."""
    # Find 'th' and extract the number after it
    parts = filename.split('_')
    for part in parts:
        if part.startswith('th'):
            threshold_str = part[2:].replace('.txt', '')
            return float(threshold_str)
    raise ValueError(f"Could not extract threshold from filename: {filename}")


def plot_precision_recall(df, pr_auc, output_path):
    """Plot precision-recall curve."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the PR curve
    ax.plot(
        df['recall'].values,
        df['precision'].values,
        'o-',
        linewidth=2,
        markersize=6,
        alpha=0.85,
        color=COLORS['red'],
        label=f'Version 4 (AUC={pr_auc:.3f})'
    )
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - Version 4, Dataset 1')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")



def main():
    print("="*60)
    print("Performance Evaluation for Version4 - Dataset 1")
    print("="*60)
    
    # Read true change points
    print(f"\nReading true change points from: {PARAM_FILE}")
    true_cps = read_true_change_points(PARAM_FILE)
    print(f"Found {len(true_cps)} true change points: {true_cps}")
    
    # Get all result files
    result_files = sorted([f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.txt')])
    print(f"\nFound {len(result_files)} result files")
    
    # Evaluate each threshold
    results = []
    for result_file in result_files:
        threshold = extract_threshold_from_filename(result_file)
        result_path = os.path.join(RESULTS_FOLDER, result_file)
        
        detected_cps = read_detected_change_points(result_path)
        precision, recall = precision_recall_one_to_one(detected_cps, true_cps, WINDOW_SIZE)
        
        results.append({
            'threshold': threshold,
            'num_detected': len(detected_cps),
            'num_true': len(true_cps),
            'precision': precision,
            'recall': recall,
        })
        
        print(f"  Threshold {threshold:5.2f}: Detected={len(detected_cps):3d}, "
              f"Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Create DataFrame and compute AUC
    df = pd.DataFrame(results).sort_values('threshold')
    
    # Compute PR AUC
    recalls = df['recall'].values
    precisions = df['precision'].values
    sort_idx = np.argsort(recalls)
    pr_auc = auc(recalls[sort_idx], precisions[sort_idx])
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"\nPrecision-Recall Curve:")
    print(df[['threshold', 'recall', 'precision', 'num_detected']].to_string(index=False))
    
    # Generate plot
    plot_file = "temp_version4_dataset1_pr_curve.png"
    plot_precision_recall(df, pr_auc, plot_file)
    
    # Save results
    output_file = "temp_version4_dataset1_performance.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save summary
    summary_file = "temp_version4_dataset1_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Version4 Performance - Dataset 1\n")
        f.write("="*60 + "\n")
        f.write(f"True change points: {len(true_cps)}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n")
        f.write("\n" + df[['threshold', 'recall', 'precision', 'num_detected']].to_string(index=False))
    # print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
