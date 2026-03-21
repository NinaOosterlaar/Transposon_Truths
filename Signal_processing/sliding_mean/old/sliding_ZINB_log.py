import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Signal_processing.log_likelihoods import zinb_log_likelihood


def load_density_lookup_tables(nucleosome_file, centromere_file):
    """Load the density lookup tables from CSV files.
    
    Returns:
        nucleosome_df: DataFrame with 'distance' and 'mean_density' columns
        centromere_df: DataFrame with 'Bin_Center' and 'mean_density' columns
    """
    nucleosome_df = pd.read_csv(nucleosome_file)
    centromere_df = pd.read_csv(centromere_file)
    return nucleosome_df, centromere_df


def interpolate_density(distance, lookup_df, distance_col, density_col='NonZero_Density'):
    """Interpolate density value for a given distance using linear interpolation.
    
    Args:
        distance: The distance value to interpolate for
        lookup_df: DataFrame containing distance and mean_density columns
        distance_col: Name of the distance column 
        density_col: Name of the density column (default: 'mean_density')
    
    Returns:
        Interpolated mean_density value
    
    Raises:
        ValueError: If distance is outside the range of available data
    """
    distances = lookup_df[distance_col].values
    densities = lookup_df[density_col].values
    
    # Check bounds
    if distance < distances.min() or distance > distances.max():
        raise ValueError(f"Distance {distance} is outside the range [{distances.min()}, {distances.max()}]")
    
    # Use numpy's interp for linear interpolation
    return np.interp(distance, distances, densities)

def sliding_ZINB_CPD_v3(data, nucleosome_distances, centromere_distances, window_size, overlap, threshold, eps=1e-10, theta_global=None, tol=1e-6, max_iter=10, nucleosome_file="Data_exploration/results/densities/nucleosome_new/combined_All_Boolean_True/ALL_combined_Boolean_True_nucleosome_density.csv", centromere_file="Data_exploration/results/densities/centromere_new/combined_All_Boolean_True/ALL_combined_Boolean_True_centromere_density.csv"):
    data = np.asarray(data, dtype=np.float64)
    step_size = max(1, int(window_size * (1 - overlap)))
    n = len(data)
    max_nucl_distance = np.max(np.array([nucleosome_distances]))
    nucleosome_df, centromere_df = load_density_lookup_tables(nucleosome_file, centromere_file)
    # Create a lookup table for distance to mean density for nucleosomes
    distance_to_density = nucleosome_df.set_index('Nucleosome_Distance_Bin')['NonZero_Density']
    # fill up all the missing values up until max_nucl_distance with a mean density of 0
    distance_to_density = distance_to_density.reindex(range(max_nucl_distance + 1), fill_value=0)
    # Create a lookup table for distance to mean density for centromeres
    centromere_distance_to_density = centromere_df.set_index('Centromere_Distance_Bin')['NonZero_Density']
    # fill up all the missing values up until max_centromere_distance with a mean density of 0

    if theta_global is None or theta_global <= 0:
        theta_global = initialize_theta_global(data, eps=eps)
        
    print(theta_global)

    change_points, scores = [], []
    last_cp, last_score = -np.inf, 0.0

    for start in range(0, n - 2 * window_size + 1, step_size):
        w1 = data[start : start + window_size]
        w2 = data[start + window_size : start + 2 * window_size]
        w0 = data[start : start + 2 * window_size]  # exactly w1+w2
        
        middle0 = start + window_size  # middle point between w1 and w2
        centr_dist_middle = centromere_distances[middle0]
        if centr_dist_middle > centromere_distance_to_density.index.max():
            centr_dist_middle = centromere_distance_to_density.index.max()

        # For the middle point of centromere, find the corresponding centromere distance, and interpolate from the centromere density lookup table to get the mean density for that distance, which we will use as pi0 for the ZINB model
        pi0 = interpolate_density(centr_dist_middle, centromere_distance_to_density.reset_index(), 'Centromere_Distance_Bin', 'NonZero_Density')
        
        # Compute the mean densities for each window, adjusting for the zero-inflation using pi0. We divide the mean by (1-pi0) to get the mean of the non-zero part of the distribution, which is what we use for the log-likelihood calculation. We also clip the values to avoid issues with zero or negative means.
        mu1 = np.clip(np.mean(w1) / (1 - pi0), eps, None)
        mu2 = np.clip(np.mean(w2) / (1 - pi0), eps, None)
        mu0 = np.clip(np.mean(w0) / (1 - pi0), eps, None)
        

        # For each position in the window find the corresponding nucleosome distance and density, and compute the average density for the window
        nucl_dist0 = nucleosome_distances[start : start + 2 * window_size]
        nucl_dist1 = nucleosome_distances[start : start + window_size]
        nucl_dist2 = nucleosome_distances[start + window_size : start + 2 * window_size]


        temp0_nucl = distance_to_density.loc[nucl_dist0].mean()
        temp1_nucl = distance_to_density.loc[nucl_dist1].mean()
        temp2_nucl = distance_to_density.loc[nucl_dist2].mean()

        pi1 = np.clip(pi0 * (temp1_nucl / temp0_nucl), eps, 1 - eps)
        pi2 = np.clip(pi0 * (temp2_nucl / temp0_nucl), eps, 1 - eps)
        mu1 = np.clip(np.mean(w1) / max(1 - pi1, eps), eps, None)
        mu2 = np.clip(np.mean(w2) / max(1 - pi2, eps), eps, None)
        
        # print(f"Window {start}-{start+2*window_size}: pi0={pi0:.4f}, mu0={mu0:.4f}, pi1={pi1:.4f}, mu1={mu1:.4f}, pi2={pi2:.4f}, mu2={mu2:.4f}")

        ll1 = zinb_log_likelihood(w1, mu1, theta_global, pi1, eps=eps)
        ll2 = zinb_log_likelihood(w2, mu2, theta_global, pi2, eps=eps)
        ll0 = zinb_log_likelihood(w0, mu0, theta_global, pi0, eps=eps)

        score = 2.0 * ((ll1 + ll2) - ll0)
        scores.append(score)

        if score > threshold:
            cp = start + window_size
            if (cp - last_cp) >= window_size:
                change_points.append(cp)
                last_cp, last_score = cp, score
            elif score > last_score:
                change_points[-1] = cp
                last_cp, last_score = cp, score

    return change_points, scores
    
def initialize_theta_global(data, eps=1e-10, theta_max=10):
    results = estimate_zinb(data, eps=eps, theta_max = theta_max)
    theta_global = results['theta']
    print(f"Estimated global theta: {theta_global:.4f}")
    print(f"(Estimated global pi: {results['pi']:.4f}, mu: {results['mu']:.4f})")
    return theta_global

def save_results(output_folder, dataset_name, change_points, scores, theta_global, window_size, overlap, threshold):  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f"{dataset_name}_ws{window_size}_ov{int(overlap*100)}_th{threshold:.2f}.txt")  
    with open(output_file, "w") as f:
        for cp in change_points:
            f.write(f"{cp} \n")
        f.write(f"scores: {scores}\n")
        f.write(f"theta_global: {theta_global}\n")
        f.write(f"window_size: {window_size}, overlap: {overlap}, threshold: {threshold}\n")
    

def process_window_size(ws, data, nucleosome_distances, centromere_distances, overlap, thresholds, theta_global, output_folder, dataset_name, nucleosome_file, centromere_file):
    """Process all thresholds for a given window size."""
    window_output_folder = os.path.join(output_folder, f"window{ws}")
    for threshold in thresholds:
        print(f"Processing window size: {ws}, threshold: {threshold:.2f}")
        change_points, scores = sliding_ZINB_CPD_v3(data, nucleosome_distances, centromere_distances, ws, overlap, threshold, theta_global=theta_global, nucleosome_file=nucleosome_file, centromere_file=centromere_file)
        save_results(window_output_folder, dataset_name, change_points, scores, theta_global, ws, overlap, threshold)
    return ws


def precision_recall_one_to_one(detected_cps, true_cps, tol):
    """Calculate precision and recall with one-to-one greedy matching."""
    detected_cps = np.asarray(detected_cps, dtype=int)
    true_cps = np.asarray(true_cps, dtype=int)

    if len(detected_cps) == 0 or len(true_cps) == 0:
        return 0.0, 0.0

    matched_true = set()
    matched_detected = set()

    pairs = []
    for i, det_cp in enumerate(detected_cps):
        for j, true_cp in enumerate(true_cps):
            dist = abs(det_cp - true_cp)
            if dist <= tol:
                pairs.append((i, j, dist))

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
    """Read true CPs from SATAY_without_pi_params.csv."""
    params_df = pd.read_csv(param_file)
    if "region_start" not in params_df.columns:
        raise ValueError(f"Missing 'region_start' column in {param_file}")
    return params_df["region_start"].values[1:].astype(int).tolist()


def parse_change_points_file(result_file):
    """Parse detected change points from one result file."""
    change_points = []
    with open(result_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("scores:") or stripped.startswith("theta_global:") or stripped.startswith("window_size:"):
                break

            token = stripped.split()[0]
            try:
                cp = int(float(token))
            except ValueError:
                continue
            change_points.append(cp)

    return change_points


def extract_threshold_from_filename(filename):
    """Extract threshold from file name ending with _thX.XX.txt."""
    match = re.search(r"_th([0-9]+(?:\.[0-9]+)?)\.txt$", filename)
    if not match:
        return None
    return float(match.group(1))


def collect_change_points_by_threshold(results_root, dataset_num, window_size, overlap):
    """Collect {threshold: change_points} for one dataset output folder."""
    window_folder = os.path.join(results_root, str(dataset_num), f"window{window_size}")
    if not os.path.isdir(window_folder):
        return {}

    expected_prefix = f"dataset_{dataset_num}_ws{window_size}_ov{int(overlap * 100)}_th"
    threshold_map = {}

    for filename in os.listdir(window_folder):
        if not filename.startswith(expected_prefix) or not filename.endswith(".txt"):
            continue

        threshold = extract_threshold_from_filename(filename)
        if threshold is None:
            continue

        result_file = os.path.join(window_folder, filename)
        threshold_map[threshold] = parse_change_points_file(result_file)

    return threshold_map


def build_precision_recall_results(base_data_folder, original_results_folder, log_results_folder, dataset_numbers, window_size, overlap):
    """Build detailed precision/recall rows for original and log1p outputs."""
    rows = []

    for dataset_num in dataset_numbers:
        dataset_folder = os.path.join(base_data_folder, str(dataset_num))
        param_file = os.path.join(dataset_folder, "SATAY_without_pi_params.csv")
        if not os.path.exists(param_file):
            print(f"Warning: Missing parameter file for dataset {dataset_num}: {param_file}")
            continue

        true_cps = read_true_change_points(param_file)

        original_threshold_map = collect_change_points_by_threshold(
            original_results_folder,
            dataset_num,
            window_size,
            overlap,
        )
        log_threshold_map = collect_change_points_by_threshold(
            log_results_folder,
            dataset_num,
            window_size,
            overlap,
        )

        common_thresholds = sorted(set(original_threshold_map).intersection(log_threshold_map))
        if not common_thresholds:
            print(
                f"Warning: No overlapping thresholds for dataset {dataset_num} "
                f"between {original_results_folder} and {log_results_folder}."
            )
            continue

        for threshold in common_thresholds:
            precision_original, recall_original = precision_recall_one_to_one(
                original_threshold_map[threshold],
                true_cps,
                window_size,
            )
            precision_log, recall_log = precision_recall_one_to_one(
                log_threshold_map[threshold],
                true_cps,
                window_size,
            )

            rows.append(
                {
                    "dataset_id": int(dataset_num),
                    "method": "original",
                    "threshold": float(threshold),
                    "precision": float(precision_original),
                    "recall": float(recall_original),
                    "num_detected": int(len(original_threshold_map[threshold])),
                    "num_true": int(len(true_cps)),
                }
            )
            rows.append(
                {
                    "dataset_id": int(dataset_num),
                    "method": "log1p",
                    "threshold": float(threshold),
                    "precision": float(precision_log),
                    "recall": float(recall_log),
                    "num_detected": int(len(log_threshold_map[threshold])),
                    "num_true": int(len(true_cps)),
                }
            )

    columns = [
        "dataset_id",
        "method",
        "threshold",
        "precision",
        "recall",
        "num_detected",
        "num_true",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dataset_id", "method", "threshold"]
    ).reset_index(drop=True)


def aggregate_precision_recall_curves(results_df):
    """Aggregate mean/std precision and recall per method and threshold."""
    grouped = results_df.groupby(["method", "threshold"])
    agg = grouped.agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        n_datasets=("dataset_id", "nunique"),
    ).reset_index()

    agg["precision_std"] = agg["precision_std"].fillna(0.0)
    agg["recall_std"] = agg["recall_std"].fillna(0.0)
    return agg


def _curve_auc(curve_df):
    """Compute PR AUC from a curve dataframe with recall/precision mean columns."""
    if curve_df.empty:
        return np.nan

    recall = curve_df["recall_mean"].values
    precision = curve_df["precision_mean"].values
    sort_idx = np.argsort(recall)
    return float(np.trapz(precision[sort_idx], recall[sort_idx]))


def plot_precision_recall_original_vs_log(agg_curve_df, output_path):
    """Plot original vs log1p precision-recall curves."""
    method_style = {
        "original": {"label": "Original", "color": "#1f77b4"},
        "log1p": {"label": "Log1p", "color": "#d62728"},
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for method in ["original", "log1p"]:
        method_curve = agg_curve_df[agg_curve_df["method"] == method]
        if method_curve.empty:
            continue

        recall = method_curve["recall_mean"].values
        precision = method_curve["precision_mean"].values
        sort_idx = np.argsort(recall)

        auc_value = np.trapz(precision[sort_idx], recall[sort_idx])
        label = f"{method_style[method]['label']} (AUC={auc_value:.3f})"

        ax.plot(
            recall[sort_idx],
            precision[sort_idx],
            "o-",
            linewidth=2,
            markersize=4,
            alpha=0.9,
            color=method_style[method]["color"],
            label=label,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall: Original vs Log1p (mean over datasets)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_precision_recall_comparison(base_data_folder, original_results_folder, log_results_folder, dataset_numbers, window_size, overlap):
    """Create and save original-vs-log precision-recall comparison artifacts."""
    results_df = build_precision_recall_results(
        base_data_folder,
        original_results_folder,
        log_results_folder,
        dataset_numbers,
        window_size,
        overlap,
    )

    if results_df.empty:
        print("No precision-recall rows were generated; skipping PR plot creation.")
        return

    output_folder = os.path.join(log_results_folder, "precision_recall")
    os.makedirs(output_folder, exist_ok=True)

    detailed_csv = os.path.join(output_folder, "original_vs_log_pr_detailed.csv")
    results_df.to_csv(detailed_csv, index=False)
    print(f"Saved detailed PR rows to: {detailed_csv}")

    agg_curve_df = aggregate_precision_recall_curves(results_df)
    agg_csv = os.path.join(output_folder, "original_vs_log_pr_aggregated.csv")
    agg_curve_df.to_csv(agg_csv, index=False)
    print(f"Saved aggregated PR stats to: {agg_csv}")

    plot_path = os.path.join(output_folder, "original_vs_log_precision_recall.png")
    plot_precision_recall_original_vs_log(agg_curve_df, plot_path)
    print(f"Saved PR comparison plot to: {plot_path}")

    for method in ["original", "log1p"]:
        method_curve = agg_curve_df[agg_curve_df["method"] == method]
        if method_curve.empty:
            continue
        auc_value = _curve_auc(method_curve)
        print(f"{method} mean PR AUC: {auc_value:.4f}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply a sliding window mean change point detection algorithm on discrete count data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing the count data.")
    parser.add_argument("--output_folder", type=str, default="Signal_processing/results/sliding_mean/sliding_ZINB_CPD", help="Output folder for results.")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the dataset being processed.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers/CPUs to use.")
    parser.add_argument("--theta_global", type=float, default=0, help="Global theta value to use for all windows (if not provided, it will be estimated from the data).")
    return parser.parse_args()


if __name__ == "__main__":
    # Configuration
    base_data_folder = "Signal_processing/final/SATAY_synthetic"
    base_original_results_folder = "Signal_processing/results/version3"
    base_output_folder = os.path.join(base_original_results_folder, "log")
    
    window_size = [100]
    overlap = 0.5
    thresholds = np.linspace(0, 15, 16)  # 16 thresholds from 0 to 15
    
    print(f"Thresholds: {thresholds}")
    print(f"Processing datasets 1-10 from {base_data_folder}")
    
    # Process each dataset (1-10)
    for dataset_num in range(1, 11):
        print(f"\n{'='*60}")
        print(f"Processing dataset {dataset_num}")
        print(f"{'='*60}")
        
        # Build input file path and density file paths for this dataset
        dataset_folder = os.path.join(base_data_folder, str(dataset_num))
        input_file = os.path.join(dataset_folder, "SATAY_with_pi.csv")
        nucleosome_file = os.path.join(dataset_folder, "density_vs_distance_nucleosome_density.csv")
        centromere_file = os.path.join(dataset_folder, "density_vs_distance_centromere_density.csv")
        
        # Check if files exist
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found: {input_file}")
            continue
        if not os.path.exists(nucleosome_file):
            print(f"Warning: Nucleosome density file not found: {nucleosome_file}")
            continue
        if not os.path.exists(centromere_file):
            print(f"Warning: Centromere density file not found: {centromere_file}")
            continue
        
        # Build output folder path
        output_folder = os.path.join(base_output_folder, str(dataset_num))
        dataset_name = f"dataset_{dataset_num}"
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Read data from CSV
        with open(input_file, "r") as f:
            lines = f.readlines()[1:]  # Skip header
            raw_data = np.array([float(line.strip().split(",")[1]) for line in lines], dtype=np.float64)
            # Column indices: Position(0), Value(1), Centromere_distance(2), Nucleosome_distance(3)
            nucleosome_distance = [int(float(line.strip().split(",")[3])) for line in lines]
            centromere_distance = [int(float(line.strip().split(",")[2])) for line in lines]

        data = np.log1p(raw_data)
        
        print(f"Loaded {len(data)} data points")
        print("Applied log1p transform to the Value column")
        
        # Estimate theta globally for this dataset
        theta_global = initialize_theta_global(data, )
        print(f"Using global theta: {theta_global:.4f} for all window sizes and thresholds.")
        
        # Process different window sizes (currently just one: 100)
        n_workers = min(1, len(window_size))
        print(f"Using {n_workers} worker(s) to process {len(window_size)} window size(s) in parallel")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_window_size, ws, data, nucleosome_distance, centromere_distance, overlap, thresholds, theta_global, output_folder, dataset_name, nucleosome_file=nucleosome_file, centromere_file=centromere_file)
                for ws in window_size
            ]
            for future in futures:
                ws_completed = future.result()
                print(f"Completed processing window size: {ws_completed}")
        
        print(f"Finished processing dataset {dataset_num}")
    
    print(f"\n{'='*60}")
    print("All log1p datasets processed!")
    print(f"Log1p results saved to: {base_output_folder}")
    print(f"{'='*60}")

    print("Building precision-recall comparison between original and log1p outputs...")
    create_precision_recall_comparison(
        base_data_folder=base_data_folder,
        original_results_folder=base_original_results_folder,
        log_results_folder=base_output_folder,
        dataset_numbers=range(1, 11),
        window_size=window_size[0],
        overlap=overlap,
    )

    print(f"\n{'='*60}")
    print("Precision-recall comparison complete")
    print(f"{'='*60}")
        
