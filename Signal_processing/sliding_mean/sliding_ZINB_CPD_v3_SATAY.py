import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Signal_processing.log_likelihoods import zinb_log_likelihood


DEFAULT_CHROMS = [
    "ChrI",
    "ChrII",
    "ChrIII",
    "ChrIV",
    "ChrV",
    "ChrVI",
    "ChrVII",
    "ChrVIII",
    "ChrIX",
    "ChrX",
    "ChrXI",
    "ChrXII",
    "ChrXIII",
    "ChrXIV",
    "ChrXV",
    "ChrXVI",
]

DEFAULT_INPUT_FILE_REL = "Signal_processing/sample_data/Centromere_region/ChrI_centromere_window.csv"
DEFAULT_CENTROMERE_REGION_DIR_REL = "Signal_processing/sample_data/Centromere_region"
# Default output path matches retrieve_pred_from_cpd.py expectations.
DEFAULT_OUTPUT_FOLDER_REL = "Signal_processing/results/sliding_mean_SATAY/sliding_ZINB_CPD"
DEFAULT_NUCLEOSOME_FILE_REL = "Data_exploration/results/densities/nucleosome_strains/combined_Datasets_Boolean_True/dataset-strain_yEK23_combined_Boolean_True_nucleosome_density.csv"
DEFAULT_CENTROMERE_FILE_REL = "Data_exploration/results/densities/centromere_strains/combined_Datasets_Boolean_True_bin_10000/dataset-strain_yEK23_combined_centromere_density_Boolean_True_bin_10000.csv"


def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_density_lookup_tables(nucleosome_file, centromere_file):
    """Load the density lookup tables from CSV files."""
    nucleosome_df = pd.read_csv(resolve_path(nucleosome_file))
    centromere_df = pd.read_csv(resolve_path(centromere_file))
    return nucleosome_df, centromere_df


def interpolate_density(distance, lookup_df, distance_col, density_col="mean_density"):
    """Interpolate a density value for a given distance."""
    distances = lookup_df[distance_col].values
    densities = lookup_df[density_col].values

    # Clamp to lookup range to avoid edge-window failures.
    distance = np.clip(distance, distances.min(), distances.max())
    return float(np.interp(distance, distances, densities))


def estimate_local_theta_blocks(data, block_size=2000, eps=1e-10, theta_max=1000):
    """
    Estimate theta for non-overlapping blocks of data.
    
    Args:
        data: Count data array
        block_size: Size of each block in base pairs (default: 2000)
        eps: Small value for numerical stability
        theta_max: Maximum allowed theta value
        
    Returns:
        theta_blocks: List of (start, end, theta) tuples for each block
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    theta_blocks = []
    
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block_data = data[start:end]
        
        # Skip empty blocks
        if len(block_data) == 0:
            continue
        
        try:
            results = estimate_zinb(block_data, eps=eps)
            theta = results["theta"]
            
            # Sanity check
            if theta >= theta_max or theta <= 0:
                print(f"    Warning: Block [{start}:{end}] has invalid theta={theta:.4f}, using fallback")
                # Use chromosome-wide estimate as fallback
                theta = None
        except Exception as e:
            print(f"    Warning: Block [{start}:{end}] theta estimation failed: {e}")
            theta = None
        
        theta_blocks.append((start, end, theta))
    
    return theta_blocks


def get_theta_for_position(position, theta_blocks, theta_fallback):
    """
    Get the appropriate theta value for a given position.
    
    Args:
        position: Position in the chromosome
        theta_blocks: List of (start, end, theta) tuples
        theta_fallback: Fallback theta if block theta is None
        
    Returns:
        theta: The theta value for this position
    """
    for start, end, theta in theta_blocks:
        if start <= position < end:
            return theta if theta is not None else theta_fallback
    
    # If position not found (shouldn't happen), use fallback
    return theta_fallback


def sliding_ZINB_CPD_v3(
    data,
    nucleosome_distances,
    centromere_distances,
    window_size,
    overlap,
    threshold,
    eps=1e-10,
    theta_global=None,
    theta_blocks=None,
    tol=1e-6,
    max_iter=10,
    nucleosome_file="Data_exploration/results/densities/nucleosome_new/combined_All_Boolean_True/ALL_combined_Boolean_True_nucleosome_density.csv",
    centromere_file="Data_exploration/results/densities/centromere_new/combined_All_Boolean_True/ALL_combined_Boolean_True_centromere_density.csv"
):
    data = np.asarray(data, dtype=np.float64)
    step_size = max(1, int(window_size * (1 - overlap)))
    n = len(data)
    max_nucl_distance = np.max(np.array([nucleosome_distances]))
    nucleosome_df, centromere_df = load_density_lookup_tables(nucleosome_file, centromere_file)

    # Create a lookup table for distance to mean density for nucleosomes
    distance_to_density = nucleosome_df.set_index('distance')['mean_density']
    distance_to_density = distance_to_density.reindex(range(max_nucl_distance + 1), fill_value=0)

    # Create a lookup table for distance to mean density for centromeres
    centromere_distance_to_density = centromere_df.set_index('Bin_Center')['mean_density']

    # Initialize theta (global fallback if using local blocks)
    if theta_global is None or theta_global <= 0:
        theta_global = initialize_theta_global(data, eps=eps)

    # Determine if using local or global theta
    use_local_theta = theta_blocks is not None
    if not use_local_theta:
        print(f"Using global theta: {theta_global:.4f}")

    change_points, scores = [], []
    last_cp, last_score = -np.inf, 0.0

    for start in range(0, n - 2 * window_size + 1, step_size):
        w1 = data[start : start + window_size]
        w2 = data[start + window_size : start + 2 * window_size]

        middle0 = start + window_size
        
        # Get theta for this window's center position
        if use_local_theta:
            theta = get_theta_for_position(middle0, theta_blocks, theta_global)
        else:
            theta = theta_global
        
        centr_dist_middle = centromere_distances[middle0]
        if centr_dist_middle > centromere_distance_to_density.index.max():
            centr_dist_middle = centromere_distance_to_density.index.max()

        # Version 3: pi0 from centromere-dependent saturation/density
        pi0 = interpolate_density(
            centr_dist_middle,
            centromere_distance_to_density.reset_index(),
            'Bin_Center',
            'mean_density'
        )

        # Nucleosome-based scaling for pi1 and pi2
        nucl_dist0 = nucleosome_distances[start : start + 2 * window_size]
        nucl_dist1 = nucleosome_distances[start : start + window_size]
        nucl_dist2 = nucleosome_distances[start + window_size : start + 2 * window_size]

        temp0_nucl = distance_to_density.loc[nucl_dist0].mean()
        temp1_nucl = distance_to_density.loc[nucl_dist1].mean()
        temp2_nucl = distance_to_density.loc[nucl_dist2].mean()
        
        s0 = 1 - pi0

        s1 = np.clip(s0 * (temp1_nucl / max(temp0_nucl, eps)), eps, 1 - eps)
        s2 = np.clip(s0 * (temp2_nucl / max(temp0_nucl, eps)), eps, 1 - eps)

        pi1 = 1 - s1
        pi2 = 1 - s2

        pi1 = np.clip(pi0 * (temp1_nucl / max(temp0_nucl, eps)), eps, 1 - eps)
        pi2 = np.clip(pi0 * (temp2_nucl / max(temp0_nucl, eps)), eps, 1 - eps)

        # Alternative model: separate mu per window
        mu1 = np.clip(np.mean(w1) / max(1 - pi1, eps), eps, None)
        mu2 = np.clip(np.mean(w2) / max(1 - pi2, eps), eps, None)

        # Null model: shared mu0 across both windows, but keep pi1 and pi2 fixed
        sum_y = np.sum(w1) + np.sum(w2)
        denom = window_size * (1 - pi1) + window_size * (1 - pi2)
        mu0 = np.clip(sum_y / max(denom, eps), eps, None)

        # Likelihoods (using local theta for this window position)
        ll0_w1 = zinb_log_likelihood(w1, mu0, theta, pi1, eps=eps)
        ll0_w2 = zinb_log_likelihood(w2, mu0, theta, pi2, eps=eps)
        ll0 = ll0_w1 + ll0_w2

        ll1 = zinb_log_likelihood(w1, mu1, theta, pi1, eps=eps)
        ll2 = zinb_log_likelihood(w2, mu2, theta, pi2, eps=eps)
        ll_alt = ll1 + ll2

        score = 2.0 * (ll_alt - ll0)
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


def apply_threshold_to_scores(scores, threshold, window_size, overlap, step_size):
    """
    Apply a threshold to pre-computed LRT scores to identify change points.
    
    This function allows applying multiple thresholds to the same set of scores
    without re-computing the statistical test.
    
    Args:
        scores: List of LRT scores from sliding_ZINB_CPD_v3
        threshold: LRT score threshold for detecting change points
        window_size: Size of sliding window (used for minimum distance between CPs)
        overlap: Overlap fraction between windows (0 to 1)
        step_size: Step size between windows
        
    Returns:
        change_points: List of detected change point positions
    """
    change_points = []
    last_cp, last_score = -np.inf, 0.0
    
    for idx, score in enumerate(scores):
        if score > threshold:
            start = idx * step_size
            cp = start + window_size
            
            if (cp - last_cp) >= window_size:
                change_points.append(cp)
                last_cp, last_score = cp, score
            elif score > last_score:
                change_points[-1] = cp
                last_cp, last_score = cp, score
    
    return change_points


def remove_top_quantile_outliers(data, quantile=0.99, threshold=None):
    """
    Remove outliers by capping non-zero values above the specified quantile.
    
    Args:
        data: List or array of count values
        quantile: Values above this quantile will be capped (default: 0.99 for top 1%)
        threshold: Pre-computed threshold value (if None, computed from non-zero data)
        
    Returns:
        filtered_data: Data with outliers capped at the threshold value
        threshold: The threshold value used
        n_affected: Number of values that were capped
    """
    data_array = np.asarray(data, dtype=np.float64)
    
    if threshold is None:
        # Compute threshold only on non-zero values
        non_zero_data = data_array[data_array > 0]
        if len(non_zero_data) > 0:
            threshold = np.quantile(non_zero_data, quantile)
        else:
            threshold = 0.0
    
    # Count how many values exceed the threshold (excluding zeros)
    n_affected = np.sum(data_array > threshold)
    
    # Cap values at the threshold (but keep zeros as zeros)
    filtered_data = np.clip(data_array, None, threshold)
    
    return filtered_data.tolist(), threshold, n_affected


def initialize_theta_global(data, eps=1e-10, theta_max=1000):
    results = estimate_zinb(data, eps=eps)
    theta_global = results["theta"]
    print(f"Estimated global theta: {theta_global:.4f}")
    print(f"(Estimated global pi: {results['pi']:.4f}, mu: {results['mu']:.4f})")
    if theta_global >= theta_max:
        raise ValueError("Estimated global theta is very large, indicating a failure in estimation.")
    return theta_global


def save_results(output_folder, dataset_name, change_points, scores, theta_global, window_size, overlap, threshold):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(
        output_folder,
        f"{dataset_name}_ws{window_size}_ov{int(overlap*100)}_th{threshold:.2f}.txt",
    )
    with open(output_file, "w") as f:
        for cp in change_points:
            f.write(f"{cp} \n")
        f.write(f"scores: {scores}\n")
        f.write(f"theta_global: {theta_global}\n")
        f.write(f"window_size: {window_size}, overlap: {overlap}, threshold: {threshold}\n")


def process_window_size(
    ws,
    data,
    nucleosome_distances,
    centromere_distances,
    overlap,
    thresholds,
    theta_global,
    theta_blocks,
    output_folder,
    dataset_name,
    nucleosome_file,
    centromere_file,
):
    """
    Process all thresholds for a given window size.
    
    OPTIMIZED: Computes LRT scores once, then applies all thresholds.
    This is ~40x faster than running the detector separately for each threshold.
    """
    window_output_folder = os.path.join(output_folder, f"window{ws}")
    step_size = max(1, int(ws * (1 - overlap)))
    
    # Compute scores ONCE with threshold=0 (no filtering)
    print(f"Processing window size: {ws} (computing scores once for {len(thresholds)} thresholds)")
    change_points_dummy, scores = sliding_ZINB_CPD_v3(
        data,
        nucleosome_distances,
        centromere_distances,
        ws,
        overlap,
        threshold=0,  # No filtering, get all scores
        theta_global=theta_global,
        theta_blocks=theta_blocks,
        nucleosome_file=nucleosome_file,
        centromere_file=centromere_file,
    )
    
    # Now apply each threshold to the pre-computed scores
    for threshold in thresholds:
        print(f"  Applying threshold: {threshold:.2f}")
        change_points = apply_threshold_to_scores(scores, threshold, ws, overlap, step_size)
        save_results(
            window_output_folder,
            dataset_name,
            change_points,
            scores,
            theta_global,
            ws,
            overlap,
            threshold,
        )
    
    return ws


def _pick_column_by_keyword(df, keyword, fallback_index):
    keyword = keyword.lower()
    for col in df.columns:
        if keyword in col.lower():
            return col

    if 0 <= fallback_index < len(df.columns):
        return df.columns[fallback_index]

    raise ValueError(
        f"Could not find a '{keyword}' column and fallback index {fallback_index} is out of range."
    )


def read_input_data(input_file):
    input_file = resolve_path(input_file)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    df = pd.read_csv(input_file)
    if df.empty:
        raise ValueError(f"Input file is empty: {input_file}")

    value_col = _pick_column_by_keyword(df, "value", 1)
    nucleosome_col = _pick_column_by_keyword(df, "nucleosome", 2)
    centromere_col = _pick_column_by_keyword(df, "centromere", 3)

    data = [int(float(v)) for v in df[value_col].values]
    nucleosome_distance = [int(float(v)) for v in df[nucleosome_col].values]
    centromere_distance = [int(float(v)) for v in df[centromere_col].values]
    
    # Extract chromosome name from filename if possible
    # Assumes format: path/to/ChrXV_distances.csv
    chrom_name = None
    basename = os.path.basename(input_file)
    if basename.startswith("Chr") and "_distances" in basename:
        chrom_name = basename.split("_distances")[0]
    
    return data, nucleosome_distance, centromere_distance, value_col, nucleosome_col, centromere_col, chrom_name


def remove_problematic_positions(data, chrom_name):
    """
    Remove known problematic positions by setting them to zero.
    
    Args:
        data: List or array of count values
        chrom_name: Name of chromosome (e.g., 'ChrXV'), or None if unknown
    
    Returns:
        cleaned_data: Data with problematic positions set to zero
        n_removed: Number of positions removed
    """
    if chrom_name is None:
        return data, 0
    
    data_cleaned = list(data)  # Make a copy
    n_removed = 0
    
    # Known problematic positions: (chromosome, position)
    problematic_positions = [
        ('ChrXV', 565596),
    ]
    
    for prob_chrom, prob_pos in problematic_positions:
        if chrom_name == prob_chrom and prob_pos < len(data_cleaned):
            if data_cleaned[prob_pos] != 0:
                data_cleaned[prob_pos] = 0
                n_removed += 1
    
    return data_cleaned, n_removed


def run_dataset(
    input_file,
    output_folder,
    dataset_name,
    window_sizes,
    overlap,
    thresholds,
    n_workers,
    theta_global,
    theta_block_size,
    outlier_threshold,
    nucleosome_file,
    centromere_file,
):
    input_file = resolve_path(input_file)
    output_folder = resolve_path(output_folder)
    nucleosome_file = resolve_path(nucleosome_file)
    centromere_file = resolve_path(centromere_file)

    data, nucleosome_distance, centromere_distance, value_col, nuc_col, cent_col, chrom_name = read_input_data(input_file)
    print(f"Loaded {len(data)} rows from: {input_file}")
    print(f"Using columns -> value: {value_col}, nucleosome: {nuc_col}, centromere: {cent_col}")
    
    # Remove problematic positions
    data, n_problematic = remove_problematic_positions(data, chrom_name)
    if n_problematic > 0:
        print(f"Removed {n_problematic} problematic position(s) (set to zero)")
    
    # Remove top 1% outliers before CPD (using pre-computed threshold if provided, non-zero values only)
    data, actual_threshold, n_outliers = remove_top_quantile_outliers(data, quantile=0.99, threshold=outlier_threshold)
    if outlier_threshold is not None:
        print(f"Outlier removal: using global threshold {actual_threshold:.1f}")
    if n_outliers > 0:
        print(f"Outlier removal: capped {n_outliers} values ({100*n_outliers/len(data):.2f}%) above threshold {actual_threshold:.1f}")
    else:
        print(f"Outlier removal: no values exceeded threshold ({actual_threshold:.1f})")

    if theta_global == 0:
        theta_global = initialize_theta_global(data)
    print(f"Global theta (fallback): {theta_global:.4f}")
    
    # Estimate local theta blocks
    theta_blocks = None
    if theta_block_size > 0:
        print(f"\nEstimating local theta in {theta_block_size}bp blocks...")
        theta_blocks = estimate_local_theta_blocks(data, block_size=theta_block_size)
        
        # Print statistics
        valid_thetas = [theta for _, _, theta in theta_blocks if theta is not None]
        if valid_thetas:
            print(f"  Estimated {len(valid_thetas)}/{len(theta_blocks)} blocks successfully")
            print(f"  Theta range: [{min(valid_thetas):.4f}, {max(valid_thetas):.4f}]")
            print(f"  Theta mean: {np.mean(valid_thetas):.4f} ± {np.std(valid_thetas):.4f}")
            print(f"  Using local theta per window (fallback: {theta_global:.4f})")
        else:
            print(f"  Warning: No valid theta estimates, using global theta")
            theta_blocks = None
    else:
        print(f"Using global theta for all positions: {theta_global:.4f}")

    os.makedirs(output_folder, exist_ok=True)

    n_workers = max(1, min(n_workers, len(window_sizes)))
    print(f"Using {n_workers} workers to process {len(window_sizes)} window sizes")

    if n_workers == 1:
        for ws in window_sizes:
            process_window_size(
                ws,
                data,
                nucleosome_distance,
                centromere_distance,
                overlap,
                thresholds,
                theta_global,
                theta_blocks,
                output_folder,
                dataset_name,
                nucleosome_file,
                centromere_file,
            )
        return

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                process_window_size,
                ws,
                data,
                nucleosome_distance,
                centromere_distance,
                overlap,
                thresholds,
                theta_global,
                theta_blocks,
                output_folder,
                dataset_name,
                nucleosome_file,
                centromere_file,
            )
            for ws in window_sizes
        ]
        for future in futures:
            ws_completed = future.result()
            print(f"Completed processing window size: {ws_completed}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Apply a sliding-window ZINB change point detector on count data."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to input CSV containing value, nucleosome distance and centromere distance.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER_REL,
        help="Output folder for results.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset (defaults to input file stem).",
    )
    parser.add_argument(
        "--window_sizes",
        type=int,
        nargs="+",
        default=[100],
        help="Window sizes to evaluate.",
    )
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap in [0, 1).")
    parser.add_argument("--threshold_start", type=float, default=0.0, help="Minimum threshold.")
    parser.add_argument("--threshold_end", type=float, default=40.0, help="Maximum threshold.")
    parser.add_argument("--threshold_step", type=float, default=1.0, help="Threshold step size.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument(
        "--theta_global",
        type=float,
        default=0,
        help="Global theta value (0 means estimate from data).",
    )
    parser.add_argument(
        "--theta_block_size",
        type=int,
        default=2000,
        help="Block size for local theta estimation in bp (0 means use global theta only). Default: 2000",
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=None,
        help="Pre-computed outlier threshold value. If not provided, computed as 95th percentile of data.",
    )
    parser.add_argument(
        "--nucleosome_file",
        type=str,
        default=DEFAULT_NUCLEOSOME_FILE_REL,
        help="CSV lookup file with nucleosome density by distance.",
    )
    parser.add_argument(
        "--centromere_file",
        type=str,
        default=DEFAULT_CENTROMERE_FILE_REL,
        help="CSV lookup file with centromere density by distance.",
    )
    parser.add_argument(
        "--all_chromosomes",
        action="store_true",
        help="Process all sample chromosomes (ChrI..ChrXVI).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    process_all_chromosomes = args.all_chromosomes or args.input_file is None

    if args.threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")

    if process_all_chromosomes and not np.isclose(args.overlap, 0.5):
        print("For retrieve_pred_from_cpd.py compatibility, forcing overlap from")
        print(f"{args.overlap} to 0.5 so filenames use _ov50_.")
        args.overlap = 0.5

    thresholds = np.arange(
        args.threshold_start,
        args.threshold_end + (args.threshold_step / 2.0),
        args.threshold_step,
    )
    window_sizes = sorted(set(args.window_sizes))

    nucleosome_file = resolve_path(args.nucleosome_file)
    centromere_file = resolve_path(args.centromere_file)

    if not os.path.exists(nucleosome_file) or not os.path.exists(centromere_file):
        print("Density lookup file(s) not found yet.")
        print(f"Nucleosome file: {nucleosome_file}")
        print(f"Centromere file: {centromere_file}")
        print("Add the files later, or pass --nucleosome_file and --centromere_file.")
        return

    output_root = resolve_path(args.output_folder)

    if process_all_chromosomes:
        if args.dataset_name:
            print(
                "Ignoring --dataset_name in all-chromosome mode to preserve "
                "retrieve_pred_from_cpd.py naming compatibility."
            )

        for chrom in DEFAULT_CHROMS:
            input_file = os.path.join(
                PROJECT_ROOT,
                DEFAULT_CENTROMERE_REGION_DIR_REL,
                f"{chrom}_centromere_window.csv",
            )
            if not os.path.exists(input_file):
                print(f"Skipping {chrom}, input file missing: {input_file}")
                continue

            dataset_name = f"{chrom}_centromere_window"
            output_folder = os.path.join(output_root, chrom)

            print(f"\nProcessing chromosome: {chrom}")
            run_dataset(
                input_file=input_file,
                output_folder=output_folder,
                dataset_name=dataset_name,
                window_sizes=window_sizes,
                overlap=args.overlap,
                thresholds=thresholds,
                n_workers=args.n_workers,
                theta_global=args.theta_global,
                theta_block_size=args.theta_block_size,
                outlier_threshold=args.outlier_threshold,
                nucleosome_file=nucleosome_file,
                centromere_file=centromere_file,
            )
        return

    input_file = resolve_path(args.input_file)

    dataset_name = args.dataset_name or os.path.splitext(os.path.basename(input_file))[0]
    output_folder = os.path.join(output_root, dataset_name)

    run_dataset(
        input_file=input_file,
        output_folder=output_folder,
        dataset_name=dataset_name,
        window_sizes=window_sizes,
        overlap=args.overlap,
        thresholds=thresholds,
        n_workers=args.n_workers,
        theta_global=args.theta_global,
        theta_block_size=args.theta_block_size,
        outlier_threshold=args.outlier_threshold,
        nucleosome_file=nucleosome_file,
        centromere_file=centromere_file,
    )


if __name__ == "__main__":
    main()