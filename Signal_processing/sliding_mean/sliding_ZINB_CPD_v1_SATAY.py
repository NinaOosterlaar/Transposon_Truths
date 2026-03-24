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


def sliding_ZINB_CPD_v3(
    data,
    nucleosome_distances,
    centromere_distances,
    window_size,
    overlap,
    threshold,
    eps=1e-10,
    theta_global=None,
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

    if theta_global is None or theta_global <= 0:
        theta_global = initialize_theta_global(data, eps=eps)

    print(theta_global)

    change_points, scores = [], []
    last_cp, last_score = -np.inf, 0.0

    for start in range(0, n - 2 * window_size + 1, step_size):
        w1 = data[start : start + window_size]
        w2 = data[start + window_size : start + 2 * window_size]

        middle0 = start + window_size
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

        pi1 = np.clip(pi0 * (temp1_nucl / max(temp0_nucl, eps)), eps, 1 - eps)
        pi2 = np.clip(pi0 * (temp2_nucl / max(temp0_nucl, eps)), eps, 1 - eps)

        # Alternative model: separate mu per window
        mu1 = np.clip(np.mean(w1) / max(1 - pi1, eps), eps, None)
        mu2 = np.clip(np.mean(w2) / max(1 - pi2, eps), eps, None)

        # Null model: shared mu0 across both windows, but keep pi1 and pi2 fixed
        sum_y = np.sum(w1) + np.sum(w2)
        denom = window_size * (1 - pi1) + window_size * (1 - pi2)
        mu0 = np.clip(sum_y / max(denom, eps), eps, None)

        # Likelihoods
        ll0_w1 = zinb_log_likelihood(w1, mu0, theta_global, pi1, eps=eps)
        ll0_w2 = zinb_log_likelihood(w2, mu0, theta_global, pi2, eps=eps)
        ll0 = ll0_w1 + ll0_w2

        ll1 = zinb_log_likelihood(w1, mu1, theta_global, pi1, eps=eps)
        ll2 = zinb_log_likelihood(w2, mu2, theta_global, pi2, eps=eps)
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
    output_folder,
    dataset_name,
    nucleosome_file,
    centromere_file,
):
    """Process all thresholds for a given window size."""
    window_output_folder = os.path.join(output_folder, f"window{ws}")
    for threshold in thresholds:
        print(f"Processing window size: {ws}, threshold: {threshold:.2f}")
        change_points, scores = sliding_ZINB_CPD_v3(
            data,
            nucleosome_distances,
            centromere_distances,
            ws,
            overlap,
            threshold,
            theta_global=theta_global,
            nucleosome_file=nucleosome_file,
            centromere_file=centromere_file,
        )
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
    return data, nucleosome_distance, centromere_distance, value_col, nucleosome_col, centromere_col


def run_dataset(
    input_file,
    output_folder,
    dataset_name,
    window_sizes,
    overlap,
    thresholds,
    n_workers,
    theta_global,
    nucleosome_file,
    centromere_file,
):
    input_file = resolve_path(input_file)
    output_folder = resolve_path(output_folder)
    nucleosome_file = resolve_path(nucleosome_file)
    centromere_file = resolve_path(centromere_file)

    data, nucleosome_distance, centromere_distance, value_col, nuc_col, cent_col = read_input_data(input_file)
    print(f"Loaded {len(data)} rows from: {input_file}")
    print(f"Using columns -> value: {value_col}, nucleosome: {nuc_col}, centromere: {cent_col}")

    if theta_global == 0:
        theta_global = initialize_theta_global(data)
    print(f"Using global theta: {theta_global:.4f}")

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
        nucleosome_file=nucleosome_file,
        centromere_file=centromere_file,
    )


if __name__ == "__main__":
    main()