import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Signal_processing.ZINB_MLE.EM import em_zinb_step
from Signal_processing.log_likelihoods import zinb_log_likelihood


def load_density_lookup_tables(nucleosome_file):
    """Load the density lookup table from CSV."""
    nucleosome_df = pd.read_csv(nucleosome_file)
    return nucleosome_df


def sliding_ZINB_CPD_v2(
    data,
    nucleosome_distances,
    window_size,
    overlap,
    threshold,
    eps=1e-10,
    theta_global=None,
    tol=1e-6,
    max_iter=10,
    nucleosome_file="Data_exploration/results/densities/nucleosome_new/combined_All_Boolean_True/ALL_combined_Boolean_True_nucleosome_density.csv",
):
    data = np.asarray(data, dtype=np.float64)
    step_size = max(1, int(window_size * (1 - overlap)))
    n = len(data)
    max_nucl_distance = np.max(np.asarray(nucleosome_distances))

    nucleosome_df = load_density_lookup_tables(nucleosome_file)
    distance_to_density = nucleosome_df.set_index("Nucleosome_Distance_Bin")["NonZero_Density"]
    distance_to_density = distance_to_density.reindex(range(max_nucl_distance + 1), fill_value=0)

    if theta_global is None or theta_global <= 0:
        theta_global = initialize_theta_global(data, eps=eps)

    print(theta_global)

    change_points, scores = [], []
    last_cp, last_score = -np.inf, 0.0

    for start in range(0, n - 2 * window_size + 1, step_size):
        w1 = data[start : start + window_size]
        w2 = data[start + window_size : start + 2 * window_size]
        w0 = data[start : start + 2 * window_size]

        # Initial pi0 and mu0 for EM on the combined window
        pi0 = np.clip(np.mean(w0 == 0), eps, 1 - eps)
        mu0_em = np.clip(np.mean(w0) / max(1 - pi0, eps), eps, None)

        for _ in range(max_iter):
            pi0_old, mu0_old = pi0, mu0_em
            r = em_zinb_step(w0, pi0, mu0_em, theta_global, eps=eps)
            pi0, mu0_em = r["pi"], r["mu"]
            if abs(pi0 - pi0_old) < tol and abs(mu0_em - mu0_old) < tol:
                break

        # Nucleosome-based scaling
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

        # Alternative model: separate mu for each window
        mu1 = np.clip(np.mean(w1) / max(1 - pi1, eps), eps, None)
        mu2 = np.clip(np.mean(w2) / max(1 - pi2, eps), eps, None)

        # Null model: shared mu0 across both windows, but keep pi1 and pi2 fixed
        sum_y = np.sum(w1) + np.sum(w2)
        denom = window_size * (1 - pi1) + window_size * (1 - pi2)
        mu0 = np.clip(sum_y / max(denom, eps), eps, None)

        # Likelihoods under null model
        ll0_w1 = zinb_log_likelihood(w1, mu0, theta_global, pi1, eps=eps)
        ll0_w2 = zinb_log_likelihood(w2, mu0, theta_global, pi2, eps=eps)
        ll0 = ll0_w1 + ll0_w2

        # Likelihoods under alternative model
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
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
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


def process_window_size(ws, data, nucleosome_distances, overlap, thresholds, theta_global, output_folder, dataset_name, nucleosome_file):
    """Process all thresholds for a given window size."""
    window_output_folder = os.path.join(output_folder, f"window{ws}")
    for threshold in thresholds:
        print(f"Processing window size: {ws}, threshold: {threshold:.2f}")
        change_points, scores = sliding_ZINB_CPD_v2(
            data,
            nucleosome_distances,
            ws,
            overlap,
            threshold,
            theta_global=theta_global,
            nucleosome_file=nucleosome_file,
        )
        save_results(window_output_folder, dataset_name, change_points, scores, theta_global, ws, overlap, threshold)
    return ws


def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply a sliding window mean change point detection algorithm on discrete count data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing the count data.")
    parser.add_argument("--output_folder", type=str, default="Signal_processing/results/sliding_mean/sliding_ZINB_CPD", help="Output folder for results.")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the dataset being processed.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers/CPUs to use.")
    parser.add_argument("--theta_global", type=float, default=0, help="Global theta value to use for all windows (if not provided, it will be estimated from the data).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    nucleosome_file = "Signal_processing/sample_data/SATAY_synthetic/density_vs_distance_nucleosome_density.csv"  # Change this when using real data
    window_size = [80, 10, 30, 50]
    overlap = 0.5
    thresholds = np.linspace(0, 40, 41)
    print(thresholds)

    output_folder = args.output_folder
    dataset_name = args.dataset_name
    output_folder = os.path.join(output_folder, dataset_name)
    n_workers = args.n_workers
    theta_global = args.theta_global

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = [int(float(line.strip().split(",")[1])) for line in lines]
        # CHANGE THIS WHEN USING REAL DATA TO 2 INSTEAD OF 3
        nucleosome_distance = [int(float(line.strip().split(",")[3])) for line in lines]

    if theta_global == 0:
        theta_global = initialize_theta_global(data)
    print(f"Using global theta: {theta_global:.4f} for all window sizes and thresholds.")

    n_workers = min(n_workers, len(window_size))
    print(f"Using {n_workers} workers to process {len(window_size)} window sizes in parallel")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                process_window_size,
                ws,
                data,
                nucleosome_distance,
                overlap,
                thresholds,
                theta_global,
                output_folder,
                dataset_name,
                nucleosome_file=nucleosome_file,
            )
            for ws in window_size
        ]
        for future in futures:
            ws_completed = future.result()
            print(f"Completed processing window size: {ws_completed}")