import numpy as np
import os, sys
import matplotlib.pyplot as plt
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Signal_processing.ZINB_MLE.log_likelihoods import nb_log_likelihood 

def fit_global_theta(data, eps=1e-10, theta_max=1e8):
    x = np.asarray(data, dtype=np.float64)
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0

    if mu < eps:
        return theta_max  # all ~0 counts

    if var <= mu + eps:
        return theta_max  # ~Poisson / underdispersed -> very large theta

    theta = (mu * mu) / (var - mu)
    return float(np.clip(theta, eps, theta_max))

def sliding_NB_CPD(data, window_size, overlap, threshold, theta_global=None, eps=1e-10):
    """
    GLR sliding CPD for Negative Binomial with global theta (size).
    Compares adjacent windows of length window_size.

    Returns:
      change_points (list of indices at the boundary between windows),
      scores (list of GLR scores for each tested boundary)  [optional but useful]
    """
    data = np.asarray(data, dtype=np.float64)
    step_size = max(1, int(window_size * (1 - overlap)))
    n = len(data)

    if theta_global is None:
        theta_global = fit_global_theta(data, eps=eps)

    change_points = []
    scores = []

    last_cp = -np.inf
    last_score = 0.0

    # need 2*window_size points to form w1 and w2
    for start in range(0, n - 2 * window_size + 1, step_size):
        w1 = data[start : start + window_size]
        w2 = data[start + step_size : start + step_size + window_size]
        w0 = data[start : start + step_size + window_size]

        mu1 = float(np.mean(w1))
        mu2 = float(np.mean(w2))
        mu0 = float(np.mean(w0))

        ll1 = nb_log_likelihood(w1, mu1, theta_global, eps=eps)
        ll2 = nb_log_likelihood(w2, mu2, theta_global, eps=eps)
        ll0 = nb_log_likelihood(w0, mu0, theta_global, eps=eps)

        # Likelihood ratio test
        glr = 2.0 * ((ll1 + ll2) - ll0)
        scores.append(glr)

        cp_loc = start + window_size  # boundary between w1 and w2

        if glr > threshold:
            # de-dup logic (same spirit as your original)
            if (cp_loc - last_cp) >= window_size:
                change_points.append(cp_loc)
                last_cp = cp_loc
                last_score = glr
            elif glr > last_score:
                last_score = glr
                change_points[-1] = cp_loc

    return change_points, scores, theta_global


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
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply a sliding window mean change point detection algorithm on discrete count data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing the count data.")
    parser.add_argument("--output_folder", type=str, default="Signal_processing/results/sliding_mean/sliding_NB_CPD", help="Output folder for results.")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the dataset being processed.")
    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_arguments()
    input_file = "Signal_processing/sample_data/SATAY_synthetic/SATAY_with_pi.csv"
    window_size = [10, 30, 50, 80]
    overlap = 0.5
    thresholds = np.linspace(0, 40, 41)
    print(thresholds)
    output_folder = "Signal_processing/results/sliding_mean/sliding_NB_CPD"
    dataset_name = "SATAY_synthetic"
    # Add dataset name to output folder path
    output_folder = os.path.join(output_folder, dataset_name)
    
    # Read data
    # datasets = read_csv_file_with_distances(input_file)
    with open(input_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = [int(line.strip().split(",")[1]) for line in lines]
    for ws in window_size:
        # Create a subfolder for each window size
        window_output_folder = os.path.join(output_folder, f"window{ws}")
        for threshold in thresholds:
            print(f"Processing window size: {ws}, threshold: {threshold:.2f}")
            change_points, scores, theta_global = sliding_NB_CPD(data, ws, overlap, threshold)
            save_results(window_output_folder, dataset_name, change_points, scores, theta_global, ws, overlap, threshold)
    





