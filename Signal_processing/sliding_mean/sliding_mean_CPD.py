import numpy as np
import os, sys
import matplotlib.pyplot as plt
import argparse


"""
Var(X_i-X_i-1) = Var(X_i) + Var(X_i-1) - 2Cov(X_i, X_i-1)
2Cov(X_i, X_i-1) = 2*rho*sigma_overlap^2/N^2
Var(X_i) = sigma_i^2/N
Var(X_i-X_i-1) = 2*sigma_i-1^2/N *
"""

def MAD(x):
    """Compute the Median Absolute Deviation of the data."""
    median = np.median(x)
    mad = np.median(np.abs(x - median)) 
    return mad

def sigma_MAD(x):
    """Convert MAD to standard deviation estimate."""
    # dx = np.diff(x)
    mad = MAD(x)
    if mad < 1e-10:
        return np.std(x) 
    return mad * 1.4826   # For normal distribution

def sliding_mean_CPD(data, window_size, overlap, threshold):
    step_size = int(window_size * (1 - overlap))
    n = len(data)
    change_points = []
    
    means = []
    sigmas = []
    prev_mean = None
    prev_sigma = None
    last_cp = -np.inf
    last_z_score = 0

    for start in range(0, n - window_size + 1, step_size):
        window = data[start:start + window_size]
        window_mean = np.mean(window)
        window_sigma = sigma_MAD(window)
        means.append(window_mean)
        sigmas.append(window_sigma)
        
        if prev_mean is not None:
            sigma_diff = np.sqrt((prev_sigma ** 2 + window_sigma ** 2) / window_size)
            if sigma_diff < 1e-10:
                sigma_diff = 1e-10  # Prevent division by zero
            z_score = np.abs(window_mean - prev_mean) / sigma_diff  # Avoid division by zero
            if z_score > threshold:
                if (start - last_cp) >= window_size:
                    change_points.append(start)  # Mark the center of the window as change point
                    last_cp = start
                    last_z_score = z_score
                elif z_score > last_z_score:
                    last_z_score = z_score
                    change_points[-1] = start  # Update to the new position if z_score is higher
                    
        prev_mean = window_mean
        prev_sigma = window_sigma
    
    return change_points, means, sigmas

def save_results(output_folder, dataset_name, change_points, means, sigmas, window_size, overlap, threshold):  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f"{dataset_name}_ws{window_size}_ov{int(overlap*100)}_th{threshold:.2f}.txt")  
    with open(output_file, "w") as f:
        for cp in change_points:
            f.write(f"{cp} \n")
        f.write(f"means: {means}\n")
        f.write(f"sigmas: {sigmas}\n")
        f.write(f"window_size: {window_size}, overlap: {overlap}, threshold: {threshold}\n")
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply a sliding window mean change point detection algorithm on discrete count data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing the count data.")
    parser.add_argument("--output_folder", type=str, default="Signal_processing/results/sliding_mean/sliding_mean_CPD", help="Output folder for results.")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the dataset being processed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    window_size = [10, 30, 50, 80]
    overlap = 0.5
    thresholds = np.concatenate([
        np.linspace(0.5, 2, 8),   # Low thresholds for high recall
        np.linspace(2, 6, 15),     # Your current range
        np.linspace(6, 10, 8)      # High thresholds for high precision
    ])
    output_folder = args.output_folder + f"/{args.dataset_name}"
    dataset_name = args.dataset_name
    
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
            change_points, means, sigma = sliding_mean_CPD(data, ws, overlap, threshold)
            save_results(window_output_folder, dataset_name, change_points, means, sigma, ws, overlap, threshold)
    
    
