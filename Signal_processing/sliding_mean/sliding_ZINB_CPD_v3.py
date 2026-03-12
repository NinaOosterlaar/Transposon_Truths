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

    if theta_global is None:
        theta_global = initialize_theta_global(data, eps=eps)

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
    
def initialize_theta_global(data, eps=1e-10, theta_max=1000):
    results = estimate_zinb(data, eps=eps)
    theta_global = results['theta']
    print(f"Estimated global theta: {theta_global:.4f}")
    print(f"(Estimated global pi: {results['pi']:.4f}, mu: {results['mu']:.4f})")
    if theta_global >= theta_max:
        # Throw an error that the estimation of theta failed
        raise ValueError("Estimated global theta is very large, indicating a failure in estimation. ")
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
        change_points, scores = sliding_ZINB_CPD(data, nucleosome_distances, centromere_distances, ws, overlap, threshold, theta_global=theta_global, nucleosome_file=nucleosome_file, centromere_file=centromere_file)
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
    # nucleosome_file = "Data_exploration/results/densities/nucleosome_new/combined_All_Boolean_True/ALL_combined_Boolean_True_nucleosome_density.csv"
    nucleosome_file = "Signal_processing/sample_data/SATAY_synthetic/density_vs_distance_nucleosome_density.csv" # Change this when using real data 
    centromere_file = "Signal_processing/sample_data/SATAY_synthetic/density_vs_distance_centromere_density.csv" # Change this when using real data
    window_size = [80, 10, 30, 50]
    overlap = 0.5
    thresholds = np.linspace(0, 40, 41)  # 41 thresholds from 0 to 40
    print(thresholds)
    output_folder = args.output_folder
    dataset_name = args.dataset_name
    # Add dataset name to output folder path
    output_folder = os.path.join(output_folder, dataset_name)
    n_workers = args.n_workers
    theta_global = args.theta_global
    
    # CHROMS = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI"]
    # for chrom in CHROMS:
    #     chrom = f"Chr{chrom}"
    #     print(f"Processing chromosome: {chrom}")
    #     input_file = f"Signal_processing/sample_data/Centromere_region/{chrom}_centromere_window.csv"
    #     window_size = [50, 80]
    #     overlap = 0.5
    #     thresholds = np.linspace(0, 40, 41)  
    #     print(thresholds)
    #     dataset_name = f"{chrom}_centromere_window"
    #     output_folder = f"Signal_processing/temp/{chrom}"
    #     n_workers = 1
    #     theta_global = 0
            
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Read data
    # datasets = read_csv_file_with_distances(input_file)
    with open(input_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = [int(float(line.strip().split(",")[1])) for line in lines]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        nucleosome_distance = [int(float(line.strip().split(",")[3])) for line in lines] # CHANGE THIS WHEN USING REAL DATA TO 2 INSTEAD OF 3!!!!!!!!!!!
        centromere_distance = [int(float(line.strip().split(",")[2])) for line in lines] # CHANGE THIS WHEN USING REAL DATA TO 3 INSTEAD OF 2!!!!!!!!!!!
    if theta_global == 0:
        theta_global = initialize_theta_global(data)
    print(f"Using global theta: {theta_global:.4f} for all window sizes and thresholds.")
    
    # Parallelize processing of different window sizes
    n_workers = min(n_workers, len(window_size))
    print(f"Using {n_workers} workers to process {len(window_size)} window sizes in parallel")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_window_size, ws, data, nucleosome_distance, centromere_distance, overlap, thresholds, theta_global, output_folder, dataset_name, nucleosome_file = nucleosome_file, centromere_file = centromere_file)
            for ws in window_size
        ]
        for future in futures:
            ws_completed = future.result()
            print(f"Completed processing window size: {ws_completed}")
        
