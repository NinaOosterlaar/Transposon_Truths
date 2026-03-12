import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.plot_config import setup_plot_style, COLORS
from Utils.reader import read_csv_file_with_distances

# Set up standardized plot style
setup_plot_style()

def fisher_z(r):
    # guard against exactly +/-1
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def inv_fisher_z(z):
    return np.tanh(z)

def weighted_average_acf(acf_list, w_list):
    """
    Combine many ACF curves into one using weighted Fisher-z averaging.

    Args:
        acf_list: list of arrays, each shape (max_lag,)
        w_list:   list of arrays, same shapes, weights (#pairs per lag)

    Returns:
        acf_avg: array (max_lag,)
        w_sum:   array (max_lag,) total weight used per lag
    """
    acf_stack = np.vstack(acf_list)      # (M, L)
    w_stack   = np.vstack(w_list).astype(float)

    # ignore NaNs and nonpositive weights
    valid = np.isfinite(acf_stack) & (w_stack > 0)

    z = fisher_z(acf_stack)
    z[~valid] = 0.0
    w_stack[~valid] = 0.0

    w_sum = w_stack.sum(axis=0)
    z_avg = np.divide((w_stack * z).sum(axis=0), w_sum, out=np.full_like(w_sum, np.nan), where=w_sum>0)

    return inv_fisher_z(z_avg), w_sum

def stabilize(x):
    """Apply a numerically stable log(1 + x) transformation."""
    return np.log1p(x)

def correlation(x1, x2, mean):
    """Compute the correlation between two signals."""
    x1_centered = x1 - mean
    x2_centered = x2 - mean
    numerator = np.sum(x1_centered * x2_centered)
    denominator = np.sqrt(np.sum(x1_centered ** 2) * np.sum(x2_centered ** 2)) + 1e-10
    return numerator / denominator

def acf_zeros_as_values(x, max_lag, transform=True, use_global_mean=True):
    """
    Autocorrelation where zeros are treated as valid values.

    Args:
        x (array-like): 1D raw count signal.
        max_lag (int): maximum lag (in indices) to compute.
        transform (bool): whether to apply log1p stabilization.
        use_global_mean (bool): if True, use mean over full series; else mean over each lagged overlap.

    Returns:
        acf (np.ndarray): autocorrelation for lags 1..max_lag
        n_pairs (np.ndarray): number of pairs used at each lag (here always N-k)
    """
    x = np.asarray(x, dtype=float)

    if transform:
        x = stabilize(x)

    # Global mean (textbook-ish)
    global_mean = np.mean(x)

    acf = np.empty(max_lag, dtype=float)
    n_pairs = np.empty(max_lag, dtype=int)

    for k in range(1, max_lag + 1):
        x1 = x[:-k]
        x2 = x[k:]
        mean = global_mean if use_global_mean else np.mean(np.concatenate([x1, x2]))
        acf[k - 1] = correlation(x1, x2, mean)
        n_pairs[k - 1] = len(x1)  # equals n-k

    return acf, n_pairs

def acf_zeros_as_missing(x, max_lag, transform=True, use_global_mean=True):
    """
    Autocorrelation where zeros are treated as missing values.

    Args:
        x (array-like): 1D raw count signal.
        max_lag (int): maximum lag (in indices) to compute.
        transform (bool): whether to apply log1p stabilization.
        use_global_mean (bool): if True, use mean over observed (non-zero) values
                                globally; else use mean over each lagged overlap.

    Returns:
        acf (np.ndarray): autocorrelation for lags 1..max_lag
        n_pairs (np.ndarray): number of valid pairs used at each lag
    """
    x = np.asarray(x, dtype=float)

    # Treat zeros as missing
    x[x == 0] = np.nan

    if transform:
        x = stabilize(x)

    n = len(x)
    max_lag = min(max_lag, n - 1)

    # Global mean over observed values only
    global_mean = np.nanmean(x)

    acf = np.empty(max_lag, dtype=float)
    n_pairs = np.empty(max_lag, dtype=int)

    for k in range(1, max_lag + 1):
        x1 = x[:-k]
        x2 = x[k:]

        # Keep only pairs where both are observed
        mask = (~np.isnan(x1)) & (~np.isnan(x2))
        n_pairs[k - 1] = mask.sum()

        if n_pairs[k - 1] < 2:
            acf[k - 1] = np.nan
            continue

        x1_valid = x1[mask]
        x2_valid = x2[mask]

        mean = global_mean if use_global_mean else np.mean(
            np.concatenate([x1_valid, x2_valid])
        )

        acf[k - 1] = correlation(x1_valid, x2_valid, mean)

    return acf, n_pairs

def process_datasets(folder_name, max_lag = 1000, zeros = True, output_folder="Signal_processing/results/autocorrelation", plot=True):
    acf_list = []
    w_list = []
    output_folder += f"/zeros_{zeros}"

    datasets = read_csv_file_with_distances(folder_name)
    for dataset_name in datasets:
        chroms = datasets[dataset_name]
        for chrom in chroms:
            print(f"Processing dataset: {chrom}")
            x = datasets[dataset_name][chrom]["Value"].values
            if zeros:
                acf, n_pairs = acf_zeros_as_values(x, max_lag=max_lag, transform=True)
            else:
                acf, n_pairs = acf_zeros_as_missing(x, max_lag=max_lag, transform=True)
            acf_list.append(acf)
            w_list.append(n_pairs)
    acf_avg, w_sum = weighted_average_acf(acf_list, w_list)
    
    # Save results to CSV
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "average_acf.csv")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Lag", "Average_ACF", "Total_Contributing_Pairs"])
        for lag in range(1, len(acf_avg) + 1):
            writer.writerow([lag, acf_avg[lag - 1], w_sum[lag - 1]])

    if plot:
        plot_acf_with_weights(acf_avg, w_sum, title="Average ACF across datasets", output_folder=output_folder)

    return acf_avg, w_sum

def plot_acf_with_weights(acf_avg, w_sum, bin_size_bp=None, title="Average ACF", output_folder="Signal_processing/results/autocorrelation"):
    lags = np.arange(1, len(acf_avg) + 1)

    # Optionally convert lags to basepairs
    x = lags if bin_size_bp is None else lags * bin_size_bp
    xlab = "Lag (positions)" if bin_size_bp is None else "Lag (bp)"

    # --- Plot ACF ---
    plt.figure()
    plt.plot(x, acf_avg)
    # plt.axhline(0, linewidth=1)
    plt.xlabel(xlab)
    plt.ylabel("Autocorrelation")
    plt.title(title)
    plt.locator_params(axis="x", nbins=8)
    plt.grid(True, which="major", ls="--")
    plt.savefig(os.path.join(output_folder, "average_acf.png"))

    # --- Plot total contributing pairs (reliability) ---
    # round w_sum 
    w_sum = np.round(w_sum) 
    plt.figure()
    plt.plot(x, w_sum)
    plt.yscale("log") 
    plt.xlabel(xlab)
    plt.ylabel("Total contributing pairs")
    plt.locator_params(axis="x", nbins=8)
    plt.grid(True, which="major", ls="--")
    plt.savefig(os.path.join(output_folder, "average_acf_pair_counts.png"))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process datasets to compute average autocorrelation function (ACF).")
    parser.add_argument("folder_name", type=str, help="Path to the folder containing datasets.")
    parser.add_argument("--max_lag", type=int, default=1000, help="Maximum lag (in indices) to compute.")
    parser.add_argument("--zeros", action="store_true", help="Treat zeros as valid values.")
    parser.add_argument("--output_folder", type=str, default="Signal_processing/results/autocorrelation", help="Output folder for results.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    process_datasets(args.folder_name, max_lag=args.max_lag, zeros=args.zeros, output_folder=args.output_folder, plot=args.plot)
    