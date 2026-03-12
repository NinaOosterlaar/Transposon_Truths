import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.plot_config import setup_plot_style, COLORS
# from trash.preprocessing import preprocess_data
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr, pearsonr

# Set up standardized plot style
setup_plot_style()

def aggregate_bins(signal, k=10):
    """Average every k bins."""
    L = len(signal)
    m = (L // k) * k
    return signal[:m].reshape(-1, k).mean(axis=1)

def build_dataset_vector(train_data, ds):
    """
    For one dataset `ds`, concatenate all valid bins (column 0)
    across all chromosomes and regions, in a deterministic order.

    Returns:
        vec (1D np.array of floats)
    """
    pieces = []

    # sort chromosomes for deterministic order (ChrI, ChrII, ...)
    for chrom in sorted(train_data[ds].keys(), key=lambda c: list(train_data[ds].keys()).index(c)):
        regions = train_data[ds][chrom]
        for region_dict in regions:
            data_sample = region_dict['data']
            L = region_dict['actual_length']
            if L == 0:
                continue
            signal = data_sample[:L, 0].astype(float)  # column 0 = logCPM signal
            signal = aggregate_bins(signal, k=100)
            pieces.append(signal)

    if len(pieces) == 0:
        return np.array([])

    return np.concatenate(pieces)

def compute_all_pairwise_binlevel_correlations_simple(train_data):
    """
    1. Build one genome-wide bin vector per dataset
    2. Do Spearman + Pearson for every dataset pair
    """
    datasets = list(train_data.keys())

    # precompute vectors once
    vectors = {
        ds: build_dataset_vector(train_data, ds)
        for ds in datasets
    }

    results = []
    for dsA, dsB in combinations(datasets, 2):
        vecA = vectors[dsA]
        vecB = vectors[dsB]

        # sanity check: if for some reason lengths diverge, truncate to min
        if len(vecA) != len(vecB):
            print(f"WARNING: dataset vectors {dsA} and {dsB} have different lengths ({len(vecA)} vs {len(vecB)}). Truncating to minimum length.")
            m = min(len(vecA), len(vecB))
            vecA = vecA[:m]
            vecB = vecB[:m]

        # remove any NaNs just in case 
        mask = np.isfinite(vecA) & np.isfinite(vecB)
        vecA = vecA[mask]
        vecB = vecB[mask]

        if len(vecA) < 2:
            spearman_r = np.nan
            spearman_p = np.nan
            pearson_r = np.nan
            pearson_p = np.nan
        else:
            spearman_r, spearman_p = spearmanr(vecA, vecB)
            pearson_r, pearson_p = pearsonr(vecA, vecB)

        results.append({
            "dataset_A": dsA,
            "dataset_B": dsB,
            "n_bins_compared": len(vecA),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        })

    return pd.DataFrame(results)


def plot_correlation_from_data(data_file):
    """Plot correlation heatmap from a CSV file containing correlation results."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv(data_file)

    # Pivot the DataFrame to create a matrix for heatmap
    spearman_matrix = df.pivot(index='dataset_A', columns='dataset_B', values='spearman_r')
    pearson_matrix = df.pivot(index='dataset_A', columns='dataset_B', values='pearson_r')
    
    # Make the matrices symmetric by filling in the lower triangle
    # The correlation matrix should be symmetric: corr(A, B) = corr(B, A)
    spearman_matrix = spearman_matrix.combine_first(spearman_matrix.T)
    pearson_matrix = pearson_matrix.combine_first(pearson_matrix.T)
    
    # Fill diagonal with 1.0 (perfect correlation with itself)
    np.fill_diagonal(spearman_matrix.values, 1.0)
    np.fill_diagonal(pearson_matrix.values, 1.0)

    # Plot Spearman correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_matrix, cmap='coolwarm', center=0, annot=False, 
                square=True, vmin=-1, vmax=1, cbar_kws={'label': 'Spearman Correlation'})
    plt.title('Spearman Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # Plot Pearson correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_matrix, cmap='coolwarm', center=0, annot=False,
                square=True, vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Pearson Correlation Heatmap')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_correlation_from_data("Data_exploration/results/binlevel_dataset_correlations.csv")
    # train_data, val_data, test_data, scalers = preprocess_data(
    # input_folder="Data/distances_with_zeros",
    # gene_file="SGD_API/architecture_info/yeast_genes_with_info.json",
    # train_val_test_split=[1, 0, 0],
    # scaling=True
    # )

    # corr_df = compute_all_pairwise_binlevel_correlations_simple(train_data)
    # print(corr_df)
    
    # # Save to CSV
    # output_csv = "Data_exploration/binlevel_dataset_correlations.csv"
    # corr_df.to_csv(output_csv, index=False)
    # print(f"\nCorrelation results saved to {output_csv}")