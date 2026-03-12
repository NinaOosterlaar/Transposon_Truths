import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.plot_config import setup_plot_style, COLORS
from Utils.reader import read_wig, label_from_filename

# Set up standardized plot style
setup_plot_style()
                    
chromosome_length = {
    "ChrI": 230218,
    "ChrII": 813184,
    "ChrIII": 316620,
    "ChrIV": 1531933,
    "ChrV": 576874,
    "ChrVI": 270161,
    "ChrVII": 1090940,
    "ChrVIII": 562643,
    "ChrIX": 439888,
    "ChrX": 745751,
    "ChrXI": 666816,
    "ChrXII": 1078171,
    "ChrXIII": 924431,
    "ChrXIV": 784333,
    "ChrXV": 1091291,
    "ChrXVI": 948066,
    "ChrM": 85779,          # mitochondrial genome (approx for S288C)   
}

def calculate_Szymkiewicz_Simpson(set_a, set_b):
    """Szymkiewicz–Simpson between chrom->set mappings."""
    inter = 0
    size_a = 0
    size_b = 0
    for chrom in set(set_a.keys()) | set(set_b.keys()):
        a = set_a.get(chrom, set())
        b = set_b.get(chrom, set())
        inter += len(a & b)
        size_a += len(a)
        size_b += len(b)
    denom = min(size_a, size_b)
    return 0.0 if denom == 0 else inter / denom
                    
def calculate_jaccard_index(set_a, set_b):
    """Calculate the Jaccard index between two sets."""
    intersection = 0
    union = 0
    for chrom in set_a.keys():
        intersection += len(set_a[chrom].intersection(set_b.get(chrom, set())))
        union += len(set_a[chrom].union(set_b.get(chrom, set())))
    if union == 0:
        return 0.0
    return intersection / union 

def bin_positions(positions, bin_size):
    """Bin positions into globally consistent bins of fixed size."""
    positions_new = {}
    for chrom, chrom_len in chromosome_length.items():
        if chrom not in positions:
            continue  # skip if no insertions on this chrom
        binned = set()
        for pos in positions[chrom]:
            # Compute 1-based bin start coordinate
            bstart = ((int(pos) - 1) // bin_size) + 1
            binned.add(bstart)
        positions_new[chrom] = binned
    return positions_new


def compute_indices_from_folder(folder_path, bin_size=1, Jaccard=True, Szymkiewicz_Simpson=True):
    """Compute Jaccard indices for all pairs of datasets in a folder.

    Args:
        folder_path (str): Path to the folder containing wig files with transposon data.
    """
    datasets = {}
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".wig") or filename.endswith(".wiggle"):
                file_path = os.path.join(root, filename)
                data = read_wig(file_path)
                label = label_from_filename(filename)
                positions_set = {}
                for chrom in data:
                    positions = data[chrom][data[chrom].columns[0]].tolist()
                    positions_set[chrom] = set(positions)
                if bin_size > 1:
                    positions_set = bin_positions(positions_set, bin_size)
                    print("Binned positions for", label)
                datasets[label] = positions_set

    jaccard_indices = {}
    szymkiewicz_simpson_indices = {}
    labels = list(datasets.keys())
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            label_a = labels[i]
            label_b = labels[j]
            if Jaccard:
                jaccard_index = calculate_jaccard_index(datasets[label_a], datasets[label_b])
                jaccard_indices[(label_a, label_b)] = jaccard_index
                print(jaccard_index)
            if Szymkiewicz_Simpson:
                szymkiewicz_simpson_index = calculate_Szymkiewicz_Simpson(datasets[label_a], datasets[label_b])
                szymkiewicz_simpson_indices[(label_a, label_b)] = szymkiewicz_simpson_index
                print(szymkiewicz_simpson_index)
    result = {'jaccard_indices': jaccard_indices, 'szymkiewicz_simpson_indices': szymkiewicz_simpson_indices}

    return result

def save_and_plot_heatmap(output_folder, bin_size=1, Jaccard_indices = {}, Szymkiewicz_Simpson_indices = {}):
    """Save Jaccard indices to a file and plot a heatmap.

    Args:
        jaccard_indices (dict): Dictionary with Jaccard indices.
        output_folder (str): Path to the output folder for saving results.
        output_file (str): Path to the output file for saving the heatmap.
    """
    if Jaccard_indices:
        with open(os.path.join(output_folder, "jaccard_indices_binned_" + str(bin_size) + ".txt"), 'w') as f:
            for (label_a, label_b), index in jaccard_indices.items():
                f.write(f"{label_a}\t{label_b}\t{index}\n")

        labels = set()
        for pair in jaccard_indices.keys():
            labels.update(pair)
        labels = sorted(labels)
        
        matrix = np.zeros((len(labels), len(labels)))
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        
        for (label_a, label_b), index in jaccard_indices.items():
            i = label_to_index[label_a]
            j = label_to_index[label_b]
            matrix[i][j] = index
            matrix[j][i] = index  # Symmetric matrix

        df = pd.DataFrame(matrix, index=labels, columns=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, cmap="YlGnBu")
        plt.title("Jaccard Index Heatmap")
        plt.savefig(os.path.join(output_folder, "jaccard_index_heatmap_binned_" + str(bin_size) + ".png"))
        plt.close()
        
    if Szymkiewicz_Simpson_indices:
        with open(os.path.join(output_folder, "szymkiewicz_simpson_indices_binned_" + str(bin_size) + ".txt"), 'w') as f:
            for (label_a, label_b), index in Szymkiewicz_Simpson_indices.items():
                f.write(f"{label_a}\t{label_b}\t{index}\n")

        labels = set()
        for pair in Szymkiewicz_Simpson_indices.keys():
            labels.update(pair)
        labels = sorted(labels)
        
        matrix = np.zeros((len(labels), len(labels)))
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        
        for (label_a, label_b), index in Szymkiewicz_Simpson_indices.items():
            i = label_to_index[label_a]
            j = label_to_index[label_b]
            matrix[i][j] = index
            matrix[j][i] = index  # Symmetric matrix

        df = pd.DataFrame(matrix, index=labels, columns=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, cmap="YlGnBu")
        plt.title("Szymkiewicz-Simpson Index Heatmap")
        plt.savefig(os.path.join(output_folder, "szymkiewicz_simpson_index_heatmap_binned_" + str(bin_size) + ".png"))
        plt.close()




if __name__ == "__main__":
    folder_path = "Data/wiggle_format"
    # find_minimal_distance_between_transposons(folder_path)
    
    # for pair in jaccard_indices:
    #     print(f"Jaccard index between {pair[0]} and {pair[1]}: {jaccard_indices[pair]}")
    # with open("Data_exploration/results/jaccard_indices/jaccard_indices.txt", 'r') as f:
    #     jaccard_indices = {}
    #     for line in f:
    #         label_a, label_b, index = line.strip().split('\t')
    #         jaccard_indices[(label_a, label_b)] = float(index)
    bin_size = 1
    jaccard_indices, szymkiewicz_simpson_indices = compute_indices_from_folder(folder_path, bin_size=bin_size, Jaccard=False, Szymkiewicz_Simpson=True).values()
    save_and_plot_heatmap("Data_exploration/results/jaccard_indices", bin_size=bin_size, Jaccard_indices=jaccard_indices, Szymkiewicz_Simpson_indices=szymkiewicz_simpson_indices)