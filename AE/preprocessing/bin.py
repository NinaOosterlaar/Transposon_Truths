import numpy as np
import sys, os
import matplotlib.pyplot as plt
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 
from Utils.plot_config import setup_plot_style, COLORS

# Set up standardized plot style
setup_plot_style()

def bin_data(data, bin_size, method, maximum_region_size):
    """Bin all data in the dictionary to reduce data dimensionality.
    
    The shape of arrays is reduced to reflect the binning (maximum_region_size / bin_size).
    
    Args:
        data (Dictionary): Dictionary containing region dictionaries.
        bin_size (int): Size of bins for data aggregation.
        method (str): Method for binning ('average', 'sum', 'max', 'min', 'median').
        maximum_region_size (int): Original maximum region size before binning.
        
    Returns:
        binned_data (Dictionary): Binned data with updated actual_length values and reduced array shapes.
    """
    
    for dataset in data:
        for chrom in data[dataset]:
            binned_regions = []
            for region_dict in data[dataset][chrom]:
                data_sample = region_dict['data']
                actual_length = region_dict['actual_length']

                binned_region, num_bins = bin_data_single_array(data_sample, actual_length, bin_size, method)

                # Update the region dictionary
                region_dict['data'] = binned_region
                region_dict['actual_length'] = num_bins
                binned_regions.append(region_dict)
            data[dataset][chrom] = binned_regions
    return data

def bin_data_single_array(data_array, length, bin_size, method):
    """Bin a single data array into specified bin size using the given method.
    
    Args:
        data_array (np.ndarray): Input data array.
                                 Shape (L,) or (L, F).
        length (int): Actual length of non-padded values (<= L).
        bin_size (int): Size of bins for data aggregation.
        method (str): Method for binning first column
                      ('average', 'sum', 'max', 'min', 'median', 'average_non_zero').
        
    Returns:
        binned_region (np.ndarray): Binned data array.
            - If original was 1D: shape (num_bins,)
            - If original was 2D: shape (num_bins, F)
        num_bins (int): Number of bins after binning.
    """
    # Remember if we started with 1D
    was_1d = (data_array.ndim == 1)
    if was_1d:
        # Make it (L, 1) so we can treat everything as 2D
        data_array = data_array.reshape(-1, 1)

    # Only bin the actual (non-padded) data
    num_bins = (length + bin_size - 1) // bin_size

    # Create array with the new maximum size (based on binning)
    num_features = data_array.shape[1]
    binned_region = np.zeros((num_bins, num_features))

    # Choose aggregation function for the main signal (column 0)
    if method == 'average':
        agg = np.mean
    elif method == 'sum':
        agg = np.sum
    elif method == 'max':
        agg = np.max
    elif method == 'min':
        agg = np.min
    elif method == 'median':
        agg = np.median
    elif method == 'average_non_zero':
        agg = lambda x: np.mean(x[x != 0]) if np.any(x != 0) else 0
    else:
        raise ValueError(f"Unknown method: {method}")

    for i in range(num_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, length)
        bin_data = data_array[start:end, :]  # always 2D now

        # For DataFrames with structure [Position, Value, features..., Value_Raw, Size_Factor]:
        if num_features > 1 and method == 'average_non_zero':
            # Column 0 (Position): regular mean
            binned_region[i, 0] = np.mean(bin_data[:, 0])
            # Column 1 (Value): non-zero average
            value_col = bin_data[:, 1]
            binned_region[i, 1] = np.mean(value_col[value_col != 0]) if np.any(value_col != 0) else 0
            # Remaining features: process individually
            if num_features > 2:
                for col_idx in range(2, num_features):
                    # Check if this is the second-to-last column (likely Value_Raw in ZINB mode)
                    if col_idx == num_features - 2 and num_features >= 5:
                        # Apply non-zero average to Value_Raw
                        value_raw = bin_data[:, col_idx]
                        binned_region[i, col_idx] = np.mean(value_raw[value_raw != 0]) if np.any(value_raw != 0) else 0
                    else:
                        # Regular mean for other features (positions, distances, size_factor)
                        binned_region[i, col_idx] = np.mean(bin_data[:, col_idx])
        else:
            # Original behavior for 1D arrays or other methods
            binned_region[i, 0] = agg(bin_data[:, 0])
            # Aggregate remaining features by mean, if present
            if num_features > 1:
                binned_region[i, 1:] = np.mean(bin_data[:, 1:], axis=0)

    # If original array was 1D, return 1D result
    if was_1d:
        return binned_region[:, 0], num_bins

    return binned_region, num_bins


def sliding_window(data, window_size, step_size, moving_average=False):
    """Apply sliding window to data.
    
    Args:
        data (np.ndarray): Input data array.
        window_size (int): Size of the sliding window.
        step_size (int or float): Step size for the sliding window. If float, interpreted as fraction of window_size.
        moving_average (bool): Whether to compute moving average within the window.
    """
    step_size = int(step_size)  # Ensure it's an integer
    
    current_window = 0
    length = len(data)
    windows = []
    
    # Print mean non-zero value before moving average if moving_average is enabled
    if moving_average:
        # Handle 1D or 2D arrays
        if data.ndim == 1:
            non_zero_values = data[data != 0]
        else:
            # Column 1 is the Value column (column 0 is Position in dataframes)
            value_col_idx = 1 if data.shape[1] > 1 else 0
            non_zero_values = data[:, value_col_idx][data[:, value_col_idx] != 0]
        mean_non_zero_before = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0
        print(f"Mean non-zero value before moving average: {mean_non_zero_before:.4f}")

    def _moving_average_window(window_data):
        """Compute moving-average output while preserving input dimensionality.

        For 2D input from DataFrames with structure [Position, Value, features..., Value_Raw, Size_Factor]:
        - Column 0 (Position): standard mean
        - Column 1 (Value): non-zero average (log-transformed input)
        - Column -2 (Value_Raw): non-zero average (raw count target for ZINB)
        - Column -1 (Size_Factor): standard mean
        - Other columns: standard mean
        """
        num_features = window_data.shape[1]
        averaged = np.zeros(num_features, dtype=np.float32)

        # Standard mean for position (column 0)
        averaged[0] = np.mean(window_data[:, 0])
        
        if num_features > 1:
            # Non-zero average for Value column (column 1)
            value_signal = window_data[:, 1]
            averaged[1] = np.mean(value_signal[value_signal != 0]) if np.any(value_signal != 0) else 0
            
            # Process remaining features (columns 2+)
            if num_features > 2:
                # Apply standard mean to middle columns
                for col_idx in range(2, num_features):
                    # Check if this is the second-to-last column (likely Value_Raw in ZINB mode)
                    if col_idx == num_features - 2 and num_features >= 5:
                        # Apply non-zero average to Value_Raw
                        value_raw = window_data[:, col_idx]
                        averaged[col_idx] = np.mean(value_raw[value_raw != 0]) if np.any(value_raw != 0) else 0
                    else:
                        # Standard mean for other features (positions, distances, size_factor)
                        averaged[col_idx] = np.mean(window_data[:, col_idx])

        return averaged

    while current_window + window_size <= length:
        window_data = data[current_window:current_window + window_size]
        if moving_average:
            # window_data = np.mean(window_data, axis=0)
            window_data = _moving_average_window(window_data)
        windows.append(window_data)
        current_window += step_size
    # Last data point should be the last window_size points, but skip if already included
    if current_window < length:
        window_data = data[-window_size:]
        if moving_average:
            # window_data = np.mean(window_data, axis=0)
            window_data = _moving_average_window(window_data)
        windows.append(window_data)
    
    # Print mean non-zero value after moving average if moving_average is enabled
    if moving_average:
        windows_array = np.array(windows)
        # Handle 1D or 2D result arrays
        if windows_array.ndim == 1:
            non_zero_values_after = windows_array[windows_array != 0]
        else:
            # Column 1 is the Value column (column 0 is Position in dataframes)
            value_col_idx = 1 if windows_array.shape[1] > 1 else 0
            non_zero_values_after = windows_array[:, value_col_idx][windows_array[:, value_col_idx] != 0]
        mean_non_zero_after = np.mean(non_zero_values_after) if len(non_zero_values_after) > 0 else 0
        print(f"Mean non-zero value after moving average: {mean_non_zero_after:.4f}")
    
    return windows

def saturation_against_bin_size(data, bin_sizes, plot = True):
    """Compute saturation of data against different bin sizes.
    Saturation is defined as the proportion of non-zero bins to total bins.

    Args:
        data (Dictionary): Dictionary containing different datasets and chromosomes.
        bin_sizes (list): List of bin sizes for saturation calculation.
    Returns:
        densities (Dictionary): For sliding window and non-sliding window densities per dataset and bin size.
    """
    densities = {}
    for dataset in data:
        print(dataset)
        densities[dataset] = {'bins': [], 'moving_average': []}
        for bin_size in bin_sizes:
            print(bin_size)
            chrom_windows = {"moving_average": [], "binned": []}
            for chrom in data[dataset]:
                print(chrom)
                data_array = np.array(data[dataset][chrom]['Value'])
                
                # Calculate mean non-zero value before moving average
                non_zero_values_before = data_array[data_array != 0]
                mean_non_zero_before = np.mean(non_zero_values_before) if len(non_zero_values_before) > 0 else 0
                
                moving_average = sliding_window(data_array, window_size=bin_size, step_size=1, moving_average=True)
                bins = bin_data_single_array(data_array, length=len(data_array), bin_size=bin_size, method='average')[0]
                
                # Calculate mean non-zero value after moving average
                moving_average_array = np.array(moving_average)
                non_zero_values_after = moving_average_array[moving_average_array != 0]
                mean_non_zero_after = np.mean(non_zero_values_after) if len(non_zero_values_after) > 0 else 0
                
                print(f"  Mean non-zero value before moving average: {mean_non_zero_before:.4f}")
                print(f"  Mean non-zero value after moving average: {mean_non_zero_after:.4f}")
                
                chrom_windows["moving_average"].extend(moving_average)
                chrom_windows["binned"].extend(bins)
            densities[dataset]['bins'].append(compute_saturation(np.array(chrom_windows["binned"])))
            densities[dataset]['moving_average'].append(compute_saturation(np.array(chrom_windows["moving_average"])))
    if plot:
        plot_saturation_vs_bin_size(densities, bin_sizes)
    return densities

def plot_saturation_vs_bin_size(densities, bin_sizes, output_folder="AE/results/"):
    """Plot saturation against bin sizes for different datasets.
    For each dataset create a plot that shows both the moving_average and binned values.
    Also show a plot with all datasets together for comparison, one with moving_average and one with binned.

    Args:
        densities (Dictionary): Saturation densities per dataset and bin size.
        bin_sizes (list): List of bin sizes used.
    """
    # for dataset in densities:
    #     plt.figure()
    #     plt.plot(bin_sizes, densities[dataset]['bins'], marker='o', label='Binned')
    #     plt.plot(bin_sizes, densities[dataset]['moving_average'], marker='o', label='Moving Average')
    #     plt.xlabel('Bin Size')
    #     plt.ylabel('Saturation')
    #     plt.title(f'Saturation vs Bin Size for {dataset}')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     # Save plot
    #     plt.savefig(os.path.join(output_folder, f"saturation_vs_bin_size_{dataset}.png"))
    # Plot binned saturation for all datasets
    fig, ax = plt.subplots(figsize=(12, 8))
    for dataset in densities:
        ax.plot(bin_sizes, densities[dataset]['bins'], marker='o', label=dataset, markersize=4)
    ax.set_xlabel('Bin Size', fontsize=12)
    ax.set_ylabel('Saturation (Binned)', fontsize=12)
    ax.set_title('Saturation vs Bin Size (Binned) for All Datasets', fontsize=14)
    ax.set_ylim(0, 1)  # Saturation is between 0 and 1
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.08)
    plt.savefig(os.path.join(output_folder, "saturation_vs_bin_size_binned_all_datasets.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot moving average saturation for all datasets
    fig, ax = plt.subplots(figsize=(12, 8))
    for dataset in densities:
        ax.plot(bin_sizes, densities[dataset]['moving_average'], marker='o', label=dataset, markersize=4)
    ax.set_xlabel('Bin Size', fontsize=12)
    ax.set_ylabel('Saturation (Moving Average)', fontsize=12)
    ax.set_title('Saturation vs Bin Size (Moving Average) for All Datasets', fontsize=14)
    ax.set_ylim(0, 1)  # Saturation is between 0 and 1
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.08)
    plt.savefig(os.path.join(output_folder, "saturation_vs_bin_size_moving_average_all_datasets.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def compute_saturation(bins):
    """Compute saturation for a single data array given a bin size."""
    num_bins = len(bins)
    if num_bins == 0:
        return 0.0
    non_zero_bins = np.sum(bins > 0)
    saturation = non_zero_bins / num_bins
    return saturation


def calculate_saturation_without_bins(data):
    """Calculate saturation without binning.
    Saturation is defined as the proportion of non-zero positions to total positions.

    Args:
        data (Dictionary): Dictionary containing different datasets and chromosomes.
    Returns:
        saturations (Dictionary): Saturation per dataset.
    """
    saturations = {}
    for dataset in data:
        total_positions = 0
        non_zero_positions = 0
        for chrom in data[dataset]:
            data_array = np.array(data[dataset][chrom]['Value'])
            total_positions += len(data_array)
            non_zero_positions += np.sum(data_array > 0)
        if total_positions == 0:
            saturations[dataset] = 0.0
        else:
            saturations[dataset] = non_zero_positions / total_positions
    return saturations

if __name__ == "__main__":
    bin_sizes = [1, 5, 10, 20, 30, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    # input_folder = "Data/distances_with_zeros"
    # output_folder = "AE/results/"
    # transposon_data = read_csv_file_with_distances(input_folder)
    # # initial_saturations = calculate_saturation_without_bins(transposon_data)
    # # print(initial_saturations)
    # densities = saturation_against_bin_size(transposon_data, bin_sizes)
    # # Save densities to a file
    # with open(os.path.join(output_folder, "saturation_densities.json"), 'w') as f:
    #     json.dump(densities, f, indent=4)
    with open("AE/results/saturation_densities.json", 'r') as f:
        densities = json.load(f)
    plot_saturation_vs_bin_size(densities, bin_sizes)
    
    

