import json
import numpy as np
import pandas as pd
import os, sys
import gc
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 
from AE.preprocessing.bin import bin_data, sliding_window, bin_data_single_array
from Utils.reader import read_csv_file_with_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
}

def clip_outliers(data, percentile=95, multiplier=1.5):
    """Clip outliers in transposon count data based on percentile analysis.
    
    Calculates the 95th percentile (excluding zeros) and caps all values above 
    multiplier * 95th_percentile to that threshold.
    
    Args:
        data (Dictionary): Dictionary containing {dataset: {chromosome: DataFrame}} structure.
        percentile (float): Percentile to use for threshold calculation. Default=95.
        multiplier (float): Multiplier for the percentile threshold. Default=1.5.
        
    Returns:
        data (Dictionary): Data with clipped outliers (same object, modified in-place).
        clip_stats (Dictionary): Statistics about clipping per dataset.
    """
    clip_stats = {}
    
    for dataset in data:
        # Collect all non-zero values for this dataset
        all_values = []
        for chrom in data[dataset]:
            df = data[dataset][chrom]
            if 'Value' in df.columns:
                non_zero_values = df[df['Value'] > 0]['Value'].values
                all_values.extend(non_zero_values)
        
        if len(all_values) == 0:
            print(f"Warning: No non-zero values found for {dataset}")
            continue
        
        # Calculate percentile threshold (excluding zeros)
        all_values = np.array(all_values)
        percentile_value = np.percentile(all_values, percentile)
        clip_threshold = multiplier * percentile_value
        
        # Count values that will be clipped
        values_above_threshold = np.sum(all_values > clip_threshold)
        
        # Apply clipping to all chromosomes in this dataset
        for chrom in data[dataset]:
            df = data[dataset][chrom]
            if 'Value' in df.columns:
                data[dataset][chrom]['Value'] = df['Value'].clip(upper=clip_threshold)
        
        # Store statistics
        clip_stats[dataset] = {
            'percentile': percentile,
            'percentile_value': percentile_value,
            'multiplier': multiplier,
            'clip_threshold': clip_threshold,
            'total_non_zero': len(all_values),
            'clipped_count': int(values_above_threshold),
            'clipped_percentage': (values_above_threshold / len(all_values)) * 100
        }
        
        print(f"{dataset}: Clipping values above {clip_threshold:.2f} (P{percentile}={percentile_value:.2f}). "
              f"Clipped {values_above_threshold}/{len(all_values)} values ({clip_stats[dataset]['clipped_percentage']:.2f}%)")
    
    return data, clip_stats

def standardize_data(train_data, val_data, test_data, features):
    """Standardize features to have mean 0 and standard deviation 1.
    Fits scalers on training data and applies to all splits.
    Does NOT standardize 'Value' (log-normalized counts) or 'Chrom' (categorical).
    
    Args:
        train_data, val_data, test_data: Dictionaries containing {dataset: {chromosome: DataFrame}}.
        features: List of features to use (e.g., ['Pos', 'Chrom', 'Nucl', 'Centr']).
        
    Returns:
        train_data, val_data, test_data: Data with standardized features (same objects, modified in-place).
        scalers: Dictionary of StandardScaler objects for each feature.
    """
    scalers = {}
    
    # Features to standardize (exclude 'Value' and 'Chrom')
    features_to_standardize = []
    
    feature_to_column = {
        'Value': 'Value',
        'Pos': 'Position',
        'Nucl': 'Nucleosome_Distance',
        'Centr': 'Centromere_Distance'
    }
    
    for feature in features:
        if feature in feature_to_column:
            features_to_standardize.append(feature)
    
    print(f"Standardizing features: {features_to_standardize}")
    
    # For each feature, fit scaler on training data
    for feature in features_to_standardize:
        column_name = feature_to_column[feature]
        scaler = StandardScaler()
        
        # Collect all values for this feature from training data
        all_values = []
        for dataset in train_data:
            for chrom in train_data[dataset]:
                df = train_data[dataset][chrom]
                if column_name in df.columns:
                    all_values.extend(df[column_name].values)
        
        if len(all_values) > 0:
            all_values = np.array(all_values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(all_values)
            scalers[feature] = scaler
            
            # Clean up
            del all_values
            gc.collect()
            
            # Apply scaler to training data (in-place)
            for dataset in train_data:
                for chrom in train_data[dataset]:
                    df = train_data[dataset][chrom]
                    if column_name in df.columns:
                        train_data[dataset][chrom][column_name] = scaler.transform(
                            df[column_name].values.reshape(-1, 1)
                        ).flatten().astype(np.float32)
            
            # Apply scaler to validation data (in-place)
            for dataset in val_data:
                for chrom in val_data[dataset]:
                    df = val_data[dataset][chrom]
                    if column_name in df.columns:
                        val_data[dataset][chrom][column_name] = scaler.transform(
                            df[column_name].values.reshape(-1, 1)
                        ).flatten().astype(np.float32)
            
            # Apply scaler to test data (in-place)
            for dataset in test_data:
                for chrom in test_data[dataset]:
                    df = test_data[dataset][chrom]
                    if column_name in df.columns:
                        test_data[dataset][chrom][column_name] = scaler.transform(
                            df[column_name].values.reshape(-1, 1)
                        ).flatten().astype(np.float32)
    
    return train_data, val_data, test_data, scalers

def preprocess_counts(data, zinb_mode=False):
    """Preprocess transposon insertion counts.
    1. CPM (Counts Per Million) normalization per dataset
    2. Log-transform counts: log1p(CPM)
    3. For ZINB mode: save raw counts in 'Value_Raw' and add size factor column
    
    Args:
        data (Dictionary): Dictionary containing {dataset: {chromosome: DataFrame}} structure.
        zinb_mode (bool): If True, save raw counts and add size factor column. Default=False.
        
    Returns:
        data (Dictionary): Preprocessed counts data (same object, modified in-place).
        stats (Dictionary): Statistics about the preprocessing (total insertions and CPM scale factor per dataset).
    """
    stats = {}

    for dataset in data:
        # 1. Compute total insertions in this dataset across all chromosomes
        total_insertions = 0.0
        for chrom in data[dataset]:
            df = data[dataset][chrom]
            total_insertions += df['Value'].sum()

        if total_insertions == 0:
            total_insertions = 1.0  # avoid div-by-zero

        # CPM scale factor = counts * (1e6 / total_insertions)
        cpm_scale_factor = 1e6 / total_insertions
        # Size factor for ZINB = total_insertions / 1e6 (library size normalization)
        size_factor = total_insertions / 1e6
        
        stats[dataset] = {
            "total_insertions": float(total_insertions),
            "cpm_scale_factor": float(cpm_scale_factor),
            "size_factor": float(size_factor),
        }

        # 2. Apply CPM + log1p(CPM) to every chromosome in this dataset (in-place)
        for chrom in data[dataset]:
            df = data[dataset][chrom]
            
            if zinb_mode:
                # Save raw counts before normalization
                data[dataset][chrom]['Value_Raw'] = df['Value'].copy()
                # Add size factor as constant column (same value for all rows in this dataset)
                data[dataset][chrom]['Size_Factor'] = size_factor
            
            # CPM normalization
            cpm = df['Value'] * cpm_scale_factor  # counts per million

            # log1p transform
            log_cpm = np.log1p(cpm).astype(np.float32)  # Use float32 to save memory

            # Write back into the DataFrame (in-place)
            data[dataset][chrom]['Value'] = log_cpm
            
            # Clean up temporary variables
            del cpm, log_cpm

    return data, stats

def _split_items(items, train_size, val_size, test_size):
    """Helper function to split a list of items into train/val/test."""
    if not items or (val_size == 0 and test_size == 0):
        return items, [], []
    if len(items) == 1 or train_size >= 1.0:
        return items, [], []
    if train_size <= 0.0:
        return [], items, []
    
    # Split train from (val + test)
    train_items, temp_items = train_test_split(items, train_size=train_size, random_state=42)
    
    # Split val from test
    if not temp_items or test_size == 0:
        return train_items, temp_items, []
    if val_size == 0:
        return train_items, [], temp_items
    if len(temp_items) == 1:
        return train_items, temp_items, []
    
    val_items, test_items = train_test_split(temp_items, test_size=test_size/(test_size + val_size), random_state=42)
    return train_items, val_items, test_items

def determine_chromosome_split(input_folder, train_val_test_split):
    """Determine which chromosomes go to train/val/test splits.
    
    This function should be called ONCE before running multiple preprocessing operations
    to ensure consistent splits across different hyperparameter configurations.
    
    Args:
        input_folder (str): Folder containing the raw CSV files
        train_val_test_split (list): Proportions for training, validation, and testing sets
        
    Returns:
        tuple: (train_chroms, val_chroms, test_chroms) - lists of chromosome names
    """
    # Read data to get all available chromosomes
    data = read_csv_file_with_distances(input_folder)
    
    # Get all unique chromosomes and SORT them for deterministic ordering
    all_chroms = sorted(list(set(chrom for dataset in data.values() for chrom in dataset.keys())))
    
    train_size, val_size, test_size = train_val_test_split
    train_chroms, val_chroms, test_chroms = _split_items(all_chroms, train_size, val_size, test_size)
    
    print(f"\nChromosome split determined:")
    print(f"  Train chromosomes: {sorted(train_chroms)}")
    print(f"  Validation chromosomes: {sorted(val_chroms)}")
    print(f"  Test chromosomes: {sorted(test_chroms)}")
    print()
    
    return train_chroms, val_chroms, test_chroms

def split_data(data, train_val_test_split, split_on, chunk_size=50000):
    """Split data into training, validation, and testing sets.
    
    Args:
        data (DataFrame): DataFrame containing the data to be split.
        train_val_test_split (list): Proportions for training, validation, and testing sets.
        split_on (str): Feature to split data on ('Chrom', 'Dataset', 'Random').
        chunk_size (int): Size of chunks in base pairs for random splitting. Default: 50000.
        
    Returns:
        train_data (DataFrame): Training data.
        val_data (DataFrame): Validation data.
        test_data (DataFrame): Testing data.
    """
    train_size, val_size, test_size = train_val_test_split
    
    if split_on == 'Dataset':
        # Sort dataset keys for deterministic splitting
        train_datasets, val_datasets, test_datasets = _split_items(sorted(list(data.keys())), train_size, val_size, test_size)
        
        train_data = {d: data[d] for d in train_datasets}
        val_data = {d: data[d] for d in val_datasets}
        test_data = {d: data[d] for d in test_datasets}
        
        print(f"Train datasets: {sorted(train_datasets)}")
        print(f"Validation datasets: {sorted(val_datasets)}")
        print(f"Test datasets: {sorted(test_datasets)}")
    
    elif split_on == 'Chrom':
        # Get all unique chromosomes and SORT them for consistency
        all_chroms = sorted(list(set(chrom for dataset in data.values() for chrom in dataset.keys())))
        train_chroms, val_chroms, test_chroms = _split_items(all_chroms, train_size, val_size, test_size)
        
        print(f"Train chromosomes: {sorted(train_chroms)}")
        print(f"Validation chromosomes: {sorted(val_chroms)}")
        print(f"Test chromosomes: {sorted(test_chroms)}")
        
        # Assign chromosomes to splits for each dataset
        train_data = {d: {c: data[d][c] for c in data[d] if c in train_chroms} for d in data}
        val_data = {d: {c: data[d][c] for c in data[d] if c in val_chroms} for d in data}
        test_data = {d: {c: data[d][c] for c in data[d] if c in test_chroms} for d in data}
        
        # Remove empty dataset dictionaries
        train_data = {d: v for d, v in train_data.items() if v}
        val_data = {d: v for d, v in val_data.items() if v}
        test_data = {d: v for d, v in test_data.items() if v}
    
    elif split_on == 'Random':
        # Create chunks from all chromosomes across all datasets
        all_chunks = []
        
        for dataset in data:
            for chrom in data[dataset]:
                df = data[dataset][chrom]
                if df.empty:
                    continue
                
                # Get min and max positions for this chromosome
                min_pos = df['Position'].min()
                max_pos = df['Position'].max()
                
                # Create chunks
                current_pos = min_pos
                while current_pos <= max_pos:
                    chunk_end = min(current_pos + chunk_size, max_pos + 1)
                    all_chunks.append({
                        'dataset': dataset,
                        'chrom': chrom,
                        'start': current_pos,
                        'end': chunk_end
                    })
                    current_pos = chunk_end
        
        # Split chunks into train/val/test
        train_chunks, val_chunks, test_chunks = _split_items(all_chunks, train_size, val_size, test_size)
        
        # Initialize data structures
        train_data = {d: {} for d in data.keys()}
        val_data = {d: {} for d in data.keys()}
        test_data = {d: {} for d in data.keys()}
        
        # Assign data points to splits based on chunk assignments
        def assign_chunks_to_split(chunks, split_data):
            for chunk in chunks:
                dataset = chunk['dataset']
                chrom = chunk['chrom']
                start = chunk['start']
                end = chunk['end']
                
                # Filter DataFrame for this chunk
                df = data[dataset][chrom]
                mask = (df['Position'] >= start) & (df['Position'] < end)
                chunk_df = df[mask].copy()
                
                if not chunk_df.empty:
                    # Add to existing chromosome data or create new
                    if chrom in split_data[dataset]:
                        split_data[dataset][chrom] = pd.concat([split_data[dataset][chrom], chunk_df], ignore_index=True)
                    else:
                        split_data[dataset][chrom] = chunk_df
        
        assign_chunks_to_split(train_chunks, train_data)
        assign_chunks_to_split(val_chunks, val_data)
        assign_chunks_to_split(test_chunks, test_data)
        
        # Remove empty datasets/chromosomes and sort by position
        for split_data in [train_data, val_data, test_data]:
            datasets_to_remove = []
            for dataset in split_data:
                chroms_to_remove = []
                for chrom in split_data[dataset]:
                    if split_data[dataset][chrom].empty:
                        chroms_to_remove.append(chrom)
                    else:
                        # Sort by position
                        split_data[dataset][chrom] = split_data[dataset][chrom].sort_values('Position').reset_index(drop=True)
                
                for chrom in chroms_to_remove:
                    del split_data[dataset][chrom]
                
                if not split_data[dataset]:
                    datasets_to_remove.append(dataset)
            
            for dataset in datasets_to_remove:
                del split_data[dataset]
    
    else:
        train_data, val_data, test_data = {}, {}, {}
    
    return train_data, val_data, test_data

def _find_consecutive_segments(df, gap_tolerance=1):
    """Find consecutive segments in a DataFrame based on Position column.
    
    Args:
        df (DataFrame): DataFrame with a 'Position' column.
        gap_tolerance (int): Maximum gap between consecutive positions. Default is 1.
        
    Returns:
        List of DataFrames, each containing a consecutive segment.
    """
    if df.empty:
        return []
    
    # Sort by position
    df = df.sort_values('Position').reset_index(drop=True)
    
    segments = []
    current_segment = [0]  # Start with first row
    
    for i in range(1, len(df)):
        prev_pos = df.iloc[i-1]['Position']
        curr_pos = df.iloc[i]['Position']
        
        # Check if positions are consecutive (within tolerance)
        if curr_pos - prev_pos <= gap_tolerance:
            current_segment.append(i)
        else:
            # Gap detected, save current segment and start new one
            segments.append(df.iloc[current_segment].copy())
            current_segment = [i]
    
    # Add the last segment
    if current_segment:
        segments.append(df.iloc[current_segment].copy())
    
    return segments

def _add_chromosome_encoding(df, chrom):
    """Add chromosome encoding to DataFrame.
    
    Args:
        df (DataFrame): DataFrame to add chromosome column to.
        chrom (str): Chromosome name (e.g., 'ChrI', 'ChrII', ..., 'ChrXVI').
        
    Returns:
        DataFrame with chromosome encoding added as categorical integers (1-16).
    """
    # Map chromosome names to integers (1-16)
    chrom_to_int = {
        'ChrI': 1, 'ChrII': 2, 'ChrIII': 3, 'ChrIV': 4,
        'ChrV': 5, 'ChrVI': 6, 'ChrVII': 7, 'ChrVIII': 8,
        'ChrIX': 9, 'ChrX': 10, 'ChrXI': 11, 'ChrXII': 12,
        'ChrXIII': 13, 'ChrXIV': 14, 'ChrXV': 15, 'ChrXVI': 16
    }
    
    chrom_int = chrom_to_int.get(chrom, 0)
    df['Chromosome'] = chrom_int
    
    return df

def process_data(transposon_data, features, bin_size, moving_average, step_size, data_point_length, split_on='Dataset', zinb_mode=False):
    """Process data: bin/window and convert to 3D array for autoencoder input.
    
    Args:
        zinb_mode (bool): If True, include Value_Raw and Size_Factor columns in output. Default=False.
    
    Returns:
        tuple: (data_points, metadata)
            - data_points (np.ndarray): 3D array of shape (num_samples, window_length, num_features)
            - metadata (list): List of dicts with keys: 'dataset', 'chromosome', 'start_pos', 'end_pos', 'window_index'
    """
    
    # Check if chromosome encoding is needed
    use_chrom = 'Chrom' in features
    
    # Only check for non-consecutive segments when using Random split
    if split_on == 'Random':
        # Split each chromosome into consecutive segments to handle gaps from random splitting
        segmented_data = {}
        for dataset in transposon_data:
            segmented_data[dataset] = {}
            for chrom in transposon_data[dataset]:
                df = transposon_data[dataset][chrom].copy()
                segments = _find_consecutive_segments(df, gap_tolerance=1)
                segmented_data[dataset][chrom] = segments
        
        # Apply binning/windowing to each consecutive segment
        processed_segments = {}
        for dataset in segmented_data:
            processed_segments[dataset] = {}
            for chrom in segmented_data[dataset]:
                processed_segments[dataset][chrom] = []
                for segment_df in segmented_data[dataset][chrom]:
                    if segment_df.empty:
                        continue
                    
                    # Apply binning or moving average
                    if moving_average:
                        binned_values = sliding_window(segment_df.values, bin_size, step_size=1, moving_average=True)
                    else:
                        binned_values = bin_data(segment_df.values, bin_size)
                    
                    # Convert back to DataFrame
                    if len(binned_values) > 0:
                        # Use the original dataframe's columns to handle both normal and ZINB modes
                        binned_df = pd.DataFrame(binned_values, columns=segment_df.columns.tolist())
                        
                        # Add chromosome encoding if needed
                        if use_chrom:
                            binned_df = _add_chromosome_encoding(binned_df, chrom)
                        
                        processed_segments[dataset][chrom].append(binned_df)
        
        # Select and order columns based on features
        cols_to_keep = ['Value']
        if 'Pos' in features:
            cols_to_keep.append('Position')
        if 'Nucl' in features:
            cols_to_keep.append('Nucleosome_Distance')
        if 'Centr' in features:
            cols_to_keep.append('Centromere_Distance')
        if use_chrom:
            # Add chromosome column (categorical encoding)
            cols_to_keep.append('Chromosome')
        
        print(f"Column order in final array (Random split): {cols_to_keep}")
        
        # Filter columns for each segment
        for dataset in processed_segments:
            for chrom in processed_segments[dataset]:
                for i, segment_df in enumerate(processed_segments[dataset][chrom]):
                    existing_cols = [col for col in cols_to_keep if col in segment_df.columns]
                    processed_segments[dataset][chrom][i] = segment_df[existing_cols]
        
        # Apply sliding window to create data points from consecutive segments only
        data_points = []
        metadata = []
        window_idx = 0
        
        for dataset in processed_segments:
            for chrom in processed_segments[dataset]:
                for segment_df in processed_segments[dataset][chrom]:
                    if segment_df.empty or len(segment_df) < data_point_length:
                        continue
                    
                    # Extract position information before converting to array
                    if 'Position' in segment_df.columns:
                        positions = segment_df['Position'].values
                    else:
                        # If Position not available, use indices as proxy
                        positions = np.arange(len(segment_df))
                    
                    data_array = segment_df.values.astype(np.float32)
                    windows = sliding_window(data_array, data_point_length, step_size)
                    
                    # Track metadata for each window
                    current_pos = 0
                    for _ in windows:
                        window_start_idx = current_pos
                        window_end_idx = min(current_pos + data_point_length - 1, len(positions) - 1)
                        
                        metadata.append({
                            'window_index': window_idx,
                            'dataset': dataset,
                            'chromosome': chrom,
                            'start_pos': int(positions[window_start_idx]),
                            'end_pos': int(positions[window_end_idx]),
                            'bin_size': bin_size,
                            'moving_average': moving_average
                        })
                        window_idx += 1
                        current_pos += step_size
                    
                    data_points.extend(windows)
                    
                    # Clean up
                    del segment_df, data_array, windows, positions
                
                # Clean up processed segments as we go
                processed_segments[dataset][chrom] = None
            
            # Clean up dataset
            processed_segments[dataset] = None
        
        # Clean up
        del processed_segments
        gc.collect()
    
    else:
        # For Dataset and Chrom splits, data is already consecutive - use simpler processing
        for dataset in transposon_data:
            for chrom in transposon_data[dataset]:
                df = transposon_data[dataset][chrom].copy()
                if moving_average:
                    binned_values = sliding_window(df.values, bin_size, step_size=1, moving_average=True)
                else:
                    binned_values, _ = bin_data_single_array(df.values, len(df), bin_size, 'average_non_zero')
                
                # Use the original dataframe's columns to handle both normal and ZINB modes
                binned_df = pd.DataFrame(binned_values, columns=df.columns.tolist())
                
                # Add chromosome encoding if needed
                if use_chrom:
                    binned_df = _add_chromosome_encoding(binned_df, chrom)
                
                transposon_data[dataset][chrom] = binned_df
                
                # Clean up
                del df, binned_values
        
        # Select and order columns based on features
        cols_to_keep = ['Value']
        if 'Pos' in features:
            cols_to_keep.append('Position')
        if 'Nucl' in features:
            cols_to_keep.append('Nucleosome_Distance')
        if 'Centr' in features:
            cols_to_keep.append('Centromere_Distance')
        if use_chrom:
            # Add chromosome column (categorical encoding)
            cols_to_keep.append('Chromosome')
    
        
        # For ZINB mode, add raw counts and size factor columns at the end
        if zinb_mode:
            cols_to_keep.append('Value_Raw')
            cols_to_keep.append('Size_Factor')
            
        print(f"Column order in final array (Dataset/Chrom split): {cols_to_keep}")
        
        # Filter columns
        for dataset in transposon_data:
            for chrom in transposon_data[dataset]:
                df = transposon_data[dataset][chrom]
                existing_cols = [col for col in cols_to_keep if col in df.columns]
                transposon_data[dataset][chrom] = df[existing_cols]
        
        # Apply sliding window to create data points
        data_points = []
        metadata = []
        window_idx = 0
        
        for dataset in transposon_data:
            for chrom in transposon_data[dataset]:
                df = transposon_data[dataset][chrom]
                if df.empty or len(df) < data_point_length:
                    continue
                
                # Extract position information before converting to array
                if 'Position' in df.columns:
                    positions = df['Position'].values
                else:
                    # If Position not available, use indices as proxy
                    positions = np.arange(len(df))
                
                data_array = df.values.astype(np.float32)
                windows = sliding_window(data_array, data_point_length, step_size, moving_average=False)
                
                # Track metadata for each window
                current_pos = 0
                for _ in windows:
                    window_start_idx = current_pos
                    window_end_idx = min(current_pos + data_point_length - 1, len(positions) - 1)
                    
                    # Handle the last window which may start from the end
                    if current_pos + data_point_length > len(positions):
                        window_start_idx = len(positions) - data_point_length
                        window_end_idx = len(positions) - 1
                    
                    metadata.append({
                        'window_index': window_idx,
                        'dataset': dataset,
                        'chromosome': chrom,
                        'start_pos': int(positions[window_start_idx]),
                        'end_pos': int(positions[window_end_idx]),
                        'bin_size': bin_size,
                        'moving_average': moving_average
                    })
                    window_idx += 1
                    current_pos += step_size
                
                data_points.extend(windows)
                
                # Clean up
                del df, data_array, windows, positions
            
            # Clean up dataset data as we process
            transposon_data[dataset] = None
        
        # Clean up
        del transposon_data
        gc.collect()
    
    # Convert list of arrays to 3D numpy array: (num_samples, window_length, num_features)
    if len(data_points) > 0:
        data_points = np.array(data_points, dtype=np.float32)
    else:
        data_points = np.array([], dtype=np.float32).reshape(0, data_point_length, len(cols_to_keep))
    
    return data_points, metadata

def remove_empty_datapoints(data, metadata=None):
    """Remove data points that are completely empty (all zeros) from the dataset.
    
    Args:
        data (np.ndarray): 3D array of shape (num_samples, window_length, num_features).
        metadata (list, optional): List of metadata dictionaries corresponding to each data point.
    
    Returns:
        tuple: (filtered_data, filtered_metadata) if metadata provided, else just filtered_data
            - filtered_data (np.ndarray): Filtered data with empty data points removed.
            - filtered_metadata (list): Filtered metadata (if provided).
    """    
    if data.size == 0:
        if metadata is not None:
            return data, metadata
        return data
    
    # Identify non-empty data points
    non_empty_mask = np.any(data != 0, axis=(1, 2))
    print(f"Removing {np.sum(~non_empty_mask)} empty data points out of {data.shape[0]}")
    filtered_data = data[non_empty_mask]
    
    # Filter metadata if provided
    if metadata is not None:
        filtered_metadata = [meta for i, meta in enumerate(metadata) if non_empty_mask[i]]
        # Re-index the filtered metadata
        for idx, meta in enumerate(filtered_metadata):
            meta['window_index'] = idx
        return filtered_data, filtered_metadata
    
    return filtered_data
            
def preprocess_with_split(input_folder,
                          train_chroms,
                          val_chroms,
                          test_chroms,
                          features = ['Nucl', 'Centr'],
                          normalize_counts = True,
                          zinb_mode = True,
                          bin_size = 10,
                          moving_average = True,
                          data_point_length = 2000,
                          step_size = 500,
                          clip_outliers_flag = True,
                          outlier_percentile = 95,
                          outlier_multiplier = 1.5
                          ):
    """Preprocessing with pre-determined chromosome splits for consistent data splitting.
    
    This function uses pre-determined chromosome assignments instead of calculating splits internally.
    Use determine_chromosome_split() first to get train_chroms, val_chroms, test_chroms.

    Args:
        input_folder (Str): The folder with the raw csv files
        train_chroms (list): List of chromosome names for training set
        val_chroms (list): List of chromosome names for validation set
        test_chroms (list): List of chromosome names for test set
        features (list, optional): The features to be used. Defaults to ['Nucl', 'Centr'].
        normalize_counts (bool, optional): Whether to apply CPM normalization and log transform to counts. Defaults to True.
        zinb_mode (bool, optional): If True, save raw counts in 'Value_Raw' column for ZINB models. Defaults to True.
        bin_size (int, optional): The bin size for binning the data. Defaults to 10.
        moving_average (bool, optional): Whether to apply a moving average to the data. Defaults to True.
        data_point_length (int, optional): The length of each data point. Defaults to 2000.
        step_size (int, optional): The step size for sliding window. Defaults to 500.
        clip_outliers_flag (bool, optional): Whether to clip outliers. Defaults to True.
        outlier_percentile (float, optional): Percentile for outlier threshold. Defaults to 95.
        outlier_multiplier (float, optional): Multiplier for the percentile threshold. Defaults to 1.5.
    
    Returns:
        train (np.ndarray): Training data
        val (np.ndarray): Validation data
        test (np.ndarray): Test data
        train_metadata (list): Metadata for training windows
        val_metadata (list): Metadata for validation windows
        test_metadata (list): Metadata for test windows
        scalers (dict): Dictionary of StandardScaler objects
        count_stats (dict): Statistics from count normalization
        clip_stats (dict): Statistics from outlier clipping
    """
    transposon_data = read_csv_file_with_distances(input_folder)
   
    # Clip outliers if requested (before any normalization)
    clip_stats = None
    if clip_outliers_flag:
        print(f"\nClipping outliers using {outlier_percentile}th percentile * {outlier_multiplier}...")
        transposon_data, clip_stats = clip_outliers(transposon_data, percentile=outlier_percentile, multiplier=outlier_multiplier)
    
    # Optionally normalize counts before splitting
    count_stats = None
    if normalize_counts:
        transposon_data, count_stats = preprocess_counts(transposon_data, zinb_mode=zinb_mode)
    
    # Split data using pre-determined chromosome assignments
    print(f"\nSplitting data using pre-determined chromosomes...")
    train = {d: {c: transposon_data[d][c] for c in transposon_data[d] if c in train_chroms} for d in transposon_data}
    val = {d: {c: transposon_data[d][c] for c in transposon_data[d] if c in val_chroms} for d in transposon_data}
    test = {d: {c: transposon_data[d][c] for c in transposon_data[d] if c in test_chroms} for d in transposon_data}
    
    # Remove empty dataset dictionaries
    train = {d: v for d, v in train.items() if v}
    val = {d: v for d, v in val.items() if v}
    test = {d: v for d, v in test.items() if v}
    
    # Standardize features (fit on train, transform on val/test)
    train, val, test, scalers = standardize_data(train, val, test, features)
    
    # Bin/window and convert to 3D arrays
    train, train_metadata = process_data(train, features, bin_size, moving_average, step_size, data_point_length, 'Chrom', zinb_mode=zinb_mode)
    train, train_metadata = remove_empty_datapoints(train, train_metadata)
    gc.collect()
    
    val, val_metadata = process_data(val, features, bin_size, moving_average, step_size, data_point_length, 'Chrom', zinb_mode=zinb_mode)
    val, val_metadata = remove_empty_datapoints(val, val_metadata)
    gc.collect()
    
    test, test_metadata = process_data(test, features, bin_size, moving_average, step_size, data_point_length, 'Chrom', zinb_mode=zinb_mode)
    test, test_metadata = remove_empty_datapoints(test, test_metadata)
    gc.collect()

    return train, val, test, train_metadata, val_metadata, test_metadata, scalers, count_stats, clip_stats

def preprocess(input_folder, 
               features = ['Nucl', 'Centr'], 
               train_val_test_split = [0.7, 0.15, 0.15], 
               split_on = 'Chrom',
               chunk_size = 50000,
               normalize_counts = True,
               zinb_mode = True,
               bin_size = 10, 
               moving_average = True,
               data_point_length = 2000,
               step_size = 500,
               clip_outliers_flag = True,
               outlier_percentile = 95,
               outlier_multiplier = 1.5
               ):
    """Preprocessing the data before using at as an input for the Autoencoder

    Args:
        input_folder (Str): The folder with the raw csv files
        features (list, optional): The features to be used. Defaults to ['Pos', 'Chrom', 'Nucl', 'Centr'].
            - 'Value': Transposon insertion counts (always included)
            - 'Pos': Position along chromosome
            - 'Chrom': Chromosome (categorical encoding: ChrI=1, ChrII=2, ..., ChrXVI=16)
            - 'Nucl': Distance to nearest nucleosome
            - 'Centr': Distance to centromere
        train_val_test_split (list, optional): The proportions for training, validation, and testing sets. Defaults to [0.7, 0.15, 0.15].
        split_on (str, optional): The feature to split data on ('Chrom', 'Dataset', 'Random'). Defaults to 'Dataset'.
        chunk_size (int, optional): Size of chunks in base pairs for random splitting. Defaults to 50000.
        normalize_counts (bool, optional): Whether to apply CPM normalization and log transform to counts. Defaults to True.
        zinb_mode (bool, optional): If True, save raw counts in 'Value_Raw' column for ZINB models. Defaults to False.
        bin_size (int, optional): The bin size for binning the data of moving average to overcome sparsity. Defaults to 10.
        moving_average (bool, optional): Whether to apply a moving average to the data or use separate bins. Defaults to True.
        data_point_length (int, optional): The length of each data point. Defaults to 2000.
        step_size (int, optional): The step size for sliding window for the data points. Defaults to 200.
        clip_outliers_flag (bool, optional): Whether to clip outliers based on percentile analysis. Defaults to False.
        outlier_percentile (float, optional): Percentile to use for outlier threshold. Defaults to 95.
        outlier_multiplier (float, optional): Multiplier for the percentile threshold. Defaults to 1.5.
    
    Returns:
        train (np.ndarray): Training data, shape (n_train_samples, window_length, n_features)
        val (np.ndarray): Validation data, shape (n_val_samples, window_length, n_features)
        test (np.ndarray): Test data, shape (n_test_samples, window_length, n_features)
        train_metadata (list): Metadata for training windows
        val_metadata (list): Metadata for validation windows
        test_metadata (list): Metadata for test windows
        scalers (dict): Dictionary of StandardScaler objects for each feature.
        count_stats (dict, optional): Statistics from count normalization if normalize_counts=True.
        clip_stats (dict, optional): Statistics from outlier clipping if clip_outliers_flag=True.
    """
    transposon_data = read_csv_file_with_distances(input_folder)
   
    # First step: Clip outliers if requested (before any normalization)
    clip_stats = None
    if clip_outliers_flag:
        print(f"\nClipping outliers using {outlier_percentile}th percentile * {outlier_multiplier}...")
        transposon_data, clip_stats = clip_outliers(transposon_data, percentile=outlier_percentile, multiplier=outlier_multiplier)
    
    # Optionally normalize counts before splitting
    count_stats = None
    if normalize_counts:
        transposon_data, count_stats = preprocess_counts(transposon_data, zinb_mode=zinb_mode)
    
    # Split data
    train, val, test = split_data(transposon_data, train_val_test_split, split_on, chunk_size)
    
    # Standardize features (fit on train, transform on val/test)
    train, val, test, scalers = standardize_data(train, val, test, features)
    
    # Bin/window and convert to 3D arrays
    train, train_metadata = process_data(train, features, bin_size, moving_average, step_size, data_point_length, split_on, zinb_mode=zinb_mode)
    train, train_metadata = remove_empty_datapoints(train, train_metadata)
    gc.collect()  # Clean up memory after processing train
    
    val, val_metadata = process_data(val, features, bin_size, moving_average, step_size, data_point_length, split_on, zinb_mode=zinb_mode)
    val, val_metadata = remove_empty_datapoints(val, val_metadata)
    gc.collect()  # Clean up memory after processing val
    
    test, test_metadata = process_data(test, features, bin_size, moving_average, step_size, data_point_length, split_on, zinb_mode=zinb_mode)
    test, test_metadata = remove_empty_datapoints(test, test_metadata)
    gc.collect()  # Clean up memory after processing test

    return train, val, test, train_metadata, val_metadata, test_metadata, scalers, count_stats, clip_stats

def parse_args():
    """Parse command line arguments for preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess transposon insertion data for autoencoder training')
    
    # Input/Output
    parser.add_argument('--input_folder', type=str, default='Data/combined_strains/',
                        help='Folder containing the raw CSV files (default: Data/combined_strains/)')
    parser.add_argument('--output_dir', type=str, default='Data/processed_data/',
                        help='Directory to save processed data (default: Data/processed_data/)')
    
    # Features
    parser.add_argument('--features', type=str, nargs='+', 
                        default=['Nucl', 'Centr'],
                        choices=['Pos', 'Nucl', 'Centr', 'Chrom'],
                        help='Features to use (default: Nucl Centr)')
    
    # Data splitting
    parser.add_argument('--train_val_test_split', type=float, nargs=3, 
                        default=[0.7, 0.15, 0.15],
                        help='Train, validation, and test split proportions (default: 0.7 0.15 0.15)')
    parser.add_argument('--split_on', type=str, default='Chrom',
                        choices=['Chrom', 'Dataset', 'Random'],
                        help='Feature to split data on (default: Chrom)')
    parser.add_argument('--chunk_size', type=int, default=50000,
                        help='Size of chunks in base pairs for random splitting (default: 50000)')
    
    # Normalization
    parser.add_argument('--normalize_counts', action='store_true', default=True,
                        help='Apply CPM normalization and log transform to counts (default: True)')
    parser.add_argument('--no_normalize_counts', action='store_false', dest='normalize_counts',
                        help='Disable count normalization')
    parser.add_argument('--zinb_mode', action='store_true', default=False,
                        help='ZINB mode: save raw counts in Value_Raw column for ZINB models (default: False)')
    
    # Binning/Windowing
    parser.add_argument('--bin_size', type=int, default=10,
                        help='Bin size for binning the data (default: 10)')
    parser.add_argument('--moving_average', action='store_true', default=True,
                        help='Apply moving average to the data (default: True)')
    parser.add_argument('--no_moving_average', action='store_false', dest='moving_average',
                        help='Use separate bins instead of moving average')
    parser.add_argument('--data_point_length', type=int, default=2000,
                        help='Length of each data point (default: 2000)')
    parser.add_argument('--step_size', type=float, default=0.25,
                        help='Step size for sliding window relative to sequence length (default: 0.25 of data_point_length)')
    parser.add_argument('--no_clip_outliers', action='store_false', dest='clip_outliers',
                        help='Do not clip outliers')
    
    return parser.parse_args()

            

if __name__ == "__main__":
    args = parse_args()
    print("Starting preprocessing with the following parameters:")
    print(args)
    clip_outliers_flag = getattr(args, 'clip_outliers', True)
    step_size = int(args.data_point_length * args.step_size)
    
    # Run preprocessing with parsed arguments
    train, val, test, train_metadata, val_metadata, test_metadata, scalers, count_stats, clip_stats = preprocess(
        input_folder=args.input_folder,
        features=args.features,
        train_val_test_split=args.train_val_test_split,
        split_on=args.split_on,
        chunk_size=args.chunk_size,
        normalize_counts=args.normalize_counts,
        zinb_mode=args.zinb_mode,
        bin_size=args.bin_size,
        moving_average=args.moving_average,
        data_point_length=args.data_point_length,
        step_size=step_size,
        clip_outliers_flag=clip_outliers_flag
    )

    # Print some info
    print(f"\nProcessing complete!")
    print(f"Train data shape: {train.shape}")
    print(f"Validation data shape: {val.shape}")
    print(f"Test data shape: {test.shape}")
    
    # Print mean and standard deviation of each feature in training data
    if train.shape[0] > 0:
        feature_means = np.mean(train, axis=(0,1))
        feature_stds = np.std(train, axis=(0,1))
        print("\nTraining data feature means and standard deviations:")
        for i in range(train.shape[2]):
            print(f"  Feature {i}: mean={feature_means[i]:.4f}, std={feature_stds[i]:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    zinb_suffix = "_ZINB" if args.zinb_mode else ""
    output_name = args.output_dir + f"BinSize{args.bin_size}_MovingAvg{args.moving_average}{zinb_suffix}_"
    
    # Save the train, validation, and test data as .npy files
    train_file = output_name + f"train_data.npy"
    val_file = output_name + f"val_data.npy"
    test_file = output_name + f"test_data.npy"
    
    np.save(train_file, train)
    np.save(val_file, val)
    np.save(test_file, test)
    
    # Save metadata as JSON files
    train_meta_file = output_name + f"train_metadata.json"
    val_meta_file = output_name + f"val_metadata.json"
    test_meta_file = output_name + f"test_metadata.json"
    
    import json
    with open(train_meta_file, 'w') as f:
        json.dump(train_metadata, f, indent=2)
    with open(val_meta_file, 'w') as f:
        json.dump(val_metadata, f, indent=2)
    with open(test_meta_file, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"\nData saved to:")
    print(f"  Train: {train_file}")
    print(f"  Validation: {val_file}")
    print(f"  Test: {test_file}")
    print(f"\nMetadata saved to:")
    print(f"  Train: {train_meta_file}")
    print(f"  Validation: {val_meta_file}")
    print(f"  Test: {test_meta_file}")
