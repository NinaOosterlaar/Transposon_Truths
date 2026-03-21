import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from Utils.reader import read_csv_file_with_distances
import random
import itertools
import shutil

# Mapping of replicate names to their strain folders
replicate_to_strain = {
    "FD7": "strain_FD",
    "FD9": "strain_FD",
    "FD11": "strain_FD",
    "FD12": "strain_FD",
    "dnrp1-1": "strain_dnrp",
    "dnrp1-2": "strain_dnrp",
}

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


def get_strain_folder(dataset_name):
    """Determine the strain folder for a dataset."""
    # Check if it's a combined replicate
    if dataset_name in replicate_to_strain:
        return replicate_to_strain[dataset_name]
    
    # Try to infer from dataset name
    if dataset_name.startswith("FD"):
        return "strain_FD"
    elif dataset_name.startswith("dnrp"):
        return "strain_dnrp"
    elif dataset_name.startswith("yEK19"):
        return "strain_yEK19"
    elif dataset_name.startswith("yEK23"):
        return "strain_yEK23"
    elif dataset_name.startswith("yTW001"):
        return "strain_yTW001"
    elif dataset_name.startswith("yWT03"):
        return "strain_yWT03a"
    elif dataset_name.startswith("yWT04"):
        return "strain_yWT04a"
    elif dataset_name.startswith("yLIC") or dataset_name.startswith("ylic"):
        return "strain_ylic137"
    else:
        return "strain_unknown"


def combine_replicates(data, replicate_names, method="average", save=True, output_folder="Data/combined_replicates/"):
    """Combine replicate datasets by averaging or summing their data.
    Assumes replicate datasets have names containing replicate identifiers.
    Every dataset point in the dataset is combined using the specified method.
    
    Saves all datasets in the same folder structure as the original data:
    - Combined replicates (e.g., FD7, FD9) go into their strain folders (strain_FD)
    - Non-replicate datasets are copied as-is into their strain folders
    
    Args:
        data (Dictionary): Dictionary containing chromosome DataFrames for each dataset.
        replicate_names (list): List of replicate names to combine (e.g., ["FD7", "FD9"]).
        method (str): Method to combine replicates, either "average" or "sum".
        save (bool): Whether to save combined data to CSV files.
        output_folder (str): Base output folder path.
    Returns:
        new_data (Dictionary): Dictionary with combined replicate datasets.
    """
    new_data = {}
    datasets_to_remove = []
    
    # Step 1: Combine replicates
    for replicate_name in replicate_names:
        # Find all datasets that match this replicate name
        matching_datasets = [dataset for dataset in data if replicate_name in dataset]
        
        if not matching_datasets:
            print(f"No datasets found for replicate: {replicate_name}")
            continue
        
        print(f"Combining {len(matching_datasets)} datasets for replicate: {replicate_name}")
        combined_regions = {}
        
        for chrom in chromosome_length.keys():
            # Initialize a dictionary to accumulate values by position
            position_data = {}
            
            # Accumulate data from all matching datasets
            for dataset in matching_datasets:
                if chrom not in data[dataset]:
                    continue
                
                df = data[dataset][chrom]
                
                for _, row in df.iterrows():
                    pos = int(row['Position'])
                    value = row['Value']
                    nuc_dist = row['Nucleosome_Distance']
                    cent_dist = row['Centromere_Distance']
                    
                    if pos not in position_data:
                        position_data[pos] = {
                            'values': [],
                            'nucleosome_distance': nuc_dist,
                            'centromere_distance': cent_dist
                        }
                    position_data[pos]['values'].append(value)
            
            # Compute combined values for this chromosome
            combined_data = []
            for pos in sorted(position_data.keys()):
                values = position_data[pos]['values']
                
                if method == "average":
                    # Only consider non-zero values for averaging
                    non_zero_values = [v for v in values if v != 0]
                    
                    if len(non_zero_values) == 0:
                        # All values are zero
                        combined_value = 0
                    elif len(non_zero_values) == 1:
                        # Only one non-zero value, use it directly
                        combined_value = non_zero_values[0]
                    else:
                        # Two or more non-zero values, take the average
                        combined_value = np.mean(non_zero_values)
                elif method == "sum":
                    combined_value = np.sum(values)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                combined_data.append({
                    'Position': pos,
                    'Value': combined_value,
                    'Nucleosome_Distance': position_data[pos]['nucleosome_distance'],
                    'Centromere_Distance': position_data[pos]['centromere_distance']
                })
            
            # Convert to DataFrame
            if combined_data:
                combined_regions[chrom] = pd.DataFrame(combined_data)
            else:
                combined_regions[chrom] = pd.DataFrame(columns=['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance'])
        
        new_data[replicate_name] = combined_regions
        
        # Mark original replicate datasets for removal
        for dataset in matching_datasets:
            if dataset != replicate_name:
                datasets_to_remove.append(dataset)
    
    # Step 2: Remove original replicate datasets from data
    for dataset in datasets_to_remove:
        del data[dataset]
    
    # Step 3: Add combined data to the data dictionary
    data.update(new_data)
    
    # Step 4: Save all datasets if requested
    if save:
        os.makedirs(output_folder, exist_ok=True)
        
        # Save all datasets
        for dataset in data:
            strain_folder = get_strain_folder(dataset)
            dataset_folder = os.path.join(output_folder, strain_folder, dataset)
            os.makedirs(dataset_folder, exist_ok=True)
            
            for chrom in data[dataset]:
                output_path = os.path.join(dataset_folder, f"{chrom}_distances.csv")
                df = data[dataset][chrom]
                df.to_csv(output_path, index=False)
            
            print(f"Saved data for {dataset} to {dataset_folder}")
    
    return data


def combine_strain_datasets(input_folder, output_folder, method="average"):
    """
    Combine all biological replicates within each strain folder into a single dataset per strain.
    
    For example, in strain_FD folder with subfolders FD7, FD9, FD11, FD12,
    this will combine all of them into a single strain_FD dataset.
    
    Args:
        input_folder (str): Path to combined_replicates folder containing strain subfolders.
        output_folder (str): Path where to save the combined strain datasets.
        method (str): Method to combine datasets, either "average" or "sum".
    
    Returns:
        strain_data (dict): Dictionary with strain names as keys and chromosome DataFrames as values.
    """
    strain_data = {}
    
    # Iterate through each strain folder
    for strain_folder in os.listdir(input_folder):
        strain_path = os.path.join(input_folder, strain_folder)
        
        if not os.path.isdir(strain_path) or strain_folder.startswith('.'):
            continue
        
        print(f"\nProcessing strain: {strain_folder}")
        
        # Get all replicate folders within this strain
        replicate_folders = [f for f in os.listdir(strain_path) 
                           if os.path.isdir(os.path.join(strain_path, f)) and not f.startswith('.')]
        
        if not replicate_folders:
            print(f"  No replicate folders found in {strain_folder}")
            continue
        
        print(f"  Found {len(replicate_folders)} replicates: {replicate_folders}")
        
        # Initialize combined data for this strain
        combined_strain = {}
        
        # Process each chromosome
        for chrom in chromosome_length.keys():
            position_data = {}
            
            # Collect data from all replicates for this chromosome
            for replicate in replicate_folders:
                csv_file = os.path.join(strain_path, replicate, f"{chrom}_distances.csv")
                
                if not os.path.exists(csv_file):
                    print(f"  Warning: {csv_file} not found, skipping")
                    continue
                
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    pos = int(row['Position'])
                    value = row['Value']
                    nuc_dist = row['Nucleosome_Distance']
                    cent_dist = row['Centromere_Distance']
                    
                    if pos not in position_data:
                        position_data[pos] = {
                            'values': [],
                            'nucleosome_distance': nuc_dist,
                            'centromere_distance': cent_dist
                        }
                    position_data[pos]['values'].append(value)
            
            # Compute combined values for this chromosome
            combined_data = []
            for pos in sorted(position_data.keys()):
                values = position_data[pos]['values']
                
                if method == "average":
                    # Only consider non-zero values for averaging
                    non_zero_values = [v for v in values if v != 0]
                    
                    if len(non_zero_values) == 0:
                        # All values are zero
                        combined_value = 0.0
                    elif len(non_zero_values) == 1:
                        # Only one non-zero value, use it directly
                        combined_value = non_zero_values[0]
                    else:
                        # Two or more non-zero values, take the average
                        combined_value = np.mean(non_zero_values)
                elif method == "sum":
                    combined_value = np.sum(values)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                combined_data.append({
                    'Position': pos,
                    'Value': combined_value,
                    'Nucleosome_Distance': position_data[pos]['nucleosome_distance'],
                    'Centromere_Distance': position_data[pos]['centromere_distance']
                })
            
            # Convert to DataFrame
            if combined_data:
                combined_strain[chrom] = pd.DataFrame(combined_data)
            else:
                combined_strain[chrom] = pd.DataFrame(columns=['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance'])
        
        # Save this strain's combined data
        strain_data[strain_folder] = combined_strain
        
        # Save to CSV files
        output_strain_folder = os.path.join(output_folder, strain_folder)
        os.makedirs(output_strain_folder, exist_ok=True)
        
        for chrom, df in combined_strain.items():
            output_path = os.path.join(output_strain_folder, f"{chrom}_distances.csv")
            df.to_csv(output_path, index=False)
        
        print(f"  Saved combined {strain_folder} data to {output_strain_folder}")
    
    return strain_data


def create_saturation_test_combinations(
    source_folder="Data/test_CPD/1",
    output_base="Data/test_CPD",
    max_combinations=6,
    random_seed=42
):
    """
    Create random combinations of datasets for saturation testing.
    
    For k in [2, 3, 4, 5], generates up to max_combinations random combinations
    of k datasets from the source folder and saves them to output_base/{k}/yEK23_{i}/.
    
    The combining method averages non-zero values per position, matching the
    behavior in AE/test_saturation.py.
    
    Args:
        source_folder (str): Path to folder containing individual datasets (folder 1)
        output_base (str): Base path for test_CPD folders
        max_combinations (int): Maximum number of combinations to create per k
        random_seed (int): Random seed for reproducibility
    """

    
    # Set random seed for reproducibility
    rng = random.Random(random_seed)
    
    # Get list of source dataset folders
    source_datasets = sorted([
        d for d in os.listdir(source_folder)
        if os.path.isdir(os.path.join(source_folder, d)) and not d.startswith('.')
    ])
    
    n_datasets = len(source_datasets)
    print(f"Found {n_datasets} source datasets in {source_folder}:")
    for ds in source_datasets:
        print(f"  {ds}")
    
    # Get list of all chromosomes from the first dataset
    first_dataset_path = os.path.join(source_folder, source_datasets[0])
    chromosomes = sorted([
        fname.split("_")[0]
        for fname in os.listdir(first_dataset_path)
        if fname.endswith("_distances.csv") and not fname.startswith("ChrM")
    ])
    
    print(f"\nFound {len(chromosomes)} chromosomes to process")
    
    # Process k=2,3,4,5
    for k in range(2, 6):
        print(f"\n{'='*60}")
        print(f"Processing k={k} (combining {k} datasets)")
        print(f"{'='*60}")
        
        # Generate all possible combinations and randomly sample
        all_combos = list(itertools.combinations(range(n_datasets), k))
        selected_combos = rng.sample(all_combos, min(max_combinations, len(all_combos)))
        
        print(f"Selected {len(selected_combos)} combinations out of {len(all_combos)} possible")
        
        # Create output folder for this k
        k_folder = os.path.join(output_base, str(k))
        
        # Process each selected combination
        for combo_idx, combo in enumerate(selected_combos, 1):
            # Get dataset names for this combination
            datasets_in_combo = [source_datasets[i] for i in combo]
            output_folder = os.path.join(k_folder, f"yEK23_{combo_idx}")
            
            print(f"\n  [{combo_idx}/{len(selected_combos)}] Creating: {output_folder}")
            print(f"    Combining: {', '.join(datasets_in_combo)}")
            
            # Remove existing folder if it exists (overwrite)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder, exist_ok=True)
            
            # Process each chromosome
            for chrom in chromosomes:
                # Read all datasets for this chromosome
                frames = []
                for dataset in datasets_in_combo:
                    csv_path = os.path.join(source_folder, dataset, f"{chrom}_distances.csv")
                    if os.path.exists(csv_path):
                        frames.append(pd.read_csv(csv_path))
                
                if not frames:
                    continue
                
                # Combine all datasets
                all_data = pd.concat(frames, ignore_index=True)
                
                # For each position, keep first-seen auxiliary columns (distances)
                aux = (
                    all_data.groupby('Position', sort=True)
                    [['Nucleosome_Distance', 'Centromere_Distance']]
                    .first()
                )
                
                # Average non-zero values per position
                def avg_nonzero(series):
                    nz = series[series != 0]
                    return nz.mean() if len(nz) > 0 else 0.0
                
                combined_values = (
                    all_data.groupby('Position', sort=True)['Value']
                    .agg(avg_nonzero)
                )
                
                # Construct output DataFrame
                result = aux.copy()
                result['Value'] = combined_values
                result = result.reset_index()[['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance']]
                
                # Save to CSV
                out_path = os.path.join(output_folder, f"{chrom}_distances.csv")
                result.to_csv(out_path, index=False)
            
            print(f"    ✓ Saved {len(chromosomes)} chromosome files")
    
    print(f"\n{'='*60}")
    print("Saturation test combinations complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Create saturation test combinations
    create_saturation_test_combinations(
        source_folder="Data/test_CPD/1",
        output_base="Data/test_CPD",
        max_combinations=6,
        random_seed=42
    )
    
    # Original code (commented out):
    # data = read_csv_file_with_distances("Data/distances_with_zeros")
    # # Example usage: Combine all biological replicates within each strain
    # combine_replicates(
    #     data=data,  # Load your data dictionary here
    #     replicate_names=["FD7", "FD9", "dnrp1-1", "dnrp1-2"],
    #     method="average",
    #     save=True,
    #     output_folder="Data/combined_replicates/"
    # )
    # combine_strain_datasets(
    #     input_folder="Data/combined_replicates",
    #     output_folder="Data/combined_strains",
    #     method="average"
    # )
