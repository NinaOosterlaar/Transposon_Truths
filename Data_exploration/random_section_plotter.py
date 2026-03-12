import os
import sys
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.plot_config import setup_plot_style, COLORS

# Set up standardized plot style
setup_plot_style()


def get_all_datasets(base_path):
    """
    Get all available datasets from the distances_with_zeros directory.
    
    Returns:
        list: List of tuples (strain_folder, dataset_folder, dataset_path)
    """
    datasets = []
    
    for strain_folder in os.listdir(base_path):
        strain_path = os.path.join(base_path, strain_folder)
        
        if not os.path.isdir(strain_path):
            continue
            
        for dataset_folder in os.listdir(strain_path):
            dataset_path = os.path.join(strain_path, dataset_folder)
            
            if os.path.isdir(dataset_path):
                datasets.append((strain_folder, dataset_folder, dataset_path))
    
    return datasets


def get_chromosomes(dataset_path):
    """
    Get all available chromosome files from a dataset.
    
    Args:
        dataset_path (str): Path to the dataset folder
        
    Returns:
        list: List of tuples (chromosome_name, file_path)
    """
    chromosomes = []
    
    for file in os.listdir(dataset_path):
        if file.endswith('_distances.csv'):
            chromosome_name = file.replace('_distances.csv', '')
            file_path = os.path.join(dataset_path, file)
            chromosomes.append((chromosome_name, file_path))
    
    return chromosomes


def plot_random_section(data_path, num_plots=10, section_size=10000):
    """
    Plot random sections from random datasets.
    
    Args:
        data_path (str): Path to the distances_with_zeros directory
        num_plots (int): Number of random plots to generate
        section_size (int): Size of each section to plot (default: 10000)
    """
    # Get all available datasets
    datasets = get_all_datasets(data_path)
    
    if not datasets:
        print("No datasets found!")
        return
    
    print(f"Found {len(datasets)} datasets")
    print(f"Generating {num_plots} random plots...\n")
    
    for i in range(num_plots):
        # Randomly select a dataset
        strain_folder, dataset_folder, dataset_path = random.choice(datasets)
        
        # Get available chromosomes
        chromosomes = get_chromosomes(dataset_path)
        
        if not chromosomes:
            print(f"No chromosomes found in {dataset_folder}")
            continue
        
        # Randomly select a chromosome
        chromosome_name, chromosome_file = random.choice(chromosomes)
        
        # Read the data
        df = pd.read_csv(chromosome_file)
        
        # Get data dimensions
        total_positions = len(df)
        
        if total_positions < section_size:
            print(f"Chromosome {chromosome_name} in {dataset_folder} has fewer than {section_size} positions")
            # Use the entire chromosome if it's smaller than section_size
            start_pos = 0
            end_pos = total_positions
        else:
            # Randomly select a start position
            max_start = total_positions - section_size
            start_pos = random.randint(0, max_start)
            end_pos = start_pos + section_size
        
        # Extract the section
        section = df.iloc[start_pos:end_pos]
        
        # Create individual figure for each plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot
        ax.plot(section['Position'], section['Value'], linewidth=0.8, color='#2E86AB')
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Value (Count)', fontsize=12)
        
        # Create a clean dataset name
        dataset_display = dataset_folder.split('_')[0]  # Get the first part before underscore
        
        ax.set_title(f'{dataset_display} - {chromosome_name}\n(Positions {section["Position"].iloc[0]:,} - {section["Position"].iloc[-1]:,})', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        print(f"Plot {i+1}/{num_plots}: {dataset_folder} - {chromosome_name} (positions {start_pos}-{end_pos})")
        
        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    # Set the base path to the distances_with_zeros directory
    base_path = '/Users/ninaoosterlaar/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/Master/MEP/Thesis/Data/distances_with_zeros'
    
    # Generate random plots
    plot_random_section(
        data_path=base_path,
        num_plots=10,
        section_size=5000
    )


if __name__ == "__main__":
    main()
