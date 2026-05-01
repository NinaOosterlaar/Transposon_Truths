"""
Analyze ZINB parameter statistics for specific strains.

Filters data where theta < 1 and pi > 0.4, then calculates mean and 
standard deviation of mu and theta for selected strains.
"""

import pandas as pd
import numpy as np
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "results/ZINB_estimates/zinb_estimates_windows_size2000.csv")
STRAINS = ['strain_FD', 'strain_yEK19', 'strain_yEK23', 'strain_dnrp']
THETA_THRESHOLD = 1.0
PI_THRESHOLD = 0.4

def analyze_zinb_parameters():
    """Analyze ZINB parameters with filtering criteria."""
    
    # Load data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Total records loaded: {len(df)}")
    
    # Filter by strains
    df_filtered = df[df['strain'].isin(STRAINS)].copy()
    print(f"\nRecords for selected strains {STRAINS}: {len(df_filtered)}")
    
    # Apply filtering criteria: theta < 1 AND pi > 0.4
    mask = (df_filtered['theta'] < THETA_THRESHOLD) & (df_filtered['pi'] > PI_THRESHOLD)
    df_filtered = df_filtered[mask]
    
    print(f"\nRecords after filtering (theta < {THETA_THRESHOLD} AND pi > {PI_THRESHOLD}): {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("\nNo data points match the filtering criteria!")
        return
    
    print("\n" + "="*80)
    print("STATISTICS BY STRAIN")
    print("="*80)
    
    # Calculate statistics per strain
    for strain in STRAINS:
        strain_data = df_filtered[df_filtered['strain'] == strain]
        
        if len(strain_data) == 0:
            print(f"\n{strain}: No data points match criteria")
            continue
        
        mu_mean = strain_data['mu'].mean()
        mu_std = strain_data['mu'].std()
        theta_mean = strain_data['theta'].mean()
        theta_std = strain_data['theta'].std()
        
        print(f"\n{strain} (n={len(strain_data)} windows):")
        print(f"  mu:    mean = {mu_mean:.6f}, std = {mu_std:.6f}")
        print(f"  theta: mean = {theta_mean:.6f}, std = {theta_std:.6f}")
    
    print("\n" + "="*80)
    print("COMBINED STATISTICS (ALL STRAINS)")
    print("="*80)
    
    # Calculate combined statistics
    mu_mean_combined = df_filtered['mu'].mean()
    mu_std_combined = df_filtered['mu'].std()
    theta_mean_combined = df_filtered['theta'].mean()
    theta_std_combined = df_filtered['theta'].std()
    
    print(f"\nCombined (n={len(df_filtered)} windows across all strains):")
    print(f"  mu:    mean = {mu_mean_combined:.6f}, std = {mu_std_combined:.6f}")
    print(f"  theta: mean = {theta_mean_combined:.6f}, std = {theta_std_combined:.6f}")
    
    # Additional summary statistics
    print("\n" + "="*80)
    print("ADDITIONAL SUMMARY")
    print("="*80)
    
    print("\nDistribution across strains:")
    strain_counts = df_filtered['strain'].value_counts()
    for strain in STRAINS:
        count = strain_counts.get(strain, 0)
        percentage = (count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        print(f"  {strain}: {count} windows ({percentage:.1f}%)")
    
    print("\nParameter ranges:")
    print(f"  mu:    min = {df_filtered['mu'].min():.6f}, max = {df_filtered['mu'].max():.6f}")
    print(f"  theta: min = {df_filtered['theta'].min():.6f}, max = {df_filtered['theta'].max():.6f}")
    print(f"  pi:    min = {df_filtered['pi'].min():.6f}, max = {df_filtered['pi'].max():.6f}")

if __name__ == "__main__":
    analyze_zinb_parameters()
