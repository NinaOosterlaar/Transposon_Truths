#!/usr/bin/env python3
"""
Create overview plots comparing detected vs true change points on centromere windows.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Signal_processing.evaluation.sliding_performance import read_change_points

setup_plot_style()

CHROMS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]

# True change points for each chromosome (centromere coordinates)
CPD = {
    "I": [-910, -780, -500, -350, -200, -70, 80, 760],
    "II": [-740, -580, -70, 65, 800],
    "III": [-900, -200, -80, 80, 750],
    "IV": [-900, -800, -740, -500, -380, -60, 60],
    "V": [-620, -280, -70, 90, 200, 480],
    "VI": [-960, -860, 460],
    "VII": [-440, -110, 70, 260, 320, 730],
    "VIII": [-860, -510, -80, 80, 200, 370, 580],
    "IX": [-400, -60, 80, 200, 300, 760],
    "X": [-920, -480, -80, 60, 200],
    "XI": [-660, -380, -70, 60, 180, 280, 470],
    "XII": [-180, -75, 70, 460],
    "XIII": [-770, -420, -300, -140, -65, 80, 780],
    "XIV": [-920, -140, -65, 80],
    "XV": [-740, -120, 60, 380, 930],
    "XVI": [-580, -170, -70, 80],
}

SATURATION = [3.2, 6.6, 13.1, 20.5, 28.8, 34.4, 39.2, 43.1]


def moving_average(data, window_size):
    """Apply a moving average filter to smooth the data, ignoring zero values.
    
    Args:
        data: 1D numpy array of values
        window_size: Size of the moving average window
    
    Returns:
        Smoothed data (same length as input), averaging only non-zero values
    """
    if window_size <= 1:
        return data
    
    result = np.zeros_like(data, dtype=float)
    half_window = window_size // 2
    
    for i in range(len(data)):
        # Define window boundaries (centered window)
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        
        # Get window data and filter out zeros
        window_data = data[start:end]
        non_zero_values = window_data[window_data != 0]
        
        # Calculate average of non-zero values only
        if len(non_zero_values) > 0:
            result[i] = np.mean(non_zero_values)
        else:
            # If all values in window are zero, keep zero
            result[i] = 0.0
    
    return result


def read_centromere_window_data(csv_path):
    """Read centromere window data with positions and values."""
    df = pd.read_csv(csv_path)
    
    if "Centromere_Distance" in df.columns:
        positions = df["Centromere_Distance"].to_numpy()
    elif "Position" in df.columns:
        positions = df["Position"].to_numpy()
    else:
        positions = np.arange(len(df)) - 1000
    
    if "value" in df.columns:
        values = pd.to_numeric(df["value"], errors="coerce").fillna(0).to_numpy()
    elif len(df.columns) >= 2:
        values = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).to_numpy()
    else:
        raise ValueError(f"Could not find signal column in {csv_path}")
    
    return positions, values


def find_result_file(results_path, chrom, threshold):
    """Find the result file for a specific threshold."""
    if not os.path.exists(results_path):
        return None
    
    result_files = [f for f in os.listdir(results_path) if f.endswith('.txt')]
    
    for result_file in result_files:
        match = re.search(r'th(\d+\.\d+)', result_file)
        if match:
            file_threshold = float(match.group(1))
            if np.isclose(file_threshold, threshold):
                return os.path.join(results_path, result_file)
    
    return None


def plot_change_points_comparison(data_path, results_path, chrom, threshold, 
                                  true_cps, output_path, saturation_level, strain):
    """Create a comparison plot of detected vs true change points."""
    
    # Read the signal data
    try:
        positions, values = read_centromere_window_data(data_path)
    except Exception as e:
        print(f"    Error reading data: {e}")
        return False
    
    # Apply 20-point moving average (ignoring zeros)
    smoothed_values = moving_average(values, window_size=20)
    
    # Find and read detected change points
    result_file = find_result_file(results_path, chrom, threshold)
    
    if result_file is None:
        print(f"    No result file found for threshold {threshold}")
        return False
    
    detected_positions = read_change_points(result_file)
    # Convert array indices to centromere coordinates
    detected_cps = [pos - 1000 for pos in detected_positions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot the smoothed signal
    ax.plot(positions, smoothed_values, linewidth=1.0, color=COLORS['blue'], alpha=0.7, label='Smoothed Signal (MA=20)')
    
    # Mark detected change points
    y_min, y_max = ax.get_ylim()
    marker_height = (y_max - y_min) * 0.05
    
    for cp in detected_cps:
        ax.axvline(x=cp, color=COLORS['orange'], linestyle='-', linewidth=1.5, alpha=0.6)
        ax.plot(cp, y_min + marker_height, marker='v', color=COLORS['orange'], 
               markersize=8, markeredgewidth=1.5)
    
    # Mark true change points
    for cp in true_cps:
        ax.axvline(x=cp, color=COLORS['green'], linestyle='--', linewidth=1.5, alpha=0.6)
        ax.plot(cp, y_min + marker_height * 2, marker='^', color=COLORS['green'], 
               markersize=8, markeredgewidth=1.5)
    
    # Mark centromere
    ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.4)
    
    # Labels and title
    ax.set_xlabel('Distance from Centromere (bp)', fontsize=12)
    ax.set_ylabel('Signal Value', fontsize=12)
    
    # Map saturation level (1-6) to saturation percentage (using 1-6 as indices in SATURATION array)
    sat_pct = SATURATION[saturation_level] if saturation_level < len(SATURATION) else None
    sat_label = f"{sat_pct:.1f}%" if sat_pct is not None else f"Level {saturation_level}"
    
    title = f'Chromosome {chrom} - Saturation {sat_label} ({strain})\n'
    title += f'Threshold: {threshold:.1f} | Detected: {len(detected_cps)} CPs | True: {len(true_cps)} CPs'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['blue'], linewidth=2, label='Smoothed Signal (MA=20)', alpha=0.7),
        Line2D([0], [0], color=COLORS['orange'], linewidth=2, label='Detected CPs', alpha=0.6),
        Line2D([0], [0], color=COLORS['green'], linewidth=2, linestyle='--', label='True CPs', alpha=0.6),
        Line2D([0], [0], color='red', linewidth=2, linestyle=':', label='Centromere', alpha=0.4)
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return True


def main():
    """Generate overview plots for all saturation levels and chromosomes."""
    
    # Configuration
    results_base = "Signal_processing/results/centromere_cpd_results"
    data_base = "Data/test_CPD/centromere_windows"
    output_base = "Signal_processing/results/centromere_cpd_results/overview_plots"
    
    saturation_levels = list(range(1, 7))  # Folders 1-6
    strain = "yEK23_1"
    threshold = 7.0
    window_size = 100
    
    print("="*80)
    print("Generating Change Point Overview Plots")
    print("="*80)
    print(f"Results base: {results_base}")
    print(f"Data base: {data_base}")
    print(f"Output base: {output_base}")
    print(f"Strain: {strain}")
    print(f"Threshold: {threshold}")
    print(f"Saturation levels: {saturation_levels}")
    print("="*80)
    
    total_plots = 0
    successful_plots = 0
    
    for sat_level in saturation_levels:
        print(f"\nProcessing Saturation Level {sat_level}...")
        
        sat_plots = 0
        
        for chrom in CHROMS:
            if chrom not in CPD or len(CPD[chrom]) == 0:
                continue
            
            # Paths
            data_path = os.path.join(data_base, str(sat_level), strain, f"Chr{chrom}_centromere_window.csv")
            
            results_path = os.path.join(
                results_base,
                str(sat_level),
                strain,
                f"Chr{chrom}",
                f"Chr{chrom}_centromere_window",
                f"window{window_size}"
            )
            
            output_path = os.path.join(
                output_base,
                f"saturation_{sat_level}",
                f"Chr{chrom}_sat{sat_level}_th{threshold:.0f}.png"
            )
            
            # Check if data exists
            if not os.path.exists(data_path):
                print(f"  Chr{chrom}: Data file not found")
                continue
            
            # Get true change points
            true_cps = CPD[chrom]
            
            # Create plot
            total_plots += 1
            success = plot_change_points_comparison(
                data_path, results_path, chrom, threshold,
                true_cps, output_path, sat_level, strain
            )
            
            if success:
                successful_plots += 1
                sat_plots += 1
            else:
                print(f"  Chr{chrom}: Failed to create plot")
        
        print(f"  Created {sat_plots} plots for saturation level {sat_level}")
    
    print("\n" + "="*80)
    print(f"Plot generation complete!")
    print(f"Successfully created {successful_plots}/{total_plots} plots")
    print(f"Plots saved to: {output_base}")
    print("="*80)


if __name__ == "__main__":
    main()
