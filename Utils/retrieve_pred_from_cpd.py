"""
Script to retrieve predicted change points from sliding ZINB CPD results
and generate the pred dictionary for use in plot_SATAY.py
"""

import os, sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def retrieve_pred_from_cpd(window_size, threshold, base_dir):
    """
    Retrieve predicted change points from sliding ZINB CPD results.
    
    Parameters:
    -----------
    window_size : int
        The window size used in the sliding window analysis (e.g., 50, 80)
    threshold : float
        The threshold value used to detect change points (e.g., 10.0)
    base_dir : str or Path, optional
        Base directory containing the results. If None, uses default path.
    
    Returns:
    --------
    dict
        Dictionary mapping chromosome names to lists of change points
        Format: {"I": [640, 800, ...], "II": [...], ...}
    """

    
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Chromosome names (Roman numerals)
    CHROMS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", 
              "XI", "XII", "XIII", "XIV", "XV", "XVI"]
    
    pred = {}
    
    for chrom in CHROMS:
        chrom_dir = base_dir / f"Chr{chrom}"
        
        if not chrom_dir.exists():
            print(f"Warning: Directory not found for chromosome {chrom}: {chrom_dir}")
            pred[chrom] = []
            continue
        
        window_dir = chrom_dir / f"window{window_size}"
        
        if not window_dir.exists():
            print(f"Warning: Window directory not found for Chr{chrom}: {window_dir}")
            pred[chrom] = []
            continue
        
        # Construct filename based on pattern: Chr{X}_centromere_window_ws{W}_ov50_th{T}.txt
        filename = f"Chr{chrom}_centromere_window_ws{window_size}_ov50_th{threshold:.2f}.txt"
        filepath = window_dir / filename
        
        if not filepath.exists():
            print(f"Warning: File not found for Chr{chrom}: {filepath}")
            pred[chrom] = []
            continue
        
        # Read change points from file
        change_points = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Stop when we hit the scores line
                    if line.startswith("scores:"):
                        break
                    # Try to parse as integer
                    if line:
                        try:
                            change_points.append(int(line))
                        except ValueError:
                            # Skip lines that aren't integers
                            pass
            
            pred[chrom] = change_points
            print(f"Chr{chrom}: Found {len(change_points)} change points")
            
        except Exception as e:
            print(f"Error reading file for Chr{chrom}: {e}")
            pred[chrom] = []
    
    return pred


def format_pred_dict(pred_dict):
    """
    Format the pred dictionary as Python code for easy copying.
    
    Parameters:
    -----------
    pred_dict : dict
        Dictionary of change points
    
    Returns:
    --------
    str
        Formatted Python dictionary code
    """
    lines = ["pred = {"]
    for chrom, points in pred_dict.items():
        points_str = ", ".join(map(str, points))
        lines.append(f'    "{chrom}": [{points_str}],')
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    window_size = 100
    threshold = 10.0
    base_dir = Path("Signal_processing/results/sliding_mean_SATAY/sliding_ZINB_CPD")
    
    print(f"Retrieving change points for window_size={window_size}, threshold={threshold}")
    print("-" * 70)
    
    pred = retrieve_pred_from_cpd(window_size, threshold, base_dir)
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    # Print formatted dictionary
    formatted = format_pred_dict(pred)
    print(formatted)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    total_points = sum(len(points) for points in pred.values())
    print(f"Total change points across all chromosomes: {total_points}")
    for chrom, points in pred.items():
        print(f"  Chr{chrom}: {len(points)} change points")
