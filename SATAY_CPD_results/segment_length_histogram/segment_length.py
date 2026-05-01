"""
Analyze segment lengths from change point detection algorithm.
Creates histograms for unmerged and merged segments across multiple strains.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils.plot_config import setup_plot_style, COLORS


# Configuration
STRAINS = ['strain_FD', 'strain_yEK19', 'strain_yEK23']
CHROMOSOMES = ['ChrI', 'ChrII', 'ChrIII', 'ChrIV', 'ChrV', 'ChrVI', 
               'ChrVII', 'ChrVIII', 'ChrIX', 'ChrX', 'ChrXI', 'ChrXII',
               'ChrXIII', 'ChrXIV', 'ChrXV', 'ChrXVI']
THRESHOLD = 3
MU_Z = 0.25
WINDOW_SIZE = 100
OVERLAP = 50
BASE_DIR = Path(__file__).parent.parent.parent  # Thesis directory
OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'segment_length_histogram'
PERCENTILE_CUTOFF = 95  # Remove top 5%


def load_unmerged_segments(strain: str, chromosome: str):
    """Load unmerged segment file and calculate segment lengths."""
    file_path = (BASE_DIR / 'SATAY_CPD_results' / 'CPD_SATAY_results' / strain / 
                 chromosome / f'{chromosome}_distances' / f'window{WINDOW_SIZE}' /
                 f'{chromosome}_distances_ws{WINDOW_SIZE}_ov{OVERLAP}_th{THRESHOLD:.2f}.txt')
    
    if not file_path.exists():
        return np.array([])
    
    # Read file line by line, stopping at metadata
    change_points = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ',' in line or ':' in line:
                break
            try:
                change_points.append(float(line))
            except ValueError:
                break
    
    change_points = np.array(change_points)
    if len(change_points) > 1:
        return np.diff(change_points)
    return np.array([])


def load_merged_segments(strain: str, chromosome: str):
    """Load merged segment file and extract segment lengths."""
    file_path = (BASE_DIR / 'SATAY_CPD_results' / 'CPD_SATAY_results' / strain / 
                 chromosome / f'{chromosome}_distances' / f'window{WINDOW_SIZE}' /
                 'merged_segments' / f'{chromosome}_th{THRESHOLD:.2f}_merged_segments_muZ{MU_Z:.2f}.csv')
    
    if not file_path.exists():
        return np.array([])
    
    df = pd.read_csv(file_path)
    return df['length'].values


def load_merged_segments_full(strain: str, chromosome: str):
    """Load merged segment file with all columns."""
    file_path = (BASE_DIR / 'SATAY_CPD_results' / 'CPD_SATAY_results' / strain / 
                 chromosome / f'{chromosome}_distances' / f'window{WINDOW_SIZE}' /
                 'merged_segments' / f'{chromosome}_th{THRESHOLD:.2f}_merged_segments_muZ{MU_Z:.2f}.csv')
    
    if not file_path.exists():
        return None
    
    return pd.read_csv(file_path)


def collect_all_segment_lengths(segment_type='unmerged'):
    """Collect segment lengths from all chromosomes and strains."""
    data = {strain: [] for strain in STRAINS}
    load_func = load_unmerged_segments if segment_type == 'unmerged' else load_merged_segments
    
    for strain in STRAINS:
        print(f"Processing {strain} ({segment_type})...")
        for chromosome in CHROMOSOMES:
            lengths = load_func(strain, chromosome)
            if len(lengths) > 0:
                data[strain].extend(lengths)
        data[strain] = np.array(data[strain])
        print(f"  {strain}: {len(data[strain])} segments")
    
    return data


def collect_scores_for_length_100():
    """Collect mu_z_scores for all segments of length 100."""
    scores_dict = {strain: [] for strain in STRAINS}
    
    for strain in STRAINS:
        print(f"Processing {strain} for length-100 scores...")
        for chromosome in CHROMOSOMES:
            df = load_merged_segments_full(strain, chromosome)
            if df is not None and not df.empty:
                # Filter for segments of length 100
                length_100 = df[df['length'] == 100]
                if not length_100.empty:
                    scores_dict[strain].extend(length_100['mu_z_score'].values)
        
        scores_dict[strain] = np.array(scores_dict[strain])
        print(f"  {strain}: {len(scores_dict[strain])} segments of length 100")
    
    return scores_dict


def filter_outliers(data, percentile=95):
    """Remove top percentile values from each strain's data."""
    filtered = {}
    for strain, lengths in data.items():
        cutoff = np.percentile(lengths, percentile)
        filtered[strain] = lengths[lengths <= cutoff]
        removed = len(lengths) - len(filtered[strain])
        print(f"{strain}: Removed {removed} outliers (>{cutoff:.0f} bp), kept {len(filtered[strain])} segments")
    return filtered


def save_summary_statistics(data, segment_type):
    """Save summary statistics to CSV file."""
    stats_list = []
    for strain in STRAINS:
        stats = {
            'strain': strain,
            'type': segment_type,
            'count': len(data[strain]),
            'mean': np.mean(data[strain]),
            'median': np.median(data[strain]),
            'std': np.std(data[strain]),
            'min': np.min(data[strain]),
            'max': np.max(data[strain]),
            'q25': np.percentile(data[strain], 25),
            'q75': np.percentile(data[strain], 75),
            'q95': np.percentile(data[strain], 95),
        }
        stats_list.append(stats)
    
    combined = np.concatenate([data[strain] for strain in STRAINS])
    stats_list.append({
        'strain': 'combined',
        'type': segment_type,
        'count': len(combined),
        'mean': np.mean(combined),
        'median': np.median(combined),
        'std': np.std(combined),
        'min': np.min(combined),
        'max': np.max(combined),
        'q25': np.percentile(combined, 25),
        'q75': np.percentile(combined, 75),
        'q95': np.percentile(combined, 95),
    })
    
    df = pd.DataFrame(stats_list)
    output_file = OUTPUT_DIR / f'segment_length_statistics_{segment_type}.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved statistics to {output_file}")


def create_histogram(data, title, filename, color='steelblue'):
    """Create and save a histogram as a bar plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    unique_lengths, counts = np.unique(data, return_counts=True)
    mask = unique_lengths >= 100
    unique_lengths = unique_lengths[mask]
    counts = counts[mask]
    
    ax.bar(unique_lengths, counts, width=45, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Segment Length (bp)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    max_length = np.max(unique_lengths)
    if max_length <= 1000:
        tick_interval = 50
    elif max_length <= 2000:
        tick_interval =100
    else:
        tick_interval = 200
    ax.set_xticks(np.arange(100, max_length + tick_interval, tick_interval))
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {OUTPUT_DIR / filename}")


def create_score_histogram(scores, title, filename, color='steelblue'):
    """Create histogram of mu_z_scores."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram with reasonable bins
    ax.hist(scores, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('μ Z-score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved score histogram to {OUTPUT_DIR / filename}")


def create_comparison_plot(unmerged_data, merged_data):
    """Create side-by-side comparison of unmerged and merged segment lengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    unique_lengths_unm, counts_unm = np.unique(unmerged_data, return_counts=True)
    mask_unm = unique_lengths_unm >= 100
    unique_lengths_unm = unique_lengths_unm[mask_unm]
    counts_unm = counts_unm[mask_unm]
    
    unique_lengths_mer, counts_mer = np.unique(merged_data, return_counts=True)
    mask_mer = unique_lengths_mer >= 100
    unique_lengths_mer = unique_lengths_mer[mask_mer]
    counts_mer = counts_mer[mask_mer]
    
    max_count = max(np.max(counts_unm), np.max(counts_mer))
    y_limit = max_count * 1.05
    
    max_length = max(np.max(unique_lengths_unm), np.max(unique_lengths_mer))
    if max_length <= 1000:
        tick_interval = 100
    elif max_length <= 2000:
        tick_interval = 200
    else:
        tick_interval = 250
    
    ax1.bar(unique_lengths_unm, counts_unm, width=45, color=COLORS['blue'], 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Segment Length (bp)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Unmerged Segments')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, y_limit])
    ax1.set_xticks(np.arange(100, max_length + tick_interval, tick_interval))
    ax1.tick_params(axis='x', rotation=45)
    ax1.text(-0.1, 1.05, 'a', transform=ax1.transAxes, fontsize=24, 
             fontweight='bold', va='bottom', ha='right')
    
    ax2.bar(unique_lengths_mer, counts_mer, width=45, color=COLORS['blue'], 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Segment Length (bp)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Merged Segments')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, y_limit])
    ax2.set_xticks(np.arange(100, max_length + tick_interval, tick_interval))
    ax2.tick_params(axis='x', rotation=45)
    ax2.text(-0.1, 1.05, 'b', transform=ax2.transAxes, fontsize=24, 
             fontweight='bold', va='bottom', ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'segment_length_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {OUTPUT_DIR / 'segment_length_comparison.png'}")


def create_score_histograms(scores_dict):
    """Create histograms of scores for segments of length 100."""
    print(f"\nCreating score histograms for length-100 segments...")
    colors = {'strain_FD': COLORS['blue'], 'strain_yEK19': COLORS['orange'], 'strain_yEK23': COLORS['green']}
    
    # Individual strain histograms
    for strain in STRAINS:
        title = f'μ Z-score Distribution - {strain}\n(Segments of length 100, th={THRESHOLD}, muZ={MU_Z})'
        filename = f'scores_length100_{strain}_histogram.png'
        create_score_histogram(scores_dict[strain], title, filename, color=colors[strain])
    
    # Combined histogram
    combined = np.concatenate([scores_dict[strain] for strain in STRAINS])
    title = f'μ Z-score Distribution - All Strains Combined\n(Segments of length 100, th={THRESHOLD}, muZ={MU_Z})'
    filename = f'scores_length100_combined_histogram.png'
    create_score_histogram(combined, title, filename, color=COLORS['blue'])
    
    # Save statistics
    stats_list = []
    for strain in STRAINS:
        stats = {
            'strain': strain,
            'count': len(scores_dict[strain]),
            'mean': np.mean(scores_dict[strain]),
            'median': np.median(scores_dict[strain]),
            'std': np.std(scores_dict[strain]),
            'min': np.min(scores_dict[strain]),
            'max': np.max(scores_dict[strain]),
            'q25': np.percentile(scores_dict[strain], 25),
            'q75': np.percentile(scores_dict[strain], 75),
        }
        stats_list.append(stats)
    
    stats_list.append({
        'strain': 'combined',
        'count': len(combined),
        'mean': np.mean(combined),
        'median': np.median(combined),
        'std': np.std(combined),
        'min': np.min(combined),
        'max': np.max(combined),
        'q25': np.percentile(combined, 25),
        'q75': np.percentile(combined, 75),
    })
    
    df = pd.DataFrame(stats_list)
    output_file = OUTPUT_DIR / 'scores_length100_statistics.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved score statistics to {output_file}")


def create_all_histograms(data, segment_type):
    """Create all histograms for a given segment type."""
    print(f"\nCreating histograms for {segment_type} segments...")
    colors = {'strain_FD': COLORS['blue'], 'strain_yEK19': COLORS['orange'], 'strain_yEK23': COLORS['green']}
    
    for strain in STRAINS:
        title = f'Segment Length Distribution - {strain}\n({segment_type.capitalize()}, th={THRESHOLD}, filtered top {100-PERCENTILE_CUTOFF}%)'
        filename = f'{segment_type}_{strain}_histogram.png'
        create_histogram(data[strain], title, filename, color=colors[strain])
    
    combined = np.concatenate([data[strain] for strain in STRAINS])
    title = f'Segment Length Distribution - All Strains Combined\n({segment_type.capitalize()}, th={THRESHOLD}, filtered top {100-PERCENTILE_CUTOFF}%)'
    filename = f'{segment_type}_combined_histogram.png'
    create_histogram(combined, title, filename, color=COLORS['blue'])


def main():
    """Main execution function."""
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    print("=" * 60)
    print("UNMERGED SEGMENTS")
    print("=" * 60)
    unmerged_data = collect_all_segment_lengths('unmerged')
    print(f"\nFiltering outliers (removing top {100-PERCENTILE_CUTOFF}%)...")
    unmerged_filtered = filter_outliers(unmerged_data, PERCENTILE_CUTOFF)
    save_summary_statistics(unmerged_filtered, 'unmerged')
    create_all_histograms(unmerged_filtered, 'unmerged')
    
    print("\n" + "=" * 60)
    print("MERGED SEGMENTS")
    print("=" * 60)
    merged_data = collect_all_segment_lengths('merged')
    print(f"\nFiltering outliers (removing top {100-PERCENTILE_CUTOFF}%)...")
    merged_filtered = filter_outliers(merged_data, PERCENTILE_CUTOFF)
    save_summary_statistics(merged_filtered, 'merged')
    create_all_histograms(merged_filtered, 'merged')
    
    print("\n" + "=" * 60)
    print("CREATING COMPARISON PLOT")
    print("=" * 60)
    unmerged_combined = np.concatenate([unmerged_filtered[strain] for strain in STRAINS])
    merged_combined = np.concatenate([merged_filtered[strain] for strain in STRAINS])
    create_comparison_plot(unmerged_combined, merged_combined)
    
    print("\n" + "=" * 60)
    print("SCORE ANALYSIS FOR LENGTH-100 SEGMENTS")
    print("=" * 60)
    scores_dict = collect_scores_for_length_100()
    create_score_histograms(scores_dict)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
