"""
Plotting utilities for mu_offset correlation analysis.

Two public functions:
  - plot_pi_correlation_heatmaps: 5×5 Pearson/Spearman heatmaps, one subplot
    per split (train / val / test / all-combined). Cell value = mean r over
    strains; annotation includes ±std.
  - plot_pi_tracks_per_chromosome: for a single chromosome, one subplot per
    strain, all mu_offset models overlaid as coloured lines.
"""

import os
import sys
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Utils.plot_config import setup_plot_style, COLORS

setup_plot_style()

# One colour per model (up to 8 models supported)
_MODEL_PALETTE = [
    COLORS['blue'],
    COLORS['orange'],
    COLORS['green'],
    COLORS['red'],
    COLORS['pink'],
    COLORS['light_blue'],
    COLORS['yellow'],
    COLORS['black'],
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_mean_std_matrices(corr_split, model_labels, metric):
    """
    Return (mean_mat, std_mat) of shape (n, n) for one split and one metric.

    Diagonal is set to 1.0 / 0.0. Off-diagonal NaN where no data was found.
    """
    n = len(model_labels)
    mean_mat = np.full((n, n), np.nan)
    std_mat  = np.full((n, n), np.nan)
    np.fill_diagonal(mean_mat, 1.0)
    np.fill_diagonal(std_mat,  0.0)

    label_idx = {label: i for i, label in enumerate(model_labels)}
    for (li, lj), data in corr_split.items():
        r_list = data.get(metric, [])
        if not r_list:
            continue
        i, j = label_idx[li], label_idx[lj]
        mean_r = float(np.mean(r_list))
        std_r  = float(np.std(r_list))
        mean_mat[i, j] = mean_mat[j, i] = mean_r
        std_mat[i, j]  = std_mat[j, i]  = std_r

    return mean_mat, std_mat


def _annotate_heatmap(ax, mean_mat, std_mat, n):
    """Add text annotations (mean ± std) to each heatmap cell."""
    for i in range(n):
        for j in range(n):
            val = mean_mat[i, j]
            if not np.isfinite(val):
                continue
            if i == j:
                text = f"{val:.2f}"
            else:
                text = f"{val:.2f}\n±{std_mat[i, j]:.2f}"
            # Choose black or white text based on background brightness
            bg = ax.images[0].cmap(ax.images[0].norm(val))
            brightness = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=8, color='black' if brightness > 0.5 else 'white',
                    fontweight='bold' if i == j else 'normal')


# ── Public plotting functions ──────────────────────────────────────────────────

def plot_pi_correlation_heatmaps(corr_results, model_labels, output_dir):
    """
    Plot Pearson and Spearman 5×5 correlation heatmaps.

    One figure per metric, with one subplot per split key present in
    corr_results (train / val / test / all). Each cell shows mean r over
    strains and ±std.

    Parameters
    ----------
    corr_results : dict
        Output of compute_pi_correlations():
        split_name -> {(label_i, label_j): {'pearson': [...], 'spearman': [...]}}
    model_labels : list[str]
        Ordered list of model labels (used as axis tick labels).
    output_dir : str
        Directory to save PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(model_labels)
    split_keys = list(corr_results.keys())  # e.g. ['train', 'val', 'test', 'all']
    n_splits = len(split_keys)

    cmap = plt.get_cmap('RdYlGn')

    for metric in ('pearson', 'spearman'):
        fig, axes = plt.subplots(1, n_splits, figsize=(4.5 * n_splits, 4.8), squeeze=False)

        im = None
        for col, split_key in enumerate(split_keys):
            ax = axes[0, col]
            mean_mat, std_mat = _build_mean_std_matrices(
                corr_results[split_key], model_labels, metric
            )
            im = ax.imshow(mean_mat, vmin=0, vmax=1, cmap=cmap, aspect='auto')
            _annotate_heatmap(ax, mean_mat, std_mat, n)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(model_labels, fontsize=9)
            ax.set_title(split_key.capitalize(), fontsize=11)

        if im is not None:
            cbar = fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04)
            cbar.set_label(f"Mean {metric.capitalize()} r", fontsize=10)

        fig.suptitle(
            f"π correlation across mu_offset models — {metric.capitalize()} r\n"
            f"(mean ± std over strains; strains kept separate before aggregation)",
            fontsize=11, y=1.03
        )
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"pi_correlation_{metric}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_pi_tracks_per_chromosome(merged_df, chromosome, model_labels, output_dir):
    """
    Plot pi value tracks for all mu_offset models along a single chromosome.

    One subplot per strain/dataset; all models overlaid as coloured lines on
    the same axes. Saves one PNG per chromosome to output_dir/pi_tracks/.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Columns: dataset, chromosome, position, pi_<label> for each label.
    chromosome : str
        Chromosome name (e.g. 'ChrI').
    model_labels : list[str]
        Ordered list of model labels matching the pi_<label> column names.
    output_dir : str
        Base output directory; PNGs go into output_dir/pi_tracks/.
    """
    chrom_df = merged_df[merged_df['chromosome'] == chromosome]
    datasets = sorted(chrom_df['dataset'].unique())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    fig, axes = plt.subplots(
        n_datasets, 1,
        figsize=(14, 2.8 * n_datasets),
        sharex=True,
        squeeze=False
    )

    pi_cols = [f'pi_{label}' for label in model_labels]
    colors = _MODEL_PALETTE[:len(model_labels)]

    for row_idx, dataset in enumerate(datasets):
        ax = axes[row_idx, 0]
        grp = chrom_df[chrom_df['dataset'] == dataset].sort_values('position')

        for label, pi_col, color in zip(model_labels, pi_cols, colors):
            if pi_col not in grp.columns:
                continue
            ax.plot(
                grp['position'].values,
                grp[pi_col].values,
                color=color,
                linewidth=0.7,
                alpha=0.85,
                label=label,
            )

        ax.set_ylabel('π', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(dataset, fontsize=10)
        ax.axhline(0.7, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)  # PI_THRESHOLD guideline
        ax.grid(True, alpha=0.2, linewidth=0.4)

        if row_idx == 0:
            ax.legend(
                loc='upper right', fontsize=8,
                ncol=min(len(model_labels), 5),
                framealpha=0.7
            )

    axes[-1, 0].set_xlabel('Genomic position (bp)', fontsize=11)
    fig.suptitle(
        f"Zero-inflation probability (π) across mu_offset models — {chromosome}",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    tracks_dir = os.path.join(output_dir, "pi_tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    out_path = os.path.join(tracks_dir, f"pi_tracks_{chromosome}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
