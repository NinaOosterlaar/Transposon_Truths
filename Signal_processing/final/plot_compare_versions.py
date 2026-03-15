"""Standalone script to regenerate the PR comparison plot from saved CSVs."""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Utils.plot_config import COLORS, setup_plot_style

setup_plot_style()


def plot_precision_recall_with_std(agg_curve_df, agg_auc_df, output_path):
    method_meta = {
        "ref": {"label": "ref", "color": COLORS["blue"]},
        "v0":  {"label": "v0",  "color": COLORS["orange"]},
        "v1":  {"label": "v1",  "color": COLORS["green"]},
        "v2":  {"label": "v2",  "color": COLORS["red"]},
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    required_curve_cols = {
        "method",
        "threshold",
        "precision_mean",
        "recall_mean",
    }
    missing_curve = required_curve_cols.difference(agg_curve_df.columns)
    if missing_curve:
        raise ValueError(
            "Missing required columns in precision_recall_aggregated.csv: "
            f"{sorted(missing_curve)}"
        )

    if "precision_std" not in agg_curve_df.columns:
        agg_curve_df = agg_curve_df.copy()
        agg_curve_df["precision_std"] = 0.0

    if "recall_std" not in agg_curve_df.columns:
        agg_curve_df = agg_curve_df.copy()
        agg_curve_df["recall_std"] = 0.0

    for method in ["ref", "v0", "v1", "v2"]:
        method_curve = agg_curve_df[agg_curve_df["method"] == method].sort_values("threshold")
        if method_curve.empty:
            continue

        auc_row = agg_auc_df[agg_auc_df["method"] == method]
        if not auc_row.empty:
            auc_mean = auc_row["auc_mean"].iloc[0]
            auc_std = auc_row["auc_std"].iloc[0]
            legend_label = f"{method_meta[method]['label']} (AUC={auc_mean:.3f} ± {auc_std:.3f})"
        else:
            legend_label = method_meta[method]["label"]

        recalls = method_curve["recall_mean"].values
        precisions = method_curve["precision_mean"].values
        precision_std = method_curve["precision_std"].values
        recall_std = method_curve["recall_std"].values

        precision_low = np.clip(precisions - precision_std, 0.0, 1.0)
        precision_high = np.clip(precisions + precision_std, 0.0, 1.0)

        ax.fill_between(
            recalls,
            precision_low,
            precision_high,
            color=method_meta[method]["color"],
            alpha=0.18,
            linewidth=0,
        )

        ax.plot(
            recalls,
            precisions,
            "o-",
            linewidth=1.8,
            markersize=4,
            alpha=0.85,
            color=method_meta[method]["color"],
            label=legend_label,
        )

        # Add sparse 2D error bars so both recall and precision std are visible without clutter.
        if len(recalls) > 0:
            n_error_points = min(12, len(recalls))
            idx = np.unique(np.linspace(0, len(recalls) - 1, n_error_points, dtype=int))
            ax.errorbar(
                recalls[idx],
                precisions[idx],
                xerr=recall_std[idx],
                yerr=precision_std[idx],
                fmt="none",
                ecolor=method_meta[method]["color"],
                alpha=0.35,
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Comparison Across Versions\n(mean ± std over datasets)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate PR comparison plot from saved CSVs.")
    parser.add_argument(
        "--agg_curve_csv",
        type=str,
        default="Signal_processing/final/results/compare_versions_ws100/precision_recall_aggregated.csv",
        help="Path to precision_recall_aggregated.csv",
    )
    parser.add_argument(
        "--agg_auc_csv",
        type=str,
        default=None,
        help=(
            "Optional path to auc_aggregated.csv. If not provided, the script will look for "
            "auc_aggregated.csv in the same folder as --agg_curve_csv."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output path for the plot (default: <agg_curve_csv folder>/precision_recall_compare_versions.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    agg_curve_csv = args.agg_curve_csv
    if not os.path.exists(agg_curve_csv):
        raise FileNotFoundError(f"Required file not found: {agg_curve_csv}")

    if args.agg_auc_csv is not None:
        agg_auc_csv = args.agg_auc_csv
    else:
        agg_auc_csv = os.path.join(os.path.dirname(agg_curve_csv), "auc_aggregated.csv")

    agg_curve_df = pd.read_csv(agg_curve_csv)
    if os.path.exists(agg_auc_csv):
        agg_auc_df = pd.read_csv(agg_auc_csv)
    else:
        agg_auc_df = pd.DataFrame(columns=["method", "auc_mean", "auc_std"])
        print(f"AUC file not found at {agg_auc_csv}; plotting without AUC values in legend.")

    output_path = args.output_file or os.path.join(
        os.path.dirname(agg_curve_csv), "precision_recall_compare_versions2.png"
    )
    plot_precision_recall_with_std(agg_curve_df, agg_auc_df, output_path)


if __name__ == "__main__":
    main()
