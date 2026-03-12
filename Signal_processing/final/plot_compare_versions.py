"""Standalone script to regenerate the PR comparison plot from saved CSVs."""
import argparse
import os
import sys

import matplotlib.pyplot as plt
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

        ax.plot(
            method_curve["recall_mean"].values,
            method_curve["precision_mean"].values,
            "o-",
            linewidth=1.8,
            markersize=4,
            alpha=0.85,
            color=method_meta[method]["color"],
            label=legend_label,
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
        "--results_folder",
        type=str,
        default="Signal_processing/final/results/compare_versions_theta0_ws100",
        help="Folder containing precision_recall_aggregated.csv and auc_aggregated.csv",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output path for the plot (default: <results_folder>/precision_recall_compare_versions.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    agg_curve_csv = os.path.join(args.results_folder, "precision_recall_aggregated.csv")
    agg_auc_csv = os.path.join(args.results_folder, "auc_aggregated.csv")

    for path in [agg_curve_csv, agg_auc_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    agg_curve_df = pd.read_csv(agg_curve_csv)
    agg_auc_df = pd.read_csv(agg_auc_csv)

    output_path = args.output_file or os.path.join(
        args.results_folder, "precision_recall_compare_versions.png"
    )
    plot_precision_recall_with_std(agg_curve_df, agg_auc_df, output_path)


if __name__ == "__main__":
    main()
