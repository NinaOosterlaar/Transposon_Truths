import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS

# This file plots the results of saturation level against metrics

# # =========================================================
# # Plot style
# # =========================================================
setup_plot_style()


# -----------------------------
# Load data from multiple k files
# -----------------------------
data_dir = "AE/results/main_results/saturation"
aggregated_files = sorted(glob.glob(os.path.join(data_dir, "saturation_results_aggregated_k*.csv")))

# Read and combine all k files
df_list = []
for file in aggregated_files:
    df_temp = pd.read_csv(file)
    df_list.append(df_temp)

df_true = pd.concat(df_list, ignore_index=True)

# Sort by saturation or n_datasets
df_true = df_true.sort_values("n_datasets").reset_index(drop=True)

# -----------------------------
# Plot settings
# -----------------------------
plt.rcParams.update({
    "figure.figsize": (10, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


split_styles = {
    "train": ":",
    "test": "-",
}

metrics = [
    ("zinb_nll", "ZINB NLL"),
    ("mae", "MAE"),
    ("r2", r"$R^2$"),
    ("masked_loss", "Masked recon. loss"),
]



# Use saturation as x-axis (convert to percentage)
x_values = df_true["saturation_mean"] * 100
x_errors = df_true["saturation_std"] * 100

# Create 2x2 plot instead of 2x3
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

# -----------------------------
# All 4 panels: metrics with saturation as x-axis
# -----------------------------
for ax, (metric_key, metric_label) in zip(axes[:4], metrics):
    # Train: no error bars
    ax.plot(
        x_values,
        df_true[f"train_{metric_key}_mean"],
        color="black",
        linestyle=split_styles["train"],
        marker="o",
        linewidth=2,
        markersize=5,
        alpha=0.95,
        label="train",
    )

    # Test: with error bars
    ax.errorbar(
        x_values,
        df_true[f"test_{metric_key}_mean"],
        yerr=df_true[f"test_{metric_key}_std"],
        color="black",
        linestyle=split_styles["test"],
        marker="o",
        linewidth=2,
        markersize=5,
        capsize=3,
        alpha=0.95,
        label="test",
    )

    ax.set_title(metric_label)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Saturation level (%)")
    
    # Set y-axis limits with padding
    # Calculate max value including error bars
    max_vals = []
    for split in ["train", "test"]:
        vals = df_true[f"{split}_{metric_key}_mean"].values
        errs = df_true[f"{split}_{metric_key}_std"].values
        max_vals.append((vals + errs).max())
    
    y_max = max(max_vals) * 1.15  # Add 15% padding at the top
    


    ax.set_ylim(top=y_max)


    ax.set_ylim(0, y_max)
    
    ax.grid(True, alpha=0.3)

# panel labels
for ax, label in zip(axes[:4], ["a", "b", "c", "d"]):
    ax.text(
        -0.12, 1.05, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )

# -----------------------------
# Legend
# -----------------------------
legend_handles = [
    Line2D([0], [0], color="black", lw=2, linestyle=":", label="Train"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="Test"),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=2,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

