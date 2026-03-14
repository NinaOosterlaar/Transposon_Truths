import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from Utils.plot_config import setup_plot_style, COLORS

# =========================================================
# Plot style
# =========================================================
setup_plot_style()

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})

# =========================================================
# Labels / colors
# =========================================================
models = ["Combined", "ZINB NLL", "Masked recon.", "Combined bins"]
splits = ["Train", "Val", "Test"]

model_colors = {
    "Combined": COLORS["blue"],
    "ZINB NLL": COLORS["orange"],
    "Masked recon.": COLORS["green"],
    "Combined bins": COLORS["red"],
}

# =========================================================
# Data
# =========================================================
recon_core = {
    "ZINB NLL": {
        "values": {
            "Train": [1.07, 0.90, 1.21, 1.37],
            "Val":   [1.10, 0.90, 1.24, 1.52],
            "Test":  [1.13, 0.92, 1.25, 1.57],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    r"$R^2$": {
        "values": {
            "Train": [0.90, -0.66, 0.81, 0.63],
            "Val":   [0.74, -0.66, 0.71, 0.41],
            "Test":  [0.79, -0.54, 0.71, 0.24],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    "MAE": {
        "values": {
            "Train": [0.62, 8.97, 0.99, 1.42],
            "Val":   [0.79, 8.87, 1.10, 1.82],
            "Test":  [0.84, 9.11, 1.17, 1.96],
        },
        "errors": {
            "Train": [1.44, 11.24, 1.93, 2.60],
            "Val":   [2.68, 11.0, 2.75, 3.47],
            "Test":  [2.79, 11.5, 3.21, 5.12],
        },
    },
}

imp_core = {
    "Masked recon.": {
        "values": {
            "Train": [0.85, 13.8, 1.29, 1.56],
            "Val":   [1.06, 13.8, 1.41, 2.33],
            "Test":  [1.13, 13.5, 1.48, 2.64],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    r"$R^2$": {
        "values": {
            "Train": [0.91, 0.18, 0.81, 0.74],
            "Val":   [0.76, 0.17, 0.69, 0.31],
            "Test":  [0.79, 0.019, 0.69, 0.18],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    "MAE": {
        "values": {
            "Train": [0.85, 11.97, 1.29, 1.56],
            "Val":   [1.11, 12.29, 1.49, 2.52],
            "Test":  [1.19, 12.55, 1.60, 2.80],
        },
        "errors": {
            "Train": [1.46, 17.68, 2.11, 2.53],
            "Val":   [2.92, 18.48, 3.23, 4.30],
            "Test":  [3.19, 19.01, 3.78, 6.16],
        },
    },
}

recon_params = {
    r"$\pi$ zeros": {
        "values": {
            "Train": [0.0, 0.84, 0.0, 0.0],
            "Val":   [0.0, 0.84, 0.0, 0.0],
            "Test":  [0.0, 0.83, 0.0, 0.0],
        },
        "errors": {
            "Train": [0.0, 0.15, 0.0, 0.0],
            "Val":   [0.0, 0.15, 0.0, 0.0],
            "Test":  [0.0, 0.16, 0.0, 0.0],
        },
    },
    r"$\pi$ non-zeros": {
        "values": {
            "Train": [0.0, 0.59, 0.0, 0.0],
            "Val":   [0.0, 0.59, 0.0, 0.0],
            "Test":  [0.0, 0.58, 0.0, 0.0],
        },
        "errors": {
            "Train": [0.0, 0.23, 0.0, 0.0],
            "Val":   [0.0, 0.23, 0.0, 0.0],
            "Test":  [0.0, 0.23, 0.0, 0.0],
        },
    },
    r"$\mu$ zeros": {
        "values": {
            "Train": [0.0831, 8.33, 0.25, 0.50],
            "Val":   [0.0847, 8.33, 0.25, 0.59],
            "Test":  [0.0840, 8.42, 0.26, 0.55],
        },
        "errors": {
            "Train": [0.27, 9.11, 0.71, 0.92],
            "Val":   [0.27, 9.11, 0.73, 1.07],
            "Test":  [0.27, 9.11, 0.74, 1.00],
        },
    },
    r"$\mu$ non-zeros": {
        "values": {
            "Train": [3.24, 12.01, 3.01, 2.78],
            "Val":   [3.35, 12.06, 3.02, 2.63],
            "Test":  [3.59, 12.60, 3.24, 2.50],
        },
        "errors": {
            "Train": [5.92, 10.48, 4.72, 4.19],
            "Val":   [6.98, 10.48, 5.15, 3.16],
            "Test":  [7.37, 10.99, 5.54, 3.07],
        },
    },
    r"$\theta$": {
        "values": {
            "Train": [1061, 1.00, 3.19, 1.0],
            "Val":   [1046, 1.00, 3.34, 1.0],
            "Test":  [1066, 1.00, 3.10, 1.0],
        },
        "errors": {
            "Train": [3457, 0.0001, 12.56, 0.0],
            "Val":   [3442, 0.0001, 55.7, 0.0],
            "Test":  [3493, 0.0001, 10.95, 0.0],
        },
    },
}

imp_params = {
    r"$\pi$": {
        "values": {
            "Train": [0.0, 0.60, 0.0, 0.0],
            "Val":   [0.0, 0.60, 0.0, 0.0],
            "Test":  [0.0, 0.58, 0.0, 0.0],
        },
        "errors": {
            "Train": [0.0, 0.23, 0.0, 0.0],
            "Val":   [0.0, 0.23, 0.0, 0.0],
            "Test":  [0.0, 0.23, 0.0, 0.0],
        },
    },
    r"$\mu$": {
        "values": {
            "Train": [3.06, 12.07, 3.02, 2.77],
            "Val":   [3.11, 11.90, 3.305, 2.57],
            "Test":  [3.35, 12.44, 3.27, 2.44],
        },
        "errors": {
            "Train": [5.50, 11.16, 4.80, 4.57],
            "Val":   [6.29, 10.3, 5.23, 3.14],
            "Test":  [6.81, 10.79, 5.62, 3.05],
        },
    },
    r"$\theta$": {
        "values": {
            "Train": [1008, 1.00, 3.38, 1.0],
            "Val":   [969, 1.00, 3.53, 1.0],
            "Test":  [949, 1.00, 3.29, 1.0],
        },
        "errors": {
            "Train": [3237, 0.0001, 10.25, 0.0],
            "Val":   [3172, 0.0003, 50.09, 0.0],
            "Test":  [3143, 0.0002, 12.52, 0.0],
        },
    },
}

# =========================================================
# Helpers
# =========================================================
def get_ylim(title, values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    data_min = np.min(values - errors)
    data_max = np.max(values + errors)

    # Lower bound:
    # - for R^2: allow negatives, but make sure 0 is included
    # - for everything else: start at 0
    if "R^2" in title:
        ymin = min(0.0, data_min)
    else:
        ymin = 0.0

    # Upper bound:
    # - for R^2 and pi plots: fix at 1
    # - otherwise: add a bit of padding
    if "R^2" in title or r"$\pi$" in title:
        ymax = 1.0
    else:
        ymax = data_max
        if ymax == 0:
            ymax = 0.1
        else:
            ymax = ymax * 1.12

    return ymin, ymax


def plot_single_metric(ax, metric_dict, title, panel_label=None, width=0.22):
    x = np.arange(len(splits))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    all_vals = []
    all_errs = []

    for m_idx, model in enumerate(models):
        vals = [metric_dict["values"][split][m_idx] for split in splits]
        errs = [metric_dict["errors"][split][m_idx] for split in splits]

        all_vals.extend(vals)
        all_errs.extend(errs)

        ax.bar(
            x + offsets[m_idx],
            vals,
            width=width,
            yerr=errs,
            capsize=2,
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.5,
            label=model,
            zorder=3,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )

    ymin, ymax = get_ylim(title, all_vals, all_errs)
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="x", length=0)

    if panel_label is not None:
        ax.text(
            -0.14, 1.05, panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )
# =========================================================
# Core metric figure (2 x 3)
# =========================================================
fig_metrics, axes_metrics = plt.subplots(2, 3, figsize=(15, 8))
fig_metrics.subplots_adjust(
    top=0.84,
    bottom=0.10,
    left=0.07,
    right=0.98,
    wspace=0.30,
    hspace=0.42,
)

plot_single_metric(axes_metrics[0, 0], recon_core["ZINB NLL"], "Reconstruction: ZINB NLL", "a")
plot_single_metric(axes_metrics[0, 1], recon_core[r"$R^2$"], r"Reconstruction: $R^2$", "b")
plot_single_metric(axes_metrics[0, 2], recon_core["MAE"], "Reconstruction: MAE", "c")

plot_single_metric(axes_metrics[1, 0], imp_core["Masked recon."], "Masked Values: Masked recon.", "d")
plot_single_metric(axes_metrics[1, 1], imp_core[r"$R^2$"], r"Masked Values: $R^2$", "e")
plot_single_metric(axes_metrics[1, 2], imp_core["MAE"], "Masked Values: MAE", "f")

handles_metrics, labels_metrics = axes_metrics[0, 0].get_legend_handles_labels()
fig_metrics.legend(
    handles_metrics,
    labels_metrics,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.98),
    columnspacing=1.2,
    handletextpad=0.5,
)

# =========================================================
# Parameter figure with 3 / 2 / 3 layout
# =========================================================
fig = plt.figure(figsize=(15, 11))
gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

# Row 1
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# Row 2
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])

# Row 3
ax6 = fig.add_subplot(gs[2, 0])
ax7 = fig.add_subplot(gs[2, 1])
ax8 = fig.add_subplot(gs[2, 2])

plot_single_metric(ax1, recon_params[r"$\pi$ non-zeros"], r"Reconstruction: $\pi$ non-zeros", "a")
plot_single_metric(ax2, recon_params[r"$\mu$ non-zeros"], r"Reconstruction: $\mu$ non-zeros", "b")
plot_single_metric(ax3, recon_params[r"$\theta$"], r"Reconstruction: $\theta$", "c")

plot_single_metric(ax4, recon_params[r"$\pi$ zeros"], r"Reconstruction: $\pi$ zeros", "d")
plot_single_metric(ax5, recon_params[r"$\mu$ zeros"], r"Reconstruction: $\mu$ zeros", "e")

plot_single_metric(ax6, imp_params[r"$\pi$"], r"Masked Values: $\pi$", "f")
plot_single_metric(ax7, imp_params[r"$\mu$"], r"Masked Values: $\mu$", "g")
plot_single_metric(ax8, imp_params[r"$\theta$"], r"Masked Values: $\theta$", "h")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.98),
    columnspacing=1.2,
    handletextpad=0.5,
)

# plt.show()

# # Optional save
fig_metrics.savefig("core_metric_comparison.png", dpi=300, bbox_inches="tight")
fig_metrics.savefig("core_metric_comparison.pdf", bbox_inches="tight")

fig.savefig("parameter_comparison_custom_layout.png", dpi=300, bbox_inches="tight")
fig.savefig("parameter_comparison_custom_layout.pdf", bbox_inches="tight")


# import pandas as pd
# import matplotlib.pyplot as plt

# moving_avg_true = pd.DataFrame({
#     "bin_size": [1, 5, 10, 15, 20],
#     "train_zinb_nll": [0.816443, 1.189480, 1.254648, 1.262064, 1.345470],
#     "test_zinb_nll":  [0.914835, 1.272952, 1.359939, 1.367745, 1.436980],
#     "train_mae":      [6.3273, 2.1120, 1.5719, 1.2916, 1.0920],
#     "test_mae":       [5.9665, 2.3812, 1.8305, 1.5325, 1.3434],
#     "train_mae_sd":   [11.8787, 5.2055, 2.6937, 1.6982, 1.1167],
#     "test_mae_sd":    [15.0405, 8.0631, 5.4239, 3.8609, 2.9470],
#     "train_r2":       [-0.5662, 0.2034, 0.6695, 0.8245, 0.8978],
#     "test_r2":        [-0.8862, 0.1669, 0.2777, 0.5848, 0.7329],
# })

# moving_avg_false = pd.DataFrame({
#     "bin_size": [1, 5, 10, 15, 20],
#     "train_zinb_nll": [0.914544, 1.36, 1.371983, 1.410180, 1.382333],
#     "test_zinb_nll":  [0.983101, 1.46, 1.505194, 1.466934, 1.555772],
#     "train_mae":      [11.0060, 2.69, 1.6677, 1.5817, 1.3896],
#     "test_mae":       [11.0128, 2.88, 1.9949, 1.8933, 1.8739],
#     "train_mae_sd":   [11.6757, 4.64, 3.3328, 2.8953, 2.5610],
#     "test_mae_sd":    [12.3275, 6.78, 4.8085, 4.4102, 4.5869],
#     "train_r2":       [-1.2213, 0.30, 0.5379, 0.5803, 0.6427],
#     "test_r2":        [-0.9654, -0.0079, 0.2925, 0.3306, 0.2877],
# })

# def plot_row(ax_row, df, row_title):
#     x = df["bin_size"]

#     # ZINB NLL
#     ax_row[0].plot(x, df["train_zinb_nll"], marker="o", label="Train")
#     ax_row[0].plot(x, df["test_zinb_nll"], marker="o", label="Test")
#     ax_row[0].set_title("ZINB NLL")
#     ax_row[0].set_xlabel("Bin size")
#     ax_row[0].set_ylabel("ZINB NLL")
#     ax_row[0].set_xticks(x)
#     ax_row[0].legend()

#     # MAE + SD
#     ax_row[1].errorbar(
#         x, df["train_mae"], yerr=df["train_mae_sd"],
#         marker="o", capsize=4, label="Train"
#     )
#     ax_row[1].errorbar(
#         x, df["test_mae"], yerr=df["test_mae_sd"],
#         marker="o", capsize=4, label="Test"
#     )
#     ax_row[1].set_title("MAE")
#     ax_row[1].set_xlabel("Bin size")
#     ax_row[1].set_ylabel("MAE")
#     ax_row[1].set_xticks(x)
#     ax_row[1].legend()

#     # R²
#     ax_row[2].plot(x, df["train_r2"], marker="o", label="Train")
#     ax_row[2].plot(x, df["test_r2"], marker="o", label="Test")
#     ax_row[2].set_title(r"$R^2$")
#     ax_row[2].set_xlabel("Bin size")
#     ax_row[2].set_ylabel(r"$R^2$")
#     ax_row[2].set_xticks(x)
#     ax_row[2].legend()

#     # row label = subfigure label
#     ax_row[0].text(
#         -0.28, 1.18, row_title,
#         transform=ax_row[0].transAxes,
#         fontsize=14, fontweight="bold",
#         va="top", ha="left"
#     )

# fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# plot_row(axes[0], moving_avg_true, "a) Moving average = True")
# plot_row(axes[1], moving_avg_false, "b) Moving average = False")

# # fig.suptitle("Performance across bin sizes", fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# # plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# # -----------------------------
# # Load data
# # -----------------------------
# df_true = pd.read_csv("AE/results/results_saturation/saturation.csv")
# df_false = pd.read_csv("AE/results/saturation_bins/saturation_results_aggregated.csv")

# # Sort both the same way
# df_true = df_true.sort_values("n_datasets").reset_index(drop=True)
# df_false = df_false.sort_values("n_datasets").reset_index(drop=True)

# # -----------------------------
# # Plot settings
# # -----------------------------
# plt.rcParams.update({
#     "figure.figsize": (12, 9),
#     "font.size": 12,
#     "axes.titlesize": 14,
#     "axes.labelsize": 12,
#     "legend.fontsize": 11,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
# })

# model_colors = {
#     "Moving average": "tab:blue",
#     "Binned": "tab:orange",
# }

# split_styles = {
#     "train": ":",
#     "test": "-",
# }

# metrics = [
#     ("zinb_nll", "ZINB NLL"),
#     ("mae", "MAE"),
#     ("r2", r"$R^2$"),
# ]

# models = [
#     ("Moving average", df_true),
#     ("Binned", df_false),
# ]

# fig, axes = plt.subplots(2, 2, sharex=True)
# axes = axes.flatten()

# # -----------------------------
# # First 3 panels: metrics
# # -----------------------------
# for ax, (metric_key, metric_label) in zip(axes[:3], metrics):
#     for model_name, df in models:
#         color = model_colors[model_name]

#         for split in ["train", "test"]:
#             ax.errorbar(
#                 df["n_datasets"],
#                 df[f"{split}_{metric_key}_mean"],
#                 yerr=df[f"{split}_{metric_key}_std"],
#                 color=color,
#                 linestyle=split_styles[split],
#                 marker="o",
#                 linewidth=2,
#                 markersize=5,
#                 capsize=3,
#                 alpha=0.95,
#             )

#     ax.set_title(metric_label)
#     ax.set_ylabel(metric_label)
#     ax.grid(True, alpha=0.3)

# # -----------------------------
# # 4th panel: one black saturation line
# # -----------------------------
# ax = axes[3]
# ax.errorbar(
#     df_true["n_datasets"],
#     df_true["saturation_mean"] * 100,
#     yerr=df_true["saturation_std"] * 100,
#     color="black",
#     linestyle="-",
#     marker="o",
#     linewidth=2,
#     markersize=5,
#     capsize=3,
#     alpha=0.95,
# )

# ax.set_title("Saturation level")
# ax.set_ylabel("Saturation level (%)")
# ax.grid(True, alpha=0.3)

# # -----------------------------
# # X labels
# # -----------------------------
# axes[2].set_xlabel("Number of combined datasets")
# axes[3].set_xlabel("Number of combined datasets")

# # -----------------------------
# # Legend
# # -----------------------------
# legend_handles = [
#     Line2D([0], [0], color=model_colors["Moving average"], lw=2, label="Moving average"),
#     Line2D([0], [0], color=model_colors["Binned"], lw=2, label="Binned"),
#     Line2D([0], [0], color="black", lw=2, linestyle=":", label="Train"),
#     Line2D([0], [0], color="black", lw=2, linestyle="-", label="Test"),
#     Line2D([0], [0], color="black", lw=2, marker="o", label="Saturation"),
# ]

# fig.legend(
#     handles=legend_handles,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 0.99),
#     ncol=5,
#     frameon=False,
# )

# plt.tight_layout(rect=[0, 0, 1, 0.92])
# plt.show()