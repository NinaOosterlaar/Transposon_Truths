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
            "Train": [1.84, 0.91, 3.56, 2.82],
            "Val":   [1.88, 0.886, 1.24, 2.83],
            "Test":  [1.87, 0.98, 1.25, 2.82],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    r"$R^2$": {
        "values": {
            "Train": [0.88, -2.4, 0.45, 0.09],
            "Val":   [0.85, -0.84, 0.71, 0.09],
            "Test":  [0.86, -0.71, 0.71, 0.10],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    "MAE": {
        "values": {
            "Train": [2.1, 13.04, 5.46, 8.28],
            "Val":   [2.3, 8.21, 1.10, 8.38],
            "Test":  [2.3, 9.19, 1.17, 8.32],
        },
        "errors": {
            "Train": [5.40, 15.03, 10.81, 13.21],
            "Val":   [6.01, 12.70, 2.75, 13.32],
            "Test":  [6.36, 13.03, 3.21, 13.32],
        },
    },
}

imp_core = {
    "Masked recon.": {
        "values": {
            "Train": [2.81, 14.0, 8.07, 9.45],
            "Val":   [3.15, 14.4, 1.41, 9.56],
            "Test":  [3.09, 14.43, 1.48, 9.28],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    r"$R^2$": {
        "values": {
            "Train": [0.88, -0.013, 0.35, 0.082],
            "Val":   [0.85, -0.073, 0.69, 0.078],
            "Test":  [0.86, 0.019, 0.69, 0.088],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    "MAE": {
        "values": {
            "Train": [2.62, 14.9, 7.98, 9.17],
            "Val":   [2.92, 12.31, 1.49, 9.31],
            "Test":  [2.88, 12.69, 1.60, 9.31],
        },
        "errors": {
            "Train": [5.69, 18.2, 12.66, 15.6],
            "Val":   [6.56, 21.96, 3.23, 15.83],
            "Test":  [6.36, 21.59, 3.78, 15.62],
        },
    },
}

recon_params = {
    r"$\pi$ zeros": {
        "values": {
            "Train": [0.78, 0.91, 0.52, 0.14],
            "Val":   [0.77, 0.84, 0.0, 0.14],
            "Test":  [0.76, 0.84, 0.0, 0.14],
        },
        "errors": {
            "Train": [0.32, 0.10, 0.38, 0.0002],
            "Val":   [0.32, 0.16, 0.0, 0.0002],
            "Test":  [0.32, 0.16, 0.0, 0.0002],
        },
    },
    r"$\pi$ non-zeros": {
        "values": {
            "Train": [0.0185, 0.72, 0.069, 0.14],
            "Val":   [0.0206, 0.59, 0.0, 0.14],
            "Test":  [0.0202, 0.57, 0.0, 0.14],
        },
        "errors": {
            "Train": [0.0696, 0.21, 0.17, 0.0003],
            "Val":   [0.0807, 0.24, 0.0, 0.0003],
            "Test":  [0.080, 0.25, 0.0, 0.0003],
        },
    },
    r"$\mu$ zeros": {
        "values": {
            "Train": [1.025, 12.7, 0.47, 6.56],
            "Val":   [1.00, 7.46, 0.25, 6.60],
            "Test":  [0.98, 7.32, 0.26, 6.42],
        },
        "errors": {
            "Train": [5.14, 14.5, 2.55, 6.22],
            "Val":   [5.04, 10.56, 0.73, 6.17],
            "Test":  [4.99, 10.61, 0.74, 6.60],
        },
    },
    r"$\mu$ non-zeros": {
        "values": {
            "Train": [11.75, 19.9, 4.82, 7.26],
            "Val":   [11.87, 12.76, 3.02, 7.41],
            "Test":  [11.93, 12.41, 3.24, 7.47],
        },
        "errors": {
            "Train": [18.06, 19.3, 11.62, 4.81],
            "Val":   [18.41, 17.02, 5.15, 4.87],
            "Test":  [18.51, 16.0, 5.54, 5.06],
        },
    },
    r"$\theta$": {
        "values": {
            "Train": [2607, 1.00, 17, 1.0],
            "Val":   [2568, 1.05, 3.34, 1.0],
            "Test":  [2541, 1.04, 3.10, 1.0],
        },
        "errors": {
            "Train": [6384, 1.4, 109, 0.0],
            "Val":   [6336, 28.09, 55.7, 0.0],
            "Test":  [6305, 21.24, 10.95, 0.0],
        },
    },
}

imp_params = {
    r"$\pi$": {
        "values": {
            "Train": [0.0206, 0.59, 0.072, 0.14],
            "Val":   [0.0228, 0.60, 0.0, 0.14],
            "Test":  [0.0223, 0.57, 0.0, 0.14],
        },
        "errors": {
            "Train": [0.0772, 0.26, 0.17, 0.0003],
            "Val":   [0.0877, 0.23, 0.0, 0.0003],
            "Test":  [0.0866, 0.26, 0.0, 0.0003],
        },
    },
    r"$\mu$": {
        "values": {
            "Train": [11.77, 19.2, 4.38, 7.27],
            "Val":   [11.88, 12.35, 3.305, 7.41],
            "Test":  [11.96, 12.02, 3.27, 7.47],
        },
        "errors": {
            "Train": [18.23, 18.5, 10.12, 4.81],
            "Val":   [18.61, 18.37, 5.23, 4.87],
            "Test":  [18.52, 16.05, 5.62, 5.06],
        },
    },
    r"$\theta$": {
        "values": {
            "Train": [3245, 1.00, 21.65, 1.0],
            "Val":   [3192, 1.36, 3.53, 1.0],
            "Test":  [3153, 1.30, 3.29, 1.0],
        },
        "errors": {
            "Train": [7007, 0.913, 126, 0.0],
            "Val":   [6336, 80.70, 50.09, 0.0],
            "Test":  [6919, 66.61, 12.52, 0.0],
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
