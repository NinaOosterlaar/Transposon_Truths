import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from Utils.plot_config import setup_plot_style, COLORS

# =========================================================
# Plot style
# =========================================================
setup_plot_style()

# =========================================================
# Load and prepare data
# =========================================================
df = pd.read_csv('AE/results/noise_sweep_metrics.csv')

# Filter rows where trained_noise_level == eval_noise_level
df_filtered = df[df['trained_noise_level'] == df['eval_noise_level']].copy()

# Filter only train and test splits
df_filtered = df_filtered[df_filtered['split'].isin(['train', 'test'])]

# Sort by noise level for plotting
df_filtered = df_filtered.sort_values('trained_noise_level')

# Separate train and test
df_train = df_filtered[df_filtered['split'] == 'train']
df_test = df_filtered[df_filtered['split'] == 'test']

noise_levels = df_train['trained_noise_level'].values

# =========================================================
# Figure 1: Loss metrics (ZINB NLL, MAE, R2, Masked Recon)
# =========================================================
fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
fig1.subplots_adjust(hspace=0.3, wspace=0.3, top=0.93)

# Subplot labels
labels = ['a', 'b', 'c', 'd']
titles = ['ZINB NLL Loss', 'MAE', 'R²', 'Masked Reconstruction Loss']
metrics = ['zinb_nll', 'mae', 'r2', 'masked_loss']

for idx, (ax, label, title, metric) in enumerate(zip(axes.flat, labels, titles, metrics)):
    # Plot train (dashed black) and test (solid black)
    ax.plot(noise_levels, df_train[metric].values, 
            marker='o', linewidth=2, markersize=6, 
            color='black', linestyle='--', label='Train')
    
    # Add error bars for MAE test data
    if metric == 'mae':
        ax.errorbar(noise_levels, df_test[metric].values, 
                    yerr=df_test['mae_sd'].values,
                    marker='s', linewidth=2, markersize=6, 
                    color='black', linestyle='-', label='Test',
                    capsize=3, capthick=1.5)
    else:
        ax.plot(noise_levels, df_test[metric].values, 
                marker='s', linewidth=2, markersize=6, 
                color='black', linestyle='-', label='Test')
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set y-axis to start at 0
    ylim = ax.get_ylim()
    if metric == 'r2':
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, max(ylim[1], 0.1))
    
    # Add subplot label on top left, outside the plot
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom')

plt.savefig('noise_sweep_figure1.png', dpi=300, bbox_inches='tight')
print("Saved Figure 1: noise_sweep_figure1.png")

# =========================================================
# Figure 2: Distribution parameters (Pi, Mu, Theta)
# =========================================================
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig2.subplots_adjust(wspace=0.3, top=0.88)

# Subplot a: Pi (zero and non-zero for train and test)
ax = axes[0]
ax.plot(noise_levels, df_train['pi_zero'].values, 
        marker='o', linewidth=2, markersize=6, linestyle='--',
        color=COLORS['blue'], label='Train Zero')
ax.plot(noise_levels, df_train['pi_non_zero'].values, 
        marker='s', linewidth=2, markersize=6, linestyle='--',
        color=COLORS['light_blue'], label='Train Non-Zero')
ax.plot(noise_levels, df_test['pi_zero'].values, 
        marker='o', linewidth=2, markersize=6, linestyle='-',
        color=COLORS['blue'], label='Test Zero')
ax.plot(noise_levels, df_test['pi_non_zero'].values, 
        marker='s', linewidth=2, markersize=6, linestyle='-',
        color=COLORS['light_blue'], label='Test Non-Zero')
ax.set_xlabel('Noise Level')
ax.set_ylabel(r'$\pi$')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.text(-0.15, 1.05, 'a', transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='bottom')

# Subplot b: Mu (zero and non-zero for train and test)
ax = axes[1]
ax.plot(noise_levels, df_train['mu_zero'].values, 
        marker='o', linewidth=2, markersize=6, linestyle='--',
        color=COLORS['blue'], label='Train Zero')
ax.plot(noise_levels, df_train['mu_non_zero'].values, 
        marker='s', linewidth=2, markersize=6, linestyle='--',
        color=COLORS['light_blue'], label='Train Non-Zero')
ax.plot(noise_levels, df_test['mu_zero'].values, 
        marker='o', linewidth=2, markersize=6, linestyle='-',
        color=COLORS['blue'], label='Test Zero')
ax.plot(noise_levels, df_test['mu_non_zero'].values, 
        marker='s', linewidth=2, markersize=6, linestyle='-',
        color=COLORS['light_blue'], label='Test Non-Zero')
ax.set_xlabel('Noise Level')
ax.set_ylabel(r'$\mu$')
ylim = ax.get_ylim()
ax.set_ylim(0, max(ylim[1], 0.1))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.text(-0.15, 1.05, 'b', transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='bottom')

# Subplot c: Theta (train and test)
ax = axes[2]
ax.plot(noise_levels, df_train['theta'].values, 
        marker='o', linewidth=2, markersize=6, linestyle='--',
        color=COLORS['blue'], label='Train')
ax.plot(noise_levels, df_test['theta'].values, 
        marker='s', linewidth=2, markersize=6, linestyle='-',
        color=COLORS['blue'], label='Test')
ax.set_xlabel('Noise Level')
ax.set_ylabel(r'$\theta$')
ylim = ax.get_ylim()
ax.set_ylim(0, max(ylim[1], 0.1))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.text(-0.15, 1.05, 'c', transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='bottom')

plt.savefig('noise_sweep_figure2.png', dpi=300, bbox_inches='tight')
print("Saved Figure 2: noise_sweep_figure2.png")

plt.show()
#     r"$R^2$": {
#         "values": {
#             "Train": [0.88, -2.4, 0.75, 0.09],
#             "Val":   [0.85, -0.84, 0.71, 0.09],
#             "Test":  [0.86, -0.71, 0.70, 0.10],
#         },
#         "errors": {
#             "Train": [0.0, 0.0, 0.0, 0.0],
#             "Val":   [0.0, 0.0, 0.0, 0.0],
#             "Test":  [0.0, 0.0, 0.0, 0.0],
#         },
#     },
#     "MAE": {
#         "values": {
#             "Train": [2.1, 13.04, 3.41, 8.28],
#             "Val":   [2.3, 8.21, 3.65, 8.38],
#             "Test":  [2.3, 9.19, 3.69, 8.32],
#         },
#         "errors": {
#             "Train": [5.40, 15.03, 7.52, 13.21],
#             "Val":   [6.01, 12.70, 8.14, 13.32],
#             "Test":  [6.36, 13.03, 8.32, 13.32],
#         },
#     },
# }

# imp_core = {
#     "Masked recon.": {
#         "values": {
#             "Train": [2.81, 14.0, 4.17, 9.45],
#             "Val":   [3.15, 14.4, 4.85, 9.56],
#             "Test":  [3.09, 14.43, 4.89, 9.28],
#         },
#         "errors": {
#             "Train": [0.0, 0.0, 0.0, 0.0],
#             "Val":   [0.0, 0.0, 0.0, 0.0],
#             "Test":  [0.0, 0.0, 0.0, 0.0],
#         },
#     },
#     r"$R^2$": {
#         "values": {
#             "Train": [0.88, -0.013, 0.75, 0.082],
#             "Val":   [0.85, -0.073, 0.69, 0.078],
#             "Test":  [0.86, 0.019, 0.68, 0.088],
#         },
#         "errors": {
#             "Train": [0.0, 0.0, 0.0, 0.0],
#             "Val":   [0.0, 0.0, 0.0, 0.0],
#             "Test":  [0.0, 0.0, 0.0, 0.0],
#         },
#     },
#     "MAE": {
#         "values": {
#             "Train": [2.62, 14.9, 4.14, 9.17],
#             "Val":   [2.92, 12.31, 4.59, 9.31],
#             "Test":  [2.88, 12.69, 4.66, 9.31],
#         },
#         "errors": {
#             "Train": [5.69, 18.2, 8.31, 15.6],
#             "Val":   [6.56, 21.96, 9.37, 15.83],
#             "Test":  [6.36, 21.59, 9.60, 15.62],
#         },
#     },
# }

# recon_params = {
#     r"$\pi$ zeros": {
#         "values": {
#             "Train": [0.78, 0.91, 0.42, 0.14],
#             "Val":   [0.77, 0.84, 0.39, 0.14],
#             "Test":  [0.76, 0.84, 0.40, 0.14],
#         },
#         "errors": {
#             "Train": [0.32, 0.10, 0.32, 0.0002],
#             "Val":   [0.32, 0.16, 0.30, 0.0002],
#             "Test":  [0.32, 0.16, 0.31, 0.0002],
#         },
#     },
#     r"$\pi$ non-zeros": {
#         "values": {
#             "Train": [0.0185, 0.72, 0.04, 0.14],
#             "Val":   [0.0206, 0.59, 0.046, 0.14],
#             "Test":  [0.0202, 0.57, 0.045, 0.14],
#         },
#         "errors": {
#             "Train": [0.0696, 0.21, 0.095, 0.0003],
#             "Val":   [0.0807, 0.24, 0.104, 0.0003],
#             "Test":  [0.080, 0.25, 0.11, 0.0003],
#         },
#     },
#     r"$\mu$ zeros": {
#         "values": {
#             "Train": [1.025, 12.7, 1.81, 6.56],
#             "Val":   [1.00, 7.46, 1.79, 6.60],
#             "Test":  [0.98, 7.32, 1.75, 6.42],
#         },
#         "errors": {
#             "Train": [5.14, 14.5, 5.16, 6.22],
#             "Val":   [5.04, 10.56, 5.0, 6.17],
#             "Test":  [4.99, 10.61, 4.93, 6.60],
#         },
#     },
#     r"$\mu$ non-zeros": {
#         "values": {
#             "Train": [11.75, 19.9, 11.78, 7.26],
#             "Val":   [11.87, 12.76, 11.7, 7.41],
#             "Test":  [11.93, 12.41, 11.6, 7.47],
#         },
#         "errors": {
#             "Train": [18.06, 19.3, 16.24, 4.81],
#             "Val":   [18.41, 17.02, 16.3, 4.87],
#             "Test":  [18.51, 16.0, 16.2, 5.06],
#         },
#     },
#     r"$\theta$": {
#         "values": {
#             "Train": [2607, 1.00, 4.71, 1.0],
#             "Val":   [2568, 1.05, 4.48, 1.0],
#             "Test":  [2541, 1.04, 4.50, 1.0],
#         },
#         "errors": {
#             "Train": [6384, 1.4, 10.32, 0.0],
#             "Val":   [6336, 28.09, 9.3, 0.0],
#             "Test":  [6305, 21.24, 9.39, 0.0],
#         },
#     },
# }

# imp_params = {
#     r"$\pi$": {
#         "values": {
#             "Train": [0.0206, 0.59, 0.043, 0.14],
#             "Val":   [0.0228, 0.60, 0.048, 0.14],
#             "Test":  [0.0223, 0.57, 0.047, 0.14],
#         },
#         "errors": {
#             "Train": [0.0772, 0.26, 0.0968, 0.0003],
#             "Val":   [0.0877, 0.23, 0.106, 0.0003],
#             "Test":  [0.0866, 0.26, 0.11, 0.0003],
#         },
#     },
#     r"$\mu$": {
#         "values": {
#             "Train": [11.77, 19.2, 11.2, 7.27],
#             "Val":   [11.88, 12.35, 11.04, 7.41],
#             "Test":  [11.96, 12.02, 10.97, 7.47],
#         },
#         "errors": {
#             "Train": [18.23, 18.5, 15.2, 4.81],
#             "Val":   [18.61, 18.37, 15.05, 4.87],
#             "Test":  [18.52, 16.05, 15.0, 5.06],
#         },
#     },
#     r"$\theta$": {
#         "values": {
#             "Train": [3245, 1.00, 5.81, 1.0],
#             "Val":   [3192, 1.36, 5.5, 1.0],
#             "Test":  [3153, 1.30, 5.5, 1.0],
#         },
#         "errors": {
#             "Train": [7007, 0.913, 11.55, 0.0],
#             "Val":   [6336, 80.70, 10.2, 0.0],
#             "Test":  [6919, 66.61, 10.19, 0.0],
#         },
#     },
# }

# # =========================================================
# # Helpers
# # =========================================================
# def get_ylim(title, values, errors):
#     values = np.asarray(values, dtype=float)
#     errors = np.asarray(errors, dtype=float)

#     data_min = np.min(values - errors)
#     data_max = np.max(values + errors)

#     # Lower bound:
#     # - for R^2: allow negatives, but make sure 0 is included
#     # - for everything else: start at 0
#     if "R^2" in title:
#         ymin = min(0.0, data_min)
#     else:
#         ymin = 0.0

#     # Upper bound:
#     # - for R^2 and pi plots: fix at 1
#     # - otherwise: add a bit of padding
#     if "R^2" in title or r"$\pi$" in title:
#         ymax = 1.0
#     else:
#         ymax = data_max
#         if ymax == 0:
#             ymax = 0.1
#         else:
#             ymax = ymax * 1.12

#     return ymin, ymax


# def plot_single_metric(ax, metric_dict, title, panel_label=None, width=0.22):
#     x = np.arange(len(splits))
#     offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

#     all_vals = []
#     all_errs = []

#     for m_idx, model in enumerate(models):
#         vals = [metric_dict["values"][split][m_idx] for split in splits]
#         errs = [metric_dict["errors"][split][m_idx] for split in splits]

#         all_vals.extend(vals)
#         all_errs.extend(errs)

#         ax.bar(
#             x + offsets[m_idx],
#             vals,
#             width=width,
#             yerr=errs,
#             capsize=2,
#             color=model_colors[model],
#             edgecolor="black",
#             linewidth=0.5,
#             label=model,
#             zorder=3,
#             error_kw={"elinewidth": 0.8, "capthick": 0.8},
#         )

#     ymin, ymax = get_ylim(title, all_vals, all_errs)
#     ax.set_ylim(ymin, ymax)

#     ax.set_xticks(x)
#     ax.set_xticklabels(splits)
#     ax.set_title(title, pad=8)
#     ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8, zorder=0)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
#     ax.tick_params(axis="x", length=0)

#     if panel_label is not None:
#         ax.text(
#             -0.14, 1.10, panel_label,
#             transform=ax.transAxes,
#             fontsize=14,
#             fontweight="bold",
#             ha="left",
#             va="top",
#         )
# # =========================================================
# # Core metric figure (2 x 3)
# # =========================================================
# fig_metrics, axes_metrics = plt.subplots(2, 3, figsize=(15, 8))
# fig_metrics.subplots_adjust(
#     top=0.84,
#     bottom=0.10,
#     left=0.07,
#     right=0.98,
#     wspace=0.30,
#     hspace=0.42,
# )

# plot_single_metric(axes_metrics[0, 0], recon_core["ZINB NLL"], "Reconstruction: ZINB NLL", "a")
# plot_single_metric(axes_metrics[0, 1], recon_core[r"$R^2$"], r"Reconstruction: $R^2$", "b")
# plot_single_metric(axes_metrics[0, 2], recon_core["MAE"], "Reconstruction: MAE", "c")

# plot_single_metric(axes_metrics[1, 0], imp_core["Masked recon."], "Masked Values: Masked recon.", "d")
# plot_single_metric(axes_metrics[1, 1], imp_core[r"$R^2$"], r"Masked Values: $R^2$", "e")
# plot_single_metric(axes_metrics[1, 2], imp_core["MAE"], "Masked Values: MAE", "f")

# handles_metrics, labels_metrics = axes_metrics[0, 0].get_legend_handles_labels()
# fig_metrics.legend(
#     handles_metrics,
#     labels_metrics,
#     loc="upper center",
#     ncol=4,
#     frameon=False,
#     bbox_to_anchor=(0.5, 0.98),
#     columnspacing=1.2,
#     handletextpad=0.5,
# )

# # =========================================================
# # Parameter figure with 3 / 2 / 3 layout
# # =========================================================
# fig = plt.figure(figsize=(15, 11))
# gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

# # Row 1
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[0, 2])

# # Row 2
# ax4 = fig.add_subplot(gs[1, 0])
# ax5 = fig.add_subplot(gs[1, 1])

# # Row 3
# ax6 = fig.add_subplot(gs[2, 0])
# ax7 = fig.add_subplot(gs[2, 1])
# ax8 = fig.add_subplot(gs[2, 2])

# plot_single_metric(ax1, recon_params[r"$\pi$ non-zeros"], r"Reconstruction: $\pi$ non-zeros", "a")
# plot_single_metric(ax2, recon_params[r"$\mu$ non-zeros"], r"Reconstruction: $\mu$ non-zeros", "b")
# plot_single_metric(ax3, recon_params[r"$\theta$"], r"Reconstruction: $\theta$", "c")

# plot_single_metric(ax4, recon_params[r"$\pi$ zeros"], r"Reconstruction: $\pi$ zeros", "d")
# plot_single_metric(ax5, recon_params[r"$\mu$ zeros"], r"Reconstruction: $\mu$ zeros", "e")

# plot_single_metric(ax6, imp_params[r"$\pi$"], r"Masked Values: $\pi$", "f")
# plot_single_metric(ax7, imp_params[r"$\mu$"], r"Masked Values: $\mu$", "g")
# plot_single_metric(ax8, imp_params[r"$\theta$"], r"Masked Values: $\theta$", "h")

# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(
#     handles,
#     labels,
#     loc="upper center",
#     ncol=4,
#     frameon=False,
#     bbox_to_anchor=(0.5, 0.98),
#     columnspacing=1.2,
#     handletextpad=0.5,
# )

# # plt.show()

# # # Optional save
# fig_metrics.savefig("core_metric_comparison.png", dpi=300, bbox_inches="tight")
# fig_metrics.savefig("core_metric_comparison.pdf", bbox_inches="tight")

# fig.savefig("parameter_comparison_custom_layout.png", dpi=300, bbox_inches="tight")
# fig.savefig("parameter_comparison_custom_layout.pdf", bbox_inches="tight")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from Utils.plot_config import setup_plot_style, COLORS

# =========================================================
# Input data
# =========================================================
recon_core = {
    "ZINB NLL": {
        "values": {
            "Train": [1.84, 0.91, 2.05, 2.82],
            "Val":   [1.88, 0.886, 2.10, 2.83],
            "Test":  [1.87, 0.98, 2.10, 2.82],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    r"$R^2$": {
        "values": {
            "Train": [0.88, -2.4, 0.75, 0.09],
            "Val":   [0.85, -0.84, 0.71, 0.09],
            "Test":  [0.86, -0.71, 0.70, 0.10],
        },
        "errors": {
            "Train": [0.0, 0.0, 0.0, 0.0],
            "Val":   [0.0, 0.0, 0.0, 0.0],
            "Test":  [0.0, 0.0, 0.0, 0.0],
        },
    },
    "MAE": {
        "values": {
            "Train": [2.1, 13.04, 3.41, 8.28],
            "Val":   [2.3, 8.21, 3.65, 8.38],
            "Test":  [2.3, 9.19, 3.69, 8.32],
        },
        "errors": {
            "Train": [5.40, 15.03, 7.52, 13.21],
            "Val":   [6.01, 12.70, 8.14, 13.32],
            "Test":  [6.36, 13.03, 8.32, 13.32],
        },
    },
}

recon_params = {
    r"$\pi$ zeros": {
        "values": {
            "Train": [0.78, 0.91, 0.42, 0.14],
            "Val":   [0.77, 0.84, 0.39, 0.14],
            "Test":  [0.76, 0.84, 0.40, 0.14],
        },
        "errors": {
            "Train": [0.32, 0.10, 0.32, 0.0002],
            "Val":   [0.32, 0.16, 0.30, 0.0002],
            "Test":  [0.32, 0.16, 0.31, 0.0002],
        },
    },
    r"$\pi$ non-zeros": {
        "values": {
            "Train": [0.0185, 0.72, 0.04, 0.14],
            "Val":   [0.0206, 0.59, 0.046, 0.14],
            "Test":  [0.0202, 0.57, 0.045, 0.14],
        },
        "errors": {
            "Train": [0.0696, 0.21, 0.095, 0.0003],
            "Val":   [0.0807, 0.24, 0.104, 0.0003],
            "Test":  [0.080, 0.25, 0.11, 0.0003],
        },
    },
    r"$\mu$ zeros": {
        "values": {
            "Train": [1.025, 12.7, 1.81, 6.56],
            "Val":   [1.00, 7.46, 1.79, 6.60],
            "Test":  [0.98, 7.32, 1.75, 6.42],
        },
        "errors": {
            "Train": [5.14, 14.5, 5.16, 6.22],
            "Val":   [5.04, 10.56, 5.0, 6.17],
            "Test":  [4.99, 10.61, 4.93, 6.60],
        },
    },
    r"$\mu$ non-zeros": {
        "values": {
            "Train": [11.75, 19.9, 11.78, 7.26],
            "Val":   [11.87, 12.76, 11.7, 7.41],
            "Test":  [11.93, 12.41, 11.6, 7.47],
        },
        "errors": {
            "Train": [18.06, 19.3, 16.24, 4.81],
            "Val":   [18.41, 17.02, 16.3, 4.87],
            "Test":  [18.51, 16.0, 16.2, 5.06],
        },
    },
    r"$\theta$": {
        "values": {
            "Train": [2607, 1.00, 4.71, 1.0],
            "Val":   [2568, 1.05, 4.48, 1.0],
            "Test":  [2541, 1.04, 4.50, 1.0],
        },
        "errors": {
            "Train": [6384, 1.4, 10.32, 0.0],
            "Val":   [6336, 28.09, 9.3, 0.0],
            "Test":  [6305, 21.24, 9.39, 0.0],
        },
    },
}

# =========================================================
# Style / settings
# =========================================================
setup_plot_style()

# smaller subplot titles
plt.rcParams.update({
    "axes.titlesize": 14,
})

test_split = "Test"

# only first, third, fourth columns
selected_idx = [0, 2, 3]
models = ["Combined", "Masked recon.", "Bins"]

model_colors = {
    "Combined": COLORS["blue"],
    "Masked recon.": COLORS["green"],
    "Bins": COLORS["red"],
}

# =========================================================
# Helpers
# =========================================================
def get_ylim(title, values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    data_min = np.min(values - errors)
    data_max = np.max(values + errors)

    if "R^2" in title:
        ymin = min(0.0, data_min)
    else:
        ymin = 0.0

    if "R^2" in title or r"$\pi$" in title:
        ymax = 1.0
    else:
        ymax = data_max
        if ymax == 0:
            ymax = 0.1
        else:
            ymax = ymax * 1.12

    return ymin, ymax


def plot_test_metric(ax, metric_dict, title, panel_label=None, width=0.6):
    x = np.arange(len(models))

    vals = [metric_dict["values"][test_split][i] for i in selected_idx]
    errs = [metric_dict["errors"][test_split][i] for i in selected_idx]

    for i, model in enumerate(models):
        ax.bar(
            x[i],
            vals[i],
            width=width,
            yerr=errs[i],
            capsize=2,
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )

    ymin, ymax = get_ylim(title, vals, errs)
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_title(title, pad=8, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="x", length=0)

    if panel_label is not None:
        ax.text(
            -0.14, 1.10, panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

# =========================================================
# Figure with 2 panels
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
fig.subplots_adjust(
    top=0.82,
    bottom=0.18,
    left=0.08,
    right=0.98,
    wspace=0.32,
)

plot_test_metric(axes[0], recon_core["MAE"], "Reconstruction: MAE", "a")
plot_test_metric(axes[1], recon_params[r"$\pi$ zeros"], r"Reconstruction: $\pi$ zeros", "b")

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=model_colors[m], edgecolor="black", linewidth=0.5)
    for m in models
]
fig.legend(
    legend_handles,
    models,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.98),
    columnspacing=1.2,
    handletextpad=0.5,
)

plt.show()

# optional save
fig.savefig("test_only_mae_pi_zero_selected_models.png", dpi=300, bbox_inches="tight")
fig.savefig("test_only_mae_pi_zero_selected_models.pdf", bbox_inches="tight")