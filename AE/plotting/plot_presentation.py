import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Utils.plot_config import setup_plot_style, COLORS


splits = ["Train", "Val", "Test"]
models = ["Combined", "ZINB NLL", "Masked Recon.", "Bins"]
model_colors = {
    "Combined": COLORS["blue"],
    "ZINB NLL": COLORS["orange"],
    "Masked Recon.": COLORS["green"],
    "Bins": COLORS["red"],
}

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

split_colors = {
    "Train": "#9aa0a6",
    "Val": "#5f6368",
    "Test": "#202124",
}

plot_splits = ["Train", "Test"]

def get_metric_values(metric_dict, metric_name, model_idx, selected_splits=plot_splits):
    vals = [metric_dict[metric_name]["values"][s][model_idx] for s in selected_splits]
    errs = [metric_dict[metric_name]["errors"][s][model_idx] for s in selected_splits]
    return np.array(vals), np.array(errs)

metrics = [
    ("ZINB NLL", recon_core, "ZINB NLL"),
    ("MAE", recon_core, "MAE"),
    (r"$\pi$ zeros", recon_params, r"$\pi$ zeros"),
    (r"$\pi$ non-zeros", recon_params, r"$\pi$ non-zeros"),
]

def plot_zinb_only(save_path="AE/results/zinb_only_results.png"):
    compare_models = ["Combined", "ZINB NLL"]
    width = 0.35
    x = np.arange(len(plot_splits))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    legend_handles = []

    for ax, (title, source, key) in zip(axes, metrics):
        for i, model in enumerate(compare_models):
            model_idx = models.index(model)
            y, yerr = get_metric_values(source, key, model_idx)

            xpos = x + (i - 0.5) * width

            if model == "Combined":
                ax.bar(
                    xpos,
                    y,
                    width,
                    color="none",
                    edgecolor="none",
                    alpha=0,
                    label="_nolegend_",
                )
            else:
                bars = ax.bar(
                    xpos,
                    y,
                    width,
                    yerr=yerr,
                    capsize=4,
                    label=model,
                    color=model_colors[model],
                )
                if not legend_handles:
                    legend_handles.append(bars[0])

        ax.set_xticks(x)
        ax.set_xticklabels(plot_splits)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)
        if title == "MAE":
            ax.set_ylim(bottom=0)

    fig.legend(
        legend_handles,
        ["ZINB NLL"],
        loc="upper center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("ZINB NLL model results", fontsize=16, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_zinb_vs_combined(save_path="AE/results/zinb_vs_combined_results.png"):
    compare_models = ["Combined", "ZINB NLL"]
    width = 0.35
    x = np.arange(len(plot_splits))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    for ax, (title, source, key) in zip(axes, metrics):
        for i, model in enumerate(compare_models):
            model_idx = models.index(model)
            y, yerr = get_metric_values(source, key, model_idx)

            bars = ax.bar(
                x + (i - 0.5) * width,
                y,
                width,
                yerr=yerr,
                capsize=4,
                label=model,
                color=model_colors[model],
            )

            if title == metrics[0][0]:
                legend_handles.append(bars[0])
                legend_labels.append(model)

        ax.set_xticks(x)
        ax.set_xticklabels(plot_splits)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)
        if title == "MAE":
            ax.set_ylim(bottom=0)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("Combined vs ZINB NLL", fontsize=16, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


plot_zinb_only()
plot_zinb_vs_combined()