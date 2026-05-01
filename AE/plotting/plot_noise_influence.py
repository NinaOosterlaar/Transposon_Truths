import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Utils.plot_config import setup_plot_style, COLORS

# =========================================================
# Plot style
# =========================================================
setup_plot_style()

# =========================================================
# Load and prepare data
# =========================================================
df = pd.read_csv('AE/results/main_results/noise_sweep_metrics.csv')

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
    
    # Add error bars for MAE and masked_loss test data
    if metric == 'mae':
        ax.errorbar(noise_levels, df_test[metric].values, 
                    yerr=df_test['mae_sd'].values,
                    marker='s', linewidth=2, markersize=6, 
                    color='black', linestyle='-', label='Test',
                    capsize=3, capthick=1.5)
    elif metric == 'masked_loss':
        ax.errorbar(noise_levels, df_test[metric].values, 
                    yerr=df_test['masked_loss_sd'].values,
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

plt.savefig('AE/results/main_results/noise_results/noise_sweep_figure1.png', dpi=300, bbox_inches='tight')
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
ax.errorbar(noise_levels, df_test['pi_zero'].values, 
            yerr=df_test['pi_zero_sd'].values,
            marker='o', linewidth=2, markersize=6, linestyle='-',
            color=COLORS['blue'], label='Test Zero',
            capsize=3, capthick=0.5)
ax.errorbar(noise_levels, df_test['pi_non_zero'].values, 
            yerr=df_test['pi_non_zero_sd'].values,
            marker='s', linewidth=2, markersize=6, linestyle='-',
            color=COLORS['light_blue'], label='Test Non-Zero',
            capsize=3, capthick=0.5)
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
ax.errorbar(noise_levels, df_test['mu_zero'].values, 
            yerr=df_test['mu_zero_sd'].values,
            marker='o', linewidth=2, markersize=6, linestyle='-',
            color=COLORS['blue'], label='Test Zero',
            capsize=3, capthick=0.5)
ax.errorbar(noise_levels, df_test['mu_non_zero'].values, 
            yerr=df_test['mu_non_zero_sd'].values,
            marker='s', linewidth=2, markersize=6, linestyle='-',
            color=COLORS['light_blue'], label='Test Non-Zero',
            capsize=3, capthick=0.5)
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
ax.errorbar(noise_levels, df_test['theta'].values, 
            yerr=df_test['theta_sd'].values,
            marker='s', linewidth=2, markersize=6, linestyle='-',
            color=COLORS['blue'], label='Test',
            capsize=3, capthick=0.5)
ax.set_xlabel('Noise Level')
ax.set_ylabel(r'$\theta$')
ylim = ax.get_ylim()
ax.set_ylim(0, max(ylim[1], 0.1))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.text(-0.15, 1.05, 'c', transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='bottom')

plt.savefig('AE/results/main_results/noise_results/noise_sweep_figure2.png', dpi=300, bbox_inches='tight')
print("Saved Figure 2: noise_sweep_figure2.png")

