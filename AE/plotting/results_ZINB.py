import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, roc_curve, auc
import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 
from Utils.plot_config import setup_plot_style, COLORS
from AE.plotting.plot_helper import generate_prefix, prepare_output_dirs, clip_hi

# Set up standardized plot style
setup_plot_style()
""" 
    **Variance from ZINB**: variance = μ + μ²/θ
"""

def plot_parameter_distributions(all_reconstructions_mu, all_theta, all_pi, all_variance=None, model_type="ZINB", save_dir=None, prefix="", clip_q=99.5):
    if all_theta is not None and all_pi is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Theta distribution
        theta_flat = all_theta.flatten()
        axes[0, 0].hist(theta_flat, bins=100, alpha=0.7, color=COLORS['blue'], edgecolor='black')
        axes[0, 0].set_xlabel('Dispersion (θ)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{model_type}: Distribution of Dispersion θ\n(Controls variance: smaller θ = larger variance)')
        # axes[0, 0].axvline(x=np.median(theta_flat), color='red', linestyle='--', 
        #                   linewidth=2, label=f'Median: {np.median(theta_flat):.3f}')
        # axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pi distribution
        pi_flat = all_pi.flatten()
        axes[0, 1].set_yscale('log')
        axes[0, 1].hist(pi_flat, bins=100, alpha=0.7, color=COLORS['orange'], edgecolor='black')
        axes[0, 1].set_xlabel('Zero-inflation Probability (π)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'{model_type}: Distribution of Zero-inflation π\n(Probability of structural zero)')
        # axes[0, 1].axvline(x=np.median(pi_flat), color='red', linestyle='--', 
        #                   linewidth=2, label=f'Median: {np.median(pi_flat):.3f}')
        # axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean (mu) distribution
        mu_flat = all_reconstructions_mu.flatten()
        mu_log = np.log1p(np.clip(mu_flat, 0, None))
        axes[0, 2].hist(mu_log, bins=100, alpha=0.7, color=COLORS['green'], edgecolor='black')
        axes[0, 2].set_xlabel('log1p Mean (μ)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'{model_type}: Distribution of Mean μ')
        # axes[0, 2].axvline(x=np.median(mu_flat), color='red', linestyle='--', 
        #                   linewidth=2, label=f'Median: {np.median(mu_flat):.3f}')
        # axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Variance distribution
        if all_variance is not None:
            var_flat = np.asarray(all_variance).flatten()
            var_plot = var_flat

            # clip_note = ""
            # if clip_q is not None:
            #     var_plot, var_hi = clip_hi(var_plot, q=clip_q)
            #     clip_note = f" (clipped at p{clip_q}={var_hi:.2g})"

            # var_log = np.log1p(np.clip(var_plot, 0, None))
            axes[1, 0].hist(var_plot, bins=100, alpha=0.7, color=COLORS['pink'], edgecolor='black')
            axes[1, 0].set_xlabel('log1p(Variance = μ + μ²/θ)')
            axes[1, 0].set_ylabel('Frequency')
            # y-axis log scale
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_title(f'{model_type}: Predicted Variance ')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Variance not available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            var_flat = None
            
        # Variance-to-mean ratio
        if var_flat is not None:
            valid = mu_flat > 0
            vmr = np.zeros_like(mu_flat, dtype=float)
            vmr[valid] = var_flat[valid] / mu_flat[valid]

            vmr_plot = vmr[valid]
            # clip_note = ""
            # if clip_q is not None and vmr_plot.size > 0:
            #     vmr_plot, vmr_hi = clip_hi(vmr_plot, q=clip_q)
            #     clip_note = f" (clipped at p{clip_q}={vmr_hi:.2g})"

            # vmr_log = np.log1p(np.clip(vmr_plot, 0, None))
            axes[1, 1].hist(vmr_plot, bins=100, alpha=0.7, color=COLORS['light_blue'], edgecolor='black')
            axes[1, 1].set_xlabel('log1p(Variance-to-Mean Ratio)')
            axes[1, 1].set_ylabel('Frequency')
            # y-axis log scale
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_title(f'{model_type}: VMR ')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'VMR not available',
                            ha='center', va='center', transform=axes[1, 1].transAxes)

        
        # Theta vs Mean relationship
        # sample_idx = np.random.choice(len(theta_flat), size=min(5000, len(theta_flat)), replace=False)
        # axes[1, 2].scatter(mu_flat, theta_flat, alpha=0.3, s=1, c='blue')
        # Use a hexbin
        mu_flat = all_reconstructions_mu.flatten()
        mu_flat, var_hi = clip_hi(mu_flat, q=99.9)
        axes[1, 2].hexbin(mu_flat, theta_flat, gridsize=50, cmap='YlOrRd', mincnt=1)
        axes[1, 2].set_xlabel('Mean (μ)')
        axes[1, 2].set_ylabel('Dispersion (θ)')
        axes[1, 2].set_title(f'{model_type}: θ vs μ Relationship')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_zinb_parameter_distributions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
def density_plots(actual_counts_flat, all_reconstructions_mu, residuals, comparison_label, model_type, save_dir, prefix, r2, mae, all_pi=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    sample_indices = np.arange(0, actual_counts_flat.size, 10)
    recon_flat = all_reconstructions_mu.flatten()
    
    # # Separate points by π threshold if available
    # if all_pi is not None:
    #     pi_flat = all_pi.flatten()
    #     high_pi_mask = pi_flat > 0.5
    #     low_pi_mask = ~high_pi_mask
        
    #     # Sample from each group
    #     low_pi_sample = sample_indices[low_pi_mask[sample_indices]]
    #     high_pi_sample = sample_indices[high_pi_mask[sample_indices]]
        
        # Plot low π points (reliable mean predictions)
        # axes[0].scatter(actual_counts_flat[low_pi_sample], recon_flat[low_pi_sample], 
        #                 alpha=0.3, s=1, c=COLORS['black'])
        # Plot high π points (unreliable mean predictions)
        # axes[0].scatter(actual_counts_flat[high_pi_sample], recon_flat[high_pi_sample], 
        #                alpha=0.3, s=1, c=COLORS['orange'], label='π > 0.5 (structural zeros)')
        
    # Filter out predicted values higher than 500 for better visualization
    mask = recon_flat < 500
    sample_indices_filtered = sample_indices[mask[sample_indices]]

    axes[0].scatter(actual_counts_flat[sample_indices_filtered], recon_flat[sample_indices_filtered], 
                    alpha=0.3, s=1, c=COLORS['black'])
    
    axes[0].plot([actual_counts_flat.min(), actual_counts_flat.max()], 
                [actual_counts_flat.min(), actual_counts_flat.max()], 
                'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel(f'Actual ({comparison_label})')
    axes[0].set_ylabel('Predicted Mean (μ)')
    axes[0].set_title(f'{model_type}: Actual vs Predicted μ\n(R²={r2:.4f})')
    legend = axes[0].legend(markerscale=5)
    for lh in legend.legend_handles:
        lh.set_alpha(0.7)
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot: exclude high π points
    residuals = actual_counts_flat - recon_flat
    if all_pi is not None:
        pi_flat = all_pi.flatten()
        low_pi_mask = pi_flat <= 0.5
        residuals_filtered = residuals[low_pi_mask]
        sample_indices_filtered = sample_indices[low_pi_mask[sample_indices]]
        median_residual = np.median(residuals_filtered)
        axes[1].hist(residuals_filtered[::10], bins=100, alpha=0.7, 
                    color=COLORS['pink'], edgecolor='black')
        axes[1].set_title(f'{model_type}: Residual Distribution (π ≤ 0.5 only)\n(MAE={mae:.4f})')
    else:
        median_residual = np.median(residuals)
        axes[1].hist(residuals[sample_indices], bins=100, alpha=0.7, 
                    color=COLORS['pink'], edgecolor='black')
        axes[1].set_title(f'{model_type}: Residual Distribution\n(MAE={mae:.4f})')
    
    axes[1].axvline(x=0, color=COLORS['red'], linestyle='--', linewidth=2)
    axes[1].axvline(x=median_residual, color=COLORS['green'], linestyle='--', 
                   linewidth=2, label=f'Median: {median_residual:.4f}')
    axes[1].set_xlabel('Residuals (Actual - Predicted μ)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_prediction_quality.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
def zero_inflation_analysis(all_reconstructions_mu, all_pi, all_raw_counts, model_type, save_dir, prefix):
    if all_pi is not None and all_raw_counts is not None:
        # Create a plot with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        actual_zeros = (all_raw_counts.flatten() == 0)
        actual_nonzeros = ~actual_zeros
        pi_flat = all_pi.flatten()
        
        print(len(pi_flat), np.sum(actual_zeros), np.sum(actual_nonzeros))
        
        # Histogram of pi for actual zeros vs non-zeros
        bins = np.linspace(0, 1, 51)  # 50 bins, shared
        axes[0].hist(pi_flat[actual_zeros],
            bins=bins,
            density=True,
            alpha=0.4,
            label='Observed zeros')
        axes[0].hist(pi_flat[actual_nonzeros],
            bins=bins,
            density=True,
            alpha=0.3,
            label='Observed non-zeros')


        axes[0].set_yscale("log")
        axes[0].set_ylabel("Number of positions")
        axes[0].set_xlabel('Zero-inflation Probability (π)')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'{model_type}: π Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Violin Plot
        data = [
            pi_flat[actual_zeros],
            pi_flat[actual_nonzeros]
        ]

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1-eps)
            return np.log(p/(1-p))

        data = [logit(pi_flat[actual_zeros]), logit(pi_flat[actual_nonzeros])]
        parts = axes[1].violinplot(data, showmedians=True, showextrema=True)
        
        # Color the violin plots: blue for zeros, orange for non-zeros
        colors = [COLORS['blue'], COLORS['orange']]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        # Set all lines to black
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in parts:
                parts[partname].set_edgecolor('gray')
                parts[partname].set_linewidth(1.5)
        
        axes[1].set_ylabel("logit(π)")

        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(['Observed zeros', 'Observed non-zeros'])
        axes[1].set_ylabel('Zero-inflation probability logit (π)')
        axes[1].set_title(f'{model_type}: π distribution')
        axes[1].grid(True, alpha=0.3)
                
        # # Scatter plot of pi vs mu - separated by actual zeros vs non-zeros
        # mu_clipped, mu_hi = clip_hi(all_reconstructions_mu.flatten(), q=99.9)
        # axes[1, 0].scatter(mu_clipped[actual_zeros], 
        #                   pi_flat[actual_zeros],
        #                   alpha=0.1, s=2, label='Actual Zeros', color=COLORS['blue'])
        # axes[1, 0].scatter(mu_clipped[actual_nonzeros], 
        #                   pi_flat[actual_nonzeros],
        #                   alpha=0.1, s=2, label='Actual Non-zeros', color=COLORS['orange'])
        # axes[1, 0].set_xlabel('Predicted Mean (μ)')
        # axes[1, 0].set_ylabel('Zero-inflation Probability (π)')
        # axes[1, 0].set_title(f'{model_type}: π vs μ Relationship')
        # legend = axes[1, 0].legend(markerscale=3)
        # # Set alpha for legend markers to make them more visible
        # for lh in legend.legend_handles:
        #     lh.set_alpha(0.7)
        # axes[1, 0].grid(True, alpha=0.3)   
        
        # # Zero predictions
        # predicted_zeros = pi_flat > 0.5
        # zero_accuracy = np.mean(predicted_zeros == actual_zeros)
        
        # # Confusion matrix
        # cm = confusion_matrix(actual_zeros, predicted_zeros)
        # im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # axes[1, 1].figure.colorbar(im, ax=axes[1, 1])
        # axes[1, 1].set(xticks=np.arange(cm.shape[1]),
        #               yticks=np.arange(cm.shape[0]),
        #               xticklabels=['Non-zero', 'Zero'],
        #               yticklabels=['Non-zero', 'Zero'],
        #               ylabel='Actual (from raw counts)',
        #               xlabel='Predicted (π > 0.5)')
        
        # thresh = cm.max() / 2.
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         axes[1, 1].text(j, i, format(cm[i, j], 'd'),
        #                       ha="center", va="center",
        #                       color="white" if cm[i, j] > thresh else "black",
        #                       fontsize=14)
        # axes[1, 1].set_title(f'{model_type}: Zero Prediction\n(Accuracy={zero_accuracy:.4f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_zero_inflation_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def masked_values_analysis(all_reconstructions_mu, all_pi, all_raw_counts, all_masks, all_theta, model_type, save_dir, prefix, threshold):
    """
    Analyze how well masked values (from denoising) are reconstructed.
    Only called when denoise_percent > 0.
    """
    if all_masks is None or all_raw_counts is None:
        return
    
    mask_flat = all_masks.flatten()
    masked_positions = mask_flat == True
    
    if not np.any(masked_positions):
        print("No masked values to analyze.")
        return
    
    mu_flat = all_reconstructions_mu.flatten()
    raw_flat = all_raw_counts.flatten()
    pi_flat = all_pi.flatten() if all_pi is not None else None
    theta_flat = all_theta.flatten() if all_theta is not None else None
    
    # Extract masked values
    masked_actual = raw_flat[masked_positions]
    masked_recon = mu_flat[masked_positions]
    
    print("Number of masked values:", len(masked_actual))
    
    # Compute metrics for masked values
    mae_masked = mean_absolute_error(masked_actual, masked_recon)
    abs_err = np.abs(masked_actual - masked_recon)
    mae_std_masked = abs_err.std(ddof=1)
    r2_masked = r2_score(masked_actual, masked_recon)
    
    # Compute detailed statistics for masked values (similar to overall metrics)
    zero_mask_masked = masked_actual == 0
    non_zero_mask_masked = ~zero_mask_masked
    
    # π statistics for masked values
    if pi_flat is not None:
        masked_pi = pi_flat[masked_positions]
        mean_pi_zeros_masked = masked_pi[zero_mask_masked].mean() if np.any(zero_mask_masked) else 0
        mean_pi_nonzeros_masked = masked_pi[non_zero_mask_masked].mean() if np.any(non_zero_mask_masked) else 0
        std_pi_zeros_masked = masked_pi[zero_mask_masked].std(ddof=1) if np.any(zero_mask_masked) else 0
        std_pi_nonzeros_masked = masked_pi[non_zero_mask_masked].std(ddof=1) if np.any(non_zero_mask_masked) else 0
    else:
        mean_pi_zeros_masked = mean_pi_nonzeros_masked = std_pi_zeros_masked = std_pi_nonzeros_masked = 0
    
    # μ statistics for masked values
    mean_mu_zeros_masked = masked_recon[zero_mask_masked].mean() if np.any(zero_mask_masked) else 0
    mean_mu_nonzeros_masked = masked_recon[non_zero_mask_masked].mean() if np.any(non_zero_mask_masked) else 0
    std_mu_zeros_masked = masked_recon[zero_mask_masked].std(ddof=1) if np.any(zero_mask_masked) else 0
    std_mu_nonzeros_masked = masked_recon[non_zero_mask_masked].std(ddof=1) if np.any(non_zero_mask_masked) else 0
    
    # θ statistics for masked values
    if theta_flat is not None:
        masked_theta = theta_flat[masked_positions]
        mean_theta_masked = masked_theta.mean()
        std_theta_masked = masked_theta.std(ddof=1)
    else:
        mean_theta_masked = std_theta_masked = 0
    
    # Print detailed statistics for masked values
    print("\n" + "="*60)
    print("MASKED VALUES PERFORMANCE METRICS")
    print("="*60)
    print(f"MAE: {mae_masked:.4f}, SD MAE: {mae_std_masked:.4f}")
    print(f"R²: {r2_masked:.4f}")
    print(f"Mean π (zeros): {mean_pi_zeros_masked:.4f}, Mean π (non-zeros): {mean_pi_nonzeros_masked:.4f}")
    print(f"SD π (zeros): {std_pi_zeros_masked:.4f}, SD π (non-zeros): {std_pi_nonzeros_masked:.4f}")
    print(f"Mean μ (zeros): {mean_mu_zeros_masked:.4f}, Mean μ (non-zeros): {mean_mu_nonzeros_masked:.4f}")
    print(f"SD μ (zeros): {std_mu_zeros_masked:.4f}, SD μ (non-zeros): {std_mu_nonzeros_masked:.4f}")
    print(f"Theta mean: {mean_theta_masked:.4f}, Theta SD: {std_theta_masked:.4f}")
    print("="*60 + "\n")
    
    # Baseline comparison: mean/median of neighbors
    # Reshape data to work with sequences
    n_samples = all_masks.shape[0]
    seq_length = all_masks.shape[1]
    
    baseline_results = {
        'neighborhood_sizes': list(range(1, 6)),
        'mean_mae': [],
        'mean_mae_std': [],
        'mean_r2': [],
        'median_mae': [],
        'median_mae_std': [],
        'median_r2': []
    }
    
    print("Computing baseline predictions using neighborhood mean/median of reconstructed values...")
    for n_size in baseline_results['neighborhood_sizes']:
        # Pre-allocate arrays for better performance
        all_mean_preds = []
        all_median_preds = []
        all_actual_values = []  # Actual raw values at masked positions
        
        # Create offset array once (excluding 0)
        offsets = np.concatenate([np.arange(-n_size, 0), np.arange(1, n_size + 1)])
        
        # Process each sequence
        for seq_idx in range(n_samples):
            seq_mask = all_masks[seq_idx]
            seq_recon_mu = all_reconstructions_mu[seq_idx]  # Use model's reconstructed μ values
            seq_raw = all_raw_counts[seq_idx]  # Actual raw values
            
            # Find masked positions in this sequence
            masked_pos = np.where(seq_mask)[0]
            
            if len(masked_pos) == 0:
                continue
            
            # Vectorized: compute all neighbor indices for all masked positions at once
            # Shape: (num_masked_positions, num_offsets)
            neighbor_indices = masked_pos[:, np.newaxis] + offsets[np.newaxis, :]
            
            # Create validity mask: only check within bounds (allow masked neighbors now)
            within_bounds = (neighbor_indices >= 0) & (neighbor_indices < seq_length)
            valid_mask = within_bounds
            
            # Get neighbor RECONSTRUCTED values for all positions
            neighbor_values = seq_recon_mu[np.clip(neighbor_indices, 0, seq_length - 1)]
            
            # Create masked array where invalid neighbors are masked
            masked_neighbors = np.ma.array(neighbor_values, mask=~valid_mask)
            
            # Compute mean and median for each masked position
            has_valid = valid_mask.any(axis=1)
            if np.any(has_valid):
                means = np.ma.mean(masked_neighbors[has_valid], axis=1).data
                medians = np.ma.median(masked_neighbors[has_valid], axis=1).data
                actual_values_batch = seq_raw[masked_pos[has_valid]]  # Actual raw values at masked positions
                
                all_mean_preds.extend(means)
                all_median_preds.extend(medians)
                all_actual_values.extend(actual_values_batch)
        
        # Calculate metrics for this neighborhood size
        if len(all_actual_values) > 0:
            actual_values = np.array(all_actual_values)  # Actual raw values at masked positions
            mean_preds = np.array(all_mean_preds)
            median_preds = np.array(all_median_preds)
            
            # Mean-based metrics (comparing baseline to actual values)
            mean_abs_err = np.abs(actual_values - mean_preds)
            baseline_results['mean_mae'].append(np.mean(mean_abs_err))
            baseline_results['mean_mae_std'].append(np.std(mean_abs_err, ddof=1))
            baseline_results['mean_r2'].append(r2_score(actual_values, mean_preds))
            
            # Median-based metrics (comparing baseline to actual values)
            median_abs_err = np.abs(actual_values - median_preds)
            baseline_results['median_mae'].append(np.mean(median_abs_err))
            baseline_results['median_mae_std'].append(np.std(median_abs_err, ddof=1))
            baseline_results['median_r2'].append(r2_score(actual_values, median_preds))
        else:
            # No valid predictions for this neighborhood size
            baseline_results['mean_mae'].append(np.nan)
            baseline_results['mean_mae_std'].append(np.nan)
            baseline_results['mean_r2'].append(np.nan)
            baseline_results['median_mae'].append(np.nan)
            baseline_results['median_mae_std'].append(np.nan)
            baseline_results['median_r2'].append(np.nan)
    
    # Create baseline comparison plot
    fig_baseline, axes_baseline = plt.subplots(1, 2, figsize=(14, 5))
    
    neighborhood_sizes = baseline_results['neighborhood_sizes']
    
    # MAE comparison - bar plot
    # Model bar at position 0, baselines at positions 1-10
    bar_width = 0.35
    
    # Model bar on the left (position 0)
    axes_baseline[0].bar(0, mae_masked, 
                         width=bar_width*2, label=f'Model: {mae_masked:.4f}',
                         yerr=mae_std_masked,
                         error_kw={'ecolor': 'black', 'linewidth': 1.5, 'capsize': 3},
                         alpha=0.8, color=COLORS['red'])
    
    # Baseline bars starting at position 1.5 (with gap after model)
    x_pos_baseline = np.arange(len(neighborhood_sizes)) + 1.5
    axes_baseline[0].bar(x_pos_baseline - bar_width/2, baseline_results['mean_mae'], 
                         width=bar_width, label='Mean of neighbors', 
                         yerr=baseline_results['mean_mae_std'], 
                         error_kw={'ecolor': 'black', 'linewidth': 1.5, 'capsize': 3},
                         alpha=0.8, color=COLORS['blue'])
    axes_baseline[0].bar(x_pos_baseline + bar_width/2, baseline_results['median_mae'], 
                         width=bar_width, label='Median of neighbors',
                         yerr=baseline_results['median_mae_std'],
                         error_kw={'ecolor': 'black', 'linewidth': 1.5, 'capsize': 3},
                         alpha=0.8, color=COLORS['orange'])
    
    axes_baseline[0].set_xlabel('Neighborhood Size')
    axes_baseline[0].set_ylabel('MAE ')
    axes_baseline[0].set_title(f'{model_type}: Baseline MAE')
    axes_baseline[0].set_xticks(np.concatenate([[0], x_pos_baseline]))
    axes_baseline[0].set_xticklabels(['Model'] + [str(s) for s in neighborhood_sizes])
    axes_baseline[0].legend(fontsize=9)
    axes_baseline[0].grid(True, alpha=0.3, axis='y')
    axes_baseline[0].set_ylim(bottom=0)
    
    # R² comparison - bar plot
    # Model bar on the left (position 0)
    axes_baseline[1].bar(0, r2_masked,
                         width=bar_width*2, label=f'Model: {r2_masked:.4f}',
                         alpha=0.8, color=COLORS['red'])
    
    # Baseline bars starting at position 1.5 (with gap after model)
    axes_baseline[1].bar(x_pos_baseline - bar_width/2, baseline_results['mean_r2'],
                         width=bar_width, label='Mean of neighbors',
                         alpha=0.8, color=COLORS['blue'])
    axes_baseline[1].bar(x_pos_baseline + bar_width/2, baseline_results['median_r2'],
                         width=bar_width, label='Median of neighbors',
                         alpha=0.8, color=COLORS['orange'])
    
    axes_baseline[1].set_xlabel('Neighborhood Size')
    axes_baseline[1].set_ylabel('R² ')
    axes_baseline[1].set_title(f'{model_type}: Baseline R²')
    axes_baseline[1].set_xticks(np.concatenate([[0], x_pos_baseline]))
    axes_baseline[1].set_xticklabels(['Model'] + [str(s) for s in neighborhood_sizes])
    axes_baseline[1].legend(fontsize=9)
    axes_baseline[1].grid(True, alpha=0.3, axis='y')
    axes_baseline[1].set_ylim(0, 1.2)  # Extended to 1.2 to avoid cramping
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_masked_baseline_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Baseline comparison completed and saved.")
    print(f"Note: Baseline uses mean/median of model's reconstructed values from neighbors")
    print(f"      Comparing baseline predictions to actual raw values at masked positions")
    print(f"      Model reference line shows model performance vs actual values")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Actual vs Reconstructed for masked values
    sample_size = min(10000, len(masked_actual))
    sample_idx = np.random.choice(len(masked_actual), size=sample_size, replace=False)
    
    axes[0, 0].scatter(masked_actual[sample_idx], masked_recon[sample_idx],
                      alpha=0.3, s=2, color=COLORS['black'])
    axes[0, 0].plot([masked_actual.min(), masked_actual.max()],
                   [masked_actual.min(), masked_actual.max()],
                   'r--', lw=2, label='Perfect reconstruction')
    axes[0, 0].set_xlabel('Actual Value (Raw Counts)')
    axes[0, 0].set_ylabel('Reconstructed μ')
    axes[0, 0].set_title(f'{model_type}: Masked Values Reconstruction\n(MAE={mae_masked:.4f}, R²={r2_masked:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. π distribution for masked positions
    if pi_flat is not None:
        masked_pi = pi_flat[masked_positions]
        axes[0, 1].hist(masked_pi, bins=50, alpha=0.7, color=COLORS['orange'], edgecolor='black')
        axes[0, 1].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[0, 1].set_xlabel('Zero-inflation Probability (π)')
        axes[0, 1].set_ylabel('Frequency')
        # axes[0, 1].set_yscale('log')
        axes[0, 1].set_title(f'{model_type}: π Distribution for Masked Positions\\n(Mean π={np.mean(masked_pi):.3f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals for masked values
    residuals_masked = masked_actual - masked_recon
    axes[1, 0].hist(residuals_masked, bins=100, alpha=0.7, color=COLORS['pink'], edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axvline(x=np.median(residuals_masked), color='green', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(residuals_masked):.4f}')
    axes[1, 0].set_xlabel('Residuals (Actual - Reconstructed μ)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{model_type}: Residuals for Masked Values')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. π vs μ scatter plot for masked values
    if pi_flat is not None:
        masked_pi = pi_flat[masked_positions]
        
        # Sample for scatter plot if too many points
        sample_size_scatter = min(10000, len(masked_recon))
        if len(masked_recon) > sample_size_scatter:
            scatter_idx = np.random.choice(len(masked_recon), size=sample_size_scatter, replace=False)
            mu_scatter = masked_recon[scatter_idx]
            pi_scatter = masked_pi[scatter_idx]
        else:
            mu_scatter = masked_recon
            pi_scatter = masked_pi
        
        axes[1, 1].scatter(mu_scatter, pi_scatter, alpha=0.3, s=2, color=COLORS['green'])
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[1, 1].set_xlabel('Predicted Mean (μ)')
        axes[1, 1].set_ylabel('Zero-inflation Probability (π)')
        axes[1, 1].set_title(f'{model_type}: π vs μ for Masked Values')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'π not available', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_masked_values_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Masked values analysis: {len(masked_actual)} masked positions, MAE={mae_masked:.4f}, R²={r2_masked:.4f}")

def zero_imputation_analysis(all_reconstructions_mu, all_pi, all_raw_counts, model_type, save_dir, prefix, pi_threshold=0.5):
    """
    Analyze zero imputation success: compare actual zeros vs predicted zeros.
    Always called for ZINB models.
    """
    if all_raw_counts is None or all_pi is None:
        return
    
    raw_flat = all_raw_counts.flatten()
    pi_flat = all_pi.flatten()
    mu_flat = all_reconstructions_mu.flatten()
    
    actual_zeros = (raw_flat == 0)
    predicted_structural_zeros = (pi_flat > pi_threshold)
    
    n_actual_zeros = np.sum(actual_zeros)
    n_predicted_zeros = np.sum(predicted_structural_zeros)
    n_total = len(raw_flat)
    
    # Imputation success: positions that were zero but now have non-zero μ prediction (and low π)
    imputed_positions = actual_zeros & ~predicted_structural_zeros
    n_imputed = np.sum(imputed_positions)
    
    # False structural zeros: positions that are non-zero but predicted as structural zero
    false_structural = ~actual_zeros & predicted_structural_zeros
    n_false_structural = np.sum(false_structural)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Zero counts comparison
    categories = ['Original\nZeros', 'Predicted\nZeros', 'Imputed\n(zero→non-zero)', '(non-zero→zero)']
    counts = [n_actual_zeros, n_predicted_zeros, n_imputed, n_false_structural]
    colors_bar = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red']]
    
    axes[0].bar(categories, counts, color=colors_bar, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{model_type}: Zero Imputation Overview')
    axes[0].grid(True, alpha=0.3, axis='y')
    # Add text labels and adjust y-axis to prevent overlap with title
    for i, (cat, count) in enumerate(zip(categories, counts)):
        pct = 100 * count / n_total
        axes[0].text(i, count, f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom')
    # Set y-limit to add 20% padding at the top for labels
    axes[0].set_ylim([0, max(counts) * 1.2])
    
    # 2. π distribution for original zeros (histogram)
    pi_actual_zeros = pi_flat[actual_zeros]
    axes[1].hist(pi_actual_zeros, bins=50, alpha=0.7, color=COLORS['blue'], edgecolor='black')
    axes[1].axvline(x=pi_threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold ({pi_threshold})')
    axes[1].set_xlabel('Zero-inflation Probability (π)')
    axes[1].set_ylabel('Frequency')
    # axes[1].set_yscale('log')
    axes[1].set_title(f'{model_type}: π Distribution for Original Zeros\n(Mean π={np.mean(pi_actual_zeros):.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_zero_imputation_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Zero imputation: {n_actual_zeros:,} actual zeros, {n_imputed:,} imputed ({100*n_imputed/n_actual_zeros:.1f}%), {n_false_structural:,} false structural zeros")
        
def reconstructions(all_originals, all_reconstructions_mu, all_variance=None, all_pi=None, all_raw_counts=None, n_examples=5, model_type="ZINB", save_dir=None, prefix="", pi_threshold=0.5):
    fig, axes = plt.subplots(n_examples, 1, figsize=(15, 4*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i in range(min(n_examples, len(all_reconstructions_mu))):
        ax = axes[i]
        positions = np.arange(len(all_reconstructions_mu[i]))
        
        # Use raw counts for comparison if available
        if all_raw_counts is not None:
            actual_data = all_raw_counts[i]
            actual_label = 'Actual (Raw Counts)'
        else:
            actual_data = all_originals[i]
            actual_label = 'Actual (Normalized)'
        
        ax.plot(positions, actual_data, label=actual_label, 
               linewidth=2, alpha=0.8, color=COLORS['blue'])
        
        # Apply pi threshold: set reconstruction to zero where pi > threshold
        reconstruction_plot = all_reconstructions_mu[i].copy()
        if all_pi is not None:
            reconstruction_plot[all_pi[i] > pi_threshold] = 0
        
        ax.plot(positions, reconstruction_plot, label='Predicted μ (Raw Counts)', 
               linewidth=2, alpha=0.8, color=COLORS['red'], linestyle='--')
        
        # Add uncertainty bands if variance available
        if all_variance is not None:
            std_dev = np.sqrt(all_variance[i])
            ax.fill_between(positions, 
                           all_reconstructions_mu[i] - std_dev,
                           all_reconstructions_mu[i] + std_dev,
                           alpha=0.2, color=COLORS['red'], label='μ ± σ (uncertainty)')
        
        if all_pi is not None:
            zero_pred_mask = all_pi[i] > 0.5
            if np.any(zero_pred_mask):
                ax.scatter(positions[zero_pred_mask], 
                          np.zeros(np.sum(zero_pred_mask)), marker='x', s=30, color=COLORS['orange'], 
                          label='Predicted Zero (π>0.5)', zorder=5)
        
        if all_raw_counts is not None:
            actual_zero_mask = all_raw_counts[i] == 0
            if np.any(actual_zero_mask):
                ax.scatter(positions[actual_zero_mask], 
                          all_raw_counts[i][actual_zero_mask],
                          marker='o', s=15, color=COLORS['green'], alpha=0.37,
                          label='Actual Zero (raw)', zorder=4)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Count Value [Raw Counts]')
        ax.set_title(f'{model_type}: Example {i+1} - Reconstruction with Uncertainty')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_example_reconstructions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
        
def metrics_summary(all_originals, all_reconstructions_mu, all_raw_counts, residuals, all_theta=None, all_variance=None, all_pi=None, model_type="ZINB", save_dir=None, prefix="", timestamp="", use_conv=False, metrics=None):    
    metrics_to_save = {
        'model_type': model_type,
        'timestamp': timestamp,
        'use_conv': bool(use_conv),
        'scaling_notes': {
            'all_originals': 'Normalized log counts (not used for main comparisons)',
            'all_reconstructions_mu': 'Predicted mean parameter mu in RAW COUNT SPACE',
            'all_raw_counts': 'Raw count data before normalization (PRIMARY comparison target)',
            'all_theta': 'Dispersion parameter (theta > 0, controls variance)',
            'all_pi': 'Zero-inflation probability (0 < pi < 1, represents P(zero))',
            'variance': 'Computed as mu + mu^2/theta (ZINB variance formula)'
        },
        'metrics': metrics,
        'summary_statistics': {
            'original': {
                'mean': float(np.mean(all_originals)),
                'std': float(np.std(all_originals)),
                'min': float(np.min(all_originals)),
                'max': float(np.max(all_originals)),
                'median': float(np.median(all_originals))
            },
            'predicted_mu': {
                'mean': float(np.mean(all_reconstructions_mu)),
                'std': float(np.std(all_reconstructions_mu)),
                'min': float(np.min(all_reconstructions_mu)),
                'max': float(np.max(all_reconstructions_mu)),
                'median': float(np.median(all_reconstructions_mu))
            },
            'residuals': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'median': float(np.median(residuals))
            }
        }
    }
    
    if all_theta is not None:
        metrics_to_save['zinb_parameters'] = {
            'theta': {
                'mean': float(np.mean(all_theta)),
                'std': float(np.std(all_theta)),
                'min': float(np.min(all_theta)),
                'max': float(np.max(all_theta)),
                'median': float(np.median(all_theta))
            }
        }
    
    if all_variance is not None:
        metrics_to_save['zinb_parameters'] = metrics_to_save.get('zinb_parameters', {})
        var_flat = all_variance.flatten()
        metrics_to_save['zinb_parameters']['variance'] = {
            'mean': float(np.mean(var_flat)),
            'std': float(np.std(var_flat)),
            'min': float(np.min(var_flat)),
            'max': float(np.max(var_flat)),
            'median': float(np.median(var_flat))
        }
        # Variance-to-mean ratio
        mu_flat = all_reconstructions_mu.flatten()
        vmr = np.where(mu_flat > 0, var_flat / mu_flat, 0)
        metrics_to_save['zinb_parameters']['variance_to_mean_ratio'] = {
            'mean': float(np.mean(vmr[mu_flat > 0])),
            'median': float(np.median(vmr[mu_flat > 0]))
        }
    
    if all_pi is not None:
        metrics_to_save['zinb_parameters'] = metrics_to_save.get('zinb_parameters', {})
        metrics_to_save['zinb_parameters']['pi'] = {
            'mean': float(np.mean(all_pi)),
            'std': float(np.std(all_pi)),
            'min': float(np.min(all_pi)),
            'max': float(np.max(all_pi)),
            'median': float(np.median(all_pi))
        }
    
    if all_pi is not None and all_raw_counts is not None:
        actual_zeros = (all_raw_counts.flatten() == 0)
        actual_nonzeros = ~actual_zeros
        pi_flat = all_pi.flatten()
        
        metrics_to_save['zero_inflation_analysis'] = {
            'percent_actual_zeros': float(np.mean(actual_zeros) * 100),
            'mean_pi_when_zero': float(np.mean(pi_flat[actual_zeros])) if np.any(actual_zeros) else 0,
            'mean_pi_when_nonzero': float(np.mean(pi_flat[actual_nonzeros])) if np.any(actual_nonzeros) else 0,
            'zero_prediction_accuracy': float(np.mean((pi_flat > 0.5) == actual_zeros))
        }
    
    metrics_file = os.path.join(save_dir, f'{prefix}_test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"ZINB metrics saved to {metrics_file}")

    print(f"ZINB-specific plots saved to {save_dir}/")


def plot_zinb_test_results(all_originals, all_reconstructions_mu, 
                           all_theta=None, all_pi=None, all_raw_counts=None,
                           all_masks=None, denoise_percent=0.0,
                           model_type='ZINBAE', save_dir=None, 
                           n_examples=10, metrics=None, use_conv=False, name="", subdir="testing", pi_threshold=0.5):
    """
    Create comprehensive visualizations specifically for ZINB models (ZINBAE/ZINBVAE).
    
    Inputs from training.py test() function:
    - all_originals: normalized log counts
    - all_reconstructions: mu parameter from ZINB model
    - all_theta, all_pi, all_raw_counts: from model output
    - model_type: 'ZINBAE' or 'ZINBVAE'
    - metrics: dict with 'zinb_nll', 'mae', 'r2', and optionally 'recon_loss', 'kl_loss'
    
    Outputs include:
    - Parameter distributions (zinb_parameter_distributions.png)
    - Prediction quality plots (prediction_quality.png)
    - Zero-inflation analysis (zero_inflation_analysis.png)
    - Example reconstructions (example_reconstructions.png)
    - Parameter heatmaps (parameter_heatmaps.png)
    - Metrics summary (metrics_summary.png)
    - Comprehensive JSON metrics (test_metrics.json)
    
    Parameters:
    -----------
    all_originals : np.ndarray
        Original normalized log counts (shape: [n_samples, seq_length])
    all_reconstructions_mu : np.ndarray
        Reconstructed mean parameters μ (shape: [n_samples, seq_length])
    all_theta : np.ndarray or None
        Dispersion parameters θ (shape: [n_samples, seq_length]). Default=None
    all_pi : np.ndarray or None
        Zero-inflation probabilities π (shape: [n_samples, seq_length]). Default=None
    all_raw_counts : np.ndarray or None
        Original raw count data before normalization (shape: [n_samples, seq_length]). Default=None
    model_type : str
        Type of model ('ZINBAE' or 'ZINBVAE'). Default='ZINBAE'
    save_dir : str or None
        Directory to save plots. If None, uses default 'AE/results/extra_results/testing'
    n_examples : int
        Number of example reconstructions to plot. Default=5
    metrics : dict
        Dictionary of metrics to save. Default=None
    use_conv : bool
        Whether Conv1D was used. Default=False
    name : str
        Name prefix for saved files. Default=""
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = generate_prefix(model_type, timestamp, use_conv, name)
    save_dir = prepare_output_dirs(save_dir, subdir=subdir, name=name)
    
    # Use RAW COUNTS for comparison
    if all_raw_counts is not None:
        print("Using raw counts for evaluation metrics.")
        actual_counts_flat = all_raw_counts.flatten()
        comparison_label = 'Raw Counts'
        abs_err = np.abs(actual_counts_flat - all_reconstructions_mu.flatten())
        mae = abs_err.mean()
        mae_std = abs_err.std(ddof=1)  # variability of absolute errors
        # mean and SD of pi for actual zeros vs non-zeros
        zero_mask = actual_counts_flat == 0
        non_zero_mask = ~zero_mask
        all_pi_flat = all_pi.flatten()
        mean_pi_zeros = all_pi_flat[zero_mask].mean() if np.any(zero_mask) else 0
        mean_pi_nonzeros = all_pi_flat[non_zero_mask].mean() if np.any(non_zero_mask) else 0
        std_pi_zeros = all_pi_flat[zero_mask].std(ddof=1) if np.any(zero_mask) else 0
        std_pi_nonzeros = all_pi_flat[non_zero_mask].std(ddof=1) if np.any(non_zero_mask) else 0
        # Compute the mean and standard deviation of the mu predictions for actual zeros vs non-zeros
        mean_mu_zeros = all_reconstructions_mu.flatten()[zero_mask].mean() if np.any(zero_mask) else 0
        mean_mu_nonzeros = all_reconstructions_mu.flatten()[non_zero_mask].mean() if np.any(non_zero_mask) else 0
        std_mu_zeros = all_reconstructions_mu.flatten()[zero_mask].std(ddof=1) if np.any(zero_mask) else 0
        std_mu_nonzeros = all_reconstructions_mu.flatten()[non_zero_mask].std(ddof=1) if np.any(non_zero_mask) else 0
        # Compute the mean and standard deviation of theta, no distinction between zeros and non-zeros since theta is a dispersion parameter that applies to all positions
        mean_theta = all_theta.flatten().mean() if all_theta is not None else 0
        std_theta = all_theta.flatten().std(ddof=1) if all_theta is not None else 0

        mean_theta_zeros = mean_theta_nonzeros = std_theta_zeros = std_theta_nonzeros = None
        r2 = r2_score(actual_counts_flat, all_reconstructions_mu.flatten())
    else:
        # Fallback to normalized if raw counts not available
        print("!!! Raw counts not provided; using normalized counts for evaluation metrics. !!!")
        actual_counts_flat = all_originals.flatten()
        comparison_label = 'Normalized Log Counts'
        abs_err = np.abs(actual_counts_flat - all_reconstructions_mu.flatten())
        mae = abs_err.mean()
        mae_std = abs_err.std(ddof=1)  # variability of absolute errors
        r2 = r2_score(actual_counts_flat, all_reconstructions_mu.flatten())
    
    # Save metrics to JSON for later analysis
    print(f"MAE: {mae:.4f}, SD MEAN: {mae_std:.4f} \n  R²: {r2:.4f}," )
    print(f"Mean π (zeros): {mean_pi_zeros:.4f}, Mean π (non-zeros): {mean_pi_nonzeros:.4f},  \n SD π (zeros): {std_pi_zeros:.4f}, SD π (non-zeros): {std_pi_nonzeros:.4f} \n")
    print(f"Mean μ (zeros): {mean_mu_zeros:.4f}, Mean μ (non-zeros): {mean_mu_nonzeros:.4f},  \n SD μ (zeros): {std_mu_zeros:.4f}, SD μ (non-zeros): {std_mu_nonzeros:.4f} \n")
    print(f"Theta mean: {mean_theta:.4f}, Theta SD: {std_theta:.4f}")
    # metrics_to_save = {
    #     'mae': mae,
    #     'mae_std': mae_std,
    #     'r2': r2,
    #     'mean_pi_zeros': mean_pi_zeros,
    #     'std_pi_zeros': std_pi_zeros,
    #     'mean_pi_nonzeros': mean_pi_nonzeros,
    #     'std_pi_nonzeros': std_pi_nonzeros, 
    # }
    # if metrics is not None:
    #     metrics_to_save.update(metrics)
    # metrics_file = os.path.join(save_dir, f'{prefix}_evaluation_metrics.json')
    # with open(metrics_file, 'w') as f:
    #     json.dump(metrics_to_save, f, indent=4)
    # print(f"Evaluation metrics saved to {metrics_file}")
        
    
    # Compute variance from ZINB parameters: variance = μ + μ²/θ
    if all_theta is not None:
        all_variance = all_reconstructions_mu + (all_reconstructions_mu ** 2) / all_theta
    else:
        all_variance = None

    residuals = actual_counts_flat - all_reconstructions_mu.flatten()
    

    # 1. ZINB Parameter Distributions (θ, π, μ, variance)
    # plot_parameter_distributions(all_reconstructions_mu, all_theta, all_pi, all_variance, 
    #                              model_type=model_type, save_dir=save_dir, prefix=prefix)
    # 2. Actual vs Predicted with Density Plot (RAW COUNTS)
    density_plots(actual_counts_flat, all_reconstructions_mu, residuals, comparison_label, model_type, save_dir, prefix, r2, mae, all_pi)
    # # 3. Zero-Inflation Analysis (if π available)
    # zero_inflation_analysis(all_reconstructions_mu, all_pi, all_raw_counts , 
    #                         model_type, save_dir, prefix)
    # # 4. Zero Imputation Analysis (always for ZINB models)
    # zero_imputation_analysis(all_reconstructions_mu, all_pi, all_raw_counts,
    #                         model_type, save_dir, prefix)
    # # 5. Masked Values Analysis (only when denoise_percent > 0)
    # if denoise_percent > 0 and all_masks is not None:
    masked_values_analysis(all_reconstructions_mu, all_pi, all_raw_counts, all_masks, all_theta,
                              model_type, save_dir, prefix, pi_threshold)
    # # 6. Example Reconstructions with ZINB Parameters and Uncertainty
    # reconstructions(all_originals, all_reconstructions_mu, all_variance, all_pi, all_raw_counts, 
    #                 n_examples=n_examples, model_type=model_type, save_dir=save_dir, prefix=prefix, pi_threshold=pi_threshold)
    # 7. Metrics Summary
    if metrics is not None:
        metrics_summary(all_originals, all_reconstructions_mu, all_raw_counts, residuals, 
                all_theta, all_variance, all_pi, model_type=model_type, 
                save_dir=save_dir, prefix=prefix, timestamp=timestamp, 
                use_conv=use_conv, metrics=metrics)