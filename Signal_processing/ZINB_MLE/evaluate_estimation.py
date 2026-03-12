import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root)

from ZINB_MLE.estimate_ZINB import estimate_zinb
from Utils.colors import COLORBLIND_COLORS


def evaluate_zinb_estimation(data_dir='ZINB', output_file='estimation_results_grid.csv'):
    """
    Load all ZINB datasets, estimate parameters, and save results.
    
    Parameters:
    -----------
    data_dir : str
        Directory name within sample_data folder (default: 'ZINB')
    output_file : str
        Output CSV file name for results
    """
    # Construct paths - go from ZINB_MLE up to Signal_processing, then to sample_data/ZINB
    # script_dir is Signal_processing/ZINB_MLE
    # parent_dir is Signal_processing
    full_data_dir = os.path.join(parent_dir, 'sample_data', data_dir)
    summary_path = os.path.join(full_data_dir, 'dataset_summary.csv')
    output_path = os.path.join(full_data_dir, output_file)
    
    # Load the summary file to get the order
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}")
        return
    
    summary_df = pd.read_csv(summary_path)
    print(f"Loaded summary with {len(summary_df)} datasets")
    
    # Store results
    results = []
    
    # Process each dataset in the same order as summary
    for idx, row in summary_df.iterrows():
        filename = row['filename']
        true_pi = row['pi']
        true_mu = row['mu']
        true_theta = row['theta']
        
        # Load the data from Data subfolder
        filepath = os.path.join(full_data_dir, 'Data', filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filename}")
            continue
        
        data_df = pd.read_csv(filepath)
        data = data_df['count'].values
        
        print(f"Processing {idx+1}/{len(summary_df)}: {filename}")
        
        # Estimate ZINB parameters
        try:
            estimation_result = estimate_zinb(data, max_iter=100, tol=1e-6)
            
            # Store results
            results.append({
                'filename': filename,
                'true_pi': true_pi,
                'true_mu': true_mu,
                'true_theta': true_theta,
                'estimated_pi': estimation_result['pi'],
                'estimated_mu': estimation_result['mu'],
                'estimated_theta': estimation_result['theta'],
                'iterations': estimation_result['iterations'],
                'converged': estimation_result['converged'],
                'log_likelihood': estimation_result['log_likelihood'],
                'pi_error': abs(estimation_result['pi'] - true_pi),
                'mu_error': abs(estimation_result['mu'] - true_mu),
                'theta_error': abs(estimation_result['theta'] - true_theta),
                'pi_relative_error': abs(estimation_result['pi'] - true_pi) / true_pi if true_pi > 0 else np.nan,
                'mu_relative_error': abs(estimation_result['mu'] - true_mu) / true_mu if true_mu > 0 else np.nan,
                'theta_relative_error': abs(estimation_result['theta'] - true_theta) / true_theta if true_theta > 0 else np.nan
            })
            
            print(f"  True: π={true_pi:.3f}, μ={true_mu:.1f}, θ={true_theta:.1f}")
            print(f"  Est:  π={estimation_result['pi']:.3f}, μ={estimation_result['mu']:.1f}, θ={estimation_result['theta']:.1f}")
            print(f"  Converged: {estimation_result['converged']}, Iterations: {estimation_result['iterations']}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            results.append({
                'filename': filename,
                'true_pi': true_pi,
                'true_mu': true_mu,
                'true_theta': true_theta,
                'estimated_pi': np.nan,
                'estimated_mu': np.nan,
                'estimated_theta': np.nan,
                'iterations': np.nan,
                'converged': False,
                'log_likelihood': np.nan,
                'pi_error': np.nan,
                'mu_error': np.nan,
                'theta_error': np.nan,
                'pi_relative_error': np.nan,
                'mu_relative_error': np.nan,
                'theta_relative_error': np.nan
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    print("\n=== ESTIMATION SUMMARY ===")
    print(f"Total datasets: {len(results_df)}")
    print(f"Converged: {results_df['converged'].sum()}")
    print(f"Mean iterations: {results_df['iterations'].mean():.1f}")
    print("\nMean Absolute Errors:")
    print(f"  π: {results_df['pi_error'].mean():.4f}")
    print(f"  μ: {results_df['mu_error'].mean():.4f}")
    print(f"  θ: {results_df['theta_error'].mean():.4f}")
    print("\nMean Relative Errors:")
    print(f"  π: {results_df['pi_relative_error'].mean():.4f}")
    print(f"  μ: {results_df['mu_relative_error'].mean():.4f}")
    print(f"  θ: {results_df['theta_relative_error'].mean():.4f}")
    print("\nMedian Absolute Errors:")
    print(f"  π: {results_df['pi_error'].median():.4f}")
    print(f"  μ: {results_df['mu_error'].median():.4f}")
    print(f"  θ: {results_df['theta_error'].median():.4f}")
    print("\nMedian Relative Errors:")
    print(f"  π: {results_df['pi_relative_error'].median():.4f}")
    print(f"  μ: {results_df['mu_relative_error'].median():.4f}")
    print(f"  θ: {results_df['theta_relative_error'].median():.4f}")
    
    # Generate plots
    plot_results(results_df, full_data_dir + "/plots")
    
    return results_df


def plot_results(results_df, output_dir):
    """
    Plot estimation results and save figures.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with estimation results
    output_dir : str
        Directory to save plots
    """
    df = results_df.copy()

    # combined metric: worst relative error across parameters
    df["combined_rel_error"] = df[["pi_relative_error","mu_relative_error","theta_relative_error"]].max(axis=1)
    df.sort_values("combined_rel_error", ascending=False).head(10)[["true_pi","true_mu","true_theta","estimated_pi","estimated_mu","estimated_theta","combined_rel_error","converged"]]

    # ---------------- Figure 1: overall fit quality ----------------
    # Calculate grid layout for 3 plots (2x2 grid)
    n_params = 3
    ncols1 = int(np.ceil(np.sqrt(n_params)))
    nrows1 = int(np.ceil(n_params / ncols1))
    
    fig1, axes1 = plt.subplots(nrows1, ncols1, figsize=(4.8*ncols1, 4.8*nrows1), constrained_layout=True)
    axes1 = axes1.flatten()

    # π scatter
    axes1[0].scatter(df["true_pi"], df["estimated_pi"], color=COLORBLIND_COLORS['blue'], 
                marker="o", alpha=0.6, s=30)
    mn, mx = df["true_pi"].min(), df["true_pi"].max()
    axes1[0].plot([mn, mx], [mn, mx], color=COLORBLIND_COLORS['black'], 
             linestyle='--', linewidth=1.5, label='Perfect fit')
    axes1[0].set_xlabel("True π")
    axes1[0].set_ylabel("Estimated π")
    axes1[0].set_title("π: True vs Estimated")
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)

    # μ scatter (log-log)
    axes1[1].scatter(df["true_mu"], df["estimated_mu"], color=COLORBLIND_COLORS['green'], 
                marker="o", alpha=0.6, s=30)
    mn, mx = df["true_mu"].min(), df["true_mu"].max()
    axes1[1].plot([mn, mx], [mn, mx], color=COLORBLIND_COLORS['black'], 
             linestyle='--', linewidth=1.5, label='Perfect fit')
    axes1[1].set_xscale("log")
    axes1[1].set_yscale("log")
    axes1[1].set_xlabel("True μ")
    axes1[1].set_ylabel("Estimated μ")
    axes1[1].set_title("μ: True vs Estimated (log-log)")
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)

    # θ scatter (log-log)
    axes1[2].scatter(df["true_theta"], df["estimated_theta"], color=COLORBLIND_COLORS['orange'], 
                marker="o", alpha=0.6, s=30)
    mn, mx = df["true_theta"].min(), df["true_theta"].max()
    axes1[2].plot([mn, mx], [mn, mx], color=COLORBLIND_COLORS['black'], 
             linestyle='--', linewidth=1.5, label='Perfect fit')
    axes1[2].set_xscale("log")
    axes1[2].set_yscale("log")
    axes1[2].set_xlabel("True θ")
    axes1[2].set_ylabel("Estimated θ")
    axes1[2].set_title("θ: True vs Estimated (log-log)")
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.3)
    
    # Hide unused subplot
    if n_params < len(axes1):
        axes1[3].axis('off')
    fig1_path = os.path.join(output_dir, 'parameter_comparison_grid.png')
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {fig1_path}")
    plt.close(fig1)

    # ---------------- Figure 2: where the errors are ----------------
    mus = sorted(df["true_mu"].unique())
    pis = sorted(df["true_pi"].unique())
    thetas = sorted(df["true_theta"].unique())

    def matrix_for(mu_val, value_col):
        sub = df[df["true_mu"] == mu_val]
        piv = sub.pivot_table(index="true_pi", columns="true_theta", values=value_col, aggfunc="mean")
        return piv.reindex(index=pis, columns=thetas).values

    # Prepare all matrices first to compute global vmin/vmax
    mat_log_list = []
    for mu_val in mus:
        mat = matrix_for(mu_val, "combined_rel_error")
        mat_log = np.log10(mat + 1e-12)  # log scale to see large ranges
        mat_log_list.append(mat_log)
    
    # Compute global color scale - make symmetric around 0 for diverging colormap
    global_vmin = np.nanmin([np.nanmin(m) for m in mat_log_list])
    global_vmax = np.nanmax([np.nanmax(m) for m in mat_log_list])
    # Make symmetric around 0
    global_abs_max = max(abs(global_vmin), abs(global_vmax))
    global_vmin = -global_abs_max
    global_vmax = global_abs_max
    
    # Calculate grid layout for square-like arrangement
    n_plots = len(mus)
    ncols = int(np.ceil(np.sqrt(n_plots)))
    nrows = int(np.ceil(n_plots / ncols))
    
    fig2, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 4.2*nrows), constrained_layout=True)
    # Flatten axes array for easier iteration
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    last_im = None
    for idx, (ax, mu_val, mat_log) in enumerate(zip(axes[:n_plots], mus, mat_log_list)):
        # Use diverging colormap: blue (negative/underestimate) to red (positive/overestimate)
        last_im = ax.imshow(mat_log, aspect="auto", origin="lower", cmap='RdBu_r',
                           vmin=global_vmin, vmax=global_vmax)
        ax.set_title(f"μ = {mu_val}")
        ax.set_xticks(range(len(thetas)))
        ax.set_xticklabels(thetas, rotation=45)
        ax.set_yticks(range(len(pis)))
        ax.set_yticklabels(pis)
        ax.set_xlabel("True θ")
        ax.set_ylabel("True π")
    
    # Hide any unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    cbar = fig2.colorbar(last_im, ax=axes[:n_plots], shrink=0.85)
    cbar.set_label("log10(max relative error)")
    
    fig2_path = os.path.join(output_dir, 'error_heatmap_grid.png')
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {fig2_path}")
    plt.close(fig2)
    
    
    # ---------------- Figure 4: Convergence failures / blow-ups ----------------
    # Define colors for each pi value
    pi_colors = [COLORBLIND_COLORS['blue'], COLORBLIND_COLORS['orange'], 
                 COLORBLIND_COLORS['green'], COLORBLIND_COLORS['pink']]
    df["blowup"] = (~df["converged"]) | (df["estimated_theta"] > 1e6)
    
    # Calculate grid layout
    n_plots4 = len(mus)
    ncols4 = int(np.ceil(np.sqrt(n_plots4)))
    nrows4 = int(np.ceil(n_plots4 / ncols4))
    
    fig4, axes4 = plt.subplots(nrows4, ncols4, figsize=(4.2*ncols4, 3.8*nrows4), 
                               sharey=True, constrained_layout=True)
    if n_plots4 == 1:
        axes4 = [axes4]
    else:
        axes4 = axes4.flatten()
    
    for idx, (ax, mu) in enumerate(zip(axes4[:n_plots4], mus)):
        sub_mu = df[df["true_mu"] == mu]
        for idx, pi in enumerate(pis):
            sub = sub_mu[sub_mu["true_pi"] == pi].sort_values("true_theta")
            color = pi_colors[idx % len(pi_colors)]
            ax.plot(sub["true_theta"], sub["blowup"].astype(int), 
                   marker="o", color=color, label=f"π={pi}", 
                   linewidth=2, markersize=6, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["ok", "blow-up"])
        ax.set_title(f"μ = {mu}")
        ax.set_xlabel("true θ (log)")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots4, len(axes4)):
        axes4[idx].axis('off')
    
    # Add legend to last visible plot
    axes4[n_plots4-1].legend(fontsize=8, loc="best", framealpha=0.9)
    
    fig4_path = os.path.join(output_dir, 'convergence_failures_grid.png')
    fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {fig4_path}")
    plt.close(fig4)
    
    # ---------------- Figure 5: θ log-ratio error (interpretable bias) ----------------
    df["theta_log_ratio"] = np.log10((df["estimated_theta"] + 1e-12) / df["true_theta"])
    
    # Calculate grid layout
    n_plots5 = len(mus)
    ncols5 = int(np.ceil(np.sqrt(n_plots5)))
    nrows5 = int(np.ceil(n_plots5 / ncols5))
    
    fig5, axes5 = plt.subplots(nrows5, ncols5, figsize=(4.2*ncols5, 4.2*nrows5), 
                               sharey=True, constrained_layout=True)
    if n_plots5 == 1:
        axes5 = [axes5]
    else:
        axes5 = axes5.flatten()
    
    for idx, (ax, mu) in enumerate(zip(axes5[:n_plots5], mus)):
        sub_mu = df[df["true_mu"] == mu]
        for idx, pi in enumerate(pis):
            sub = sub_mu[sub_mu["true_pi"] == pi].sort_values("true_theta")
            color = pi_colors[idx % len(pi_colors)]
            ax.plot(sub["true_theta"], sub["theta_log_ratio"], 
                   marker="o", color=color, label=f"π={pi}", 
                   linewidth=2, markersize=6, alpha=0.8)
        ax.axhline(0, color=COLORBLIND_COLORS['black'], 
                   linestyle='--', linewidth=1.5, label='Perfect estimate')
        ax.set_xscale("log")
        ax.set_title(f"μ = {mu}")
        ax.set_xlabel("True θ (log)")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots5, len(axes5)):
        axes5[idx].axis('off')
    
    axes5[0].set_ylabel("log10(estimated θ / true θ)")
    axes5[n_plots5-1].legend(fontsize=8, loc="best", framealpha=0.9)
    
    fig5_path = os.path.join(output_dir, 'theta_log_ratio_grid.png')
    fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {fig5_path}")
    plt.close(fig5)
    
    # ---------------- Figure 6: Relative errors of mu and pi by theta ----------------
    # Group by true_theta and compute mean/median relative errors
    theta_groups = df.groupby('true_theta').agg({
        'pi_relative_error': ['mean', 'median', 'std'],
        'mu_relative_error': ['mean', 'median', 'std']
    }).reset_index()
    
    theta_values = theta_groups['true_theta'].values
    pi_rel_mean = theta_groups['pi_relative_error']['mean'].values
    pi_rel_median = theta_groups['pi_relative_error']['median'].values
    pi_rel_std = theta_groups['pi_relative_error']['std'].values
    mu_rel_mean = theta_groups['mu_relative_error']['mean'].values
    mu_rel_median = theta_groups['mu_relative_error']['median'].values
    mu_rel_std = theta_groups['mu_relative_error']['std'].values
    
    fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig6.suptitle("Relative Errors of π and μ vs True θ", fontsize=14, fontweight='bold')
    
    # Plot for pi
    ax1.plot(theta_values, pi_rel_mean, 'o-', color=COLORBLIND_COLORS['blue'], 
             linewidth=2, markersize=6, label='Mean')
    ax1.plot(theta_values, pi_rel_median, 's--', color=COLORBLIND_COLORS['orange'], 
             linewidth=2, markersize=6, label='Median')
    ax1.set_xscale('log')
    ax1.set_xlabel('True θ', fontsize=12)
    ax1.set_ylabel('Relative Error of π', fontsize=12)
    ax1.set_title('π Estimation Error vs θ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for mu
    ax2.plot(theta_values, mu_rel_mean, 'o-', color=COLORBLIND_COLORS['blue'], 
             linewidth=2, markersize=6, label='Mean')
    ax2.plot(theta_values, mu_rel_median, 's--', color=COLORBLIND_COLORS['orange'], 
             linewidth=2, markersize=6, label='Median')
    ax2.set_xscale('log')
    ax2.set_xlabel('True θ', fontsize=12)
    ax2.set_ylabel('Relative Error of μ', fontsize=12)
    ax2.set_title('μ Estimation Error vs θ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig6_path = os.path.join(output_dir, 'mu_pi_error_vs_theta_grid.png')
    fig6.savefig(fig6_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {fig6_path}")
    plt.close(fig6)


if __name__ == "__main__":
    evaluate_zinb_estimation()
