import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ZINB_MLE.estimate_ZINB import estimate_zinb
from ZINB_MLE.EM import em_zinb_step
import matplotlib.pyplot as plt


def generate_zinb_sample(n, mu, theta, pi):
    """
    Generate a sample from a Zero-Inflated Negative Binomial distribution.
    
    Parameters:
    -----------
    n : int
        Sample size
    mu : float
        Mean parameter of the NB distribution
    theta : float
        Dispersion parameter of the NB distribution
    pi : float
        Zero-inflation probability (0 <= pi < 1)
    
    Returns:
    --------
    numpy.ndarray : Array of ZINB samples
    """
    # Generate zero-inflation indicators
    zero_inflation = np.random.binomial(1, pi, size=n)
    
    # Generate NB samples
    # NB parameterization: p = theta / (theta + mu)
    p = theta / (theta + mu)
    nb_samples = np.random.negative_binomial(theta, p, size=n)
    
    # Combine: if zero_inflation[i] == 1, use 0, otherwise use nb_samples[i]
    zinb_samples = np.where(zero_inflation == 1, 0, nb_samples)
    
    return zinb_samples


def estimate_with_fixed_theta(data, true_theta, max_iter=100, tol=1e-6, eps=1e-10):
    """
    Estimate pi and mu using EM algorithm with fixed (true) theta.
    
    Parameters:
    -----------
    data : array-like
        Observed count data
    true_theta : float
        The true theta value to use (fixed)
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    eps : float
        Small value for numerical stability
    
    Returns:
    --------
    dict : Dictionary with 'pi', 'mu', 'theta', 'iterations', 'converged'
    """
    data = np.asarray(data, dtype=np.float64)
    
    # Initialize pi and mu
    pi = np.clip(np.mean(data == 0), eps, 1 - eps)
    ybar = np.mean(data)
    mu = np.clip(ybar / (1 - pi), eps, None)
    
    # Run EM with fixed theta
    for iteration in range(max_iter):
        pi_old = pi
        mu_old = mu
        
        # EM step
        em_result = em_zinb_step(data, pi, mu, true_theta, eps=eps)
        pi = em_result['pi']
        mu = em_result['mu']
        
        # Check convergence
        pi_change = abs(pi - pi_old)
        mu_change = abs(mu - mu_old) / (mu_old + eps)
        
        if pi_change < tol and mu_change < tol:
            return {
                'pi': pi,
                'mu': mu,
                'theta': true_theta,
                'iterations': iteration + 1,
                'converged': True
            }
    
    # Did not converge
    return {
        'pi': pi,
        'mu': mu,
        'theta': true_theta,
        'iterations': max_iter,
        'converged': False
    }


def run_estimation_experiment(
    n_replicates=10,
    sample_size=1000,
    pi_values=None,
    mu_values=None,
    theta_values=None
):
    """
    Run estimation experiment:
    1. Generate n_replicates datasets for each (pi, mu, theta) combination
    2. Estimate parameters using both methods
    3. Calculate relative errors
    4. Return aggregated results
    
    Parameters:
    -----------
    n_replicates : int
        Number of replicate datasets per parameter combination
    sample_size : int
        Number of samples per dataset
    pi_values : list
        List of pi values to test
    mu_values : list
        List of mu values to test
    theta_values : list
        List of theta values to test
    
    Returns:
    --------
    pd.DataFrame : Results with columns for true values, estimated values, and errors
    """
    # Default parameter values
    if pi_values is None:
        pi_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    if mu_values is None:
        mu_values = [1, 2, 3, 4, 5, 10, 20, 50]
    if theta_values is None:
        theta_values = [0.01, 0.1, 1, 5, 10, 50, 75, 100, 200]
    
    all_results = []
    
    total_configs = len(pi_values) * len(mu_values) * len(theta_values)
    config_count = 0
    
    print(f"Running experiment with {total_configs} parameter combinations, "
          f"{n_replicates} replicates each...")
    print(f"Total datasets to process: {total_configs * n_replicates}\n")
    
    for true_pi in pi_values:
        for true_mu in mu_values:
            for true_theta in theta_values:
                config_count += 1
                
                if config_count % 10 == 0 or config_count == 1:
                    print(f"Processing configuration {config_count}/{total_configs}: "
                          f"pi={true_pi:.2f}, mu={true_mu}, theta={true_theta}")
                
                for replicate in range(n_replicates):
                    # Generate ZINB data
                    data = generate_zinb_sample(sample_size, true_mu, true_theta, true_pi)
                    
                    # Method 1: Estimate all parameters (pi, mu, theta)
                    try:
                        result_full = estimate_zinb(
                            data, 
                            max_iter=100, 
                            tol=1e-6,
                            theta_min=0.01,
                            theta_max=1000,
                            n_theta_grid=500
                        )
                        est_pi_full = result_full['pi']
                        est_mu_full = result_full['mu']
                        est_theta_full = result_full['theta']
                        converged_full = result_full['converged']
                    except Exception as e:
                        print(f"  Warning: Full estimation failed for replicate {replicate}: {e}")
                        est_pi_full = np.nan
                        est_mu_full = np.nan
                        est_theta_full = np.nan
                        converged_full = False
                    
                    # Method 2: Estimate pi, mu with true theta
                    try:
                        result_fixed = estimate_with_fixed_theta(
                            data, 
                            true_theta,
                            max_iter=100,
                            tol=1e-6
                        )
                        est_pi_fixed = result_fixed['pi']
                        est_mu_fixed = result_fixed['mu']
                        converged_fixed = result_fixed['converged']
                    except Exception as e:
                        print(f"  Warning: Fixed-theta estimation failed for replicate {replicate}: {e}")
                        est_pi_fixed = np.nan
                        est_mu_fixed = np.nan
                        converged_fixed = False
                    
                    # Calculate relative errors
                    rel_error_pi_full = (est_pi_full - true_pi) / true_pi if true_pi > 0 else np.nan
                    rel_error_mu_full = (est_mu_full - true_mu) / true_mu if true_mu > 0 else np.nan
                    rel_error_theta_full = (est_theta_full - true_theta) / true_theta if true_theta > 0 else np.nan
                    
                    rel_error_pi_fixed = (est_pi_fixed - true_pi) / true_pi if true_pi > 0 else np.nan
                    rel_error_mu_fixed = (est_mu_fixed - true_mu) / true_mu if true_mu > 0 else np.nan
                    
                    # Store results
                    all_results.append({
                        'true_pi': true_pi,
                        'true_mu': true_mu,
                        'true_theta': true_theta,
                        'replicate': replicate,
                        'sample_size': sample_size,
                        # Full estimation
                        'est_pi_full': est_pi_full,
                        'est_mu_full': est_mu_full,
                        'est_theta_full': est_theta_full,
                        'converged_full': converged_full,
                        'rel_error_pi_full': rel_error_pi_full,
                        'rel_error_mu_full': rel_error_mu_full,
                        'rel_error_theta_full': rel_error_theta_full,
                        # Fixed theta estimation
                        'est_pi_fixed': est_pi_fixed,
                        'est_mu_fixed': est_mu_fixed,
                        'converged_fixed': converged_fixed,
                        'rel_error_pi_fixed': rel_error_pi_fixed,
                        'rel_error_mu_fixed': rel_error_mu_fixed,
                    })
    
    print(f"\nExperiment complete! Processed {len(all_results)} datasets.")
    return pd.DataFrame(all_results)


def plot_relative_errors(results_df, output_dir='results/ZINB_estimation_errors'):
    """
    Create plots showing relative error vs theta for both estimation methods.
    Creates two separate figures: one for mu, one for pi.
    Each figure has two subplots: theta estimated vs theta provided.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_estimation_experiment
    output_dir : str
        Directory to save plots (default: 'results/ZINB_estimation_errors')
    """
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Take absolute value of relative errors
    results_df = results_df.copy()
    results_df['abs_rel_error_pi_full'] = results_df['rel_error_pi_full'].abs()
    results_df['abs_rel_error_mu_full'] = results_df['rel_error_mu_full'].abs()
    results_df['abs_rel_error_pi_fixed'] = results_df['rel_error_pi_fixed'].abs()
    results_df['abs_rel_error_mu_fixed'] = results_df['rel_error_mu_fixed'].abs()
    
    # Aggregate results by true theta: compute mean and std of absolute relative errors
    # For full estimation (theta estimated)
    agg_full = results_df.groupby('true_theta').agg({
        'abs_rel_error_pi_full': ['mean', 'std'],
        'abs_rel_error_mu_full': ['mean', 'std']
    }).reset_index()
    agg_full.columns = ['true_theta', 'pi_mean', 'pi_std', 'mu_mean', 'mu_std']
    
    # For fixed theta estimation
    agg_fixed = results_df.groupby('true_theta').agg({
        'abs_rel_error_pi_fixed': ['mean', 'std'],
        'abs_rel_error_mu_fixed': ['mean', 'std']
    }).reset_index()
    agg_fixed.columns = ['true_theta', 'pi_mean', 'pi_std', 'mu_mean', 'mu_std']
    
    # Calculate y-axis limits for mu (including error bars)
    mu_min_full = max(0, (agg_full['mu_mean'] - agg_full['mu_std']).min())
    mu_max_full = (agg_full['mu_mean'] + agg_full['mu_std']).max()
    mu_min_fixed = max(0, (agg_fixed['mu_mean'] - agg_fixed['mu_std']).min())
    mu_max_fixed = (agg_fixed['mu_mean'] + agg_fixed['mu_std']).max()
    mu_ylim = [min(mu_min_full, mu_min_fixed), max(mu_max_full, mu_max_fixed)]
    # Add 5% padding at top, keep bottom at computed min (close to 0)
    mu_range = mu_ylim[1] - mu_ylim[0]
    mu_ylim = [mu_ylim[0] - 0.02 * mu_range, mu_ylim[1] + 0.05 * mu_range]
    
    # Calculate y-axis limits for pi (including error bars)
    pi_min_full = max(0, (agg_full['pi_mean'] - agg_full['pi_std']).min())
    pi_max_full = (agg_full['pi_mean'] + agg_full['pi_std']).max()
    pi_min_fixed = max(0, (agg_fixed['pi_mean'] - agg_fixed['pi_std']).min())
    pi_max_fixed = (agg_fixed['pi_mean'] + agg_fixed['pi_std']).max()
    pi_ylim = [min(pi_min_full, pi_min_fixed), max(pi_max_full, pi_max_fixed)]
    # Add 5% padding at top, keep bottom at computed min (close to 0)
    pi_range = pi_ylim[1] - pi_ylim[0]
    pi_ylim = [pi_ylim[0] - 0.02 * pi_range, pi_ylim[1] + 0.05 * pi_range]
    
    # ---------- FIGURE 1: MU ABSOLUTE RELATIVE ERROR ----------
    fig_mu, axes_mu = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Mu with theta estimated
    ax = axes_mu[0]
    ax.errorbar(agg_full['true_theta'], agg_full['mu_mean'], 
                yerr=agg_full['mu_std'], label='μ relative error', 
                marker='s', capsize=5, alpha=0.7, color='black')
    ax.set_xscale('log')
    ax.set_xlabel('True θ (log scale)', fontsize=12)
    ax.set_ylabel('Absolute Relative Error of μ', fontsize=12)
    ax.set_title('θ Estimated', fontsize=14, fontweight='bold')
    ax.set_ylim(mu_ylim)
    ax.grid(True, alpha=0.3, which='both')
    
    # Subplot 2: Mu with theta provided
    ax = axes_mu[1]
    ax.errorbar(agg_fixed['true_theta'], agg_fixed['mu_mean'], 
                yerr=agg_fixed['mu_std'], label='μ relative error', 
                marker='s', capsize=5, alpha=0.7, color='black')
    ax.set_xscale('log')
    ax.set_xlabel('True θ (log scale)', fontsize=12)
    ax.set_ylabel('Absolute Relative Error of μ', fontsize=12)
    ax.set_title('True θ Provided', fontsize=14, fontweight='bold')
    ax.set_ylim(mu_ylim)
    ax.grid(True, alpha=0.3, which='both')
    
    fig_mu.suptitle('Absolute Relative Error of μ vs θ', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save mu figure
    plot_path_mu = os.path.join(full_output_dir, 'absolute_relative_error_mu_vs_theta.png')
    fig_mu.savefig(plot_path_mu, dpi=300, bbox_inches='tight')
    print(f"Mu plot saved to: {plot_path_mu}")
    
    # ---------- FIGURE 2: PI ABSOLUTE RELATIVE ERROR ----------
    fig_pi, axes_pi = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Pi with theta estimated
    ax = axes_pi[0]
    ax.errorbar(agg_full['true_theta'], agg_full['pi_mean'], 
                yerr=agg_full['pi_std'], label='π relative error', 
                marker='o', capsize=5, alpha=0.7, color='black')
    ax.set_xscale('log')
    ax.set_xlabel('True θ (log scale)', fontsize=12)
    ax.set_ylabel('Absolute Relative Error of π', fontsize=12)
    ax.set_title('θ Estimated', fontsize=14, fontweight='bold')
    ax.set_ylim(pi_ylim)
    ax.grid(True, alpha=0.3, which='both')
    
    # Subplot 2: Pi with theta provided
    ax = axes_pi[1]
    ax.errorbar(agg_fixed['true_theta'], agg_fixed['pi_mean'], 
                yerr=agg_fixed['pi_std'], label='π relative error', 
                marker='o', capsize=5, alpha=0.7, color='black')
    ax.set_xscale('log')
    ax.set_xlabel('True θ (log scale)', fontsize=12)
    ax.set_ylabel('Absolute Relative Error of π', fontsize=12)
    ax.set_title('True θ Provided', fontsize=14, fontweight='bold')
    ax.set_ylim(pi_ylim)
    ax.grid(True, alpha=0.3, which='both')
    
    fig_pi.suptitle('Absolute Relative Error of π vs θ', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save pi figure
    plot_path_pi = os.path.join(full_output_dir, 'absolute_relative_error_pi_vs_theta.png')
    fig_pi.savefig(plot_path_pi, dpi=300, bbox_inches='tight')
    print(f"Pi plot saved to: {plot_path_pi}")
    
    plt.show()
    
    return agg_full, agg_fixed


def generate_zinb_datasets(sample_size=1000, output_dir='ZINB'):
    """
    Generate ZINB datasets with various parameter combinations and save to files.
    
    Parameters:
    -----------
    sample_size : int
        Number of samples per dataset
    output_dir : str
        Directory to save the generated datasets (relative to script location)
    """
    # Define parameter ranges
    pi_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    mu_values = [1, 2, 3, 4, 5, 10, 20, 50]
    theta_values = [0.01, 0.1, 1, 5, 10, 50, 75, 100, 200]
    
    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Create Data subdirectory
    data_dir = os.path.join(full_output_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Store all parameter combinations and their results
    all_results = []
    
    # Generate datasets for each parameter combination
    for pi in pi_values:
        for mu in mu_values:
            for theta in theta_values:
                # Generate ZINB data
                data = generate_zinb_sample(sample_size, mu, theta, pi)
                
                # Create filename
                filename = f"zinb_pi{pi:.1f}_mu{mu}_theta{theta}.csv"
                # Save to Data subdirectory
                filepath = os.path.join(data_dir, filename)
                
                # Save data to CSV
                df = pd.DataFrame({'count': data})
                df.to_csv(filepath, index=False)
                
                
                all_results.append({
                    'filename': filename,
                    'pi': pi,
                    'mu': mu,
                    'theta': theta,
                    'sample_size': sample_size,
                })
                
                print(f"Generated: {filename}")
    
    # Save parameter summary
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(full_output_dir, 'dataset_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Total datasets generated: {len(all_results)}")


if __name__ == "__main__":
    # Option 1: Run the estimation experiment (default)
    # This generates 10 replicates for each parameter combination,
    # estimates parameters using both methods, and creates plots
    
    print("=" * 80)
    print("ZINB Parameter Estimation Experiment")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Generate 10 replicates for each (pi, mu, theta) combination")
    print("  2. Estimate parameters using two methods:")
    print("     - Method 1: Estimate all three parameters (pi, mu, theta)")
    print("     - Method 2: Estimate pi and mu with true theta provided")
    print("  3. Calculate relative errors")
    print("  4. Create plots showing mean ± std of relative errors vs theta")
    print("\n" + "=" * 80 + "\n")
    
    # Run experiment with default parameters
    # You can customize by passing different parameter lists
    results = run_estimation_experiment(
        n_replicates=10,
        sample_size=1000,
        # Uncomment and modify to use custom parameter ranges:
        pi_values=[0.3, 0.5, 0.7, 0.9],
        mu_values=[1, 2, 5, 10],
        theta_values=[0.1, 0.5, 1, 5, 10, 50, 100]
    )
    
    # Save results to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results', 'ZINB_estimation_errors')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'estimation_results.csv')
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    print("\nGenerating plots...")
    agg_full, agg_fixed = plot_relative_errors(results)
    
    # Save aggregated results
    agg_full_path = os.path.join(results_dir, 'aggregated_full_estimation.csv')
    agg_fixed_path = os.path.join(results_dir, 'aggregated_fixed_theta.csv')
    agg_full.to_csv(agg_full_path, index=False)
    agg_fixed.to_csv(agg_fixed_path, index=False)
    print(f"Aggregated results (full estimation) saved to: {agg_full_path}")
    print(f"Aggregated results (fixed theta) saved to: {agg_fixed_path}")
    
    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)
    
    # Option 2: Uncomment to use the old function that saves datasets to files
    # generate_zinb_datasets(sample_size=1000)

