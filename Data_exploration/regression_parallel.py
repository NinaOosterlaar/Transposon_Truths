import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from math import exp
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.reader import read_csv_file_with_distances
from Utils.plot_config import setup_plot_style, COLORS, NUCLEOSOME_CENTROMERE
from multiprocessing import Pool, Process
import pickle
import gc  # For explicit memory cleanup

# Set up standardized plot style
setup_plot_style()


def ZINB_regression(Y, X, Z = None, offset = None, p = 2, exposure = None, method = 'lbfgs', regularized = None, alpha = 1.0):
    """
    Perform Zero-Inflated Negative Binomial regression.
    
    Parameters:
    Y : array-like
        Dependent variable (counts).
    X : array-like
        Independent variables for the count model.
    Z : array-like
        Independent variables for the zero-inflation model. If None, uses X.
    offset : array-like, optional
        Offset for the count model.
    p : int, optional
        Power parameter for the Negative Binomial distribution.
    exposure : array-like, optional
        Exposure variable for the count model.
    method : str, optional
        Optimization method for fitting the model.
    regularized : str, optional
        Regularization method if regularization is desired.
    alpha : float, optional
        Regularization strength.
        
    Returns:
    model : statsmodels object
        The ZINB model instance.
    result : statsmodels object
        Fitted ZINB regression model.
    """
    print(X.mean(), X.std(ddof=0))
    X = (X - X.mean()) / X.std(ddof=0)  # Standardize X)
    
    if Z is None:
        Z = X
    
    # Add constant to independent variables
    X = sm.add_constant(X)
    Z = sm.add_constant(Z)

    # Fit the ZINB model
    model = sm.ZeroInflatedNegativeBinomialP(endog=Y, exog=X, exog_infl=Z, offset=offset, p=p, exposure=exposure)
    if regularized is not None:
        # 4) pen_weight: 0 for inflate_const, 1 for inflate slopes,
        #                0 for count const, 1 for count slopes,
        #                0 for alpha (last param)
        k_infl = Z.shape[1]           # includes inflate_const
        k_exog = X.shape[1]           # includes const
        pen_weight = np.r_[
            [0.0],                    # inflate_const
            np.ones(k_infl - 1),      # inflate slopes
            [0.0],                    # count const
            np.ones(k_exog - 1),      # count slopes
            [0.0]                     # alpha (dispersion)
        ]

        result = model.fit_regularized(
            method="l1_cvxopt_cp",    # use CVXOPT backend if installed
            alpha=alpha,              # start small (e.g., 1e-4 or 1e-3)
            pen_weight=pen_weight,
            maxiter=5000,
            cnvrg_tol=1e-8,
            trim_mode="size",
            size_trim_tol=1e-6,
            refit=True,
            disp=False
        )
    else:
        result = model.fit(method=method, maxiter=5000, disp=1)

    return model, result

def _fit_worker(args):
    """Run one fit in a separate process and return pickled result."""
    # avoid BLAS oversubscription per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    dataset_name, Y, X = args
    # Call your existing function (unchanged)
    _, result = ZINB_regression(Y, X, Z=None, regularized='l1_cvxopt_cp', alpha=1e-4)
    return dataset_name, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def _write_result_immediately(dataset_name: str, result, output_dir: str) -> None:
    """
    Persist results for one dataset immediately to disk (pickle + text summary).
    Falls back to raw pickle if statsmodels save() is unavailable.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save full result
    pickle_path = os.path.join(output_dir, f"{dataset_name}_poly_result.pickle")
    try:
        result.save(pickle_path)
    except Exception:
        with open(pickle_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2) Save quick human-readable summary
    summary_path = os.path.join(output_dir, f"{dataset_name}_summary.txt")
    try:
        with open(summary_path, "w") as f:
            f.write(f"--- {dataset_name} ---\n")
            f.write("params:\n")
            try:
                f.write(result.params.to_string())
                f.write("\n")
            except Exception:
                f.write("<params unavailable>\n")

            # Fit metrics if available
            llf = getattr(result, 'llf', 'NA')
            aic = getattr(result, 'aic', 'NA')
            bic = getattr(result, 'bic', 'NA')
            f.write(f"loglike: {llf}, AIC: {aic}, BIC: {bic}\n")
    except Exception:
        # Best-effort; if summary writing fails, continue
        pass

def perform_regression_on_datasets(
    input_folder: str = "Data_exploration/results/distances_with_zeros",
    range=None,
    combine_all: bool = False,
    output_dir: str = "Data_exploration/results/regression/linear",
    write_immediately: bool = True,
):
    datasets = read_csv_file_with_distances(input_folder)
    if range is not None:
        datasets = set_centromere_range(datasets, range)
    results = {}

    # Ensure output directory exists when immediate writing is enabled
    if write_immediately:
        os.makedirs(output_dir, exist_ok=True)

    if combine_all:
        combined_df = pd.DataFrame()
        for dataset_name in datasets:
            combined_df = pd.concat([combined_df] + list(datasets[dataset_name].values()), ignore_index=True)
        datasets = {"combined": {"all_chromosomes": combined_df}}

    # build tasks
    tasks = []
    for dataset_name in datasets:
        print(f"\n--- Fitting ZINB for dataset: {dataset_name} ---")
        df_all = pd.concat(datasets[dataset_name].values(), ignore_index=True)
        print(f"Combined dataframe size: {df_all.shape}")

        df_all = df_all.dropna(subset=["Value", "Nucleosome_Distance", "Centromere_Distance"]).copy()
        df_all["Centromere_Distance"] = df_all["Centromere_Distance"] / 1000.0

        Y = df_all["Value"].astype(int)
        X = df_all[["Nucleosome_Distance", "Centromere_Distance"]]
        tasks.append((dataset_name, Y, X))

    # parallel fit across datasets
    if tasks:
        with Pool(processes=os.cpu_count()) as pool:
            for dataset_name, result_bytes in pool.imap_unordered(_fit_worker, tasks):
                result = pickle.loads(result_bytes)

                # Save immediately for crash resilience
                if write_immediately:
                    _write_result_immediately(dataset_name, result, output_dir)
                    print(f"✓ Saved result for {dataset_name}", flush=True)

                # Optionally keep in memory (set to None to minimize memory when writing immediately)
                results[dataset_name] = None if write_immediately else result

                # Free memory ASAP
                del result
                gc.collect()

    return results

def set_centromere_range(datasets, distance_from_centromere):
    """
    Filter datasets to only include data within a certain distance from the centromere.
    
    Parameters:
    datasets : dict
        { dataset_name: { chromosome_name: dataframe } }
    distance_from_centromere : int
        Distance from centromere in base pairs.
    """
    filtered_datasets = {}
    for dataset_name in datasets:
        filtered_datasets[dataset_name] = {}
        for chrom in datasets[dataset_name]:
            df = datasets[dataset_name][chrom]
            df_filtered = df[np.abs(df["Centromere_Distance"]) <= distance_from_centromere]
            filtered_datasets[dataset_name][chrom] = df_filtered
    return filtered_datasets


def _logistic(x: float) -> float:
    """Safely compute logistic(x) = 1 / (1 + exp(-x)) for possibly large |x|."""
    # prevent overflow in exp()
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def _natural_sort_key(text):
    """
    Generate a key for natural sorting that handles numbers correctly.
    E.g., FD7 comes before FD12 instead of FD12 before FD7.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


def plot_regression_results(folder, output_file, transform: bool = False, input_folder: str = "Data_exploration/results/distances_with_zeros") -> None:
    """
    Generate plots for regression results stored in the specified folder.

    Args:
        folder (str): Path to the folder containing regression result txt files
        output_file (str): Base path (".png" will be modified for each figure)
        transform (bool): If False (default), plot raw model coefficients.
                          If True, convert:
                            - inflate_const -> structural zero probability (pi)
                            - other inflate_* -> odds ratio (exp)
                            - count_const -> baseline mean (exp)
                            - other count_* -> fold-change (exp)
        input_folder (str): Path to the original data folder to compute standard deviations
    """

    # ---------- 1. Read all summaries ----------
    regression_results = {}
    for file_name in os.listdir(folder):
        if file_name.endswith("_summary.txt"):
            dataset_name = file_name.replace("_summary.txt", "")
            with open(os.path.join(folder, file_name), 'r') as f:
                lines = f.readlines()
                params = {}
                for line in lines:
                    if line.startswith("params:"):
                        continue
                    elif line.startswith("---") or line.startswith("loglike:"):
                        continue
                    else:
                        key_value = line.strip().split()
                        if len(key_value) == 2:
                            key, value = key_value
                            try:
                                params[key] = float(value)
                            except ValueError:
                                params[key] = value
                regression_results[dataset_name] = params

    # ---------- 2. Extract parameters and sort with natural sorting ----------
    datasets = sorted(list(regression_results.keys()), key=_natural_sort_key)

    # raw values from file
    raw_pi_const = [regression_results[ds].get('inflate_const', np.nan) for ds in datasets]
    raw_pi_nuc   = [regression_results[ds].get('inflate_Nucleosome_Distance', np.nan) for ds in datasets]
    raw_pi_cent  = [regression_results[ds].get('inflate_Centromere_Distance', np.nan) for ds in datasets]

    raw_cnt_const = [regression_results[ds].get('const', np.nan) for ds in datasets]
    raw_cnt_nuc   = [regression_results[ds].get('Nucleosome_Distance', np.nan) for ds in datasets]
    raw_cnt_cent  = [regression_results[ds].get('Centromere_Distance', np.nan) for ds in datasets]

    # ---------- 3. Optionally transform ----------
    if transform:
        # Zero inflation part
        # inflate_const is on log-odds scale -> convert to probability pi
        pi_const = [_logistic(v) if np.isfinite(v) else np.nan for v in raw_pi_const]

        # other inflate_* coefficients are changes in log-odds per unit -> odds ratios
        pi_nuc  = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_pi_nuc]
        pi_cent = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_pi_cent]

        # Count part
        # const is log(mean) -> mean
        count_const = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_cnt_const]

        # slopes are log fold-change per unit -> fold-change multiplier
        count_nuc  = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_cnt_nuc]
        count_cent = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_cnt_cent]

        # Names for axes / legends in transformed mode
        param_names = [
            'Inflate: Baseline π (prob)',
            'Inflate: Nucleosome OR',
            'Inflate: Centromere OR',
            'Count: Baseline mean',
            'Count: Nucleosome FC',
            'Count: Centromere FC'
        ]

        heatmap_label = 'Transformed Value'
        detailed_xlabel = [
            'Baseline π (probability of structural zero)',
            'Odds ratio per unit nucleosome distance',
            'Odds ratio per unit centromere distance',
            'Baseline expected mean (growth/count)',
            'Fold-change per unit nucleosome distance',
            'Fold-change per unit centromere distance'
        ]

        zero_line_for_bars = [None, 1.0, 1.0, None, 1.0, 1.0]
        # (In transformed space, "neutral" is 1.0 for ratios/FC, not 0.)

    else:
        # No transform: just plot raw coefficients
        pi_const     = raw_pi_const
        pi_nuc       = raw_pi_nuc
        pi_cent      = raw_pi_cent
        count_const  = raw_cnt_const
        count_nuc    = raw_cnt_nuc
        count_cent   = raw_cnt_cent

        param_names = [
            'Inflate: Const',
            'Inflate: Nucleosome',
            'Inflate: Centromere',
            'Count: Const',
            'Count: Nucleosome',
            'Count: Centromere'
        ]

        heatmap_label = 'Parameter Value'

        detailed_xlabel = [
            'Value',
            'Value',
            'Value',
            'Value',
            'Value',
            'Value'
        ]

        zero_line_for_bars = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # We'll reuse these lists below
    all_param_lists = [
        pi_const, pi_nuc, pi_cent,
        count_const, count_nuc, count_cent
    ]

    # ---------- 4. Compute standard deviations from input data ----------
    std_values = {}
    try:
        datasets_data = read_csv_file_with_distances(input_folder)
        for dataset_name in datasets:
            if dataset_name in datasets_data:
                df_all = pd.concat(datasets_data[dataset_name].values(), ignore_index=True)
                df_all = df_all.dropna(subset=["Value", "Nucleosome_Distance", "Centromere_Distance"]).copy()
                df_all["Centromere_Distance"] = df_all["Centromere_Distance"] / 1000.0
                
                std_values[dataset_name] = {
                    'nucleosome_std': df_all["Nucleosome_Distance"].std(),
                    'centromere_std': df_all["Centromere_Distance"].std()
                }
    except Exception as e:
        print(f"Warning: Could not compute standard deviations: {e}")
        # Use default values if data is not available
        for dataset_name in datasets:
            std_values[dataset_name] = {'nucleosome_std': 1.0, 'centromere_std': 1.0}

    # ---------- 5. Create overview plot with 4 subplots (2x2 grid) ----------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    bar_color = COLORS['blue']  # Single blue color for all bars
    nucleosome_color = NUCLEOSOME_CENTROMERE['nucleosome']  # Red for nucleosome
    centromere_color = NUCLEOSOME_CENTROMERE['centromere']  # Green for centromere
    
    x_pos = np.arange(len(datasets))
    
    # Plot 1: Zero-probability: baseline pi (top-left)
    ax = axes[0, 0]
    ax.bar(x_pos, pi_const, color=bar_color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title('Zero-probability: baseline π', fontweight='bold')
    ax.set_ylabel('Probability of structural zero')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Plot 2: Zero-probability odds ratio (top-right)
    ax = axes[0, 1]
    bar_width = 0.35
    x_pos_nuc = x_pos - bar_width/2
    x_pos_cent = x_pos + bar_width/2
    
    ax.bar(x_pos_nuc, pi_nuc, bar_width, label='Nucleosome', color=nucleosome_color, alpha=0.7)
    ax.bar(x_pos_cent, pi_cent, bar_width, label='Centromere', color=centromere_color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title('Zero-probability odds ratio', fontweight='bold')
    ax.set_ylabel('Odds ratio per unit distance')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    ax.set_ylim(0, 2)
    
    # Add std dev info as text
    std_text_lines = []
    for ds in datasets:
        if ds in std_values:
            std_text_lines.append(f"{ds}: σ_nuc={std_values[ds]['nucleosome_std']:.2f}, σ_cent={std_values[ds]['centromere_std']:.2f}")
    if std_text_lines:
        std_text = "Standard deviations:\n" + "\n".join(std_text_lines[:5])  # Show first 5 to avoid crowding
        ax.text(0.98, 0.98, std_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 3: Count: baseline mean (bottom-left)
    ax = axes[1, 0]
    ax.bar(x_pos, count_const, color=bar_color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title('Count: baseline mean', fontweight='bold')
    ax.set_ylabel('Baseline expected mean')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Count: Fold Change (bottom-right)
    ax = axes[1, 1]
    ax.bar(x_pos_nuc, count_nuc, bar_width, label='Nucleosome', color=nucleosome_color, alpha=0.7)
    ax.bar(x_pos_cent, count_cent, bar_width, label='Centromere', color=centromere_color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title('Count: Fold Change', fontweight='bold')
    ax.set_ylabel('Fold Change per unit distance')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    ax.set_ylim(0, max(count_nuc + count_cent) * 1.05)
    
    # Add std dev info as text
    if std_text_lines:
        ax.text(0.98, 0.98, std_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Match axes for related plots
    # OR plots should have same y-axis
    or_max = max(max([v for v in pi_nuc if np.isfinite(v)] or [1]), 
                 max([v for v in pi_cent if np.isfinite(v)] or [1]))
    or_min = min(min([v for v in pi_nuc if np.isfinite(v)] or [1]), 
                 min([v for v in pi_cent if np.isfinite(v)] or [1]))
    axes[0, 1].set_ylim(0, max(2, or_max * 1.05))
    
    # Fold Change plots should have same y-axis
    fc_max = max(max([v for v in count_nuc if np.isfinite(v)] or [1]), 
                 max([v for v in count_cent if np.isfinite(v)] or [1]))
    fc_min = min(min([v for v in count_nuc if np.isfinite(v)] or [1]), 
                 min([v for v in count_cent if np.isfinite(v)] or [1]))
    axes[1, 1].set_ylim(0, fc_max * 1.05)
    
    plt.suptitle('ZINB Regression Parameters by Dataset', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file.replace(f".png", f"_overview.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Generated overview plot: {output_file.replace(f'.png', f'_overview.png')}")
    
    # ---------- 6. Create individual plots ----------
    # Plot 1: Zero-probability: baseline pi
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, pi_const, color=bar_color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title('Zero-probability: baseline π', fontweight='bold')
    ax.set_ylabel('Probability of structural zero')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_file.replace(f".png", f"_baseline_pi.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated individual plot: {output_file.replace(f'.png', f'_baseline_pi.png')}")
    
    # Plot 2: Zero-probability odds ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos_nuc, pi_nuc, bar_width, label='Nucleosome', color=nucleosome_color, alpha=0.7)
    ax.bar(x_pos_cent, pi_cent, bar_width, label='Centromere', color=centromere_color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title('Zero-probability odds ratio', fontweight='bold')
    ax.set_ylabel('Odds ratio per unit distance')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    ax.set_ylim(0, 2)
    
    # Add std dev info
    if std_text_lines:
        ax.text(0.98, 0.98, std_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file.replace(f".png", f"_odds_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated individual plot: {output_file.replace(f'.png', f'_odds_ratio.png')}")
    
    # Plot 3: Count: baseline mean
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, count_const, color=bar_color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title('Count: baseline mean', fontweight='bold')
    ax.set_ylabel('Baseline expected mean')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_file.replace(f".png", f"_baseline_mean.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated individual plot: {output_file.replace(f'.png', f'_baseline_mean.png')}")
    
    # Plot 4: Count: Fold Change
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos_nuc, count_nuc, bar_width, label='Nucleosome', color=nucleosome_color, alpha=0.7)
    ax.bar(x_pos_cent, count_cent, bar_width, label='Centromere', color=centromere_color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title('Count: Fold Change', fontweight='bold')
    ax.set_ylabel('Fold Change per unit distance')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    ax.set_ylim(0, fc_max * 1.05)
    
    # Add std dev info
    if std_text_lines:
        ax.text(0.98, 0.98, std_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file.replace(f".png", f"_fold_change.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated individual plot: {output_file.replace(f'.png', f'_fold_change.png')}")
    
    print(f"\n✓ Generated 5 plots total: 1 overview + 4 individual plots")
    
def retrieve_average_parameters(folder):
    """
    Retrieve average regression parameters across datasets in the specified folder.
    
    Args:
        folder (str): Path to the folder containing regression result txt files.

    Returns:
        dict: A dictionary containing average regression parameters for each dataset.
    """
    regression_results = {}
    for file_name in os.listdir(folder):
        if file_name.endswith("_summary.txt"):
            dataset_name = file_name.replace("_summary.txt", "")
            with open(os.path.join(folder, file_name), 'r') as f:
                lines = f.readlines()
                params = {}
                for line in lines:
                    if line.startswith("params:"):
                        continue
                    elif line.startswith("---") or line.startswith("loglike:"):
                        continue
                    else:
                        key_value = line.strip().split()
                        if len(key_value) == 2:
                            key, value = key_value
                            try:
                                params[key] = float(value)
                            except ValueError:
                                params[key] = value
                regression_results[dataset_name] = params
    # Transform parameters as in plot_regression_results
    for ds in regression_results:
        # Zero inflation part
        regression_results[ds]['inflate_const'] = _logistic(regression_results[ds]['inflate_const'])
        regression_results[ds]['inflate_Nucleosome_Distance'] = exp(regression_results[ds]['inflate_Nucleosome_Distance'])
        regression_results[ds]['inflate_Centromere_Distance'] = exp(regression_results[ds]['inflate_Centromere_Distance'])
        # Count part
        regression_results[ds]['const'] = exp(regression_results[ds]['const'])
        regression_results[ds]['Nucleosome_Distance'] = exp(regression_results[ds]['Nucleosome_Distance'])
        regression_results[ds]['Centromere_Distance'] = exp(regression_results[ds]['Centromere_Distance'])
    # Compute the mean and standard deviation for each parameter across datasets
    average_params = {}
    for param in regression_results[next(iter(regression_results))].keys():
        values = [regression_results[ds][param] for ds in regression_results if param in regression_results[ds]]
        average_params[param] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return average_params
            

if __name__ == "__main__":
    plot_regression_results("Data_exploration/results/regression/linear", "Data_exploration/results/regression/linear/regression_parameters.png", transform=True)
    # print(retrieve_average_parameters("Data_exploration/results/regression/linear"))
    # out_dir = "Data_exploration/results/regression/linear"
    # # Run per-dataset with immediate writes
    # regression_results = perform_regression_on_datasets(
    #     "Data_exploration/results/distances_with_zeros",
    #     combine_all=False,
    #     output_dir=out_dir,
    #     write_immediately=True,
    # )

    # Run combined (note: this may use more memory by design)
    # combined_results = perform_regression_on_datasets(
    #     "Data_exploration/results/distances_with_zeros",
    #     combine_all=True,
    #     output_dir=out_dir,
    #     write_immediately=True,
    # )

    