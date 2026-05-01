import os, sys
import torch
import numpy as np
import hashlib
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from AE.preprocessing.preprocessing import preprocess_with_split
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import test
from AE.reconstruction.reconstruct_output import OutputReconstructor
from sklearn.metrics import mean_absolute_error, r2_score
from AE.plotting.results_ZINB import density_plots, masked_values_analysis
import matplotlib.pyplot as plt

# ========== MODEL AND DATA CONFIGURATION ==========

# Test dataset configuration
INPUT_FOLDER = "Data/combined_strains"
test_chromosomes = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']
# test_chromosomes = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
val_chromosomes = ['ChrVIII', 'ChrXIV', 'ChrXV']
train_chromosomes = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
# No validation set needed for testing

# # Path to the trained model
# MODEL_PATH = "AE/results/models/ZINBAE_layers1600_ep150_noise0.150_muoff0.000.pt"

# !!! ZINB NLL

# # Preprocessing parameters (should match training configuration)
# FEATURES = ['Nucl']
# BIN_SIZE = 1
# MOVING_AVERAGE = False
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = 2000

# # Training parameters (should match training configuration)
# BATCH_SIZE = 128
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.3
# MASKED_RECON_WEIGHT = 0.001  # gamma
# REGULARIZER = 'none'
# REGULARIZATION_WEIGHT = 1e-5  # alpha

# !!! COMBINED


# MODEL_PATH = "AE/results/models/ZINBAE_layers1600_ep144_noise0.150_muoff0.000.pt"
# # Preprocessing parameters 
# FEATURES = ['Centr']
# BIN_SIZE = 20
# MOVING_AVERAGE = True
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = 500

# # Training parameters 
# BATCH_SIZE = 32
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.53
# MASKED_RECON_WEIGHT = 0.079  # gamma
# REGULARIZER = 'none'
# REGULARIZATION_WEIGHT = 1e-5  # alpha

# !!! Masked

MODEL_PATH = "AE/results/models/ZINBAE_layers752_ep93_noise0.150_muoff0.000.pt"

# Preprocessing parameters 
FEATURES = ['Centr', 'Nucl']
BIN_SIZE = 20
MOVING_AVERAGE = True
DATA_POINT_LENGTH = 2000
STEP_SIZE = int(0.75 * 2000)  
# Training parameters 
BATCH_SIZE = 64
NOISE_LEVEL = 0.15
PI_THRESHOLD = 0.43
MASKED_RECON_WEIGHT = 0.0033  # gamma - exact value
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 4.22e-05  # alpha - exact value

# MODEL_PATH = "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff0.000.pt"


# # Preprocessing parameters 
# FEATURES = ['Centr']
# BIN_SIZE = 19
# MOVING_AVERAGE = True
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = int(0.572 * 2000)  
# # Training parameters 
# BATCH_SIZE = 64
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.516 
# MASKED_RECON_WEIGHT = 0.0327  # gamma - exact value
# REGULARIZER = 'none'
# REGULARIZATION_WEIGHT = 4.22e-05 

# !!! Binned
# MODEL_PATH = "AE/results/models/ZINBAE_layers1168_ep116_noise0.150_muoff0.000.pt"

# # Preprocessing parameters 
# FEATURES = ['Centr']
# BIN_SIZE = 17
# MOVING_AVERAGE = False
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = int(0.81 * 2000)  
# # Training parameters 
# BATCH_SIZE = 128
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.45
# MASKED_RECON_WEIGHT = 1.37  # gamma - exact value
# REGULARIZER = 'l1'
# REGULARIZATION_WEIGHT = 0.00035

# Data caching options
USE_CACHED_DATA = True  # Set to True after first run to use cached data with correct parameters

# Output directory for results and plots
OUTPUT_DIR = "AE/results/extra_results/final"  # Where plots and metrics will be saved
MU_OFFSET = 0
PROCESSED_DATA_DIR = "Data/processed_data"  # Where preprocessed data will be cached
RECONSTRUCT = True  # Whether to reconstruct genomic coordinates from predictions and save as CSV
RECONSTRUCTION_BASE_DIR = "Data/reconstruction"

# Control which splits to process (to avoid memory issues with large datasets)
PROCESS_TRAIN = True # Set to False to skip train set (can cause memory issues)
PROCESS_VAL = False # Set to False to skip validation set
PROCESS_TEST = False   # Always process test set
# ================================================


def run_name_from_model_path(model_path):
    """Use model filename (without extension) as run folder name."""
    return os.path.splitext(os.path.basename(model_path))[0]


def generate_cache_filename(train_chroms, test_chroms, features, bin_size, moving_average, data_point_length, step_size):
    """
    Generate a unique filename for cached preprocessed data based on parameters.
    IMPORTANT: train_chroms affects normalization (size factors), so it MUST be in the cache hash!
    """
    # Create a dict of parameters that affect preprocessing
    params = {
        'train_chroms': sorted(train_chroms),  # CRITICAL: affects size_factor calculation!
        'test_chroms': sorted(test_chroms),  # Sort to ensure consistency
        'features': sorted(features),
        'bin_size': bin_size,
        'moving_average': moving_average,
        'data_point_length': data_point_length,
        'step_size': step_size
    }
    
    # Create a hash of the parameters for a unique identifier
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    features_str = "_".join(sorted(features))
    
    filename = f"{features_str}_bin{bin_size}_ma{moving_average}_len{data_point_length}_step{step_size}_{params_hash}.npy"
    
    return filename


def plot_pi_vs_mu(mu, pi, raw_counts, save_dir, prefix="", model_type="ZINB"):
    """
    Create a plot showing the relationship between pi (zero-inflation probability) and mu (mean)
    for positions where the original raw count is zero.
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean parameter from ZINB model
    pi : np.ndarray  
        Zero-inflation probability from ZINB model
    raw_counts : np.ndarray
        Original raw count data
    save_dir : str
        Directory to save the plot
    prefix : str
        Prefix for the filename
    model_type : str
        Model type for the title
    """
    if pi is None or raw_counts is None:
        print("Cannot create pi vs mu plot: pi or raw_counts not available")
        return
    
    mu_flat = mu.flatten()
    pi_flat = pi.flatten()
    raw_flat = raw_counts.flatten()
    
    # Filter to only keep positions where original raw count is zero
    zero_mask = raw_flat == 0
    mu_zeros = mu_flat[zero_mask]
    pi_zeros = pi_flat[zero_mask]
    
    n_zeros = len(mu_zeros)
    print(f"Number of original zero positions: {n_zeros} ({100*n_zeros/len(raw_flat):.2f}% of data)")
    
    if n_zeros == 0:
        print("No zero positions found, skipping pi vs mu plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Clip mu for better visualization (remove extreme outliers)
    mu_zeros_clipped = np.clip(mu_zeros, 0, np.percentile(mu_zeros, 99.5))
    
    # Subplot 1: Hexbin plot showing density of relationship for zeros
    axes[0].hexbin(mu_zeros_clipped, pi_zeros, gridsize=50, cmap='Blues', mincnt=1)
    axes[0].set_xlabel('Mean (μ)')
    axes[0].set_ylabel('Zero-inflation Probability (π)')
    axes[0].set_title(f'{model_type}: π vs μ for Original Zeros\n(n={n_zeros:,} positions)')
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot for zeros
    # Sample for better visualization if too many points
    sample_size = min(10000, len(mu_zeros))
    if len(mu_zeros) > sample_size:
        sample_indices = np.random.choice(len(mu_zeros), size=sample_size, replace=False)
        mu_sample = mu_zeros_clipped[sample_indices]
        pi_sample = pi_zeros[sample_indices]
        title_suffix = f'\n(showing {sample_size:,} of {n_zeros:,} points)'
    else:
        mu_sample = mu_zeros_clipped
        pi_sample = pi_zeros
        title_suffix = f'\n(n={n_zeros:,} points)'
    
    axes[1].scatter(mu_sample, pi_sample, alpha=0.3, s=5, c='blue')
    axes[1].set_xlabel('Mean (μ)')
    axes[1].set_ylabel('Zero-inflation Probability (π)')
    axes[1].set_title(f'{model_type}: π vs μ for Original Zeros{title_suffix}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_pi_vs_mu.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pi vs mu plot to: {save_path}")


def load_or_preprocess_data(input_folder, train_chroms, test_chroms, features, bin_size, moving_average, 
                            preprocessing_length, step_size, use_cache=True, cache_dir="Data/processed_data"):
    """
    Load preprocessed data from cache if available, otherwise preprocess and save.
    
    Returns:
        tuple: (test_set, test_metadata)
            - test_set: The preprocessed test dataset
            - test_metadata: Metadata for window reconstruction
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = generate_cache_filename(train_chroms, test_chroms, features, bin_size, moving_average, 
                                            preprocessing_length, step_size)
    cache_path = os.path.join(cache_dir, cache_filename)
    cache_meta_path = cache_path.replace('.npy', '_metadata.json')
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_path) and os.path.exists(cache_meta_path):
        print(f"\n{'='*50}")
        print("LOADING CACHED DATA")
        print(f"{'='*50}")
        print(f"Loading from: {cache_path}")
        
        test_set = np.load(cache_path)
        with open(cache_meta_path, 'r') as f:
            test_metadata = json.load(f)
        print(f"Loaded test set with shape: {test_set.shape}")
        print(f"Loaded metadata for {len(test_metadata)} windows")
        
        return test_set, test_metadata
    
    # If not cached or cache disabled, preprocess
    print(f"\n{'='*50}")
    print("PREPROCESSING DATA")
    print(f"{'='*50}")
    print("Cached data not found or caching disabled. Preprocessing from scratch...")
    
    print(f"Test chromosomes: {test_chroms}")
    print(f"Train chromosomes: {train_chroms}")
    
    # Preprocess data with explicit chromosome split
    _, _, test_set, _, _, test_metadata, _, _, _ = preprocess_with_split(
        input_folder=input_folder,
        train_chroms=train_chroms,
        val_chroms=[],  # No validation set needed for testing
        test_chroms=test_chroms,
        features=features,
        bin_size=bin_size,
        moving_average=moving_average,
        data_point_length=preprocessing_length,
        step_size=step_size
    )
    
    # Save to cache
    print(f"\nSaving preprocessed data to cache: {cache_path}")
    np.save(cache_path, test_set)
    with open(cache_meta_path, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    print(f"Cached data and metadata saved successfully!")
    
    return test_set, test_metadata


def print_parameter_statistics(predictions, mu_raw, theta, pi, raw_counts, masks=None, split_name=""):
    """
    Print detailed statistics about ZINB parameters (mu, theta, pi) similar to results_ZINB.py
    """
    print("\n" + "="*80)
    print(f"{split_name.upper()} - OVERALL PARAMETER STATISTICS")
    print("="*80)
    
    # Use RAW COUNTS for comparison
    actual_counts_flat = raw_counts.flatten()
    mu_flat = mu_raw.flatten()
    
    # Overall metrics
    abs_err = np.abs(actual_counts_flat - mu_flat)
    mae = abs_err.mean()
    mae_std = abs_err.std(ddof=1)
    r2 = r2_score(actual_counts_flat, mu_flat)
    
    # Separate zeros from non-zeros
    zero_mask = actual_counts_flat == 0
    non_zero_mask = ~zero_mask
    
    # Pi statistics
    if pi is not None:
        pi_flat = pi.flatten()
        mean_pi_zeros = pi_flat[zero_mask].mean() if np.any(zero_mask) else 0
        mean_pi_nonzeros = pi_flat[non_zero_mask].mean() if np.any(non_zero_mask) else 0
        std_pi_zeros = pi_flat[zero_mask].std(ddof=1) if np.any(zero_mask) else 0
        std_pi_nonzeros = pi_flat[non_zero_mask].std(ddof=1) if np.any(non_zero_mask) else 0
    else:
        mean_pi_zeros = mean_pi_nonzeros = std_pi_zeros = std_pi_nonzeros = 0
    
    # Mu statistics
    mean_mu_zeros = mu_flat[zero_mask].mean() if np.any(zero_mask) else 0
    mean_mu_nonzeros = mu_flat[non_zero_mask].mean() if np.any(non_zero_mask) else 0
    std_mu_zeros = mu_flat[zero_mask].std(ddof=1) if np.any(zero_mask) else 0
    std_mu_nonzeros = mu_flat[non_zero_mask].std(ddof=1) if np.any(non_zero_mask) else 0
    
    # Theta statistics
    if theta is not None:
        theta_flat = theta.flatten()
        mean_theta = theta_flat.mean()
        std_theta = theta_flat.std(ddof=1)
    else:
        mean_theta = std_theta = 0
    
    # Print overall statistics
    print(f"MAE: {mae:.4f}, SD MAE: {mae_std:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Mean π (zeros): {mean_pi_zeros:.4f}, Mean π (non-zeros): {mean_pi_nonzeros:.4f}")
    print(f"SD π (zeros): {std_pi_zeros:.4f}, SD π (non-zeros): {std_pi_nonzeros:.4f}")
    print(f"Mean μ (zeros): {mean_mu_zeros:.4f}, Mean μ (non-zeros): {mean_mu_nonzeros:.4f}")
    print(f"SD μ (zeros): {std_mu_zeros:.4f}, SD μ (non-zeros): {std_mu_nonzeros:.4f}")
    print(f"Theta mean: {mean_theta:.4f}, Theta SD: {std_theta:.4f}")
    
    # Masked values analysis if masks are provided
    if masks is not None:
        mask_flat = masks.flatten()
        masked_positions = mask_flat == True
        
        if np.any(masked_positions):
            print("\n" + "="*80)
            print(f"{split_name.upper()} - MASKED VALUES PERFORMANCE METRICS")
            print("="*80)
            
            # Extract masked values
            masked_actual = actual_counts_flat[masked_positions]
            masked_recon = mu_flat[masked_positions]
            
            print(f"Number of masked values: {len(masked_actual)}")
            
            # Compute metrics for masked values
            mae_masked = mean_absolute_error(masked_actual, masked_recon)
            abs_err_masked = np.abs(masked_actual - masked_recon)
            mae_std_masked = abs_err_masked.std(ddof=1)
            r2_masked = r2_score(masked_actual, masked_recon)
            
            # Compute detailed statistics for masked values
            zero_mask_masked = masked_actual == 0
            non_zero_mask_masked = ~zero_mask_masked
            
            # Pi statistics for masked values
            if pi is not None:
                masked_pi = pi_flat[masked_positions]
                mean_pi_zeros_masked = masked_pi[zero_mask_masked].mean() if np.any(zero_mask_masked) else 0
                mean_pi_nonzeros_masked = masked_pi[non_zero_mask_masked].mean() if np.any(non_zero_mask_masked) else 0
                std_pi_zeros_masked = masked_pi[zero_mask_masked].std(ddof=1) if np.any(zero_mask_masked) else 0
                std_pi_nonzeros_masked = masked_pi[non_zero_mask_masked].std(ddof=1) if np.any(non_zero_mask_masked) else 0
            else:
                mean_pi_zeros_masked = mean_pi_nonzeros_masked = std_pi_zeros_masked = std_pi_nonzeros_masked = 0
            
            # Mu statistics for masked values
            mean_mu_zeros_masked = masked_recon[zero_mask_masked].mean() if np.any(zero_mask_masked) else 0
            mean_mu_nonzeros_masked = masked_recon[non_zero_mask_masked].mean() if np.any(non_zero_mask_masked) else 0
            std_mu_zeros_masked = masked_recon[zero_mask_masked].std(ddof=1) if np.any(zero_mask_masked) else 0
            std_mu_nonzeros_masked = masked_recon[non_zero_mask_masked].std(ddof=1) if np.any(non_zero_mask_masked) else 0
            
            # Theta statistics for masked values
            if theta is not None:
                masked_theta = theta_flat[masked_positions]
                mean_theta_masked = masked_theta.mean()
                std_theta_masked = masked_theta.std(ddof=1)
            else:
                mean_theta_masked = std_theta_masked = 0
            
            # Print detailed statistics for masked values
            print(f"MAE: {mae_masked:.4f}, SD MAE: {mae_std_masked:.4f}")
            print(f"R²: {r2_masked:.4f}")
            print(f"Mean π (zeros): {mean_pi_zeros_masked:.4f}, Mean π (non-zeros): {mean_pi_nonzeros_masked:.4f}")
            print(f"SD π (zeros): {std_pi_zeros_masked:.4f}, SD π (non-zeros): {std_pi_nonzeros_masked:.4f}")
            print(f"Mean μ (zeros): {mean_mu_zeros_masked:.4f}, Mean μ (non-zeros): {mean_mu_nonzeros_masked:.4f}")
            print(f"SD μ (zeros): {std_mu_zeros_masked:.4f}, SD μ (non-zeros): {std_mu_nonzeros_masked:.4f}")
            print(f"Theta mean: {mean_theta_masked:.4f}, Theta SD: {std_theta_masked:.4f}")
    
    print("="*80 + "\n")


def process_split(split_name, split_chromosomes, model, model_config, chrom_embedding, seq_len):
    """Process a single split (train/val/test) and save reconstructions."""
    print("\n" + "="*80)
    print(f"PROCESSING {split_name.upper()} SPLIT ({len(split_chromosomes)} chromosomes)")
    print("="*80)
    
    preprocessing_length = DATA_POINT_LENGTH
    if preprocessing_length != seq_len:
        print(
            f"\nWARNING: preprocessing_length={preprocessing_length} does not match "
            f"checkpoint seq_length={seq_len}. Using checkpoint seq_length."
        )
        preprocessing_length = seq_len
    
    # Load or preprocess data for this split
    chrom = 'Chr' in FEATURES
    
    split_set, split_metadata = load_or_preprocess_data(
        input_folder=INPUT_FOLDER,
        train_chroms=train_chromosomes,  # CRITICAL: used for normalization calculation
        test_chroms=split_chromosomes,
        features=FEATURES,
        bin_size=BIN_SIZE,
        moving_average=MOVING_AVERAGE,
        preprocessing_length=preprocessing_length,
        step_size=STEP_SIZE,
        use_cache=USE_CACHED_DATA,
        cache_dir=PROCESSED_DATA_DIR
    )
    
    print(f"\n{split_name} set size: {len(split_set)}")
    print(f"{split_name} set shape: {split_set.shape}")
    
    # Create dataloader
    split_dataloader = dataloader_from_array(
        split_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        zinb=True,  
        chrom=chrom,
        sample_fraction=1.0,
        denoise_percentage=NOISE_LEVEL
    )
    
    print(f"{split_name} dataloader: {len(split_dataloader.dataset)} samples, {len(split_dataloader)} batches")
    
    # Run inference
    print(f"\nEvaluating model on {split_name} data...")
    predictions, latents, metrics, mu_raw, theta, pi, raw_counts, masks = test(
        model=model,
        dataloader=split_dataloader,
        chrom=chrom,
        chrom_embedding=chrom_embedding,
        plot=False,  # Disable plotting for efficiency
        denoise_percent=NOISE_LEVEL,
        gamma=MASKED_RECON_WEIGHT,
        pi_threshold=PI_THRESHOLD,
        regularizer=REGULARIZER,
        alpha=REGULARIZATION_WEIGHT,
        eval_mode="",
        name=""
    )
    
    print(f"\n{split_name.upper()} METRICS:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Print detailed parameter statistics
    print_parameter_statistics(
        predictions=predictions,
        mu_raw=mu_raw,
        theta=theta,
        pi=pi,
        raw_counts=raw_counts,
        masks=masks,
        split_name=split_name
    )
    
    # Generate plots for test set only
    if split_name == 'test':
        print("\n" + "="*80)
        print("GENERATING PLOTS FOR TEST SET")
        print("="*80)
        
        # Create output directory for plots
        plot_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Generate filename prefix
        run_name = run_name_from_model_path(MODEL_PATH)
        prefix = f"{split_name}_{run_name}"
        
        # Calculate metrics needed for plotting
        actual_counts_flat = raw_counts.flatten()
        mu_flat = mu_raw.flatten()
        residuals = actual_counts_flat - mu_flat
        mae = mean_absolute_error(actual_counts_flat, mu_flat)
        r2 = r2_score(actual_counts_flat, mu_flat)
        
        # 1. Density plots (actual vs predicted, residuals)
        print("\nGenerating density plots...")
        density_plots(
            actual_counts_flat=actual_counts_flat,
            all_reconstructions_mu=mu_raw,
            residuals=residuals,
            comparison_label='Raw Counts',
            model_type='ZINB',
            save_dir=plot_dir,
            prefix=prefix,
            r2=r2,
            mae=mae,
            all_pi=pi
        )
        
        # 2. Masked values analysis (if masks are available)
        if masks is not None and np.any(masks):
            print("Generating masked values analysis plot...")
            masked_values_analysis(
                all_reconstructions_mu=mu_raw,
                all_pi=pi,
                all_raw_counts=raw_counts,
                all_masks=masks,
                all_theta=theta,
                model_type='ZINB',
                save_dir=plot_dir,
                prefix=prefix,
                threshold=PI_THRESHOLD
            )
        else:
            print("No masked values to analyze (masks not available or empty)")
        
        # 3. Pi vs Mu relationship plot
        print("Generating pi vs mu relationship plot...")
        plot_pi_vs_mu(
            mu=mu_raw,
            pi=pi,
            raw_counts=raw_counts,
            save_dir=plot_dir,
            prefix=prefix,
            model_type='ZINB'
        )
        
        print(f"\nAll plots saved to: {plot_dir}/")
        print("="*80 + "\n")
    
    # Save reconstruction if enabled
    if RECONSTRUCT:
        run_name = run_name_from_model_path(MODEL_PATH)
        reconstruction_base = os.path.join(RECONSTRUCTION_BASE_DIR, run_name)
        split_reconstruction_dir = os.path.join(reconstruction_base, split_name)
        os.makedirs(split_reconstruction_dir, exist_ok=True)

        # Save metadata for this split
        split_metadata_path = os.path.join(split_reconstruction_dir, "metadata.json")
        with open(split_metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        print(f"\nMetadata saved to: {split_metadata_path}")
        print(f"  Contains {len(split_metadata)} window locations")

        # Reconstruct and save
        reconstructor = OutputReconstructor(split_metadata_path)
        reconstructed_df = reconstructor.reconstruct_to_dataframe(
            predictions,
            aggregation='mean',
            include_uncertainty=(theta is not None or pi is not None),
            mu_raw=mu_raw,
            theta=theta,
            pi=pi
        )

        reconstructor.save_as_csv(reconstructed_df, split_reconstruction_dir, split_by_chromosome=True)
        print(f"\nReconstructed genomic data saved to: {split_reconstruction_dir}")
    
    return metrics


def load_model_and_test():
    """
    Load a trained model and evaluate it on train, val, and test datasets.
    Creates separate reconstruction directories for each split.
    
    Uses configuration parameters defined at the top of this file.
    """
    print("="*80)
    print("LOADING TRAINED MODEL")
    print("="*80)
    
    # Load the saved model
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Extract model configuration from checkpoint
    model_config = checkpoint['model_config']
    
    print(f"\nModel Architecture from checkpoint:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Calculate expected input dimension based on model config
    seq_len = model_config.get('seq_length', 2000)
    feat_dim = model_config.get('feature_dim', 8)
    use_conv = model_config.get('use_conv', False)
    
    if use_conv:
        pool_size = model_config.get('pool_size', 2)
        conv_channels = model_config.get('conv_channels', 64)
        pooled_seq_length = seq_len // pool_size
        expected_input_dim = pooled_seq_length * conv_channels
        print(f"\n  Expected input (with conv): ({BATCH_SIZE}, {seq_len}, {feat_dim}) -> flattened after conv+pool: ({BATCH_SIZE}, {expected_input_dim})")
    else:
        expected_input_dim = seq_len * feat_dim
        print(f"\n  Expected input (no conv): ({BATCH_SIZE}, {seq_len * feat_dim})")
    
    print(f"  Model expects feature_dim={feat_dim}, use_conv={use_conv}")
    
    print(f"\nConfiguration:")
    print(f"  features: {FEATURES}")
    print(f"  bin_size: {BIN_SIZE}")
    print(f"  moving_average: {MOVING_AVERAGE}")
    print(f"  data_point_length: {DATA_POINT_LENGTH}")
    print(f"  step_size: {STEP_SIZE}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  noise_level: {NOISE_LEVEL}")
    print(f"  pi_threshold: {PI_THRESHOLD}")
    print(f"  masked_recon_weight: {MASKED_RECON_WEIGHT}")
    print(f"  regularizer: {REGULARIZER}")
    print(f"  regularization_weight: {REGULARIZATION_WEIGHT}")
    print(f"  use_cached_data: {USE_CACHED_DATA}")
    
    # Initialize model with saved configuration
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = ZINBAE(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully!")
    
    # Create chromosome embedding if needed
    chrom = 'Chr' in FEATURES
    chrom_embedding = ChromosomeEmbedding() if chrom else None
    
    # Process each split
    all_metrics = {}
    
    splits_to_process = []
    if PROCESS_TRAIN and train_chromosomes:
        splits_to_process.append(('train', train_chromosomes))
    if PROCESS_VAL and val_chromosomes:
        splits_to_process.append(('val', val_chromosomes))
    if PROCESS_TEST and test_chromosomes:
        splits_to_process.append(('test', test_chromosomes))
    
    if not splits_to_process:
        print("\nWARNING: No splits selected for processing!")
        print("Enable PROCESS_TRAIN, PROCESS_VAL, or PROCESS_TEST in configuration.")
        return {}
    
    print(f"\nProcessing {len(splits_to_process)} split(s): {[s[0] for s in splits_to_process]}")
    
    for split_name, split_chroms in splits_to_process:
        metrics = process_split(
            split_name, 
            split_chroms, 
            model, 
            model_config, 
            chrom_embedding,
            seq_len
        )
        all_metrics[split_name] = metrics
    
    print("\n" + "="*80)
    print("ALL SPLITS COMPLETE")
    print("="*80)
    
    if RECONSTRUCT and all_metrics:
        run_name = run_name_from_model_path(MODEL_PATH)
        reconstruction_base = os.path.join(RECONSTRUCTION_BASE_DIR, run_name)
        print(f"\nReconstructions saved to: {reconstruction_base}/")
        for split_name in all_metrics.keys():
            split_chroms = dict(splits_to_process).get(split_name, [])
            print(f"  - {split_name}/ : {len(split_chroms)} chromosomes")
    
    return all_metrics


if __name__ == "__main__":
    load_model_and_test()
