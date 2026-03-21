import os, sys
import torch
import numpy as np
import hashlib
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess_with_split
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import test
from AE.reconstruct_output import OutputReconstructor

# ========== MODEL AND DATA CONFIGURATION ==========

# Test dataset configuration
INPUT_FOLDER = "Data/combined_strains"
# test_chromosomes = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']
test_chromosomes = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
# test_chromosomes = ['ChrVIII', 'ChrXIV', 'ChrXV']
train_chromosomes = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
# No validation set needed for testing

# # Path to the trained model
# MODEL_PATH = "AE/results/models/ZINBAE_20260225_121351_noconv_layers1600_ep30.pt"

# # Preprocessing parameters (should match training configuration)
# FEATURES = ['Centr']
# BIN_SIZE = 1
# MOVING_AVERAGE = False
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = 500

# # Training parameters (should match training configuration)
# BATCH_SIZE = 32
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.7
# MASKED_RECON_WEIGHT = 0.001  # gamma
# REGULARIZER = 'l2'
# REGULARIZATION_WEIGHT = 1e-5  # alpha

# MODEL_PATH = "AE/results/models/ZINBAE_20260226_195349_noconv_layers752_ep141.pt"
# MODEL_PATH = "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff0.000.pt"
# # # # MODEL_PATH = "AE/results/models/ZINBAE_20260227_153016_noconv_layers752_ep141.pt"
# # # MODEL_PATH = "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff1.000.pt"

# # Preprocessing parameters 
# FEATURES = ['Centr']
# BIN_SIZE = 19
# MOVING_AVERAGE = True
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = 894

# # Training parameters 
# BATCH_SIZE = 128
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.7
# MASKED_RECON_WEIGHT = 0.00872  # gamma
# REGULARIZER = 'none'
# REGULARIZATION_WEIGHT = 1e-5  # alpha

MODEL_PATH = "AE/results/models/ZINBAE_layers752_ep92_noise0.150_muoff0.000.pt"

# Preprocessing parameters 
FEATURES = ['Nucl']
BIN_SIZE = 17
MOVING_AVERAGE = False
DATA_POINT_LENGTH = 2000
STEP_SIZE = int(0.391 * 2000)  
# Training parameters 
BATCH_SIZE = 128
NOISE_LEVEL = 0.15
PI_THRESHOLD = 0.378  
MASKED_RECON_WEIGHT = 0.127  # gamma - exact value
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 4.22e-05  # alpha - exact value

# MODEL_PATH = "AE/results/models/ZINBAE_layers432_ep54_noise0.150_muoff1.000.pt"


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

# Data caching options
USE_CACHED_DATA = True  # Set to True after first run to use cached data with correct parameters

# Output directory for results and plots
OUTPUT_DIR = "AE/results/final"  # Where plots and metrics will be saved
MU_OFFSET = 1
PROCESSED_DATA_DIR = "Data/processed_data"  # Where preprocessed data will be cached
RECONSTRUCT = True  # Whether to reconstruct genomic coordinates from predictions and save as CSV
RECONSTRUCTION_BASE_DIR = "Data/reconstruction"
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


def load_model_and_test():
    """
    Load a trained model and evaluate it on the test dataset.
    Creates visualization plots of the test results.
    
    Uses configuration parameters defined at the top of this file.
    """
    print("="*50)
    print("LOADING TRAINED MODEL")
    print("="*50)
    
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
    
    print(f"\nTest Configuration:")
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
    print(f"  output_dir: {OUTPUT_DIR}")
    
    # Mirror AE/main.py preprocessing behavior:
    # when not using moving average, DATA_POINT_LENGTH is converted to bin-count windows.

    preprocessing_length = DATA_POINT_LENGTH

    # Always prioritize the checkpoint's seq_length to avoid shape mismatches
    # when test-time constants drift from training settings.
    if preprocessing_length != seq_len:
        print(
            f"\nWARNING: preprocessing_length={preprocessing_length} does not match "
            f"checkpoint seq_length={seq_len}. Using checkpoint seq_length."
        )
        preprocessing_length = seq_len
    
    # Check if chromosome feature is used
    chrom = 'Chr' in FEATURES
    
    # Load or preprocess test data
    test_set, test_metadata = load_or_preprocess_data(
        input_folder=INPUT_FOLDER,
        train_chroms=train_chromosomes,  # CRITICAL: used for normalization calculation
        test_chroms=test_chromosomes,
        features=FEATURES,
        bin_size=BIN_SIZE,
        moving_average=MOVING_AVERAGE,
        preprocessing_length=preprocessing_length,
        step_size=STEP_SIZE,
        use_cache=USE_CACHED_DATA,
        cache_dir=PROCESSED_DATA_DIR
    )
    
    print(f"\nTest set size: {len(test_set)}")
    print(f"Test set shape: {test_set.shape}")
    print(f"  - Sequence length: {test_set.shape[1]}")
    print(f"  - Feature dimension from data: {test_set.shape[2]}")
    
    # Create chromosome embedding if needed
    chrom_embedding = ChromosomeEmbedding() if chrom else None
    
    # Create test dataloader
    test_dataloader = dataloader_from_array(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        zinb=True,  
        chrom=chrom,
        sample_fraction=1.0,  # Use all test data
        denoise_percentage=NOISE_LEVEL
    )
    
    print(f"Test dataloader: {len(test_dataloader.dataset)} samples, {len(test_dataloader)} batches")
    
    # Initialize model with saved configuration
    print("\n" + "="*50)
    print("INITIALIZING MODEL")
    print("="*50)
    
    model = ZINBAE(**model_config)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model weights loaded successfully!")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run test function to create plots
    print("\n" + "="*50)
    print("EVALUATING ON TEST DATA")
    print("="*50)
    
    # Extract just the final directory name for organizing results
    # The test function will create: AE/results/{subdir}/{name}/
    # To save to OUTPUT_DIR, we need to set subdir to the relative path from AE/results/
    # or use a simple name if OUTPUT_DIR is already under AE/results/
    
    # Check if OUTPUT_DIR is under AE/results/
    if OUTPUT_DIR.startswith("AE/results/"):
        # Extract the subdirectory structure after AE/results/
        subdir_path = OUTPUT_DIR.replace("AE/results/", "")
        # Use the path as subdir and no additional name
        eval_mode = subdir_path
        custom_name = ""
    else:
        # OUTPUT_DIR is somewhere else - use it as a simple identifier
        eval_mode = "testing"
        custom_name = OUTPUT_DIR.replace("/", "_")
    
    predictions, latents, test_metrics, mu_raw, theta, pi = test(
        model=model,
        dataloader=test_dataloader,
        chrom=chrom,
        chrom_embedding=chrom_embedding,
        plot=True,  # Enable plotting
        denoise_percent=NOISE_LEVEL,
        gamma=MASKED_RECON_WEIGHT,
        pi_threshold=PI_THRESHOLD,
        regularizer=REGULARIZER,
        alpha=REGULARIZATION_WEIGHT,
        eval_mode=eval_mode,
        name=custom_name
    )
    
    print("\n" + "="*50)
    print("TEST METRICS")
    print("="*50)
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")
    
    # Save metrics to output directory
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    if RECONSTRUCT:
        print("\n" + "="*50)
        print("RECONSTRUCTING GENOMIC COORDINATES")
        print("="*50)

        run_name = run_name_from_model_path(MODEL_PATH)
        reconstruction_dir = os.path.join(RECONSTRUCTION_BASE_DIR, run_name)
        os.makedirs(reconstruction_dir, exist_ok=True)

        metadata_path = os.path.join(reconstruction_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(test_metadata, f, indent=2)
        print(f"\nMetadata saved to: {metadata_path}")
        print(f"  Contains {len(test_metadata)} window locations")

        reconstructor = OutputReconstructor(metadata_path)
        reconstructed_df = reconstructor.reconstruct_to_dataframe(
            predictions,
            aggregation='mean',
            include_uncertainty=(theta is not None or pi is not None),
            mu_raw=mu_raw,
            theta=theta,
            pi=pi
        )

        reconstructor.save_as_csv(reconstructed_df, reconstruction_dir, split_by_chromosome=True)
        print(f"\nReconstructed genomic data saved to: {reconstruction_dir}")
        print("  One CSV per dataset/chromosome with columns:")
        print("  position, reconstruction, mu, pi, theta")

        print("\n" + "="*50)
        print("TESTING COMPLETE")
        print("="*50)
        print(f"Results saved to: {reconstruction_dir}/")

    return test_metrics


if __name__ == "__main__":
    load_model_and_test()
