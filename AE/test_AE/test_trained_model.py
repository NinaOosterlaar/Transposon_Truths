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
from AE.reconstruct_output import OutputReconstructor

# ========== MODEL AND DATA CONFIGURATION ==========

# Test dataset configuration
INPUT_FOLDER = "Data/combined_strains"
test_chromosomes = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']
# test_chromosomes = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
val_chromosomes = ['ChrVIII', 'ChrXIV', 'ChrXV']
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

MODEL_PATH = "AE/results/models/ZINBAE_layers1168_ep116_noise0.150_muoff0.000.pt"

# Preprocessing parameters 
FEATURES = ['Centr']
BIN_SIZE = 17
MOVING_AVERAGE = False
DATA_POINT_LENGTH = 2000
STEP_SIZE = int(0.807 * 2000)  
# Training parameters 
BATCH_SIZE = 128
NOISE_LEVEL = 0.15
PI_THRESHOLD = 0.45 
MASKED_RECON_WEIGHT = 1.37  # gamma - exact value
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
    predictions, latents, metrics, mu_raw, theta, pi = test(
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
    
    splits_to_process = [
        ('train', train_chromosomes),
        ('val', val_chromosomes),
        ('test', test_chromosomes)
    ]
    
    for split_name, split_chroms in splits_to_process:
        if split_chroms:  # Only process if chromosomes are defined
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
    
    run_name = run_name_from_model_path(MODEL_PATH)
    reconstruction_base = os.path.join(RECONSTRUCTION_BASE_DIR, run_name)
    print(f"\nAll reconstructions saved to: {reconstruction_base}/")
    for split_name, split_chroms in splits_to_process:
        if split_chroms:
            print(f"  - {split_name}/ : {len(split_chroms)} chromosomes")
    
    return all_metrics


if __name__ == "__main__":
    load_model_and_test()
