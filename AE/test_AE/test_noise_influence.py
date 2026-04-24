import os, sys
import torch
import numpy as np
import hashlib
import json
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess_with_split
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import test

# ========== MODEL AND DATA CONFIGURATION ==========

# Test dataset configuration
INPUT_FOLDER = "Data/combined_strains"
test_chromosomes = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']
# test_chromosomes = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
val_chromosomes = ['ChrVIII', 'ChrXIV', 'ChrXV']
train_chromosomes = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
  # No validation set needed for testing
noise_levels = [0.10, 0.15, 0.25, 0.5, 0.75, 0.9]  # Noise levels to test (as percentages of data points to corrupt)


# MODEL_PATH = "AE/results/models/ZINBAE_20260226_195349_noconv_layers752_ep141.pt"
MODEL_PATH = "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff0.000.pt"
# # # # MODEL_PATH = "AE/results/models/ZINBAE_20260227_153016_noconv_layers752_ep141.pt"
# # # MODEL_PATH = "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff1.000.pt"

# Preprocessing parameters 
FEATURES = ['Centr']
BIN_SIZE = 19
MOVING_AVERAGE = True
DATA_POINT_LENGTH = 2000
STEP_SIZE = 894

# Training parameters 
BATCH_SIZE = 128
NOISE_LEVEL = 0.15
PI_THRESHOLD = 0.7
MASKED_RECON_WEIGHT = 0.00872  # gamma
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 1e-5  # alpha


# Data caching options
USE_CACHED_DATA = True  # Set to True after first run to use cached data with correct parameters

# Output directory for optional plots and for the noise sweep CSV
OUTPUT_DIR = "AE/results/final"
RESULTS_CSV_PATH = "AE/results/noise_sweep_metrics.csv"
PLOT_SWEEP = False
PROCESSED_DATA_DIR = "Data/processed_data"  # Where preprocessed data will be cached
# ================================================


def generate_cache_filename(
    train_chroms,
    val_chroms,
    test_chroms,
    features,
    bin_size,
    moving_average,
    data_point_length,
    step_size,
):
    """Generate a cache filename keyed by all preprocessing/split settings."""
    params = {
        'train_chroms': sorted(train_chroms),
        'val_chroms': sorted(val_chroms),
        'test_chroms': sorted(test_chroms),
        'features': sorted(features),
        'bin_size': bin_size,
        'moving_average': moving_average,
        'data_point_length': data_point_length,
        'step_size': step_size,
    }

    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    features_str = "_".join(sorted(features))
    return (
        f"{features_str}_bin{bin_size}_ma{moving_average}_len{data_point_length}"
        f"_step{step_size}_{params_hash}.npz"
    )


def load_or_preprocess_data(
    input_folder,
    train_chroms,
    val_chroms,
    test_chroms,
    features,
    bin_size,
    moving_average,
    preprocessing_length,
    step_size,
    use_cache=True,
    cache_dir="Data/processed_data",
):
    """Load preprocessed train/val/test sets from cache or preprocess and cache them."""
    os.makedirs(cache_dir, exist_ok=True)

    cache_filename = generate_cache_filename(
        train_chroms,
        val_chroms,
        test_chroms,
        features,
        bin_size,
        moving_average,
        preprocessing_length,
        step_size,
    )
    cache_path = os.path.join(cache_dir, cache_filename)
    cache_meta_path = cache_path.replace('.npz', '_metadata.json')

    if use_cache and os.path.exists(cache_path) and os.path.exists(cache_meta_path):
        print(f"\n{'='*50}")
        print("LOADING CACHED SPLIT DATA")
        print(f"{'='*50}")
        print(f"Loading arrays from: {cache_path}")

        cached_arrays = np.load(cache_path)
        train_set = cached_arrays['train_set']
        val_set = cached_arrays['val_set']
        test_set = cached_arrays['test_set']

        with open(cache_meta_path, 'r') as f:
            metadata_by_split = json.load(f)

        print(f"Train shape: {train_set.shape}")
        print(f"Val shape:   {val_set.shape}")
        print(f"Test shape:  {test_set.shape}")

        return train_set, val_set, test_set, metadata_by_split

    print(f"\n{'='*50}")
    print("PREPROCESSING DATA")
    print(f"{'='*50}")
    print("Cached split data not found or caching disabled. Preprocessing from scratch...")
    print(f"Train chromosomes: {train_chroms}")
    print(f"Val chromosomes:   {val_chroms}")
    print(f"Test chromosomes:  {test_chroms}")

    train_set, val_set, test_set, train_metadata, val_metadata, test_metadata, _, _, _ = preprocess_with_split(
        input_folder=input_folder,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
        features=features,
        bin_size=bin_size,
        moving_average=moving_average,
        data_point_length=preprocessing_length,
        step_size=step_size,
    )

    metadata_by_split = {
        'train': train_metadata,
        'val': val_metadata,
        'test': test_metadata,
    }

    print(f"\nSaving preprocessed split data to cache: {cache_path}")
    np.savez_compressed(
        cache_path,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
    )
    with open(cache_meta_path, 'w') as f:
        json.dump(metadata_by_split, f, indent=2)
    print("Cached split arrays and metadata saved successfully!")

    return train_set, val_set, test_set, metadata_by_split


def to_serializable_metrics(metrics):
    """Convert numpy scalar values to JSON/CSV-safe Python scalars."""
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            serializable[key] = float(value)
        else:
            serializable[key] = value
    return serializable


def extract_raw_counts_from_dataloader(dataloader, chrom):
    """Extract unmasked raw counts from the evaluation dataloader dataset."""
    raw_count_idx = 3 if chrom else 2
    raw_counts = dataloader.dataset.tensors[raw_count_idx]
    return raw_counts.detach().cpu().numpy()


def save_rows_to_csv(rows, output_path):
    """Save a list of metric dicts to CSV with stable column ordering."""
    if not rows:
        print("No rows to save. Skipping CSV write.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base_columns = [
        'model_path',
        'trained_noise_level',
        'eval_noise_level',
        'split',
        'n_samples',
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    metric_columns = all_keys.difference(base_columns)
    preferred_metric_order = [
        'total_loss',
        'zinb_nll',
        'mse',
        'mae',
        'mae_sd',
        'r2',
        'masked_loss',
        'pi_zero',
        'pi_non_zero',
        'mu_zero',
        'mu_non_zero',
        'theta',
        'kl_loss',
        'reg_loss',
    ]

    ordered_metrics = [m for m in preferred_metric_order if m in metric_columns]
    ordered_metrics.extend(sorted(metric_columns.difference(ordered_metrics)))
    fieldnames = base_columns + ordered_metrics

    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nNoise sweep metrics saved to: {output_path}")


def evaluate_split_for_noise(model, split_name, split_set, noise_level, chrom, chrom_embedding):
    """Run one split/noise evaluation and return a row for CSV output."""
    split_dataloader = dataloader_from_array(
        split_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        zinb=True,
        chrom=chrom,
        sample_fraction=1.0,
        denoise_percentage=noise_level,
    )

    predictions, _, split_metrics, all_mu_raw, all_theta, all_pi, all_raw_counts, all_masks = test(
        model=model,
        dataloader=split_dataloader,
        chrom=chrom,
        chrom_embedding=chrom_embedding,
        plot=PLOT_SWEEP,
        denoise_percent=noise_level,
        gamma=MASKED_RECON_WEIGHT,
        pi_threshold=PI_THRESHOLD,
        regularizer=REGULARIZER,
        alpha=REGULARIZATION_WEIGHT,
        eval_mode=f"noise_sweep/{split_name}",
        name=f"noise{noise_level:.3f}_{split_name}",
    )

    raw_counts = extract_raw_counts_from_dataloader(split_dataloader, chrom=chrom)
    if raw_counts.shape != predictions.shape:
        raise ValueError(
            f"Shape mismatch for MSE calculation on split={split_name}, noise={noise_level}: "
            f"raw_counts={raw_counts.shape}, predictions={predictions.shape}"
        )

    mse = float(np.mean((raw_counts - predictions) ** 2))
    # Per-sample MAE to capture mean and spread across windows.
    abs_diff = np.abs(raw_counts - predictions)
    per_sample_mae = abs_diff.reshape(abs_diff.shape[0], -1).mean(axis=1)
    mae_mean = float(per_sample_mae.mean())
    mae_sd = float(per_sample_mae.std())

    # Compute additional ZINB parameter metrics
    # Flatten arrays for easier processing
    raw_flat = all_raw_counts.flatten()
    mu_flat = all_mu_raw.flatten()
    theta_flat = all_theta.flatten()
    pi_flat = all_pi.flatten()
    
    # Split by zero/non-zero raw counts
    zero_mask = raw_flat == 0
    non_zero_mask = raw_flat > 0
    
    pi_zero = float(pi_flat[zero_mask].mean()) if zero_mask.sum() > 0 else 0.0
    pi_non_zero = float(pi_flat[non_zero_mask].mean()) if non_zero_mask.sum() > 0 else 0.0
    mu_zero = float(mu_flat[zero_mask].mean()) if zero_mask.sum() > 0 else 0.0
    mu_non_zero = float(mu_flat[non_zero_mask].mean()) if non_zero_mask.sum() > 0 else 0.0
    theta_mean = float(theta_flat.mean())

    row = {
        'model_path': MODEL_PATH,
        'trained_noise_level': float(NOISE_LEVEL),
        'eval_noise_level': float(noise_level),
        'split': split_name,
        'n_samples': int(len(split_dataloader.dataset)),
        'mse': mse,
    }
    row.update(to_serializable_metrics(split_metrics))
    row['mae'] = mae_mean
    row['mae_sd'] = mae_sd
    row['pi_zero'] = pi_zero
    row['pi_non_zero'] = pi_non_zero
    row['mu_zero'] = mu_zero
    row['mu_non_zero'] = mu_non_zero
    row['theta'] = theta_mean
    return row


def load_model_and_evaluate_noise_sweep():
    """
    Load a trained model and evaluate train/val/test performance over noise_levels.
    Saves one CSV with one row per (split, noise_level).
    """
    print("="*50)
    print("LOADING TRAINED MODEL")
    print("="*50)

    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']

    print("\nModel architecture from checkpoint:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    seq_len = model_config.get('seq_length', DATA_POINT_LENGTH)
    preprocessing_length = DATA_POINT_LENGTH
    if preprocessing_length != seq_len:
        print(
            f"\nWARNING: preprocessing_length={preprocessing_length} does not match "
            f"checkpoint seq_length={seq_len}. Using checkpoint seq_length."
        )
        preprocessing_length = seq_len

    print("\nEvaluation configuration:")
    print(f"  features: {FEATURES}")
    print(f"  bin_size: {BIN_SIZE}")
    print(f"  moving_average: {MOVING_AVERAGE}")
    print(f"  data_point_length: {preprocessing_length}")
    print(f"  step_size: {STEP_SIZE}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  trained_noise_level: {NOISE_LEVEL}")
    print(f"  eval_noise_levels: {noise_levels}")
    print(f"  output_csv: {RESULTS_CSV_PATH}")

    chrom = 'Chr' in FEATURES
    chrom_embedding = ChromosomeEmbedding() if chrom else None

    train_set, val_set, test_set, _ = load_or_preprocess_data(
        input_folder=INPUT_FOLDER,
        train_chroms=train_chromosomes,
        val_chroms=val_chromosomes,
        test_chroms=test_chromosomes,
        features=FEATURES,
        bin_size=BIN_SIZE,
        moving_average=MOVING_AVERAGE,
        preprocessing_length=preprocessing_length,
        step_size=STEP_SIZE,
        use_cache=USE_CACHED_DATA,
        cache_dir=PROCESSED_DATA_DIR,
    )

    split_data = {
        'train': train_set,
        'val': val_set,
        'test': test_set,
    }

    print("\nLoaded split sizes:")
    for split_name, split_set in split_data.items():
        if split_set is None:
            print(f"  {split_name}: None")
        else:
            print(f"  {split_name}: {len(split_set)} windows")

    print("\n" + "="*50)
    print("INITIALIZING MODEL")
    print("="*50)
    model = ZINBAE(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully!")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_rows = []
    for noise_level in noise_levels:
        print("\n" + "="*50)
        print(f"EVALUATING NOISE LEVEL: {noise_level}")
        print("="*50)

        for split_name, split_set in split_data.items():
            if split_set is None or len(split_set) == 0:
                print(f"Skipping split '{split_name}': no data")
                continue

            print(f"\nEvaluating split='{split_name}' at noise={noise_level}")
            split_row = evaluate_split_for_noise(
                model=model,
                split_name=split_name,
                split_set=split_set,
                noise_level=noise_level,
                chrom=chrom,
                chrom_embedding=chrom_embedding,
            )
            results_rows.append(split_row)

            print("  Metrics summary:")
            print(f"    total_loss:   {split_row.get('total_loss'):.6f}")
            print(f"    zinb_nll:     {split_row.get('zinb_nll'):.6f}")
            print(f"    mse:          {split_row.get('mse'):.6f}")
            print(f"    mae:          {split_row.get('mae'):.6f}")
            print(f"    r2:           {split_row.get('r2'):.6f}")
            print(f"    masked_loss:  {split_row.get('masked_loss', 0.0):.6f}")
            print(f"    pi_zero:      {split_row.get('pi_zero'):.6f}")
            print(f"    pi_non_zero:  {split_row.get('pi_non_zero'):.6f}")
            print(f"    mu_zero:      {split_row.get('mu_zero'):.6f}")
            print(f"    mu_non_zero:  {split_row.get('mu_non_zero'):.6f}")
            print(f"    theta:        {split_row.get('theta'):.6f}")

    save_rows_to_csv(results_rows, RESULTS_CSV_PATH)
    return results_rows


if __name__ == "__main__":
    load_model_and_evaluate_noise_sweep()
