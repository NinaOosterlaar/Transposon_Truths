import os, sys
import csv
import itertools
import random
import shutil
import torch
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess_with_split
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import test

# ========== CONFIGURATION ==========

train_chroms = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
validation_chroms = ['ChrVIII', 'ChrXIV', 'ChrXV']
test_chroms = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']

MODEL_PATH = "AE/results/models/ZINBAE_20260226_195349_noconv_layers752_ep141.pt"

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

STRAIN = "yEK23"
RAW_DATA_DIR = "Data/distances_with_zeros_new"   # source data
TEMP_DIR = "Data/temp_saturation"                # temporary combined datasets
OUTPUT_DIR = "AE/results/saturation"             # where CSV results are saved

MAX_COMBINATIONS = 8   # max combos sampled per group size
RANDOM_SEED = 42

# ====================================


def get_dataset_folders(strain):
    """Return sorted list of replicate folder names for the given strain."""
    strain_dir = os.path.join(RAW_DATA_DIR, f"strain_{strain}")
    folders = sorted([
        f for f in os.listdir(strain_dir)
        if os.path.isdir(os.path.join(strain_dir, f)) and not f.startswith('.')
    ])
    return strain_dir, folders


def compute_saturation(dataset_folders, strain_dir):
    """Fraction of genome positions with at least one non-zero insertion value
    across all given dataset folders (pooled over all chromosomes).
    """
    # Discover all chromosomes from the first folder
    first_folder = os.path.join(strain_dir, dataset_folders[0])
    chroms = [
        fname.split("_")[0]
        for fname in os.listdir(first_folder)
        if fname.endswith("_distances.csv") and not fname.startswith("ChrM")
    ]

    total_positions = 0
    nonzero_positions = 0

    for chrom in chroms:
        all_pos = set()
        nonzero_pos = set()
        for folder in dataset_folders:
            csv_path = os.path.join(strain_dir, folder, f"{chrom}_distances.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            positions = df['Position'].values
            values = df['Value'].values
            all_pos.update(positions.tolist())
            nonzero_pos.update(positions[values != 0].tolist())

        total_positions += len(all_pos)
        nonzero_positions += len(nonzero_pos)

    if total_positions == 0:
        return 0.0
    return nonzero_positions / total_positions


def prepare_combo_dir(dataset_folders, strain_dir, combo_name):
    """Prepare the temp directory for a combination.

    If there is only one dataset, symlink directly to the existing folder to
    avoid any file I/O.  Otherwise combine by averaging non-zero values per
    position using vectorised pandas operations.

    Returns the path to the directory that was created/symlinked.
    """
    combo_out = os.path.join(TEMP_DIR, f"strain_{STRAIN}", combo_name)

    # --- Single dataset: hard-link individual CSV files, no combining needed ---
    if len(dataset_folders) == 1:
        src_dir = os.path.abspath(os.path.join(strain_dir, dataset_folders[0]))
        os.makedirs(combo_out, exist_ok=True)
        for fname in os.listdir(src_dir):
            if fname.endswith(".csv"):
                os.link(os.path.join(src_dir, fname), os.path.join(combo_out, fname))
        return combo_out

    # --- Multiple datasets: vectorised combine ---
    os.makedirs(combo_out, exist_ok=True)

    first_folder = os.path.join(strain_dir, dataset_folders[0])
    chroms = [
        fname.split("_")[0]
        for fname in os.listdir(first_folder)
        if fname.endswith("_distances.csv") and not fname.startswith("ChrM")
    ]

    for chrom in sorted(chroms):
        frames = []
        for folder in dataset_folders:
            csv_path = os.path.join(strain_dir, folder, f"{chrom}_distances.csv")
            if os.path.exists(csv_path):
                frames.append(pd.read_csv(csv_path))

        if not frames:
            continue

        # Stack all replicates; keep auxiliary columns from first occurrence per position
        all_data = pd.concat(frames, ignore_index=True)

        # For each position, average the non-zero Values; keep first-seen distances
        aux = (
            all_data.groupby('Position', sort=True)
            [['Nucleosome_Distance', 'Centromere_Distance']]
            .first()
        )

        def _avg_nonzero(series):
            nz = series[series != 0]
            return nz.mean() if len(nz) > 0 else 0.0

        combined_values = (
            all_data.groupby('Position', sort=True)['Value']
            .agg(_avg_nonzero)
        )

        result = aux.copy()
        result['Value'] = combined_values
        result = result.reset_index()[['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance']]

        out_path = os.path.join(combo_out, f"{chrom}_distances.csv")
        result.to_csv(out_path, index=False)

    return combo_out


def load_model():
    """Load the trained ZINBAE model."""
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    model = ZINBAE(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")
    return model


def _eval_split(model, data_split, use_chrom, chrom_embedding, split_label):
    """Run test() on a single preprocessed split array. Returns prefixed metrics dict,
    or None if the split is empty."""
    if len(data_split) == 0:
        print(f"  Warning: empty {split_label} split, skipping.")
        return None

    dataloader = dataloader_from_array(
        data_split,
        batch_size=BATCH_SIZE,
        shuffle=False,
        zinb=True,
        chrom=use_chrom,
        sample_fraction=1.0,
        denoise_percentage=NOISE_LEVEL,
    )

    _, _, metrics, _, _, _ = test(
        model=model,
        dataloader=dataloader,
        chrom=use_chrom,
        chrom_embedding=chrom_embedding,
        plot=False,
        denoise_percent=NOISE_LEVEL,
        gamma=MASKED_RECON_WEIGHT,
        pi_threshold=PI_THRESHOLD,
        regularizer=REGULARIZER,
        alpha=REGULARIZATION_WEIGHT,
        eval_mode="saturation",
        name="",
    )

    return {f"{split_label}_{k}": v for k, v in metrics.items()}


def evaluate_combination(model):
    """Preprocess data from TEMP_DIR and evaluate the model on train, validation,
    and test chromosome splits.

    TEMP_DIR must already contain the combined dataset in the expected folder structure.
    Returns a merged metrics dict with keys prefixed by split name, or None if all
    splits are empty.
    """
    use_chrom = 'Chr' in FEATURES
    chrom_embedding = ChromosomeEmbedding() if use_chrom else None

    train_set, val_set, test_set, _, _, _, _, _, _ = preprocess_with_split(
        input_folder=TEMP_DIR,
        train_chroms=train_chroms,
        val_chroms=validation_chroms,
        test_chroms=test_chroms,
        features=FEATURES,
        bin_size=BIN_SIZE,
        moving_average=MOVING_AVERAGE,
        data_point_length=DATA_POINT_LENGTH,
        step_size=STEP_SIZE,
    )

    combined_metrics = {}
    for split_label, data_split in [('train', train_set), ('val', val_set), ('test', test_set)]:
        result = _eval_split(model, data_split, use_chrom, chrom_embedding, split_label)
        if result is not None:
            combined_metrics.update(result)

    return combined_metrics if combined_metrics else None


def run_saturation_test():
    rng = random.Random(RANDOM_SEED)

    strain_dir, all_folders = get_dataset_folders(STRAIN)
    n_datasets = len(all_folders)
    print(f"Found {n_datasets} datasets for strain {STRAIN}:")
    for f in all_folders:
        print(f"  {f}")

    model = load_model()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    all_results = []  # one row per evaluated combination

    for k in range(1, n_datasets + 1):
        all_combos = list(itertools.combinations(range(n_datasets), k))
        selected = rng.sample(all_combos, min(MAX_COMBINATIONS, len(all_combos)))

        print(f"\n{'='*60}")
        print(f"k={k}: testing {len(selected)} combination(s) out of {len(all_combos)} possible")
        print(f"{'='*60}")

        for combo_idx, combo in enumerate(selected):
            folders_in_combo = [all_folders[i] for i in combo]
            combo_name = f"k{k}_combo{combo_idx}"

            print(f"\n  [{combo_idx+1}/{len(selected)}] {folders_in_combo}")

            # Compute saturation before combining
            saturation = compute_saturation(folders_in_combo, strain_dir)
            print(f"  Saturation: {saturation:.4f}")

            # Prepare temp folder (symlink for single dataset, combine otherwise)
            combo_out = prepare_combo_dir(folders_in_combo, strain_dir, combo_name)

            # Evaluate
            metrics = evaluate_combination(model)

            # Remove this combo's temp entry immediately
            shutil.rmtree(combo_out, ignore_errors=True)

            if metrics is None:
                continue

            row = {
                'n_datasets': k,
                'combo_idx': combo_idx,
                'folders': "|".join(folders_in_combo),
                'saturation': saturation,
            }
            row.update(metrics)
            all_results.append(row)
            print(f"  Metrics: { {mk: round(mv, 6) for mk, mv in metrics.items()} }")

    # --- Save detailed results (one row per combination) ---
    detailed_path = os.path.join(OUTPUT_DIR, "saturation_results_detailed.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(detailed_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nDetailed results saved to: {detailed_path}")

    # --- Aggregate per group size: mean ± std ---
    _non_metric = {'n_datasets', 'combo_idx', 'folders', 'saturation'}
    metric_keys = [k for k in (all_results[0].keys() if all_results else [])
                   if k not in _non_metric]

    aggregated = []
    for k in range(1, n_datasets + 1):
        group = [r for r in all_results if r['n_datasets'] == k]
        if not group:
            continue

        row = {'n_datasets': k, 'n_combos': len(group)}

        sats = [r['saturation'] for r in group]
        row['saturation_mean'] = float(np.mean(sats))
        row['saturation_std'] = float(np.std(sats))

        for key in metric_keys:
            vals = [r[key] for r in group if key in r]
            if vals:
                row[f'{key}_mean'] = float(np.mean(vals))
                row[f'{key}_std'] = float(np.std(vals))

        aggregated.append(row)

    agg_path = os.path.join(OUTPUT_DIR, "saturation_results_aggregated.csv")
    if aggregated:
        fieldnames = list(aggregated[0].keys())
        with open(agg_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregated)
        print(f"Aggregated results saved to: {agg_path}")

    # Clean up temp strain folder if empty
    strain_temp = os.path.join(TEMP_DIR, f"strain_{STRAIN}")
    if os.path.exists(strain_temp) and not os.listdir(strain_temp):
        shutil.rmtree(strain_temp, ignore_errors=True)

    print("\nSaturation test complete.")
    return aggregated


if __name__ == "__main__":
    run_saturation_test()