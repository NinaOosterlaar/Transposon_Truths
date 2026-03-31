import os, sys
import gc
import torch
import numpy as np
import hashlib
import json
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess_with_split
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import test
from AE.reconstruct_output import OutputReconstructor
from AE.plotting.plot_mu_offset import plot_pi_correlation_heatmaps, plot_pi_tracks_per_chromosome

# ── Chromosome splits ──────────────────────────────────────────────────────────
train_chroms = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
validation_chroms = ['ChrVIII', 'ChrXIV', 'ChrXV']
test_chroms = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']

# ── Models ─────────────────────────────────────────────────────────────────────
# Keys are used as directory names and plot labels; must be filesystem-safe.
MODELS = {
    'muoff0': "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff0.000.pt",
    'muoff1': "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff1.000.pt",
    'muoff2': "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff2.000.pt",
    'muoff4': "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff4.000.pt",
    'muoff5': "AE/results/models/ZINBAE_layers752_ep141_noise0.150_muoff5.000.pt",
}

# ── Preprocessing parameters ───────────────────────────────────────────────────
INPUT_FOLDER = "Data/combined_strains"
FEATURES = ['Centr']
BIN_SIZE = 19
MOVING_AVERAGE = True
DATA_POINT_LENGTH = 2000
STEP_SIZE = 894

# ── Training / inference parameters ───────────────────────────────────────────
BATCH_SIZE = 128
NOISE_LEVEL = 0.15
PI_THRESHOLD = 0.7
MASKED_RECON_WEIGHT = 0.00872  # gamma
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 1e-5  # alpha

# ── Output paths ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "AE/results/mu_offset"
PROCESSED_DATA_DIR = "Data/processed_data"
RECONSTRUCTION_BASE_DIR = "Data/reconstruction/mu_offset"


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def _generate_cache_filename(split_name):
    """Generate a unique cache filename for a split, tied to all preprocessing parameters."""
    params = {
        'train_chroms': sorted(train_chroms),
        'val_chroms': sorted(validation_chroms),
        'test_chroms': sorted(test_chroms),
        'features': sorted(FEATURES),
        'bin_size': BIN_SIZE,
        'moving_average': MOVING_AVERAGE,
        'data_point_length': DATA_POINT_LENGTH,
        'step_size': STEP_SIZE,
        'split': split_name,
    }
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    features_str = "_".join(sorted(FEATURES))
    return f"{features_str}_bin{BIN_SIZE}_ma{MOVING_AVERAGE}_len{DATA_POINT_LENGTH}_step{STEP_SIZE}_{params_hash}.npy"


def load_or_preprocess_all_splits(use_cache=True):
    """
    Preprocess train / val / test splits, caching each separately.

    Returns dict: {'train': (data, metadata), 'val': (...), 'test': (...)}
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Build cache paths for every split
    cache = {}
    for split_name in ('train', 'val', 'test'):
        fname = _generate_cache_filename(split_name)
        cache[split_name] = {
            'data': os.path.join(PROCESSED_DATA_DIR, fname),
            'meta': os.path.join(PROCESSED_DATA_DIR, fname.replace('.npy', '_metadata.json')),
        }

    all_cached = all(
        os.path.exists(c['data']) and os.path.exists(c['meta'])
        for c in cache.values()
    )

    if use_cache and all_cached:
        print("Loading all splits from cache...")
        result = {}
        for split_name, paths in cache.items():
            data = np.load(paths['data'])
            with open(paths['meta'], 'r') as f:
                meta = json.load(f)
            result[split_name] = (data, meta)
            print(f"  {split_name}: shape={data.shape}, windows={len(meta)}")
        return result

    print("Cache miss — preprocessing from scratch...")
    train_set, val_set, test_set, train_meta, val_meta, test_meta, _, _, _ = preprocess_with_split(
        input_folder=INPUT_FOLDER,
        train_chroms=train_chroms,
        val_chroms=validation_chroms,
        test_chroms=test_chroms,
        features=FEATURES,
        bin_size=BIN_SIZE,
        moving_average=MOVING_AVERAGE,
        data_point_length=DATA_POINT_LENGTH,
        step_size=STEP_SIZE,
    )

    splits_out = {
        'train': (train_set, train_meta),
        'val':   (val_set,   val_meta),
        'test':  (test_set,  test_meta),
    }
    for split_name, (data, meta) in splits_out.items():
        np.save(cache[split_name]['data'], data)
        with open(cache[split_name]['meta'], 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Cached {split_name}: shape={data.shape}")

    return splits_out


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_model(model_path):
    """Load a ZINBAE checkpoint. Returns (model, model_config)."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    model = ZINBAE(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_config


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference_and_save(model, data, metadata, model_label, split_name):
    """
    Run inference on one split for one model and save per-chromosome CSV files.

    Output directory: RECONSTRUCTION_BASE_DIR/<model_label>/<split_name>/
    Skips inference if output directory already contains CSV files.

    Returns the output directory path.
    """
    out_dir = os.path.join(RECONSTRUCTION_BASE_DIR, model_label, split_name)

    # Skip if already computed
    already_done = os.path.isdir(out_dir) and any(
        fname.endswith('.csv')
        for _, _, files in os.walk(out_dir)
        for fname in files
    )
    if already_done:
        print(f"  Skipping {model_label}/{split_name} — found existing CSVs at {out_dir}")
        return out_dir

    chrom = 'Chr' in FEATURES
    chrom_embedding = ChromosomeEmbedding() if chrom else None

    dataloader = dataloader_from_array(
        data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        zinb=True,
        chrom=chrom,
        sample_fraction=1.0,
        denoise_percentage=NOISE_LEVEL,
    )

    predictions, _, _, mu_raw, theta, pi, _, _ = test(
        model=model,
        dataloader=dataloader,
        chrom=chrom,
        chrom_embedding=chrom_embedding,
        plot=False,
        denoise_percent=NOISE_LEVEL,
        gamma=MASKED_RECON_WEIGHT,
        pi_threshold=PI_THRESHOLD,
        regularizer=REGULARIZER,
        alpha=REGULARIZATION_WEIGHT,
        eval_mode="mu_offset",
        name="",
    )

    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    reconstructor = OutputReconstructor(metadata_path)
    df = reconstructor.reconstruct_to_dataframe(
        predictions,
        aggregation='mean',
        include_uncertainty=True,
        mu_raw=mu_raw,
        theta=theta,
        pi=pi,
    )
    reconstructor.save_as_csv(df, out_dir, split_by_chromosome=True)
    print(f"  Saved {model_label}/{split_name} → {out_dir}")
    return out_dir


# ── Correlation analysis ───────────────────────────────────────────────────────

def _load_pi_for_model_split(model_label, split_name):
    """
    Load pi values from all per-chromosome CSVs for a given model + split.

    Returns DataFrame with columns: dataset, chromosome, position, pi.
    """
    split_dir = os.path.join(RECONSTRUCTION_BASE_DIR, model_label, split_name)
    frames = []
    for dataset_name in os.listdir(split_dir):
        dataset_dir = os.path.join(split_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for fname in os.listdir(dataset_dir):
            if not fname.endswith('.csv'):
                continue
            chrom = fname.replace('.csv', '')
            df = pd.read_csv(os.path.join(dataset_dir, fname), usecols=['position', 'pi'])
            df['dataset'] = dataset_name
            df['chromosome'] = chrom
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['dataset', 'chromosome', 'position', 'pi'])
    return pd.concat(frames, ignore_index=True)


def compute_pi_correlations(model_labels, split_names):
    """
    Compute pairwise Pearson and Spearman correlations of pi values across models.

    Correlations are computed **per strain** (to avoid combining strains), then
    aggregated across strains as mean ± std.

    Returns:
        dict: split_name -> {(label_i, label_j): {'pearson': [r_per_strain], 'spearman': [...]}}
        An extra key 'all' pools all three splits before computing correlations.
    """
    print("Loading pi reconstructions for correlation analysis...")
    # Cache loaded data to avoid reading files repeatedly
    pi_data = {label: {} for label in model_labels}
    for label in model_labels:
        for split_name in split_names:
            pi_data[label][split_name] = _load_pi_for_model_split(label, split_name)
        # 'all': concatenate across splits (per strain, so strains still separate)
        parts = [pi_data[label][s] for s in split_names if not pi_data[label][s].empty]
        pi_data[label]['all'] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    results = {}
    for split_key in split_names + ['all']:
        results[split_key] = {}
        for label_i, label_j in combinations(model_labels, 2):
            df_i = pi_data[label_i][split_key].rename(columns={'pi': 'pi_i'})
            df_j = pi_data[label_j][split_key].rename(columns={'pi': 'pi_j'})
            merged = df_i.merge(df_j, on=['dataset', 'chromosome', 'position'], how='inner')
            if merged.empty:
                continue

            pearson_rs, spearman_rs = [], []
            for _dataset, grp in merged.groupby('dataset'):
                x = grp['pi_i'].values
                y = grp['pi_j'].values
                if len(x) < 3:
                    continue
                pearson_rs.append(pearsonr(x, y).statistic)
                spearman_rs.append(spearmanr(x, y).statistic)

            results[split_key][(label_i, label_j)] = {
                'pearson':  pearson_rs,
                'spearman': spearman_rs,
            }

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Preprocess (or load from cache)
    print("=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    splits = load_or_preprocess_all_splits(use_cache=True)

    # 2. Inference loop: 5 models × 3 splits
    print("\n" + "=" * 60)
    print("STEP 2: Running inference")
    print("=" * 60)
    model_labels = list(MODELS.keys())
    split_names = ['train', 'val', 'test']

    for label, model_path in MODELS.items():
        print(f"\n--- Model: {label} ---")
        model, _ = load_model(model_path)
        for split_name in split_names:
            data, metadata = splits[split_name]
            run_inference_and_save(model, data, metadata, label, split_name)
        del model
        gc.collect()

    # 3. Correlation analysis
    print("\n" + "=" * 60)
    print("STEP 3: Computing π correlations")
    print("=" * 60)
    corr_results = compute_pi_correlations(model_labels, split_names)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save raw numbers
    serialisable = {
        split: {f"{a}_vs_{b}": v for (a, b), v in pairs.items()}
        for split, pairs in corr_results.items()
    }
    corr_json_path = os.path.join(OUTPUT_DIR, "pi_correlations.json")
    with open(corr_json_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"Correlation values saved to: {corr_json_path}")

    # 4. Plots
    print("\n" + "=" * 60)
    print("STEP 4: Generating plots")
    print("=" * 60)
    plot_pi_correlation_heatmaps(corr_results, model_labels, OUTPUT_DIR)

    # Build merged DataFrame for pi track plots (all splits pooled per model)
    print("Building merged pi DataFrame for track plots...")
    merged = None
    for label in model_labels:
        parts = [_load_pi_for_model_split(label, s) for s in split_names]
        df_label = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        df_label = df_label.rename(columns={'pi': f'pi_{label}'})
        merged = df_label if merged is None else merged.merge(
            df_label, on=['dataset', 'chromosome', 'position'], how='inner'
        )

    if merged is not None and len(merged) > 0:
        for chrom in sorted(merged['chromosome'].unique()):
            plot_pi_tracks_per_chromosome(merged, chrom, model_labels, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()