import os, sys
import gc
import torch
import json
import numpy as np
from datetime import datetime
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess_with_split
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import train, test
from AE.reconstruct_output import OutputReconstructor


SPLIT_ON = 'Chrom'
TRAIN_CHROM = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
TEST_CHROM = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']
VAL_CHROM = ['ChrVIII', 'ChrXIV', 'ChrXV']  # No validation set 
# Train model on all the data (no chromosome split)
# TRAIN_CHROM = ['ChrI', 'ChrII', 'ChrIII', 'ChrIV', 'ChrV', 'ChrVI', 'ChrVII', 'ChrVIII', 
#                'ChrIX', 'ChrX', 'ChrXI', 'ChrXII', 'ChrXIII', 'ChrXIV', 'ChrXV', 'ChrXVI']
# TEST_CHROM = []


# Preprocessings

# INPUT_FOLDER = "Data/combined_strains"
# FEATURES = ['Nucl']
# BIN_SIZE = 17
# MOVING_AVERAGE = False
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = int(DATA_POINT_LENGTH*0.391)
# SAMPLE_FRACTION = 1.0

# print("MOVING AVERAGE:", MOVING_AVERAGE)

# USE_CONV = True
# CONV_CHANNEL = 48
# POOL_SIZE = 6
# POOLING_OPERATION = 'max'
# KERNEL_SIZE = 3
# PADDING = 'same'
# STRIDE = 1

# EPOCHS = 92
# BATCH_SIZE = 128
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.38
# MASKED_RECON_WEIGHT = 0.127# gamma: weight for masked reconstruction loss
# LEARNING_RATE = 0.00619
# DROPOUT_RATE = 0.219
# LAYERS = [752]
# REGULARIZER = 'none'
# REGULARIZATION_WEIGHT = 1e-4
# MU_OFFSET = 0.0  # Offset added to mu in ZINB loss to prevent zero variance




# TRAIN_CHROM = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
# TEST_CHROM = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']
# VAL_CHROM = []  # No validation set 

# INPUT_FOLDER = "Data/combined_strains"
# FEATURES = ['Centr']
# BIN_SIZE = 1
# MOVING_AVERAGE = False
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = 500
# SAMPLE_FRACTION = 1.0

# USE_CONV = False
# CONV_CHANNEL = 85
# POOL_SIZE = 8
# POOLING_OPERATION = 'max'
# KERNEL_SIZE = 7
# PADDING = 'same'
# STRIDE = 1

# EPOCHS = 30
# BATCH_SIZE = 32
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.7
# MASKED_RECON_WEIGHT = 0.001  # gamma: weight for masked reconstruction loss
# LEARNING_RATE = 1e-5
# DROPOUT_RATE = 0
# LAYERS = [1600]
# REGULARIZER = 'l2'
# REGULARIZATION_WEIGHT = 1e-5
# MU_OFFSET = 0.0

# PLOT = True
# SAVE_MODEL = True
# MODEL_SAVE_DIR = "AE/results/models"
# MODEL_PATH_LOAD = "AE/results/models/ZINBAE_20260227_153016_noconv_layers752_ep141.pt"
# MODEL_LOAD = False

INPUT_FOLDER = "Data/combined_strains"
FEATURES = ['Centr']
BIN_SIZE = 19
MOVING_AVERAGE = True
DATA_POINT_LENGTH = 2000
STEP_SIZE = int(DATA_POINT_LENGTH * 0.45)
SAMPLE_FRACTION = 1.0

USE_CONV = False
CONV_CHANNEL = 85
POOL_SIZE = 8
POOLING_OPERATION = 'max'
KERNEL_SIZE = 7
PADDING = 'same'
STRIDE = 1

EPOCHS = 141
BATCH_SIZE = 128
NOISE_LEVEL = 0.15
PI_THRESHOLD = 0.7
MASKED_RECON_WEIGHT = 0.008  # gamma: weight for masked reconstruction loss
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.008
LAYERS = [752]
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 1e-5
MU_OFFSET = 0.0

PLOT = True
SAVE_MODEL = True
MODEL_SAVE_DIR = "AE/results/models"
MODEL_PATH_LOAD = "AE/results/models/ZINBAE_20260227_153016_noconv_layers752_ep141.pt"
MODEL_LOAD = False


def save_reconstruction_artifacts(
    predictions,
    mu_raw,
    theta,
    pi,
    metadata,
    output_dir,
    aggregation="mean",
):
    """Reconstruct model outputs to genomic coordinates and save compact artifacts."""
    if metadata is None:
        print("Skipping reconstruction: metadata is None")
        return

    if len(predictions) != len(metadata):
        print(
            "Skipping reconstruction: prediction/metadata length mismatch "
            f"({len(predictions)} vs {len(metadata)})"
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved reconstruction metadata to: {output_dir}")
    print(f"  metadata: {metadata_path}")

    print("\nReconstructing genomic coordinates...")
    reconstructor = OutputReconstructor(metadata_path)
    reconstructed_df = reconstructor.reconstruct_to_dataframe(
        predictions,
        aggregation=aggregation,
        include_uncertainty=(theta is not None or pi is not None),
        mu_raw=mu_raw,
        theta=theta,
        pi=pi,
    )

    reconstructor.save_as_csv(reconstructed_df, output_dir, split_by_chromosome=True)
    print(f"Reconstructed genomic data saved to: {output_dir}")
    print("  One CSV per dataset/chromosome with columns:")
    print("  position, reconstruction, mu, pi, theta")



def main_with_datasets(
    train_set,
    val_set,
    test_set,
    features=FEATURES,
    data_point_length=DATA_POINT_LENGTH,
    use_conv=USE_CONV, 
    conv_channel=CONV_CHANNEL, 
    pool_size=POOL_SIZE,
    kernel_size=KERNEL_SIZE,
    padding=PADDING, 
    stride=STRIDE,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    noise_level=NOISE_LEVEL,
    pi_threshold=PI_THRESHOLD,
    masked_recon_weight=MASKED_RECON_WEIGHT, 
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE, 
    layers=LAYERS, 
    regularizer=REGULARIZER, 
    regularization_weight=REGULARIZATION_WEIGHT,
    sample_fraction=SAMPLE_FRACTION,
    mu_offset=MU_OFFSET,
    plot=PLOT,
    eval_on_val=True,
    evaluate_all_splits=False,
    save_model=SAVE_MODEL,
    model_save_dir=MODEL_SAVE_DIR,
    model_load=MODEL_LOAD,
    model_path_load=MODEL_PATH_LOAD,
    train_metadata=None,
    val_metadata=None,
    test_metadata=None,
    save_reconstruction=True,
    reconstruction_output_dir=None,
    ):
    """
    Main training function that accepts pre-made datasets.
    Used by Bayesian optimization to avoid recreating data every trial.
    
    Parameters:
    -----------
    train_set, val_set, test_set : numpy arrays
        Pre-processed datasets
    eval_on_val : bool
        If True, validation metrics are used as the primary eval metrics.
        If False, test metrics are used as the primary eval metrics when available.
    evaluate_all_splits : bool
        If True, evaluate all available eval splits (validation and test).
        Primary returned eval metrics still follow eval_on_val.
    """
    # Initialize model
    if "Chr" in features: chrom = True
    else: chrom = False
    
    # Print shapes of the datasets for debugging
    print(f"\nDataset shapes:")
    print(f"  Train set: {train_set.shape if train_set is not None else 'None'}")
    print(f"  Val set: {val_set.shape if val_set is not None else 'None'}")
    print(f"  Test set: {test_set.shape if test_set is not None else 'None'}")
    
    # Create chromosome embedding once if needed
    chrom_embedding = ChromosomeEmbedding() if chrom else None
    
    # Create train dataloader
    train_dataloader = dataloader_from_array(
        train_set, batch_size=batch_size, shuffle=True, zinb=True, 
        chrom=chrom, sample_fraction=sample_fraction, denoise_percentage=noise_level
    )
    
    # Create evaluation dataloaders for available splits
    eval_dataloaders = {}

    if val_set is not None and len(val_set) > 0:
        print(f"\nCreating validation dataloader:")
        print(f"  Validation set size: {len(val_set)}")
        print(f"  Batch size: {batch_size}")
        val_dataloader = dataloader_from_array(
            val_set, batch_size=batch_size, shuffle=False, zinb=True,
            chrom=chrom, sample_fraction=1.0, denoise_percentage=noise_level
        )
        print(f"  Dataloader created: {len(val_dataloader.dataset)} samples, {len(val_dataloader)} batches")
        eval_dataloaders["VALIDATION"] = val_dataloader

    if test_set is not None and len(test_set) > 0:
        print(f"\nCreating test dataloader:")
        print(f"  Test set size: {len(test_set)}")
        print(f"  Batch size: {batch_size}")
        test_dataloader = dataloader_from_array(
            test_set, batch_size=batch_size, shuffle=False, zinb=True,
            chrom=chrom, sample_fraction=1.0, denoise_percentage=noise_level
        )
        print(f"  Dataloader created: {len(test_dataloader.dataset)} samples, {len(test_dataloader)} batches")
        eval_dataloaders["TEST"] = test_dataloader

    if evaluate_all_splits:
        eval_order = [split for split in ("VALIDATION", "TEST") if split in eval_dataloaders]
    else:
        if eval_on_val and "VALIDATION" in eval_dataloaders:
            eval_order = ["VALIDATION"]
        elif "TEST" in eval_dataloaders:
            eval_order = ["TEST"]
        elif "VALIDATION" in eval_dataloaders:
            eval_order = ["VALIDATION"]
        else:
            eval_order = []
    
    # Calculate feature dimension
    feature_dim = train_dataloader.dataset.tensors[0].shape[2] + 1
    if chrom:
        feature_dim += chrom_embedding.embedding.embedding_dim 
    
    if model_load and os.path.exists(model_path_load):
        print(f"\n{'='*50}")
        print(f"LOADING MODEL FROM: {model_path_load}")
        print(f"{'='*50}\n")
        loaded_dict = torch.load(model_path_load, map_location=torch.device('cpu'), weights_only=False)
        
        # CRITICAL: Use saved model_config, not current parameters!
        model_config = loaded_dict['model_config']
        
        print("Loaded model configuration:")
        for k, v in model_config.items():
            print(f"  {k}: {v}")
        
        zinbae_model = ZINBAE(**model_config)
        zinbae_model.load_state_dict(loaded_dict['model_state_dict'])
        print(f"Model loaded successfully!")
        # No training performed when loading model
        train_metrics = {}
    else:
        # Initialize model
        zinbae_model = ZINBAE(
            seq_length=data_point_length,
            feature_dim=feature_dim,
            layers=layers,
            use_conv=use_conv,
            conv_channels=conv_channel,
            pool_size=pool_size,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dropout=dropout_rate,
            mu_offset=mu_offset
        )
        
        # Free memory from train_set after dataloader is created
        del train_set
        if val_set is not None:
            del val_set
        if test_set is not None:
            del test_set
        gc.collect()
        # Train model
        _, train_metrics = train(
            model=zinbae_model,
            dataloader=train_dataloader,
            num_epochs=epochs,
            pi_threshold=pi_threshold,
            learning_rate=learning_rate,
            regularizer=regularizer,
            alpha=regularization_weight,
            denoise_percent=noise_level,
            gamma=masked_recon_weight,
            name=f"mu_offset{mu_offset:.1f}_noise{noise_level:.3f}",
            chrom=chrom,
            chrom_embedding=chrom_embedding,
            plot=plot,
        )
    

    
    # Free training dataloader memory before evaluation
    del train_dataloader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Evaluate model on selected split(s)
    eval_metrics_by_split = {}
    reconstruction_payloads = {}

    for eval_set_name in eval_order:
        eval_dataloader = eval_dataloaders[eval_set_name]

        print(f"\n{'='*50}")
        print(f"EVALUATING ON {eval_set_name} DATA")
        print(f"Dataset size: {len(eval_dataloader.dataset)}")
        print(f"{'='*50}\n")

        eval_predictions, _, split_eval_metrics, eval_mu_raw, eval_theta, eval_pi = test(
            model=zinbae_model,
            dataloader=eval_dataloader,
            pi_threshold=pi_threshold,
            chrom=chrom,
            chrom_embedding=chrom_embedding,
            plot=plot,
            denoise_percent=noise_level,
            alpha=regularization_weight,
            gamma=masked_recon_weight,
            name=f"mu_offset{mu_offset:.1f}_noise{noise_level:.3f}",
            regularizer=regularizer,
        )

        eval_metrics_by_split[eval_set_name] = split_eval_metrics

        if save_reconstruction and eval_predictions is not None:
            reconstruction_payloads[eval_set_name] = {
                "predictions": eval_predictions,
                "mu_raw": eval_mu_raw,
                "theta": eval_theta,
                "pi": eval_pi,
            }

    if len(eval_order) == 0:
        print(f"\nSkipping evaluation: No evaluation dataset available")

    primary_eval_set_name = eval_order[0] if len(eval_order) > 0 else None
    eval_metrics = eval_metrics_by_split.get(primary_eval_set_name, {})

    output_label = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    
    # Save model before cleanup
    if save_model:
        # Create save directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Generate filename with timestamp and key parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_str = f"conv{conv_channel}" if use_conv else "noconv"
        layers_str = "_".join(map(str, layers))
        model_filename = (
            f"ZINBAE_layers{layers_str}_ep{epochs}"
            f"_noise{noise_level:.3f}_muoff{mu_offset:.3f}.pt"
        )
        model_path = os.path.join(model_save_dir, model_filename)
        output_label = os.path.splitext(model_filename)[0]
        
        # Save model state dict and configuration
        save_dict = {
            'model_state_dict': zinbae_model.state_dict(),
            'model_config': {
                'seq_length': data_point_length,
                'feature_dim': feature_dim,
                'layers': layers,
                'use_conv': use_conv,
                'conv_channels': conv_channel,
                'pool_size': pool_size,
                'kernel_size': kernel_size,
                'padding': padding,
                'stride': stride,
                'dropout': dropout_rate,
                'mu_offset': zinbae_model.mu_offset,  
            },
            'training_config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'noise_level': noise_level,
                'pi_threshold': pi_threshold,
                'masked_recon_weight': masked_recon_weight,
                'regularizer': regularizer,
                'regularization_weight': regularization_weight,
                'features': features,
            },
            'metrics': {
                'train': train_metrics,
                'eval': eval_metrics,
            }
        }

        if primary_eval_set_name is not None:
            save_dict['metrics']['primary_eval_split'] = primary_eval_set_name.lower()

        if len(eval_metrics_by_split) > 0:
            save_dict['metrics']['by_split'] = {
                split_name.lower(): split_metrics
                for split_name, split_metrics in eval_metrics_by_split.items()
            }
        
        torch.save(save_dict, model_path)
        print(f"\n{'='*50}")
        print(f"MODEL SAVED: {model_filename}")
        print(f"Location: {model_path}")
        print(f"{'='*50}\n")

    if save_reconstruction and len(reconstruction_payloads) > 0:
        if reconstruction_output_dir is None:
            run_output_dir = os.path.join("Data", "reconstruction", output_label)
        else:
            run_output_dir = reconstruction_output_dir

        metadata_by_split = {
            "VALIDATION": val_metadata,
            "TEST": test_metadata,
        }
        use_split_subdirs = len(reconstruction_payloads) > 1

        for split_name, payload in reconstruction_payloads.items():
            split_output_dir = run_output_dir
            if use_split_subdirs:
                split_output_dir = os.path.join(run_output_dir, split_name.lower())

            save_reconstruction_artifacts(
                predictions=payload["predictions"],
                mu_raw=payload["mu_raw"],
                theta=payload["theta"],
                pi=payload["pi"],
                metadata=metadata_by_split.get(split_name),
                output_dir=split_output_dir,
                aggregation="mean",
            )
    
    # Final cleanup - delete model and dataloaders to free memory
    del zinbae_model
    del eval_dataloaders
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return train_metrics, eval_metrics


def main(
    input_folder=INPUT_FOLDER,
    features=FEATURES, 
    bin_size=BIN_SIZE, 
    moving_average=MOVING_AVERAGE,
    data_point_length=DATA_POINT_LENGTH, 
    step_size=STEP_SIZE,
    sample_fraction=SAMPLE_FRACTION, 
    train_chroms=TRAIN_CHROM,
    val_chroms=VAL_CHROM,
    test_chroms=TEST_CHROM,
    use_conv=USE_CONV, 
    conv_channel=CONV_CHANNEL, 
    pool_size=POOL_SIZE,
    kernel_size=KERNEL_SIZE,
    padding=PADDING, 
    stride=STRIDE,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    noise_level=NOISE_LEVEL,
    pi_threshold=PI_THRESHOLD,
    masked_recon_weight=MASKED_RECON_WEIGHT, 
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE, 
    layers=LAYERS, 
    regularizer=REGULARIZER, 
    regularization_weight=REGULARIZATION_WEIGHT,
    mu_offset=MU_OFFSET,
    plot=PLOT,
    save_model=SAVE_MODEL,
    model_save_dir=MODEL_SAVE_DIR):
    """
    Main function that handles data preprocessing with explicit chromosome splits.
    
    Parameters:
    -----------
    train_chroms : list
        List of chromosome names for training set
    val_chroms : list
        List of chromosome names for validation set (can be empty)
    test_chroms : list
        List of chromosome names for test set

    Notes:
    ------
    If both validation and test chromosome sets are provided and contain data,
    both splits are evaluated.
    """
    
    # Preprocess data with explicit chromosome split
    print(f"\nPreprocessing with chromosome split:")
    print(f"  Train chromosomes: {train_chroms}")
    print(f"  Val chromosomes: {val_chroms if val_chroms else 'None'}")
    print(f"  Test chromosomes: {test_chroms}")
    
    train_set, val_set, test_set, train_metadata, val_metadata, test_metadata, _, _, _ = preprocess_with_split(
        input_folder=input_folder,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
        features=features,
        bin_size=bin_size,
        moving_average=moving_average,
        data_point_length=data_point_length,
        step_size=step_size
    )
    
    # Debug: Print dataset sizes
    print(f"\nDataset sizes after preprocessing:")
    print(f"  Train: {len(train_set) if train_set is not None else 0}")
    print(f"  Val: {len(val_set) if val_set is not None else 0}")
    print(f"  Test: {len(test_set) if test_set is not None else 0}")
    
    # Determine evaluation strategy:
    # - Primary eval remains validation when available (for compatibility)
    # - When both val/test chromosome splits exist and have data, evaluate both
    has_val = val_set is not None and len(val_set) > 0
    has_test = test_set is not None and len(test_set) > 0
    eval_on_val = has_val
    evaluate_all_splits = bool(val_chroms) and bool(test_chroms) and has_val and has_test

    if evaluate_all_splits:
        eval_strategy = "VALIDATION + TEST"
    elif eval_on_val:
        eval_strategy = "VALIDATION"
    elif has_test:
        eval_strategy = "TEST"
    else:
        eval_strategy = "NONE"
    print(f"  Eval strategy: {eval_strategy}\n")
    
    # Train and evaluate
    train_metrics, eval_metrics = main_with_datasets(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        test_metadata=test_metadata,
        features=features,
        data_point_length=data_point_length,
        use_conv=use_conv,
        conv_channel=conv_channel,
        pool_size=pool_size,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        epochs=epochs,
        batch_size=batch_size,
        noise_level=noise_level,
        pi_threshold=pi_threshold,
        masked_recon_weight=masked_recon_weight,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        layers=layers,
        regularizer=regularizer,
        regularization_weight=regularization_weight,
        sample_fraction=sample_fraction,
        mu_offset=mu_offset,
        plot=plot,
        eval_on_val=eval_on_val,
        evaluate_all_splits=evaluate_all_splits,
        save_model=save_model,
        model_save_dir=model_save_dir,
        save_reconstruction=True,
    )
    
    # Return metrics (useful for Bayesian optimization or logging)
    # For normal usage, you can ignore the return value
    return train_metrics, eval_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise_level",
        type=float,
        default=NOISE_LEVEL,
        help="Denoising percentage used in dataloader/train/test (default: %(default)s).",
    )
    parser.add_argument(
        "--mu_offset",
        type=float,
        default=0.0,
        help="Offset added to global mean when initializing theta_global (default: %(default)s).",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=BIN_SIZE,
        help="Bin size used during preprocessing (default: %(default)s).",
    )
    args = parser.parse_args()

    main(noise_level=args.noise_level, mu_offset=args.mu_offset, bin_size=args.bin_size)