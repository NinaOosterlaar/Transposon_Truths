import os, sys
import gc
import torch
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess_with_split
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import train, test


# Preprocessing
# INPUT_FOLDER = "Data/combined_strains"
# FEATURES = ['Centr']
# BIN_SIZE = 19
# MOVING_AVERAGE = True
# DATA_POINT_LENGTH = 2000
# STEP_SIZE = 894
# SAMPLE_FRACTION = 0.94

# TRAIN_CHROM = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
# TEST_CHROM = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']
# VAL_CHROM = []  # No validation set 

# USE_CONV = False
# CONV_CHANNEL = 85
# POOL_SIZE = 4
# POOLING_OPERATION = 'max'
# KERNEL_SIZE = 3
# PADDING = 'same'
# STRIDE = 1

# EPOCHS = 141
# BATCH_SIZE = 128
# NOISE_LEVEL = 0.15
# PI_THRESHOLD = 0.7
# MASKED_RECON_WEIGHT = 0.0087 # gamma: weight for masked reconstruction loss
# LEARNING_RATE = 0.000102
# DROPOUT_RATE = 0.0077
# LAYERS = [752]
# REGULARIZER = 'none'
# REGULARIZATION_WEIGHT = 1e-4

INPUT_FOLDER = "Data/synthetic_data"
FEATURES = ['Centr']
BIN_SIZE = 19
MOVING_AVERAGE = True
DATA_POINT_LENGTH = 2000
STEP_SIZE = 894
SAMPLE_FRACTION = 1.0

SPLIT_ON = 'Chrom'
TRAIN_CHROM = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI', 'ChrVIII', 'ChrXV', 'ChrXIV']
TEST_CHROM = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']
VAL_CHROM = []  # No validation set 

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
MASKED_RECON_WEIGHT = 0.00872  # gamma: weight for masked reconstruction loss
LEARNING_RATE = 1e-5
DROPOUT_RATE = 0
LAYERS = [752]
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 1e-5

PLOT = True
SAVE_MODEL = True
MODEL_SAVE_DIR = "AE/results/models"


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
    plot=PLOT,
    eval_on_val=True,
    save_model=SAVE_MODEL,
    model_save_dir=MODEL_SAVE_DIR):
    """
    Main training function that accepts pre-made datasets.
    Used by Bayesian optimization to avoid recreating data every trial.
    
    Parameters:
    -----------
    train_set, val_set, test_set : numpy arrays
        Pre-processed datasets
    eval_on_val : bool
        If True, evaluate on validation set (for hyperparameter tuning).
        If False, evaluate on test set (for final evaluation).
    """
    # Initialize model
    if "Chr" in features: chrom = True
    else: chrom = False
    
    # Create chromosome embedding once if needed
    chrom_embedding = ChromosomeEmbedding() if chrom else None
    
    # Create train dataloader
    train_dataloader = dataloader_from_array(
        train_set, batch_size=batch_size, shuffle=True, zinb=True, 
        chrom=chrom, sample_fraction=sample_fraction, denoise_percentage=noise_level
    )
    
    # Create evaluation dataloader
    # Priority: val_set if available and eval_on_val=True, otherwise test_set
    eval_dataloader = None
    eval_set_name = None
    
    if eval_on_val and val_set is not None and len(val_set) > 0:
        eval_dataloader = dataloader_from_array(
            val_set, batch_size=batch_size, shuffle=False, zinb=True, 
            chrom=chrom, sample_fraction=1.0, denoise_percentage=noise_level
        )
        eval_set_name = "VALIDATION"
    elif test_set is not None and len(test_set) > 0:
        print(f"\nCreating test dataloader:")
        print(f"  Test set size: {len(test_set)}")
        print(f"  Batch size: {batch_size}")
        eval_dataloader = dataloader_from_array(
            test_set, batch_size=batch_size, shuffle=False, zinb=True, 
            chrom=chrom, sample_fraction=1.0, denoise_percentage=noise_level
        )
        print(f"  Dataloader created: {len(eval_dataloader.dataset)} samples, {len(eval_dataloader)} batches")
        eval_set_name = "TEST"
    
    # Calculate feature dimension
    feature_dim = train_dataloader.dataset.tensors[0].shape[2] + 1
    if chrom:
        feature_dim += chrom_embedding.embedding.embedding_dim 
    
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
        chrom=chrom,
        chrom_embedding=chrom_embedding,
        plot=plot,
    )
    
    # Free training dataloader memory before evaluation
    del train_dataloader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Evaluate model on validation or test set
    if eval_dataloader is not None and len(eval_dataloader.dataset) > 0:
        print(f"\n{'='*50}")
        print(f"EVALUATING ON {eval_set_name} DATA")
        print(f"Dataset size: {len(eval_dataloader.dataset)}")
        print(f"{'='*50}\n")
        
        _, _, eval_metrics = test(
            model=zinbae_model,
            dataloader=eval_dataloader,
            pi_threshold=pi_threshold,
            chrom=chrom,
            chrom_embedding=chrom_embedding,
            plot=plot,
            denoise_percent=noise_level,
            alpha=regularization_weight,
            gamma=masked_recon_weight,
            regularizer=regularizer,
        )
    else:
        print(f"\nSkipping evaluation: No evaluation dataset available")
        eval_metrics = {}
    
    # Save model before cleanup
    if save_model:
        # Create save directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Generate filename with timestamp and key parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_str = f"conv{conv_channel}" if use_conv else "noconv"
        layers_str = "_".join(map(str, layers))
        model_filename = f"ZINBAE_{timestamp}_{conv_str}_layers{layers_str}_ep{epochs}.pt"
        model_path = os.path.join(model_save_dir, model_filename)
        
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
        
        torch.save(save_dict, model_path)
        print(f"\n{'='*50}")
        print(f"MODEL SAVED: {model_filename}")
        print(f"Location: {model_path}")
        print(f"{'='*50}\n")
    
    # Final cleanup - delete model and dataloader to free memory
    del zinbae_model
    del eval_dataloader
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
    """
    if not moving_average:
        data_point_length = data_point_length // bin_size
    
    # Preprocess data with explicit chromosome split
    print(f"\nPreprocessing with chromosome split:")
    print(f"  Train chromosomes: {train_chroms}")
    print(f"  Val chromosomes: {val_chroms if val_chroms else 'None'}")
    print(f"  Test chromosomes: {test_chroms}")
    
    train_set, val_set, test_set, _, _, _, _, _, _ = preprocess_with_split(
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
    # - If val_set exists and has data, use it for evaluation (hyperparameter tuning)
    # - Otherwise, use test_set for final evaluation
    eval_on_val = (val_set is not None and len(val_set) > 0)
    print(f"  Eval strategy: {'VALIDATION' if eval_on_val else 'TEST'}\n")
    
    # Train and evaluate
    train_metrics, eval_metrics = main_with_datasets(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
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
        plot=plot,
        eval_on_val=eval_on_val,
        save_model=save_model,
        model_save_dir=model_save_dir
    )
    
    # Return metrics (useful for Bayesian optimization or logging)
    # For normal usage, you can ignore the return value
    return train_metrics, eval_metrics

if __name__ == "__main__":
    main()