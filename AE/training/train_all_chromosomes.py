"""
Train ZINBAE model on all chromosomes from combined_strains data.
This script trains on all chromosomes without train/test split.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.main import main

# All chromosomes for training
ALL_CHROM = ['ChrI', 'ChrII', 'ChrIII', 'ChrIV', 'ChrV', 'ChrVI', 'ChrVII', 'ChrVIII', 
             'ChrIX', 'ChrX', 'ChrXI', 'ChrXII', 'ChrXIII', 'ChrXIV', 'ChrXV', 'ChrXVI']

# Training configuration (from main.py current settings)
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
MASKED_RECON_WEIGHT = 0.008
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.008
LAYERS = [752]
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 1e-5
MU_OFFSET = 0.0

PLOT = True
SAVE_MODEL = True
MODEL_SAVE_DIR = "AE/results/models"

if __name__ == "__main__":
    print("="*60)
    print("TRAINING MODEL ON ALL CHROMOSOMES")
    print("="*60)
    print(f"Training chromosomes: {ALL_CHROM}")
    print(f"Features: {FEATURES}")
    print(f"Bin size: {BIN_SIZE}")
    print(f"Moving average: {MOVING_AVERAGE}")
    print(f"Data point length: {DATA_POINT_LENGTH}")
    print(f"Epochs: {EPOCHS}")
    print("="*60)
    
    train_metrics, eval_metrics = main(
        input_folder=INPUT_FOLDER,
        features=FEATURES,
        bin_size=BIN_SIZE,
        moving_average=MOVING_AVERAGE,
        data_point_length=DATA_POINT_LENGTH,
        step_size=STEP_SIZE,
        sample_fraction=SAMPLE_FRACTION,
        train_chroms=ALL_CHROM,  # Train on all chromosomes
        val_chroms=[],  # No validation set
        test_chroms=[],  # No test set
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
        model_save_dir=MODEL_SAVE_DIR,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Training metrics: {train_metrics}")
    print("="*60)
