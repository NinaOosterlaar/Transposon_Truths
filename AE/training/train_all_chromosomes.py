"""
Train ZINBAE model on all chromosomes from combined_strains data.
This script trains on all chromosomes and saves the model for CPD reconstruction.
"""
import os, sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from AE.main import main, parse_features, parse_int_list, parse_chromosomes, str_to_bool

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
SAVE_RECONSTRUCTION = False
RECONSTRUCTION_OUTPUT_DIR = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ZINBAE model on all chromosomes for later CPD reconstruction."
    )
    parser.add_argument("--input_folder", type=str, default=INPUT_FOLDER)
    parser.add_argument("--features", type=parse_features, default=FEATURES)
    parser.add_argument("--bin_size", type=int, default=BIN_SIZE)
    parser.add_argument("--moving_average", type=str_to_bool, default=MOVING_AVERAGE)
    parser.add_argument("--data_point_length", type=int, default=DATA_POINT_LENGTH)
    parser.add_argument("--step_size", type=int, default=STEP_SIZE)
    parser.add_argument("--sample_fraction", type=float, default=SAMPLE_FRACTION)
    parser.add_argument("--train_chroms", type=parse_chromosomes, default=ALL_CHROM)
    parser.add_argument(
        "--reconstruction_chroms",
        type=parse_chromosomes,
        default=[],
        help="Chromosomes to evaluate and reconstruct. Use '' to disable the eval split.",
    )
    parser.add_argument("--use_conv", type=str_to_bool, default=USE_CONV)
    parser.add_argument("--conv_channel", type=int, default=CONV_CHANNEL)
    parser.add_argument("--pool_size", type=int, default=POOL_SIZE)
    parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE)
    parser.add_argument("--padding", type=str, default=PADDING)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--noise_level", type=float, default=NOISE_LEVEL)
    parser.add_argument("--pi_threshold", type=float, default=PI_THRESHOLD)
    parser.add_argument("--masked_recon_weight", type=float, default=MASKED_RECON_WEIGHT)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout_rate", type=float, default=DROPOUT_RATE)
    parser.add_argument("--layers", type=parse_int_list, default=LAYERS)
    parser.add_argument("--regularizer", type=str, choices=["none", "l1", "l2"], default=REGULARIZER)
    parser.add_argument("--regularization_weight", type=float, default=REGULARIZATION_WEIGHT)
    parser.add_argument("--mu_offset", type=float, default=MU_OFFSET)
    parser.add_argument("--plot", type=str_to_bool, default=PLOT)
    parser.add_argument("--save_model", type=str_to_bool, default=SAVE_MODEL)
    parser.add_argument("--model_save_dir", type=str, default=MODEL_SAVE_DIR)
    parser.add_argument("--save_reconstruction", type=str_to_bool, default=SAVE_RECONSTRUCTION)
    parser.add_argument(
        "--reconstruction_output_dir",
        type=str,
        default=RECONSTRUCTION_OUTPUT_DIR,
        help="Optional explicit output directory. Defaults to Data/reconstruction/<saved-model-name>.",
    )
    return parser.parse_args()


def run_training(args):
    reconstruction_chroms = args.reconstruction_chroms if args.save_reconstruction else []

    print("="*60)
    print("TRAINING MODEL ON ALL CHROMOSOMES")
    print("="*60)
    print(f"Training chromosomes: {args.train_chroms}")
    print(f"Evaluation/reconstruction chromosomes: {reconstruction_chroms if reconstruction_chroms else 'None'}")
    print(f"Features: {args.features}")
    print(f"Bin size: {args.bin_size}")
    print(f"Moving average: {args.moving_average}")
    print(f"Data point length: {args.data_point_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Save reconstruction: {args.save_reconstruction}")
    print("="*60)
    
    train_metrics, eval_metrics = main(
        input_folder=args.input_folder,
        features=args.features,
        bin_size=args.bin_size,
        moving_average=args.moving_average,
        data_point_length=args.data_point_length,
        step_size=args.step_size,
        sample_fraction=args.sample_fraction,
        train_chroms=args.train_chroms,
        val_chroms=[],  # No validation set
        test_chroms=reconstruction_chroms,
        use_conv=args.use_conv,
        conv_channel=args.conv_channel,
        pool_size=args.pool_size,
        kernel_size=args.kernel_size,
        padding=args.padding,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_level=args.noise_level,
        pi_threshold=args.pi_threshold,
        masked_recon_weight=args.masked_recon_weight,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        layers=args.layers,
        regularizer=args.regularizer,
        regularization_weight=args.regularization_weight,
        mu_offset=args.mu_offset,
        plot=args.plot,
        save_model=args.save_model,
        model_save_dir=args.model_save_dir,
        save_reconstruction=args.save_reconstruction,
        reconstruction_output_dir=args.reconstruction_output_dir,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Training metrics: {train_metrics}")
    print(f"Evaluation metrics: {eval_metrics}")
    print("="*60)

    return train_metrics, eval_metrics


def main_cli():
    args = parse_args()
    return run_training(args)


if __name__ == "__main__":
    main_cli()
