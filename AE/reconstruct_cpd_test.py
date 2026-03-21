"""
Reconstruct test_CPD data using a trained ZINBAE model.
Processes all replicates in Data/test_CPD and saves reconstructions.
"""
import os
import sys
import torch
import numpy as np
import json
import glob
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess_with_split
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import test
from AE.reconstruct_output import OutputReconstructor


def find_latest_model(model_dir, pattern="ZINBAE_layers752_ep141_*.pt"):
    """Find the most recently created model file matching pattern."""
    model_files = glob.glob(os.path.join(model_dir, pattern))
    if not model_files:
        return None
    # Sort by modification time, most recent first
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]


def reconstruct_test_cpd(
    model_path,
    test_cpd_folder="Data/test_CPD",
    output_base_dir="Data/reconstruction_cpd_test",
    features=['Centr'],
    bin_size=19,
    moving_average=True,
    data_point_length=2000,
    step_size=900,
    batch_size=128,
    noise_level=0.15,
    pi_threshold=0.7,
    masked_recon_weight=0.008,
    regularizer='none',
    regularization_weight=1e-5,
):
    """
    Load trained model and reconstruct all test_CPD data.
    
    Args:
        model_path: Path to the trained model file
        test_cpd_folder: Root folder containing test CPD data
        output_base_dir: Base directory for saving reconstructions
        Other args: Model/preprocessing configuration (should match training)
    """
    print("="*60)
    print("RECONSTRUCTING TEST_CPD DATA")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Test CPD folder: {test_cpd_folder}")
    print(f"Output directory: {output_base_dir}")
    print("="*60)
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\nLoading model from {model_path}...")
    loaded_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model_config = loaded_dict['model_config']
    
    print("Model configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    zinbae_model = ZINBAE(**model_config)
    zinbae_model.load_state_dict(loaded_dict['model_state_dict'])
    print("Model loaded successfully!")
    
    # Get chromosome embedding if needed
    chrom = "Chr" in features
    chrom_embedding = ChromosomeEmbedding() if chrom else None
    
    # Find all replicate folders in test_CPD
    # Structure: Data/test_CPD/1/yEK23_1/, Data/test_CPD/2/yEK23_2/, etc.
    replicate_folders = []
    for subdir in sorted(os.listdir(test_cpd_folder)):
        subdir_path = os.path.join(test_cpd_folder, subdir)
        if os.path.isdir(subdir_path) and not subdir.startswith('_'):
            for replicate in sorted(os.listdir(subdir_path)):
                replicate_path = os.path.join(subdir_path, replicate)
                if os.path.isdir(replicate_path):
                    replicate_folders.append((subdir, replicate, replicate_path))
    
    print(f"\nFound {len(replicate_folders)} replicate folders to process:")
    for subdir, replicate, path in replicate_folders:
        print(f"  {subdir}/{replicate}")
    
    # Process each replicate
    all_chromosomes = ['ChrI', 'ChrII', 'ChrIII', 'ChrIV', 'ChrV', 'ChrVI', 'ChrVII', 'ChrVIII',
                       'ChrIX', 'ChrX', 'ChrXI', 'ChrXII', 'ChrXIII', 'ChrXIV', 'ChrXV', 'ChrXVI']
    
    for subdir, replicate, replicate_path in replicate_folders:
        print(f"\n{'='*60}")
        print(f"Processing: {subdir}/{replicate}")
        print(f"{'='*60}")
        
        # Preprocess this replicate's data
        # We'll use all chromosomes as "test" data for reconstruction
        test_set, _, _, test_metadata, _, _, _, _, _ = preprocess_with_split(
            input_folder=replicate_path,
            train_chroms=[],  # No training data needed
            val_chroms=[],
            test_chroms=all_chromosomes,  # All chromosomes for reconstruction
            features=features,
            bin_size=bin_size,
            moving_average=moving_average,
            data_point_length=data_point_length,
            step_size=step_size,
        )
        
        if test_set is None or len(test_set) == 0:
            print(f"Warning: No data found for {subdir}/{replicate}, skipping...")
            continue
        
        print(f"Dataset size: {test_set.shape}")
        
        # Create dataloader
        test_dataloader = dataloader_from_array(
            test_set, batch_size=batch_size, shuffle=False, zinb=True, chrom=chrom
        )
        
        # Run inference
        print("Running model inference...")
        predictions, _, metrics, mu_raw, theta, pi = test(
            model=zinbae_model,
            dataloader=test_dataloader,
            pi_threshold=pi_threshold,
            chrom=chrom,
            chrom_embedding=chrom_embedding,
            plot=False,
            denoise_percent=noise_level,
            alpha=regularization_weight,
            gamma=masked_recon_weight,
            name=f"{subdir}_{replicate}",
            regularizer=regularizer,
        )
        
        print(f"Reconstruction complete. Metrics: {metrics}")
        
        # Save reconstruction artifacts
        output_dir = os.path.join(output_base_dir, subdir, replicate)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(test_metadata, f, indent=2)
        
        print(f"\nReconstructing genomic coordinates...")
        reconstructor = OutputReconstructor(metadata_path)
        reconstructed_df = reconstructor.reconstruct_to_dataframe(
            predictions,
            aggregation="mean",
            include_uncertainty=True,
            mu_raw=mu_raw,
            theta=theta,
            pi=pi,
        )
        
        reconstructor.save_as_csv(reconstructed_df, output_dir, split_by_chromosome=True)
        print(f"Saved reconstruction to: {output_dir}")
    
    print("\n" + "="*60)
    print("ALL RECONSTRUCTIONS COMPLETE")
    print("="*60)
    print(f"Output saved to: {output_base_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reconstruct test_CPD data using trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model. If not provided, will use most recent model."
    )
    parser.add_argument(
        "--test_cpd_folder",
        type=str,
        default="Data/test_CPD",
        help="Path to test_CPD folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data/reconstruction_cpd_test",
        help="Output directory for reconstructions"
    )
    
    args = parser.parse_args()
    
    # Find model if not specified
    model_path = args.model_path
    if model_path is None:
        model_path = find_latest_model("AE/results/models")
        if model_path is None:
            print("Error: No model found. Please train a model first or specify --model_path")
            sys.exit(1)
        print(f"Using most recent model: {model_path}")
    
    # Run reconstruction with configuration matching training
    reconstruct_test_cpd(
        model_path=model_path,
        test_cpd_folder=args.test_cpd_folder,
        output_base_dir=args.output_dir,
        features=['Centr'],
        bin_size=19,
        moving_average=True,
        data_point_length=2000,
        step_size=int(2000 * 0.45),
        batch_size=128,
        noise_level=0.15,
        pi_threshold=0.7,
        masked_recon_weight=0.008,
        regularizer='none',
        regularization_weight=1e-5,
    )
