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

# Chromosome splits (must match training configuration)
# These are used to fit the standardization scaler correctly
TRAIN_CHROMS = ['ChrIII', 'ChrIV', 'ChrIX', 'ChrVI', 'ChrVII', 'ChrX', 'ChrXI', 'ChrXIII', 'ChrXVI']
VALIDATION_CHROMS = ['ChrVIII', 'ChrXIV', 'ChrXV']
TEST_CHROMS = ['ChrI', 'ChrII', 'ChrV', 'ChrXII']

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
    for subdir, replicate, replicate_path in replicate_folders:
        print(f"\n{'='*60}")
        print(f"Processing: {subdir}/{replicate}")
        print(f"{'='*60}")
        
        # Preprocess this replicate's data using proper chromosome splits
        # This ensures standardization scaler is fitted on training chromosomes
        train_set, val_set, test_set, train_metadata, val_metadata, test_metadata, _, _, _ = preprocess_with_split(
            input_folder=replicate_path,
            train_chroms=TRAIN_CHROMS,  # Required to fit standardization scaler
            val_chroms=VALIDATION_CHROMS,
            test_chroms=TEST_CHROMS,
            features=features,
            bin_size=bin_size,
            moving_average=moving_average,
            data_point_length=data_point_length,
            step_size=step_size,
        )
        
        # Combine all splits for reconstruction (reconstruct all chromosomes)
        all_data_sets = []
        all_metadata = []
        
        if train_set is not None and len(train_set) > 0:
            all_data_sets.append(train_set)
            all_metadata.extend(train_metadata)
            print(f"Train set: {train_set.shape}")
            
        if val_set is not None and len(val_set) > 0:
            all_data_sets.append(val_set)
            all_metadata.extend(val_metadata)
            print(f"Val set: {val_set.shape}")
            
        if test_set is not None and len(test_set) > 0:
            all_data_sets.append(test_set)
            all_metadata.extend(test_metadata)
            print(f"Test set: {test_set.shape}")
        
        if not all_data_sets:
            print(f"Warning: No data found for {subdir}/{replicate}, skipping...")
            continue
        
        # Concatenate all splits
        combined_data = np.concatenate(all_data_sets, axis=0)
        combined_metadata = all_metadata
        
        print(f"Combined dataset size: {combined_data.shape}")
        
        # Create dataloader
        combined_dataloader = dataloader_from_array(
            combined_data, batch_size=batch_size, shuffle=False, zinb=True, chrom=chrom
        )
        
        # Run inference
        print("Running model inference...")
        predictions, _, metrics, mu_raw, theta, pi, _, _ = test(
            model=zinbae_model,
            dataloader=combined_dataloader,
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
            json.dump(combined_metadata, f, indent=2)
        
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
