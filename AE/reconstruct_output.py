"""
Reconstruction utility for mapping autoencoder outputs back to genomic coordinates.

This script takes model predictions and metadata to create genomic coordinate files
in CSV.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class OutputReconstructor:
    """Reconstruct and map autoencoder outputs to genomic coordinates."""
    
    def __init__(self, metadata_path: str):
        """
        Initialize reconstructor with metadata.
        
        Args:
            metadata_path: Path to JSON file containing window metadata
        """
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded metadata for {len(self.metadata)} windows")
    
    def reconstruct_to_dataframe(
        self,
        predictions: np.ndarray,
        aggregation: str = 'mean',
        include_uncertainty: bool = True,
        mu_raw: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
        use_raw_mu: bool = False
    ) -> pd.DataFrame:
        """
        Reconstruct predictions into a DataFrame with genomic coordinates.
        

          The "predictions" array is expected to contain threshold-applied reconstructions:
          reconstruction = mu * (pi < pi_threshold)
        
          If mu_raw, theta, or pi are provided, they are aggregated per genomic position
          and included as additional output columns.
        
        How window-to-location mapping works:
        --------------------------------------
        1. Each window has metadata: {dataset, chromosome, start_pos, end_pos}
        2. For window i with predictions shape (seq_length,):
           - We interpolate seq_length positions between start_pos and end_pos
           - Position j in the window maps to genomic position:
             position[j] = start_pos + j * (end_pos - start_pos) / (seq_length - 1)
        3. If windows overlap, we aggregate multiple predictions for the same position
        
        Args:
            predictions: Model predictions, shape (n_windows, seq_length) or (n_windows, seq_length, n_features)
                        - If using output from test_trained_model.py: this is already threshold-applied
                        - Contains mu * (pi < pi_threshold) for each window
            aggregation: How to aggregate overlapping windows ('mean', 'median', 'max', 'first', 'last')
            include_uncertainty: Whether to include pi/theta in output when provided
            mu_raw: Raw mu values (before thresholding), shape (n_windows, seq_length)
            theta: ZINB theta (dispersion) parameters, shape (n_windows, seq_length)
            pi: ZINB pi (zero-inflation) parameters, shape (n_windows, seq_length)
            use_raw_mu: If True, treat predictions as raw mu (not threshold-applied)
        
        Returns:
            DataFrame with columns: dataset, chromosome, position,
            reconstruction, mu, pi, theta, n_overlaps
        """
        def _to_2d(array: Optional[np.ndarray], array_name: str) -> Optional[np.ndarray]:
            if array is None:
                return None

            array = np.asarray(array)
            if array.ndim == 3:
                print(f"Taking first feature from {array_name} of shape {array.shape}")
                array = array[:, :, 0]
            if array.ndim != 2:
                raise ValueError(f"{array_name} must be 2D or 3D, got shape {array.shape}")
            if len(array) != len(self.metadata):
                raise ValueError(
                    f"{array_name} length ({len(array)}) doesn't match metadata length ({len(self.metadata)})"
                )
            return array

        predictions = _to_2d(predictions, "predictions")
        mu_raw = _to_2d(mu_raw, "mu_raw")
        theta = _to_2d(theta, "theta") if include_uncertainty else None
        pi = _to_2d(pi, "pi") if include_uncertainty else None
        valid_aggregations = {'mean', 'median', 'max', 'first', 'last'}
        if aggregation not in valid_aggregations:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        for array_name, array in (('mu_raw', mu_raw), ('theta', theta), ('pi', pi)):
            if array is not None and array.shape != predictions.shape:
                raise ValueError(
                    f"{array_name} shape {array.shape} must match predictions shape {predictions.shape}"
                )

        n_windows, seq_length = predictions.shape
        if n_windows == 0 or seq_length == 0:
            return pd.DataFrame(
                columns=['dataset', 'chromosome', 'position', 'reconstruction', 'mu', 'pi', 'theta', 'n_overlaps']
            )

        datasets = np.fromiter((meta['dataset'] for meta in self.metadata), dtype=object, count=n_windows)
        chromosomes = np.fromiter((meta['chromosome'] for meta in self.metadata), dtype=object, count=n_windows)
        start_positions = np.fromiter((meta['start_pos'] for meta in self.metadata), dtype=np.int64, count=n_windows)
        end_positions = np.fromiter((meta['end_pos'] for meta in self.metadata), dtype=np.int64, count=n_windows)

        # Group windows by dataset/chromosome first to reduce object-heavy operations.
        window_groups: Dict[tuple, List[int]] = {}
        for window_idx, (dataset, chrom) in enumerate(zip(datasets, chromosomes)):
            window_groups.setdefault((dataset, chrom), []).append(window_idx)

        value_columns = ['reconstruction']
        if mu_raw is not None:
            value_columns.append('mu')
        if theta is not None:
            value_columns.append('theta')
        if pi is not None:
            value_columns.append('pi')

        grouped_frames = []
        if seq_length > 1:
            relative_positions = np.arange(seq_length, dtype=np.float64)
            denominator = seq_length - 1

        for (dataset, chrom), indices in window_groups.items():
            idx = np.asarray(indices, dtype=np.int64)
            starts = start_positions[idx]
            ends = end_positions[idx]

            if seq_length == 1:
                positions = starts[:, None]
            else:
                positions = starts[:, None] + (
                    ((ends - starts)[:, None] * relative_positions[None, :]) / denominator
                )
                positions = positions.astype(np.int64)

            group_df = pd.DataFrame(
                {
                    'position': positions.reshape(-1),
                    'reconstruction': predictions[idx].reshape(-1),
                }
            )

            if mu_raw is not None:
                group_df['mu'] = mu_raw[idx].reshape(-1)
            if theta is not None:
                group_df['theta'] = theta[idx].reshape(-1)
            if pi is not None:
                group_df['pi'] = pi[idx].reshape(-1)

            grouped = group_df.groupby('position', sort=False)

            if aggregation in {'mean', 'median', 'max'}:
                aggregated_values = grouped[value_columns].agg(aggregation)
            elif aggregation == 'first':
                aggregated_values = grouped[value_columns].first()
            else:  # aggregation == 'last'
                aggregated_values = grouped[value_columns].last()

            aggregated_values['n_overlaps'] = grouped.size()
            aggregated_values = aggregated_values.reset_index()
            aggregated_values.insert(0, 'chromosome', chrom)
            aggregated_values.insert(0, 'dataset', dataset)
            grouped_frames.append(aggregated_values)

        df = pd.concat(grouped_frames, ignore_index=True)

        for optional_column in ('mu', 'pi', 'theta'):
            if optional_column not in df.columns:
                df[optional_column] = float('nan')

        df = df[['dataset', 'chromosome', 'position', 'reconstruction', 'mu', 'pi', 'theta', 'n_overlaps']]
        df = df.sort_values(['dataset', 'chromosome', 'position']).reset_index(drop=True)
        
        print(f"Reconstructed {len(df)} unique genomic positions from {len(self.metadata)} windows")
        print(f"Average overlap per position: {df['n_overlaps'].mean():.2f}")
        
        return df
    
    def save_as_csv(
        self,
        df: pd.DataFrame,
        output_path: str,
        split_by_chromosome: bool = False,
        reconstruction_decimals: Optional[int] = 1
    ):
        """
        Save reconstructed data as CSV file(s).
        
        Args:
            df: DataFrame with reconstructed data
            output_path: Base path for output file
            split_by_chromosome: If True, create separate files per chromosome
            reconstruction_decimals: Number of decimals for reconstruction values;
                                     set to None to disable rounding
        """
        if reconstruction_decimals is not None:
            if reconstruction_decimals < 0:
                raise ValueError("reconstruction_decimals must be >= 0 or None")
            df = df.copy()
            df['reconstruction'] = df['reconstruction'].round(reconstruction_decimals)

        if split_by_chromosome:
            os.makedirs(output_path, exist_ok=True)
            
            for (dataset, chrom), group_df in df.groupby(['dataset', 'chromosome']):
                safe_dataset = str(dataset).replace(os.sep, '_')
                dataset_dir = os.path.join(output_path, safe_dataset)
                os.makedirs(dataset_dir, exist_ok=True)

                filename = f"{chrom}.csv"
                filepath = os.path.join(dataset_dir, filename)
                cols = ['position', 'reconstruction', 'mu', 'pi', 'theta']
                group_df = group_df.sort_values('position')[cols]
                group_df.to_csv(filepath, index=False)
                print(f"Saved {len(group_df)} positions to {filepath}")
        else:
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} positions to {output_path}")
    
    
    def filter_by_chromosome(self, chromosome: str) -> List[int]:
        """
        Get window indices for a specific chromosome.
        
        Args:
            chromosome: Chromosome name (e.g., 'ChrI')
        
        Returns:
            List of window indices
        """
        indices = [i for i, meta in enumerate(self.metadata) if meta['chromosome'] == chromosome]
        print(f"Found {len(indices)} windows for chromosome {chromosome}")
        return indices
    
    def filter_by_dataset(self, dataset: str) -> List[int]:
        """
        Get window indices for a specific dataset.
        
        Args:
            dataset: Dataset name
        
        Returns:
            List of window indices
        """
        indices = [i for i, meta in enumerate(self.metadata) if meta['dataset'] == dataset]
        print(f"Found {len(indices)} windows for dataset {dataset}")
        return indices


def main():
    """Command-line interface for reconstruction."""
    parser = argparse.ArgumentParser(description='Reconstruct autoencoder outputs to genomic coordinates')
    
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata JSON file')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions NPY file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for reconstructed data')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'bed', 'wiggle'],
                        help='Output format (default: csv)')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'median', 'max', 'first', 'last'],
                        help='Aggregation method for overlapping windows (default: mean)')
    parser.add_argument('--split-by-chrom', action='store_true',
                        help='Create separate files per chromosome')
    parser.add_argument('--theta', type=str, default=None,
                        help='Path to ZINB theta parameters NPY file (optional)')
    parser.add_argument('--pi', type=str, default=None,
                        help='Path to ZINB pi parameters NPY file (optional)')
    parser.add_argument('--mu', type=str, default=None,
                        help='Path to raw mu parameters NPY file (optional)')
    parser.add_argument('--reconstruction-decimals', type=int, default=1,
                        help='Number of decimals for reconstruction values in CSV output (default: 1)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    reconstructor = OutputReconstructor(args.metadata)
    predictions = np.load(args.predictions)
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Load optional ZINB parameters
    mu_raw = np.load(args.mu) if args.mu else None
    theta = np.load(args.theta) if args.theta else None
    pi = np.load(args.pi) if args.pi else None
    
    # Reconstruct to DataFrame
    print("Reconstructing genomic coordinates...")
    df = reconstructor.reconstruct_to_dataframe(
        predictions,
        aggregation=args.aggregation,
        include_uncertainty=(theta is not None or pi is not None),
        mu_raw=mu_raw,
        theta=theta,
        pi=pi
    )
    
    # Save in requested format
    print(f"Saving as {args.format}...")
    if args.format == 'csv':
        reconstructor.save_as_csv(
            df,
            args.output,
            split_by_chromosome=args.split_by_chrom,
            reconstruction_decimals=args.reconstruction_decimals
        )

    
    print("Done!")


if __name__ == "__main__":
    main()
