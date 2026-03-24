import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 
import numpy as np
import json
from torch import cuda
from datetime import datetime
import gc
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import dump, load
from AE.main import main_with_datasets
from AE.preprocessing.preprocessing import preprocess, preprocess_with_split, determine_chromosome_split
import argparse
import joblib

# Custom checkpoint callback to avoid pickle errors with nested functions
class CustomCheckpointSaver:
    """Custom checkpoint callback that avoids pickling issues."""
    def __init__(self, checkpoint_path, save_every=10):
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every
        self.iteration_count = 0
        self.x_iters = []
        self.func_vals = []
    
    def __call__(self, res):
        """Called after each iteration to save checkpoint every N iterations."""
        self.iteration_count += 1
        
        # Only save every N iterations
        if self.iteration_count % self.save_every != 0:
            return False  # Continue without saving
        
        # Extract only the data we need (no function references)
        self.x_iters = res.x_iters
        self.func_vals = res.func_vals
        
        # Create a minimal checkpoint with just the data
        checkpoint_data = {
            'x_iters': self.x_iters,
            'func_vals': self.func_vals,
            'x': res.x,
            'fun': res.fun,
            'space': res.space,
        }
        
        # Save using joblib dump
        joblib.dump(checkpoint_data, self.checkpoint_path, compress=9)
        print(f"\n*** Checkpoint saved at iteration {self.iteration_count} ***\n")
        return False  # Continue optimization

# Force numpy to not use memory mapping for large arrays (prevents bus errors)
os.environ['NUMPY_MMAP_MODE'] = 'c'  # Copy mode instead of mmap

# ============================================================================
# BAYESIAN OPTIMIZATION PARAMETER SPACE
# ============================================================================

# Preprocessing hyperparameters - CATEGORICAL
# Note: Encode features as strings to avoid skopt Categorical distance calculation issues
FEATURES_OPTIONS = ["Centr_Nucl", "Centr", "Nucl"]  # Will be decoded to lists later
MOVING_AVERAGE_OPTIONS = [False, True]

# Convolutional Layer hyperparameters
USE_CONV_OPTIONS = [True, False]
# Padding is fixed to 'same' to preserve sequence length and avoid dimension collapse
FIXED_PADDING = 'same'

# Regularization hyperparameters
REGULARIZATIONS = ["l1", "l2", "none"]


# ============================================================================
# BAYESIAN OPTIMIZATION SEARCH SPACE
# ============================================================================
search_space = [
    # Preprocessing
    Categorical(FEATURES_OPTIONS, name='features'),
    Real(0.25, 1.0, name='step_size'),  # STEP_SIZES as continuous
    Integer(1, 20, name='bin_size'),  # BIN_SIZES range
    Real(0.25, 1.0, name='sample_fraction'),  # SAMPLE_FRACTIONS as continuous
    Categorical(MOVING_AVERAGE_OPTIONS, name='moving_average'),
    
    # Model Architecture (parameterized, layer sizes divisible by 16)
    Integer(4, 100, name='first_layer_size_factor'),  # 64-1600 (multiples of 16)
    Integer(1, 4, name='num_layers'),  # Number of layers (1 to 4, will be multiplied by 2 since encoder and decoder, and one latent layer will be added in between)
    
    # Convolutional Layers
    Categorical(USE_CONV_OPTIONS, name='use_conv'),
    Integer(16, 128, name='conv_channel'),  # CONV_CHANNELS range (powers of 2 will be sampled)
    Integer(2, 8, name='pool_size'),  # POOL_SIZES range
    Categorical([3, 5, 7, 9, ], name='kernel_size'),  # Odd numbers only for symmetry
    
    # Training
    Integer(30, 150, name='epochs'),  # EPOCHS range
    Categorical([32, 64, 128], name='batch_size'),  # Powers of 2 only
    Real(0.3, 0.7, name='pi_threshold'),  # PI_THRESHOLD as continuous
    Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),  # Log scale for learning rate
    Real(0.0, 0.5, name='dropout_rate'),  # DROPOUT_RATES as continuous
    
    # Loss weights (log-scale since it can span orders of magnitude)
    Real(1e-3, 10.0, prior='log-uniform', name='masked_recon_weight'),  # gamma: weight for masked reconstruction loss
    
    # Regularization
    Categorical(REGULARIZATIONS, name='regularizer'),
    Real(1e-5, 100, prior='log-uniform', name='regularization_weight'),  # Log scale
]

# Fixed parameters (not optimized)
FIXED_PARAMS = {
    'input_folder': "Data/combined_strains",
    'split_on': 'Chrom',
    'train_val_test_split': [0.6, 0.2, 0.2],  # Proper train/val/test split
    'plot': False,
    'data_point_length': 2000,  # Fixed sequence length
    'stride': 1,  # Fixed to avoid dimension mismatch issues
    'padding': 'same',  # Fixed to 'same' to preserve sequence length
    'noise_level': 0.15,  # Fixed noise level for data augmentation
}

# Optimization metric: which loss to minimize from VALIDATION set
# 'zinb_nll': Only ZINB reconstruction loss (no masked loss)
# 'masked_loss': Only masked reconstruction loss (no ZINB loss)
# 'total_loss': Includes zinb_nll + weighted masked_loss + regularization (optimizer can game this!)
# 'combined': zinb_nll + masked_loss (unweighted, no regularization - best option)
OPTIMIZATION_METRIC = 'combined'  # Default, can be overridden by command line arg

# Budget for optimization
N_CALLS = 50  # Number of Bayesian optimization iterations 
RANDOM_STATE = 42  # For reproducibility
N_INITIAL_POINTS = 10  # Random exploration before Bayesian optimization

# Results directory
RESULTS_DIR = "AE/results/bayesian_optimization"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================
def create_objective_function(optimization_metric, train_chroms, val_chroms, test_chroms, input_folder):
    """Create objective function with specified optimization metric and chromosome splits.
    
    Args:
        optimization_metric: Which metric to optimize on validation set
        train_chroms: List of chromosomes for training
        val_chroms: List of chromosomes for validation
        test_chroms: List of chromosomes for testing (not used during optimization)
        input_folder: Path to data folder
    """
    @use_named_args(search_space)
    def objective(**params):
        # Clean up GPU memory before starting new trial
        gc.collect()
        if cuda.is_available():
            cuda.empty_cache()
            cuda.synchronize()
        """
        Objective function for Bayesian optimization.
        Creates datasets with trial-specific preprocessing hyperparameters,
        trains model, and returns validation loss.
        
        Returns:
            float: Validation loss metric to minimize (specified by optimization_metric)
        """
        # Merge with fixed parameters
        all_params = {**FIXED_PARAMS, **params}
        
        print(f"\n{'='*80}")
        print(f"Trial with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        try:
            # Extract preprocessing parameters for this trial
            features_str = all_params['features']
            bin_size = all_params['bin_size']
            moving_average = all_params['moving_average']
            data_point_length = all_params['data_point_length']
            step_size = all_params['step_size']
            
            # Decode features string to list
            if features_str == "Centr_Nucl":
                features = ["Centr", "Nucl"]
            elif features_str == "Centr":
                features = ["Centr"]
            elif features_str == "Nucl":
                features = ["Nucl"]
            else:
                features = [features_str]  # Fallback
            
            # Construct layers from parametric representation (divisible by 16)
            first_layer_size_factor = all_params['first_layer_size_factor']
            first_layer_size = first_layer_size_factor * 16  # Ensure divisible by 16
            num_layers = all_params['num_layers']
            layers = [first_layer_size // (2**i) for i in range(num_layers)]
            print(f"len(layers): {len(layers)}")
            
            # Always use the fixed output length (2000)
            # The preprocess function will internally read more nucleotides if bin_size > 1
            preprocessing_data_length = data_point_length
            
            # Convert step_size from fraction to actual step size
            # step_size is a fraction (0.25-1.0) that needs to be multiplied by data_point_length
            actual_step_size = int(preprocessing_data_length * step_size)
            
            print(f"Creating datasets with:")
            print(f"  features: {features}")
            print(f"  bin_size: {bin_size}")
            print(f"  moving_average: {moving_average}")
            print(f"  data_point_length: {preprocessing_data_length} (from {data_point_length})")
            print(f"  step_size: {actual_step_size} (from fraction {step_size})")
            print(f"  layers: {layers} (first={first_layer_size}, num={num_layers})")
            print(f"  stride: {all_params['stride']} (fixed), padding: {all_params['padding']} (fixed)")
            print(f"  Using pre-determined chromosome split\n")
            
            # Preprocess data with trial-specific parameters and pre-determined splits
            train_set, val_set, test_set, _, _, _, _, _, _ = preprocess_with_split(
                input_folder=input_folder,
                train_chroms=train_chroms,
                val_chroms=val_chroms,
                test_chroms=test_chroms,
                features=features,
                bin_size=bin_size,
                moving_average=moving_average,
                data_point_length=preprocessing_data_length,
                step_size=actual_step_size
            )
            print(f"Datasets created with shapes:" )
            if train_set is not None:
                print(f"  Train: {train_set.shape}")
            if val_set is not None:
                print(f"  Val: {val_set.shape}")
            if test_set is not None:
                print(f"  Test: {test_set.shape}")
            
            # CRITICAL: Ensure arrays are in-memory copies, not memory-mapped
            # This prevents bus errors in parallel execution
            if train_set is not None and hasattr(train_set, 'flags') and not train_set.flags['OWNDATA']:
                train_set = np.array(train_set, copy=True)
            if val_set is not None and hasattr(val_set, 'flags') and not val_set.flags['OWNDATA']:
                val_set = np.array(val_set, copy=True)
            if test_set is not None and hasattr(test_set, 'flags') and not test_set.flags['OWNDATA']:
                test_set = np.array(test_set, copy=True)
            
            print(f"Datasets created:")
            print(f"  Train: {train_set.shape}, Val: {val_set.shape if val_set is not None else 'None'}, Test: {test_set.shape if test_set is not None else 'None'}\n")
            
            try:
                # data_point_length should be the OUTPUT length (always 2000), not preprocessing length
                train_metrics, val_metrics = main_with_datasets(
                    train_set=train_set,
                    val_set=val_set,
                    test_set=test_set,
                    features=features,
                    data_point_length=preprocessing_data_length,
                    use_conv=all_params['use_conv'],
                    conv_channel=int(all_params['conv_channel']),
                    pool_size=int(all_params['pool_size']),
                    kernel_size=int(all_params['kernel_size']),
                    padding=all_params['padding'],  # Keep as string 'same'
                    stride=all_params['stride'],  # Already int from FIXED_PARAMS
                    epochs=int(all_params['epochs']),
                    batch_size=int(all_params['batch_size']),
                    noise_level=all_params['noise_level'],
                    pi_threshold=all_params['pi_threshold'],
                    masked_recon_weight=all_params['masked_recon_weight'],
                    learning_rate=all_params['learning_rate'],
                    dropout_rate=all_params['dropout_rate'],
                    layers=layers,  # Use converted list
                    regularizer=all_params['regularizer'],
                    regularization_weight=all_params['regularization_weight'],
                    sample_fraction=all_params['sample_fraction'],
                    plot=all_params['plot'],
                    eval_on_val=True,  # Use validation set for optimization
                    save_model=False  # Don't save models during optimization trials
                )
            finally:
                # Explicitly delete datasets and metrics to free memory after training
                del train_set, val_set, test_set
                if 'train_metrics' in locals():
                    del train_metrics
                # Force garbage collection and clear CUDA cache
                gc.collect()
                if cuda.is_available():
                    cuda.empty_cache()
                    cuda.synchronize()  # Wait for all GPU operations to complete
            
            # Extract the metric to optimize from VALIDATION metrics
            if optimization_metric == 'combined':
                # Custom: zinb_nll + masked_loss (both unweighted, no regularization)
                # This prevents optimizer from gaming the weight or regularization parameters
                loss = val_metrics.get('zinb_nll', 0.0)
                if 'masked_loss' in val_metrics:
                    loss += val_metrics['masked_loss']
                    print(f"Using combined metric: zinb_nll ({val_metrics['zinb_nll']:.6f}) + masked_loss ({val_metrics['masked_loss']:.6f}) = {loss:.6f}")
                else:
                    print(f"Using combined metric: zinb_nll only = {loss:.6f}")
            elif optimization_metric not in val_metrics:
                print(f"Warning: Metric '{optimization_metric}' not found in val_metrics.")
                print(f"Available metrics: {list(val_metrics.keys())}")
                # Fallback to total_loss or first available metric
                if 'total_loss' in val_metrics:
                    loss = val_metrics['total_loss']
                    print(f"Using 'total_loss' instead: {loss:.6f}")
                else:
                    loss = float(list(val_metrics.values())[0])
                    print(f"Using first metric '{list(val_metrics.keys())[0]}': {loss:.6f}")
            else:
                loss = val_metrics[optimization_metric]
            
            print(f"\n>>> Optimizing {optimization_metric} on VALIDATION set: {loss:.6f}")
            print(f">>> Full validation metrics: {val_metrics}\n")
            
            # Explicitly delete metrics to free memory
            if 'train_metrics' in locals():
                del train_metrics
            del val_metrics
            # Force garbage collection
            gc.collect()
            if cuda.is_available():
                cuda.empty_cache()
            
            return loss
            
        except Exception as e:
            print(f"\nERROR during training: {str(e)}")
            print(f"Parameters that caused error: {params}\n")
            import traceback
            traceback.print_exc()
            
            # Clean up any allocated memory before returning error
            try:
                if 'train_set' in locals():
                    del train_set
                if 'val_set' in locals():
                    del val_set
                if 'test_set' in locals():
                    del test_set
                if 'train_metrics' in locals():
                    del train_metrics
                if 'val_metrics' in locals():
                    del val_metrics
                # Force multiple garbage collection cycles
                for _ in range(3):
                    gc.collect()
                if cuda.is_available():
                    cuda.empty_cache()
                    cuda.synchronize()
            except:
                pass
            
            # Return a large penalty value instead of crashing
            return 1e6
    
    return objective


# ============================================================================
# OPTIMIZATION FUNCTION
# ============================================================================
def run_bayesian_optimization(n_calls=N_CALLS, random_state=RANDOM_STATE, 
                              n_initial_points=N_INITIAL_POINTS, n_jobs=1,
                              optimization_metric='combined', resume_from=None):
    """
    Run Bayesian hyperparameter optimization using scikit-optimize.
    
    Args:
        n_calls: Number of optimization iterations
        random_state: Random seed for reproducibility
        n_initial_points: Number of random evaluations before Gaussian Process
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        resume_from: Path to checkpoint file to resume from (optional)
        
    Returns:
        result: OptimizeResult object from skopt
    """
    # Generate timestamp for this optimization run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # DETERMINE CHROMOSOME SPLIT ONCE FOR ALL TRIALS
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"Determining chromosome split (will be consistent across all trials)...")
    print(f"{'='*80}\n")
    
    train_chroms, val_chroms, test_chroms = determine_chromosome_split(
        input_folder=FIXED_PARAMS['input_folder'],
        train_val_test_split=FIXED_PARAMS['train_val_test_split']
    )
    
    # Save chromosome split to file for reproducibility
    chrom_split_file = os.path.join(RESULTS_DIR, f"chromosome_split_{timestamp}.json")
    chrom_split_data = {
        'train_chromosomes': train_chroms,
        'val_chromosomes': val_chroms,
        'test_chromosomes': test_chroms,
        'train_val_test_split': FIXED_PARAMS['train_val_test_split'],
        'random_state': random_state,
        'timestamp': timestamp,
        'note': 'This chromosome split is used consistently across all optimization trials'
    }
    with open(chrom_split_file, 'w') as f:
        json.dump(chrom_split_data, f, indent=4)
    print(f"Chromosome split saved to: {chrom_split_file}\n")
    
    # Setup checkpoint saving with custom callback to avoid pickle errors
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{optimization_metric}_{timestamp}.pkl")
    checkpoint_saver = CustomCheckpointSaver(checkpoint_path, save_every=10)
    
    # Check if resuming from previous checkpoint
    x0 = None
    y0 = None
    if resume_from and os.path.exists(resume_from):
        print(f"\n{'='*80}")
        print(f"Resuming from checkpoint: {resume_from}")
        print(f"{'='*80}\n")
        checkpoint_data = joblib.load(resume_from)
        
        # Handle both old skopt format and new custom format
        if isinstance(checkpoint_data, dict) and 'x_iters' in checkpoint_data:
            # Custom checkpoint format
            x0 = checkpoint_data['x_iters']
            y0 = checkpoint_data['func_vals']
        else:
            # Old skopt OptimizeResult format
            x0 = checkpoint_data.x_iters
            y0 = checkpoint_data.func_vals
        
        print(f"Loaded {len(x0)} previous trials")
        print(f"Best score so far: {min(y0):.6f}\n")
        
        # Adjust n_calls to account for already completed trials
        remaining_calls = n_calls - len(x0)
        if remaining_calls <= 0:
            print(f"WARNING: Requested {n_calls} total trials, but {len(x0)} already completed!")
            print(f"No new trials will be run. Increase --n_calls to run more trials.\n")
            # Return existing results without running new trials
            return None
        
        print(f"Adjusting n_calls: {n_calls} total - {len(x0)} completed = {remaining_calls} remaining\n")
        n_calls = remaining_calls
    
    print(f"\n{'#'*80}")
    print(f"# Starting Bayesian Hyperparameter Optimization")
    print(f"# Optimizing metric: {optimization_metric} on VALIDATION set")
    if x0 is not None:
        print(f"# Resuming with {len(x0)} previous trials")
        print(f"# Running {n_calls} additional trials")
    else:
        print(f"# Number of trials: {n_calls}")
    print(f"# Initial random points: {n_initial_points}")
    print(f"# Random state: {random_state}")
    print(f"# Parallel jobs: {n_jobs}")
    print(f"# Checkpoint will be saved to: {checkpoint_path}")
    print(f"{'#'*80}\n")
    
    # Set environment variables to prevent each worker from spawning multiple threads
    # Without this, 10 workers × multicple threads each = memory explosion
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Disable tqdm progress bars to reduce output clutter
    os.environ['TQDM_DISABLE'] = '1'
    
    # Create objective function with specified metric and chromosome splits
    objective = create_objective_function(
        optimization_metric=optimization_metric,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
        input_folder=FIXED_PARAMS['input_folder']
    )
    
    # Run optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        verbose=True,
        n_jobs=n_jobs,
        x0=x0,
        y0=y0,
        callback=[checkpoint_saver]
    )
    
    # Force cleanup of any remaining resources
    gc.collect()
    
    # Save final results using custom format (same as checkpoint to avoid pickle errors)
    result_file = os.path.join(RESULTS_DIR, f"bayesian_opt_result_{optimization_metric}_{timestamp}.pkl")
    final_result_data = {
        'x_iters': result.x_iters,
        'func_vals': result.func_vals,
        'x': result.x,
        'fun': result.fun,
        'space': result.space,
    }
    joblib.dump(final_result_data, result_file, compress=9)
    print(f"\nOptimization result saved to: {result_file}")
    
    # Save final result metadata
    final_metadata = {
        'optimization_metric': optimization_metric,
        'timestamp': timestamp,
        'n_calls': n_calls,
        'n_initial_points': n_initial_points,
        'random_state': random_state,
        'n_jobs': n_jobs,
        'total_trials': len(result.x_iters),
        'best_score': float(result.fun),
        'result_file': result_file,
        'note': f'Final result of optimizing {optimization_metric} on validation set'
    }
    final_metadata_path = os.path.join(RESULTS_DIR, f"bayesian_opt_result_{optimization_metric}_{timestamp}_metadata.json")
    with open(final_metadata_path, 'w') as f:
        json.dump(final_metadata, f, indent=4)
    print(f"Result metadata saved to: {final_metadata_path}")
    
    # Extract best parameters
    best_params = {search_space[i].name: result.x[i] for i in range(len(search_space))}
    
    # Extract ALL trial results from the optimization result
    all_trials_data = {
        'optimization_info': {
            'description': f'Bayesian optimization results for minimizing {optimization_metric} on validation set',
            'optimization_metric': optimization_metric,
            'metric_description': 'The loss metric being minimized during optimization',
            'n_calls': n_calls,
            'n_initial_points': n_initial_points,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'timestamp': timestamp,
            'total_trials_completed': len(result.x_iters),
            'best_score': float(result.fun),
        },
        'best_parameters': {},
        'all_trials': []
    }
    
    # Save best parameters
    for key, value in best_params.items():
        if isinstance(value, (list, tuple)):
            all_trials_data['best_parameters'][key] = list(value)
        elif isinstance(value, np.integer):
            all_trials_data['best_parameters'][key] = int(value)
        elif isinstance(value, np.floating):
            all_trials_data['best_parameters'][key] = float(value)
        elif isinstance(value, (np.bool_, bool)):
            all_trials_data['best_parameters'][key] = bool(value)
        else:
            all_trials_data['best_parameters'][key] = value
    
    # Save all trial results from result object
    # Cache space names to avoid recreating list for each trial
    space_names = [space.name for space in search_space]
    
    for i, (params_list, score) in enumerate(zip(result.x_iters, result.func_vals)):
        trial_data = {
            'trial_number': i + 1,
            'score': float(score),
            'parameters': {}
        }
        
        # Convert parameters
        for j, param_name in enumerate(space_names):
            value = params_list[j]
            if isinstance(value, (list, tuple)):
                trial_data['parameters'][param_name] = list(value)
            elif isinstance(value, np.integer):
                trial_data['parameters'][param_name] = int(value)
            elif isinstance(value, np.floating):
                trial_data['parameters'][param_name] = float(value)
            elif isinstance(value, (np.bool_, bool)):
                trial_data['parameters'][param_name] = bool(value)
            else:
                trial_data['parameters'][param_name] = value
        
        all_trials_data['all_trials'].append(trial_data)
    
    # Save all results to single JSON file
    all_results_file = os.path.join(RESULTS_DIR, f"all_trials_{optimization_metric}_{timestamp}.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_trials_data, f, indent=4)
    
    # Clear large data structure from memory
    del all_trials_data
    
    print(f"All trial results saved to: {all_results_file}")
    
    # Print summary
    print(f"\n{'#'*80}")
    print(f"# Optimization Complete!")
    print(f"# Optimized metric: {optimization_metric} on VALIDATION set")
    print(f"# Best validation {optimization_metric}: {result.fun:.6f}")
    print(f"# Total trials: {len(result.x_iters)}")
    print(f"# Best parameters:")
    for key, value in best_params.items():
        print(f"#   {key}: {value}")
    print(f"{'#'*80}\n")
    
    # ========================================================================
    # EVALUATE BEST MODEL ON TEST SET
    # ========================================================================
    print(f"\n{'#'*80}")
    print(f"# Evaluating Best Model on TEST SET")
    print(f"{'#'*80}\n")
    
    try:
        # Extract best parameters
        features_str = best_params['features']
        if features_str == "Centr_Nucl":
            features = ["Centr", "Nucl"]
        elif features_str == "Centr":
            features = ["Centr"]
        elif features_str == "Nucl":
            features = ["Nucl"]
        else:
            features = [features_str]
        
        first_layer_size_factor = best_params['first_layer_size_factor']
        first_layer_size = first_layer_size_factor * 16
        num_layers = best_params['num_layers']
        layers = [first_layer_size // (2**i) for i in range(num_layers)]
        
        data_point_length = FIXED_PARAMS['data_point_length']
        actual_step_size = int(data_point_length * best_params['step_size'])
        
        print(f"Retraining with best parameters and evaluating on test set...")
        print(f"  features: {features}")
        print(f"  bin_size: {best_params['bin_size']}")
        print(f"  layers: {layers}\n")
        
        # Preprocess data with best parameters
        train_set, val_set, test_set, _, _, _, _, _, _ = preprocess_with_split(
            input_folder=FIXED_PARAMS['input_folder'],
            train_chroms=train_chroms,
            val_chroms=val_chroms,
            test_chroms=test_chroms,
            features=features,
            bin_size=int(best_params['bin_size']),
            moving_average=best_params['moving_average'],
            data_point_length=data_point_length,
            step_size=actual_step_size
        )
        
        # Ensure arrays are in-memory copies
        if train_set is not None and hasattr(train_set, 'flags') and not train_set.flags['OWNDATA']:
            train_set = np.array(train_set, copy=True)
        if val_set is not None and hasattr(val_set, 'flags') and not val_set.flags['OWNDATA']:
            val_set = np.array(val_set, copy=True)
        if test_set is not None and hasattr(test_set, 'flags') and not test_set.flags['OWNDATA']:
            test_set = np.array(test_set, copy=True)
        
        print(f"Datasets created:")
        print(f"  Train: {train_set.shape}")
        print(f"  Val: {val_set.shape if val_set is not None else 'None'}")
        print(f"  Test: {test_set.shape if test_set is not None else 'None'}\n")
        
        # Train and evaluate on test set (eval_on_val=False)
        train_metrics, test_metrics = main_with_datasets(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            features=features,
            data_point_length=data_point_length,
            use_conv=best_params['use_conv'],
            conv_channel=int(best_params['conv_channel']),
            pool_size=int(best_params['pool_size']),
            kernel_size=int(best_params['kernel_size']),
            padding=FIXED_PARAMS['padding'],
            stride=FIXED_PARAMS['stride'],
            epochs=int(best_params['epochs']),
            batch_size=int(best_params['batch_size']),
            noise_level=FIXED_PARAMS['noise_level'],
            pi_threshold=best_params['pi_threshold'],
            masked_recon_weight=best_params['masked_recon_weight'],
            learning_rate=best_params['learning_rate'],
            dropout_rate=best_params['dropout_rate'],
            layers=layers,
            regularizer=best_params['regularizer'],
            regularization_weight=best_params['regularization_weight'],
            sample_fraction=best_params['sample_fraction'],
            plot=FIXED_PARAMS['plot'],
            eval_on_val=False,  # Evaluate on TEST set
            save_model=True  # Save the final best model
        )
        
        print(f"\n{'#'*80}")
        print(f"# TEST SET EVALUATION COMPLETE")
        print(f"# Test metrics: {test_metrics}")
        print(f"{'#'*80}\n")
        
        # Save test results
        test_results_file = os.path.join(RESULTS_DIR, f"test_results_{optimization_metric}_{timestamp}.json")
        test_results_data = {
            'optimization_metric': optimization_metric,
            'timestamp': timestamp,
            'best_validation_score': float(result.fun),
            'best_parameters': {k: (int(v) if isinstance(v, np.integer) else 
                                   float(v) if isinstance(v, np.floating) else 
                                   bool(v) if isinstance(v, (np.bool_, bool)) else v) 
                              for k, v in best_params.items()},
            'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in test_metrics.items()},
            'train_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                            for k, v in train_metrics.items()},
            'chromosome_split': {
                'train': train_chroms,
                'val': val_chroms,
                'test': test_chroms
            },
            'note': 'Final model trained with best hyperparameters and evaluated on held-out test set'
        }
        with open(test_results_file, 'w') as f:
            json.dump(test_results_data, f, indent=4)
        print(f"Test results saved to: {test_results_file}\n")
        
        # Cleanup
        del train_set, val_set, test_set, train_metrics, test_metrics
        gc.collect()
        if cuda.is_available():
            cuda.empty_cache()
    
    except Exception as e:
        print(f"\nERROR during test set evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nContinuing without test set evaluation...\n")
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Bayesian Hyperparameter Optimization')
    parser.add_argument('--n_calls', type=int, default=N_CALLS,
                       help='Number of optimization iterations')
    parser.add_argument('--n_initial_points', type=int, default=N_INITIAL_POINTS,
                       help='Number of random initial evaluations')
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE,
                       help='Random seed for reproducibility')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs (-1 for all cores, 1 for sequential)')
    parser.add_argument('--metric', type=str, default=OPTIMIZATION_METRIC,
                       choices=['zinb_nll', 'masked_loss', 'combined', 'total_loss'],
                       help='Optimization metric to minimize (zinb_nll, masked_loss, combined, or total_loss)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint file to resume optimization from')
    
    args = parser.parse_args()
    
    # Run optimization
    result = run_bayesian_optimization(
        n_calls=args.n_calls,
        random_state=args.random_state,
        n_initial_points=args.n_initial_points,
        n_jobs=args.n_jobs,
        optimization_metric=args.metric,
        resume_from=args.resume_from
    )

