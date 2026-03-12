import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 

# Get the directory where this script is located (AE/training folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the AE folder
AE_DIR = os.path.dirname(SCRIPT_DIR)
# Get the project root (parent of AE folder)
PROJECT_ROOT = os.path.dirname(AE_DIR)

def generate_prefix(model_type, timestamp, use_conv, name=""):
    """
    Generate a consistent filename prefix for saved plots and metrics.
    
    Parameters:
    -----------
    model_type : str
        Type of model (e.g., 'AE', 'VAE', 'ZINBAE')
    timestamp : str
        Timestamp string
    use_conv : bool
        Whether Conv1D was used
    name : str
        Additional name prefix
    
    Returns:
    --------
    str
        Filename prefix
    """
    conv_suffix = "conv" if use_conv else "no_conv"
    return f"{name}_{model_type}_{timestamp}_{conv_suffix}" if name else f"{model_type}_{timestamp}_{conv_suffix}"


def prepare_output_dirs(save_dir=None, subdir="testing", name=""):
    """
    Prepare output directory, creating if necessary.
    
    Parameters:
    -----------
    save_dir : str or None
        Base directory to save to. If None, uses default.
    subdir : str
        Subdirectory name ('testing' or 'training')
    name : str
        Additional subdirectory name (e.g., 'small_data')
    
    Returns:
    --------
    str
        Full output directory path
    """
    if save_dir is None:
        if name:
            save_dir = os.path.join(AE_DIR, 'results', subdir, name)
        else:
            save_dir = os.path.join(AE_DIR, 'results', subdir)
    else:
        save_dir = os.path.dirname(save_dir) if os.path.isfile(save_dir) else save_dir
    
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def clip_hi(x, q=99.5):
    hi = np.nanpercentile(x, q)
    return np.clip(x, None, hi), hi
