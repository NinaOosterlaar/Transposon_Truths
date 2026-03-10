# Transposon Truths: Identifying Essential Gene Domains in Yeast Using SATAY Data

[![Status](https://img.shields.io/badge/status-in%20development-yellow)](https://github.com/yourusername/yourrepo)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org/)

> ⚠️ **Note**: This project is currently a work in progress as part of an ongoing Master's thesis.

## Overview

This repository presents a novel approach for identifying essential domains in the *Saccharomyces cerevisiae* genome using SATAY (SAturated Transposon Analysis in Yeast) data. Unlike traditional annotation-based methods, this project takes a **clean slate perspective** to estimate essentiality levels both within and outside of annotated genes, discovering functional domains without relying on existing gene boundaries.

The core methodology employs a **Zero-Inflated Negative Binomial Autoencoder (ZINB-AE)** to model the sparse, count-based nature of transposon insertion data, enabling the identification of essential genomic regions where transposon insertions are depleted.

### Key Features

- **ZINB Autoencoder Architecture**: Custom deep learning model designed for sparse genomic count data
- **Annotation-Free Analysis**: Identifies essential domains without prior knowledge of gene boundaries
- **Signal Processing Methods**: Change point detection for domain boundary identification
- **Multi-Strain Support**: Analysis pipeline compatible with multiple *S. cerevisiae* strains
- **HPC Compatible**: Includes SLURM batch scripts for high-performance computing environments

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training, but not required)
- 8GB+ RAM recommended

### Option 1: Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/NinaOosterlaar/Transposon-Truths.git
cd Transposon-Truths
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Using Apptainer/Singularity (HPC Environments)

For HPC environments, an Apptainer definition file is provided:

```bash
cd Utils/Apptainer
apptainer build my-container.sif Apptainer.def
apptainer shell --nv my-container.sif  
# --nv enables GPU support
```

## Data Preparation

### Using Example Data

Example wiggle format data is provided in the `Data/wiggle_format/` directory. This can be processed using the included utilities.

### Processing Your Own SATAY Data

If you have your own SATAY data in **wiggle format**, you can process it as follows:

```python
from Utils.reader import read_wig, compute_distances

# Read wiggle format data and compute distances between insertions
# Also computes distances to genomic features (centromeres, nucleosomes)
data = read_wig('path/to/your/data.wig')
distances = compute_distances(data)
```

The preprocessing pipeline will:
1. Read transposon insertion positions from wiggle files
2. Compute distances to relevant genomic features (centromeres, nucleosomes)
3. Generate processed data suitable for model training

## Usage

### Basic Workflow

#### 1. Preprocess Data

Generate preprocessed datasets from your SATAY data using the preprocessing script:

```bash
python AE/preprocessing/preprocessing.py \
    --input_folder Data/wiggle_format/your_strain \
    --output_dir Data/processed_data \
    --features Nucl Centr \
    --bin_size 10 \
    --data_point_length 2000 \
    --step_size 0.25
```

**Available preprocessing parameters:**

- **Input/Output:**
  - `--input_folder`: Path to folder containing CSV files with distance data (default: `Data/combined_strains/`)
  - `--output_dir`: Directory to save processed data (default: `Data/processed_data/`)

- **Features** (choose one or more):
  - `--features`: Genomic features to include as model inputs
    - `Pos`: Genomic position
    - `Nucl`: Distance to nearest nucleosome
    - `Centr`: Distance to nearest centromere
    - `Chrom`: Chromosome identifier
    - Default: `Nucl Centr`

- **Data Splitting:**
  - `--train_val_test_split`: Proportions for train/validation/test split (default: `0.7 0.15 0.15`)
  - `--split_on`: Feature to split data on
    - `Chrom`: Split by chromosome (recommended)
    - `Dataset`: Split by dataset/strain
    - `Random`: Random splitting in chunks
    - Default: `Chrom`
  - `--chunk_size`: Chunk size in base pairs for random splitting (default: `50000`)

- **Normalization:**
  - `--normalize_counts`: Apply CPM normalization and log transform (default: enabled)
  - `--no_normalize_counts`: Disable count normalization
  - `--zinb_mode`: Save raw counts for ZINB models (default: disabled)

- **Binning/Windowing:**
  - `--bin_size`: Bin size for data aggregation (default: `10`)
  - `--moving_average`: Use moving average/sliding window (default: enabled)
  - `--no_moving_average`: Use separate bins instead
  - `--data_point_length`: Length of each data sequence (default: `2000`)
  - `--step_size`: Sliding window step size as fraction of sequence length (default: `0.25`)
  - `--no_clip_outliers`: Disable outlier clipping

**Programmatic usage:**

```python
from AE.preprocessing.preprocessing import preprocess

# Run preprocessing with custom parameters
train_data, val_data, test_data, scalers, count_stats, clip_stats = preprocess(
    input_folder="Data/wiggle_format/your_strain",
    features=['Nucl', 'Centr'],
    bin_size=10,
    data_point_length=2000,
    step_size=500,  # 0.25 * 2000
    moving_average=True,
    normalize_counts=True,
    zinb_mode=False
)
```

This will create training, validation, and test datasets in the `Data/processed_data/` directory that can be used directly for model training.

#### 2. Train the ZINB Autoencoder

Configure and train the Zero-Inflated Negative Binomial Autoencoder:

```python
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training import train
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding

# Create data loader
train_dataloader = dataloader_from_array(
    train_data,
    batch_size=64,
    shuffle=True,
    zinb=True,
    denoise_percentage=0.3  # Mask 30% of non-zero values for denoising
)

# Initialize model
model = ZINBAE(
    seq_length=2000,
    feature_dim=3,  # Count + additional features (Nucl, Centr)
    layers=[512, 256, 128],
    use_conv=True,
    conv_channels=64,
    pool_size=2,
    kernel_size=5,
    padding='same',
    stride=1,
    dropout=0.2
)

# Train model
trained_model = train(
    model=model,
    dataloader=train_dataloader,
    num_epochs=100,
    learning_rate=1e-3,
    chrom=False,  # Set to True if using chromosome embeddings
    plot=True,
    beta=1.0,  # KL weight (only for ZINBVAE)
    denoise_percent=0.3,
    regularizer='L2',  # 'none', 'L1', or 'L2'
    alpha=1e-4,  # Regularization strength
    gamma=0.0,  # Weight for masked reconstruction loss
    pi_threshold=0.5  # Threshold for zero-inflation
)
```

**Model Architecture Parameters:**

- **Sequence Settings:**
  - `seq_length`: Length of input sequences (default: `2000`)
  - `feature_dim`: Number of features per position (count + additional features)

- **Encoder/Decoder Layers:**
  - `layers`: List of hidden layer dimensions (default: `[512, 256, 128]`)
  - `dropout`: Dropout rate for regularization (default: `0.0`)

- **Convolutional Options:**
  - `use_conv`: Enable 1D convolution preprocessing (default: `False`)
  - `conv_channels`: Number of convolutional filters (default: `64`)
  - `pool_size`: Max pooling kernel size (default: `2`)
  - `kernel_size`: Convolutional kernel size (default: `3`)
  - `padding`: Padding type - `'same'` or `'valid'` (default: `'same'`)
  - `stride`: Convolutional stride (default: `1`)

**Training Parameters:**

- **Optimization:**
  - `num_epochs`: Number of training epochs (default: `50`)
  - `learning_rate`: Learning rate for Adam optimizer (default: `1e-3`)
  - `batch_size`: Batch size for training (set in dataloader)

- **Regularization:**
  - `regularizer`: Type of regularization - `'none'`, `'L1'`, or `'L2'` (default: `'none'`)
  - `alpha`: Regularization strength (default: `0.0`)

- **Denoising & Masking:**
  - `denoise_percent`: Fraction of non-zero values to mask during training (default: `0.0`)
  - `gamma`: Weight for masked reconstruction loss (default: `0.0`)
  - `pi_threshold`: Threshold for considering values as non-zero based on dropout probability (default: `0.5`)

- **VAE Specific (ZINBVAE only):**
  - `beta`: Weight for KL divergence loss (default: `1.0`)

- **Other:**
  - `chrom`: Use chromosome embeddings as additional features (default: `False`)
  - `chrom_embedding`: ChromosomeEmbedding object if `chrom=True`
  - `plot`: Generate training loss plots (default: `True`)

#### 3. Evaluate and Analyze Results

Evaluate the trained model on test data and generate visualizations:

```python
from AE.training.training import test
from AE.plotting.results_ZINB import plot_zinb_test_results

# Create test data loader
test_dataloader = dataloader_from_array(
    test_data,
    batch_size=64,
    shuffle=False,
    zinb=True,
    denoise_percentage=0.0  # No masking for evaluation
)

# Evaluate on test data
test_metrics = test(
    model=trained_model,
    dataloader=test_dataloader,
    chrom=False,
    plot=True,
    beta=1.0,
    denoise_percent=0.0,
    regularizer='none',
    alpha=0.0,
    gamma=0.0,
    pi_threshold=0.5,
    name="test_run"
)

# Generate detailed visualizations
plot_zinb_test_results(
    test_metrics,
    name="test_run",
    model_type="ZINBAE",
    use_conv=True
)
```

**Test Metrics Returned:**

The `test()` function returns a dictionary containing:
- `test_loss`: Overall test loss
- `test_recon_loss`: Reconstruction loss (ZINB negative log-likelihood)
- `test_kl_loss`: KL divergence loss (ZINBVAE only)
- `test_masked_loss`: Masked reconstruction loss (if applicable)
- `test_reg_loss`: Regularization loss (if applicable)
- `mae`: Mean Absolute Error
- `r2`: R² score
- `reconstructions`: Reconstructed sequences
- `latents`: Latent representations
- `originals`: Original input sequences
- `mu`, `theta`, `pi`: ZINB parameters
- `raw_counts`: Unmasked raw count data

**Visualization Options:**

Results are automatically saved to `AE/results/testing/<experiment_name>/` and include:
- Loss curves over training epochs
- Reconstruction quality plots
- Latent space visualizations
- ZINB parameter distributions
- Comparison of original vs. reconstructed sequences

### Running the Complete Pipeline

A complete example is provided in `AE/main.py`:

```bash
python AE/main.py
```

Modify the configuration parameters at the top of `main.py` to customize:
- Input data location
- Model architecture (layers, convolution settings)
- Training hyperparameters (epochs, batch size, learning rate)
- Regularization settings

### Using HPC/SLURM

Batch scripts for SLURM-based HPC systems are provided in the `Batch/` directory:

```bash
sbatch Batch/training_zinb.sh
```



## Results

> 🚧 **Coming Soon**: Results and visualizations from the analysis will be added as the project progresses.

This section will include:
- Model performance metrics
- Identified essential domains
- Comparison with existing annotations
- Visualizations of essentiality patterns across the genome

## Contributing

This is an active research project. If you're interested in collaborating or have suggestions, please feel free to open an issue or reach out.

## Acknowledgments

**Supervisors:**
- Jasmijn Baaijens - TU Delft
- Liedewij Laan - TU Delft

**Contributors & Advisors:**
- David Tax - TU Delft
- Jorge Martinez - TU Delft
- Members of the Bioinformatics Lab at TU Delft

**Data Sources:**
- Genomic annotations from the [Saccharomyces Genome Database (SGD)](https://www.yeastgenome.org/)

<!-- ## License

> 📝 **Note**: Please add an appropriate license for your project (e.g., MIT, GPL-3.0, Apache-2.0). -->

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{oosterlaar2026transposon,
  title={Transposon Truths: Identifying Essential Gene Domains in Yeast Using SATAY Data},
  author={Oosterlaar, Nina},
  year={2026},
  school={Delft University of Technology},
  type={Master's Thesis}
}
``` -->

## Contact

For questions or issues, please open an issue on GitHub or contact N.I.M.Oosterlaar@dstudent.tudelft.nl.