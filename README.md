# Thesis Reproducibility Guide

This repository contains the code used to obtain the results for my thesis.

Thesis report: https://repository.tudelft.nl/record/uuid:6360bc4f-c995-4a35-b43e-f80190f1279d

## Abstract

Saccharomyces cerevisiae is a model organism used for studying fundamental eukaryotic processes and genome function. While essentiality studies provide important insight into gene function, biological pathways, and evolution, they often focus on entire genes, thereby ignoring variation in essentiality within and beyond gene boundaries. SAturated Transposon Analysis in Yeast (SATAY) is a Transposon Insertion Sequencing (TIS) technique that can measure essentiality across the entire genome of S. cerevisiae. However, the resulting data is noisy and sparse, and is affected by strong insertion biases. Most existing methods for analyzing TIS data either rely on known gene annotations or are optimized for bacterial datasets. Methods optimized for bacterial datasets are difficult to apply to SATAY data due to its higher sparsity and distinct insertion patterns. We therefore aim to develop a method tailored to SATAY data that does not rely on predefined annotations.
In this study, we developed a change-point detection (CPD) algorithm to identify genomic regions with distinct levels of essentiality directly from SATAY data.
We explored whether an autoencoder could improve the quality of  SATAY data by denoising and imputing missing values. Although the autoencoder appeared to denoise the data, it did not perform meaningful imputation. Moreover, CPD applied to the AE output was outperformed by a CPD algorithm that modeled raw SATAY data as a zero-inflated negative binomial (ZINB) distribution. The ZINB-based CPD algorithm achieved better results by explicitly accounting for sparsity, overdispersion, and insertion biases.
The regions produced by our CPD algorithm align with biological expectations, though they are oversegmented and remain affected by known biases in the data. Despite these limitations, our results show that CPD applied to SATAY data is a promising first step towards identifying essential genomic regions beyond predefined gene boundaries across the whole genome.

## Repository Structure

- `AE/`: code related to the autoencoder models.
- `Batch/`: batch scripts used to run analyses on the DAIC supercomputer.
- `Data/`: raw and generated data used throughout the thesis workflow.
- `Data_exploration/`: exploratory analyses, mainly for nucleosome and centromere insertion biases, plus additional summary statistics.
- `Enzo_Kingma/`: code for performing SATAY analysis following Kingma et al. (2024): https://doi.org/10.1371/journal.pone.0312437.
- `SATAY_CPD_results/`: analyses comparing change-point detection results on SATAY data with known annotations.
- `Signal_processing/`: change-point detection and essentiality-score code.
- `Utils/`: general utility functions used across the repository.

## Environment

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The first processing step uses the `sgd-rest` dependency through `Utils/SGD_API/yeast_architecture.py` to obtain yeast nucleosome and centromere annotations.

## 1. Process WIG Files


On GitHub, only the WIG files in `Data/wiggle_format/` are included. The processed data folders used later in the analysis must be recreated by running the steps below.

The raw WIG files are stored in:

```text
Data/wiggle_format/
```

The goal of this step is to create:

```text
Data/distances_with_zeros_new/
Data/combined_strains/
```

### 1.1 Compute Distance-Annotated Insertion Files

Use `compute_distances()` from `Utils/reader.py`.

This function:

1. Recursively reads all `.wig` files in `Data/wiggle_format/`.
2. Parses the insertion positions and counts per chromosome.
3. Computes the distance from every genomic position to the nearest nucleosome and centromere.
4. Adds missing genomic positions with `Value = 0` when `with_zeros=True`.
5. Writes one CSV file per chromosome for each replicate.

Run from the repository root:

```bash
python Utils/reader.py \
  --input_dir Data/wiggle_format \
  --output_dir Data/distances_with_zeros_new \
  --with_zeros
```

Expected output structure:

```text
Data/distances_with_zeros_new/
  strain_FD/
    FD7_1_.../
      ChrI_distances.csv
      ChrII_distances.csv
      ...
  strain_dnrp/
  strain_yEK19/
  ...
```

Each chromosome CSV contains:

```text
Position, Value, Nucleosome_Distance, Centromere_Distance
```

### 1.2 Combine Replicates into Strain-Level Datasets

Use `combine_strain_datasets()` from `Utils/combine_data.py`.

This function combines all replicate folders within each strain folder in `Data/distances_with_zeros_new/` into one strain-level dataset.

Run from the repository root:

```bash
python Utils/combine_data.py \
  --input_dir Data/distances_with_zeros_new \
  --output_dir Data/combined_strains \
  --method average
```

Expected final output:

```text
Data/combined_strains/
  strain_FD/
    ChrI_distances.csv
    ChrII_distances.csv
    ...
  strain_dnrp/
  strain_yEK19/
  strain_yEK23/
  strain_yTW001/
  strain_yWT03a/
  strain_yWT04a/
  strain_ylic137/
```

The averaging method ignores zero values when at least one replicate has a non-zero insertion count at a position. If all values are zero, the combined value remains `0`.

## 2. Create nucleosome and centromere bias data and plots

The nucleosome and centromere insertion-bias plots are generated with:

```text
Data_exploration/densities/densities.py
```

To see all available command-line options, run:

```bash
python Data_exploration/densities/densities.py --help
```

The script computes two bias summaries:

- `density`: insertion rate/count density as a function of distance.
- `mean`: mean non-zero insertion count as a function of distance.

The input can be either of the processed data folders from step 1:

- `Data/combined_strains/`: creates bias plots for the combined strain-level datasets.
- `Data/distances_with_zeros_new/`: creates bias plots for the separate replicate/dataset-level files.

### 2.1 Bias plots for combined strains

Run from the repository root:

```bash
python Data_exploration/densities/densities.py \
  --input_dir Data/combined_strains \
  --output_dir Data_exploration/results/bias_plots/combined_strains \
  --target both \
  --step all \
  --metric both \
  --boolean \
  --bin_size 10000 \
  --combine_mode All \
  --plot_combined \
  --absolute_centromere_distance
```

This creates nucleosome and centromere density and mean-count tables and plots in:

```text
Data_exploration/results/bias_plots/combined_strains/
  density/
  mean/
```

### 2.2 Bias plots for separate datasets

To show the bias for each replicate/dataset separately, use `Data/distances_with_zeros_new/` as input:

```bash
python Data_exploration/densities/densities.py \
  --input_dir Data/distances_with_zeros_new \
  --output_dir Data_exploration/results/bias_plots/separate_datasets \
  --target both \
  --step all \
  --metric both \
  --boolean \
  --bin_size 10000 \
  --combine_mode Datasets \
  --plot_combined \
  --absolute_centromere_distance
```

The `--metric both` option creates both density-based and mean-count-based bias plots. The `--boolean` option converts insertion counts to presence/absence before calculating densities; the mean-count plots always use non-zero insertion counts.

## Autoencoder training and analysis

The ZINB autoencoder workflow is split into two entry points:

- `AE/training/bayesian_hyperparameter.py` runs Bayesian optimization through `run_bayesian_optimization(...)`. The hyperparameter search space is defined directly in that file; see the thesis appendix or the code for the exact ranges. Run `python AE/training/bayesian_hyperparameter.py --help` to see the available optimization arguments.
- `AE/main.py` trains and evaluates one specific autoencoder configuration. Run `python AE/main.py --help` to see the available command-line options for setting preprocessing, architecture, and training hyperparameters.

Bayesian optimization example:

```bash
python AE/training/bayesian_hyperparameter.py \
  --n_calls 50 \
  --n_initial_points 10 \
  --metric combined
```

Specific model training example:

```bash
python AE/main.py \
  --features Centr \
  --bin_size 20 \
  --moving_average true \
  --layers 1600 \
  --epochs 144 \
  --batch_size 32 \
  --learning_rate 0.00002
```

Additional autoencoder analyses were used to test the influence of the masking noise value and dataset saturation. These are implemented in `AE/test_AE/test_noise_influence.py` and `AE/test_AE/test_saturation.py`, respectively. The corresponding plotting scripts are in `AE/plotting`, including the noise-influence and saturation plotting code.

When `save_reconstruction=True` in `AE/main.py`, reconstructed outputs are saved under `Data/reconstruction`. Differences between pi predictions for zero and non-zero insertion positions can then be calculated with the code in `AE/imputation/imputation.py`.


Note: This README was generated with the assistance of Codex.
