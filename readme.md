# Thesis Reproducibility Guide

This repository contains the code used to obtain the results for my thesis.

Thesis report: https://repository.tudelft.nl/record/uuid:6360bc4f-c995-4a35-b43e-f80190f1279d

## Abstract

<!-- Paste the thesis abstract here. -->

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


