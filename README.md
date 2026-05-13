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


## 3. Autoencoder training and analysis

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


## 4. Generate Data for CPD analysis

### Generate synthetic SATAY datasets

Synthetic SATAY datasets are generated with `Signal_processing/sample_data/SATAY_sim.py`. The script samples region-level insertion counts from a negative-binomial process, generates nucleosome and centromere distances, derives position-specific pi values from the empirical density lookup tables, and applies those pi values as dropout probabilities. By default, generated datasets are written to `Data/SATAY_synthetic`.

```bash
python Signal_processing/sample_data/SATAY_sim.py \
  --number_of_samples 10 \
  --output_folder Data/SATAY_synthetic
```

To see all configurable generation parameters, run:

```bash
python Signal_processing/sample_data/SATAY_sim.py --help
```

### Datasets with different saturations

The CPD saturation datasets are stored under `Data/test_CPD`. Higher-saturation datasets are created by combining datasets from the base saturation level, while lower-saturation datasets are created by randomly removing non-zero insertions.

Create higher saturation levels:

```bash
python Utils/create_different_saturations.py \
  --source_folder Data/test_CPD/1 \
  --output_base Data/test_CPD
```

Create lower saturation levels:

```bash
python Utils/make_sparser.py \
  --input_folder Data/test_CPD/1 \
  --output_base Data/test_CPD
```

After the saturation folders have been created, extract the centromere windows. These are saved by default in `Data/test_CPD/centromere_windows`.

```bash
python Utils/extract_centromere_windows.py test_cpd \
  --base_path Data/test_CPD
```

Finally, generate the nucleosome and centromere bias files required for CPD:

```bash
python Data_exploration/densities/generate_test_cpd_densities.py test_cpd \
  --base_path Data/test_CPD
```

### Generate AE output

First train a ZINB autoencoder on all chromosomes with `AE/training/train_all_chromosomes.py`. This creates the model used for AE-based CPD reconstruction.

```bash
python AE/training/train_all_chromosomes.py
```

To see all configurable training parameters, run:

```bash
python AE/training/train_all_chromosomes.py --help
```

Then reconstruct the `Data/test_CPD` saturation datasets with `AE/reconstruction/reconstruct_CPD_test.py`. By default, this script writes the reconstructed outputs to `Data/reconstruction_cpd_test_all_chrom`. This code reconstructs the data from all the different reconstruction levels created in the previous step. 

```bash
python AE/reconstruction/reconstruct_CPD_test.py \
  --test_cpd_folder Data/test_CPD \
  --output_dir Data/reconstruction_cpd_test_all_chrom
```

Centromere windows can be extracted from AE reconstructions with `Utils/extract_centromere_windows.py` in `reconstruction` mode. In this mode, the default reconstruction input path is `Data/reconstruction_cpd_test_all_chrom`.

```bash
python Utils/extract_centromere_windows.py reconstruction \
  --base_path Data/reconstruction_cpd_test_all_chrom
```

## 5. Run CPD analyses

### ZINB CPD on synthetic datasets

Performance for different CPD window sizes can be computed with `Signal_processing/CPD_test_parameters/window_performance.py`. The script evaluates a list of window sizes over the synthetic SATAY datasets and writes precision-recall summaries and plots to the selected output folder.

```bash
python Signal_processing/CPD_test_parameters/window_performance.py \
  --base_folder Data/SATAY_synthetic \
  --output_folder Signal_processing/results_new/compare_window_performance \
  --window_sizes 10,20,30,50,80,100,150,200 \
  --overlap 0.5 \
  --threshold_min 0 \
  --threshold_max 40 \
  --threshold_step 1 \
  --dataset_start 1 \
  --dataset_end 10
```

The informed CPD versions can be compared against the reference CPD algorithm with `Signal_processing/CPD_test_parameters/compare_versions.py`. This compares the reference method and the informed versions on the synthetic datasets for one selected window size.

```bash
python Signal_processing/CPD_test_parameters/compare_versions.py \
  --base_folder Data/SATAY_synthetic \
  --output_folder Signal_processing/results_new/compare_versions_ws100 \
  --window_size 100 \
  --overlap 0.5 \
  --theta 0 \
  --threshold_min 0 \
  --threshold_max 40 \
  --threshold_step 1 \
  --dataset_start 1 \
  --dataset_end 10 \
  --n_workers 8
```

For all available options, run each script with `--help`.

### ZINB CPD on raw data

ZINB-based CPD can be run on SATAY centromere windows with `Signal_processing/CPD_on_SATAY/run_ZINB_CPD_SATAY_v3_windows.py`. This uses the centromere-window files and the per-dataset nucleosome/centromere bias files generated above.

```bash
python Signal_processing/CPD_on_SATAY/run_ZINB_CPD_SATAY_v3_windows.py \
  --windows_base Data/test_CPD/centromere_windows \
  --test_cpd_base Data/test_CPD \
  --output_base Signal_processing/results_new/CPD_SATAY_v3_window \
  --window_size 100 \
  --overlap 0.5 \
  --threshold_start 0 \
  --threshold_end 20 \
  --threshold_step 1
```

To run ZINB-based CPD on the whole genome for the strain-level SATAY datasets, use `Signal_processing/CPD_on_SATAY/run_ZINB_CPD_SATAY_v3_strains.py`. By default, the results are saved in `SATAY_CPD_results/CPD_SATAY_results`.

```bash
python Signal_processing/CPD_on_SATAY/run_ZINB_CPD_SATAY_v3_strains.py \
  --output_base SATAY_CPD_results/CPD_SATAY_results \
  --window_size 100 \
  --overlap 0.5 \
  --threshold_start 5 \
  --threshold_end 10 \
  --threshold_step 0.5 \
  --n_strain_workers 1
```

For all available options, run each script with `--help`.

### CPD on AE output

CPD on AE output is performed with a Gaussian sliding-mean CPD, not with the ZINB-based CPD. The AE reconstructions are dense continuous signals rather than sparse count data, so the Gaussian CPD is applied to the reconstructed centromere windows with `Signal_processing/CPD_on_SATAY/Gaussian_CPD_AE_reconstruction.py`.

```bash
python Signal_processing/CPD_on_SATAY/Gaussian_CPD_AE_reconstruction.py \
  --input_dir Data/reconstruction_cpd_test_all_chrom/centromere_window \
  --output_dir Signal_processing/results_new/Gaussian_AE_CPD \
  --window_size 100 \
  --overlap 0.5 \
  --threshold_start 0 \
  --threshold_end 20 \
  --threshold_step 1 \
  --saturation_levels 0 1 2 3 4 5 6 7
```

Gaussian CPD can also be applied to raw SATAY centromere-window data after moving-average preprocessing with `Signal_processing/CPD_on_SATAY/Gaussian_CPD_Moving_average.py`.

### Evaluation of CPD on SATAY windows

Precision-recall curves for CPD on SATAY centromere windows can be generated with `Signal_processing/CPD_evaluation/SATAY_CPD_performance.py`. The script has method presets for the ZINB-based CPD, Gaussian CPD on AE reconstructions, and Gaussian CPD on moving-average raw data.

```bash
python Signal_processing/CPD_evaluation/SATAY_CPD_performance.py \
  --method zinb \
  --data_base Data/test_CPD/centromere_windows \
  --saturation_levels 0 1 2 3 4 5 6 7 \
  --window_size 100
```

For the other SATAY-window methods, use `--method gaussian_ae` or `--method gaussian_ma`. Custom result and output folders can be supplied with `--method custom`, `--results_base`, and `--output_folder`.

## 6. Calculate Essentiality Scores

### Compare informed and pure estimation

The performance comparison between informed and pure segment-level essentiality estimation can be calculated with `Signal_processing/essentiality_calculation/performance_essentiality.py`. This evaluates the `segment_mu` and `segment_mu_informed` outputs on the synthetic SATAY datasets.

```bash
python Signal_processing/essentiality_calculation/performance_essentiality.py \
  --base_data_folder Data/SATAY_synthetic \
  --base_results_folder Signal_processing/results_new/essentiality_score \
  --datasets 1 2 3 4 5 6 7 8 9 10 \
  --estimator_subdirs segment_mu segment_mu_informed \
  --threshold_min 0 \
  --threshold_max 10
```

### Calculate strain essentiality scores

Essentiality scores for the SATAY strain data are calculated with `Signal_processing/essentiality_calculation/calculate_strain_essentiality.py`. This script uses the whole-genome CPD results and writes segment-level mu estimates and mu z-scores.

```bash
python Signal_processing/essentiality_calculation/calculate_strain_essentiality.py \
  --base_data_folder Data/combined_strains \
  --base_results_folder SATAY_CPD_results/CPD_SATAY_results \
  --thresholds 3.0 \
  --output_subdir segment_mu \
  --summary_output Signal_processing/results_new/essentiality_calculation/strain_essentiality_summary.csv
```

### Merge similar segments

Segments with similar essentiality scores can be merged with `Signal_processing/essentiality_calculation/merge_segments.py`. The merge is based on adjacent segment differences in `mu_z_score`.

```bash
python Signal_processing/essentiality_calculation/merge_segments.py \
  --base-dir SATAY_CPD_results/CPD_SATAY_results \
  --input-th 3.0 \
  --merge-threshold 0.25
```

## 7. Compare CPD segments with known annotations

### Essentiality enrichment with SGD annotations

The alignment between CPD-derived essentiality scores and known SGD essentiality annotations can be analyzed with `SATAY_CPD_results/essentiality_enrichment/essentiality_enrichment_analysis.py`.

```bash
python SATAY_CPD_results/essentiality_enrichment/essentiality_enrichment_analysis.py \
  --strains yEK19 \
  --threshold 3.0 \
  --n_bins 15 \
  --output_dir SATAY_CPD_results/results/essentiality_enrichment
```

### Compare with Kingma et al. 2024

The CPD-derived position-level essentiality scores can also be compared to the gene-level method of Kingma et al. (2024) with `Enzo_Kingma/compare_essentiality_scores.py`.

```bash
python Enzo_Kingma/compare_essentiality_scores.py \
  --strain yEK19 \
  --strain_data_path Data/combined_strains/strain_yEK19 \
  --segments_base_path SATAY_CPD_results/CPD_SATAY_results \
  --threshold 3.0 \
  --mu_z 0.25
```

### Change points near genes and gene boundaries

The distance between change points and annotated gene boundaries can be analyzed with `SATAY_CPD_results/boundary_alignment/changepoint_boundary_alignment_analysis.py`. The number of change points within genes can be summarized with `SATAY_CPD_results/change_points_within_gene/change_points_within_gene_histogram.py`.

```bash
python SATAY_CPD_results/boundary_alignment/changepoint_boundary_alignment_analysis.py \
  --thresholds 3.0 \
  --strains FD yEK19 yEK23 \
  --window_size 100 \
  --output_dir SATAY_CPD_results/results/boundary_alignment
```

```bash
python SATAY_CPD_results/change_points_within_gene/change_points_within_gene_histogram.py \
  --strains FD yEK19 yEK23 \
  --threshold 3.0 \
  --merged_segments_threshold 0.25 \
  --gene_extension_bp 100
```

### Compare strains

The ARI index and Pearson correlation between strains can be compared with `SATAY_CPD_results/compare_strains.py/compare_strains.py`.

```bash
python SATAY_CPD_results/compare_strains.py/compare_strains.py \
  --base_path SATAY_CPD_results/CPD_SATAY_results \
  --strains FD yEK19 yEK23 \
  --thresholds 3.0 \
  --mu_z_threshold 0.25 \
  --tolerance 100
```

### Gene overview plots

Example gene overview plots can be generated with `SATAY_CPD_results/genes_overview_plots/genes_plot.py`. If no genes are passed, the script uses the default gene list defined in the file; passing `--genes` restricts the output to those genes.

```bash
python SATAY_CPD_results/genes_overview_plots/genes_plot.py \
  --genes PRO3 CDC28 SEC18 \
  --strains FD yEK19 yEK23 \
  --threshold 3.0 \
  --mu_z_threshold 0.25
```


---

For questions or comments, feel free to reach out to me at <a href="mailto:ninaoosterlaar@gmail.com">ninaoosterlaar@gmail.com</a>.

Note: This README was generated with the assistance of Codex.
