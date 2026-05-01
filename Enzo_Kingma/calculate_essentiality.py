import numpy as np
import os, sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from gene_reader import geneClassifier
from transposon_reader import fit_centromere_bias_from_rates, read_strain_data

def get_central_80_region(start, end):
    """Return the central 80% of a gene."""
    gene_start = min(start, end)
    gene_end = max(start, end)
    gene_length = gene_end - gene_start + 1
    central_start = gene_start + int(np.floor(0.10 * gene_length))
    central_end = gene_end - int(np.floor(0.10 * gene_length))
    return central_start, central_end

def count_observed_insertions_gene(chromosome_df, central_start, central_end):
    """Count observed insertion sites and return their read counts."""
    gene_insertions = chromosome_df[
        (chromosome_df["Position"] >= central_start) &
        (chromosome_df["Position"] <= central_end) &
        (chromosome_df["Value"] > 0)
    ]
    observed_insertions = len(gene_insertions)
    observed_reads = gene_insertions["Value"].to_numpy(dtype=float)
    return observed_insertions, observed_reads

def count_expected_insertions_gene(
    chromosome_df,
    gene_start,
    gene_end,
    central_length,
    insertion_rate,
):
    """Estimate expected insertions using centromere-distance correction."""
    gene_center = int(round((min(gene_start, gene_end) + max(gene_start, gene_end)) / 2))

    gene_centromere_distance = chromosome_df.loc[
        gene_center,
        "Centromere_Distance"
    ]
    expected_insertions = int(
        np.floor(insertion_rate(gene_centromere_distance) * central_length)
    )
    return expected_insertions, gene_centromere_distance

def add_zero_counts(observed_reads, observed_insertions, expected_insertions):
    """Add missing expected insertions as zero-read sites."""
    added_zero_insertions = max(0, expected_insertions - observed_insertions)
    reads_with_zeros = np.concatenate([
        observed_reads,
        np.zeros(added_zero_insertions),
    ])
    return reads_with_zeros, added_zero_insertions

def remove_outliers(reads):
    """Remove high read-count outliers using the paper's 5–95 percentile rule."""
    reads = np.asarray(reads, dtype=float)
    if len(reads) == 0:
        return reads
    p05 = np.percentile(reads, 5)
    p95 = np.percentile(reads, 95)
    upper_threshold = p95 + 1.5 * (p95 - p05)
    return reads[reads <= upper_threshold]

def exclude_gene(reads, min_insertions=5):
    """Return True if gene should be excluded from fitness calculation."""
    if len(reads) < min_insertions:
        return True
    if np.sum(reads) == 0:
        return True
    return False

def process_single_gene(gene, chromosome_df, insertion_rate, min_insertions=5):
    """Apply the paper's preprocessing steps to one gene."""
    gene_name = gene["name"]
    start = gene["start"]
    end = gene["end"]
    is_essential = gene["is_essential"]
    central_start, central_end = get_central_80_region(start, end)
    central_length = central_end - central_start + 1
    observed_insertions, observed_reads = count_observed_insertions_gene(
        chromosome_df,
        central_start,
        central_end,
    )
    expected_insertions, gene_centromere_distance = count_expected_insertions_gene(
        chromosome_df,
        start,
        end,
        central_length,
        insertion_rate,
    )
    reads_with_zeros, added_zero_insertions = add_zero_counts(
        observed_reads,
        observed_insertions,
        expected_insertions,
    )
    reads_filtered = remove_outliers(reads_with_zeros)
    excluded = exclude_gene(reads_filtered, min_insertions=min_insertions)
    if excluded:
        mean_read_count = np.nan
        sample_variance = np.nan
        valid_for_fitness = False
    else:
        mean_read_count = np.mean(reads_filtered)
        sample_variance = np.var(reads_filtered, ddof=1)
        valid_for_fitness = True
    return {
        "gene": gene_name,
        "start": start,
        "end": end,
        "central_start": central_start,
        "central_end": central_end,
        "central_length": central_length,
        "is_essential": is_essential,
        "gene_centromere_distance": gene_centromere_distance,
        "observed_insertions": observed_insertions,
        "expected_insertions": expected_insertions,
        "added_zero_insertions": added_zero_insertions,
        "n_used_after_outlier_removal": len(reads_filtered),
        "mean_read_count": mean_read_count,
        "sample_variance": sample_variance,
        "valid_for_fitness": valid_for_fitness,
    }

def process_genes(chr_data, classifier, insertion_rate, min_insertions=5):
    """Process all genes across all chromosomes."""
    results = []
    for chrom, chromosome_df in chr_data.items():
        genes = classifier.get_chromosome_genes(chrom)
        if len(genes) == 0:
            print(f"No genes found for chromosome name: {chrom}")
            continue
        chromosome_df = chromosome_df.copy()
        chromosome_df = chromosome_df[np.isfinite(chromosome_df["Centromere_Distance"])]
        for gene in genes:
            result = process_single_gene(
                gene=gene,
                chromosome_df=chromosome_df,
                insertion_rate=insertion_rate,
                min_insertions=min_insertions,
            )
            result["chromosome"] = chrom
            results.append(result)
    return pd.DataFrame(results)

def calculate_fitness(gene_stats_df):
    """Calculate normalized fitness score from mean read count."""
    df = gene_stats_df.copy()
    valid_mask = (
        df["valid_for_fitness"] &
        np.isfinite(df["mean_read_count"]) &
        (df["mean_read_count"] > 0)
    )
    median_log_mean = np.median(np.log(df.loc[valid_mask, "mean_read_count"]))
    df["fitness_score"] = np.nan
    df.loc[valid_mask, "fitness_score"] = (
        np.log(df.loc[valid_mask, "mean_read_count"]) / median_log_mean
    )
    return df, median_log_mean


if __name__ == "__main__":
    # Load gene information
    strain_path = "Data/combined_strains/strain_yEK19"
    gene_path = "Utils/SGD_API/architecture_info/yeast_genes_with_info.json"
    classifier = geneClassifier(gene_path)
    data = read_strain_data(strain_path)
    coeffs_rate, rate_poly, insertion_rate, rate_df = fit_centromere_bias_from_rates(data)
    gene_stats_df = process_genes(data, classifier, insertion_rate)
    gene_stats_df, median_log_mean = calculate_fitness(gene_stats_df)
    print(gene_stats_df.head())
    # Print average fitness scores for essential vs non-essential genes
    essential_avg_fitness = gene_stats_df.loc[gene_stats_df["is_essential"], "fitness_score"].mean()
    non_essential_avg_fitness = gene_stats_df.loc[~gene_stats_df["is_essential"], "fitness_score"].mean()
    print(f"\nAverage fitness score for essential genes: {essential_avg_fitness:.4f}")
    print(f"Average fitness score for non-essential genes: {non_essential_avg_fitness:.4f}")