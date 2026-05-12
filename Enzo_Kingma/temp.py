import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from gene_reader import geneClassifier


def collect_gene_lengths(classifier):
    rows = []

    for chrom, genes in classifier.genes_by_chromosome.items():
        for gene in genes:
            start = gene["start"]
            end = gene["end"]
            length = abs(end - start) + 1

            rows.append({
                "chromosome": chrom,
                "gene": gene["name"],
                "start": start,
                "end": end,
                "length_bp": length,
                "is_essential": gene["is_essential"],
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    gene_path = "Utils/SGD_API/architecture_info/yeast_genes_with_info.json"
    classifier = geneClassifier(gene_path)

    gene_lengths = collect_gene_lengths(classifier)

    essential = gene_lengths[gene_lengths["is_essential"] == True]["length_bp"]
    nonessential = gene_lengths[gene_lengths["is_essential"] == False]["length_bp"]

    print("Essential genes:")
    print(essential.describe())

    print("\nNon-essential genes:")
    print(nonessential.describe())

    plt.figure(figsize=(8, 5))

    bins = np.linspace(
        gene_lengths["length_bp"].min(),
        gene_lengths["length_bp"].quantile(0.99),
        60,
    )

    plt.hist(
        nonessential,
        bins=bins,
        alpha=0.6,
        density=True,
        color="blue",
        label=f"Non-essential genes (n={len(nonessential)})",
    )

    plt.hist(
        essential,
        bins=bins,
        alpha=0.6,
        density=True,
        color="red",
        label=f"Essential genes (n={len(essential)})",
    )

    plt.xlabel("Gene length (bp)")
    plt.ylabel("Density")
    plt.title("Distribution of gene lengths in S. cerevisiae")
    plt.legend()
    plt.tight_layout()
    plt.show()