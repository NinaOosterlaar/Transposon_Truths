import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.SGD_API.yeast_genes import SGD_Genes

def convert_chromosome_name(chrom_name):
    """Convert chromosome names to a consistent format (e.g., 'Chromosome_I' to 'ChrI')."""
    if chrom_name.startswith("Chromosome_"):
        return "Chr" + chrom_name.split("_")[1]
    return chrom_name

# Retrieve all genes, their chromosomes, essentiality status, start and end positions and length
class geneClassifier:
    def __init__(self, gene_json_path):
        self.sgd_genes = SGD_Genes(gene_json_path)
        self.genes_by_chromosome = self._organize_genes_by_chromosome()
    
    def _organize_genes_by_chromosome(self):
        """Organize genes by chromosome for efficient lookup.
        Use the gene name as key and store chromosome, start, end, and essentiality status as values."""
        genes_by_chr = {}
        for gene_name, gene_info in self.sgd_genes.list_all_gene_info().items():
            chrom = gene_info['location']['chromosome']
            chrom = convert_chromosome_name(chrom)
            if chrom not in genes_by_chr:
                genes_by_chr[chrom] = []
            genes_by_chr[chrom].append({
                'name': gene_name,
                'start': gene_info['location']['start'],
                'end': gene_info['location']['end'],
                'is_essential': gene_info['essentiality']})
        return genes_by_chr

    def get_chromosome_genes(self, chrom):
        """Get all genes on a given chromosome."""
        return self.genes_by_chromosome.get(chrom, [])
    
if __name__ == "__main__":
    gene_path = "Utils/SGD_API/architecture_info/yeast_genes_with_info.json"
    classifier = geneClassifier(gene_path)
    print(classifier.get_chromosome_genes("Chromosome_I")[:5])  # Print first 5 genes on chromosome I for verification