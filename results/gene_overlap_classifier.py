import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add parent directory to import Utils
sys.path.append(str(Path(__file__).parent.parent))
from Utils.SGD_API.yeast_genes import SGD_Genes


class PositionClassifier:
    """
    Lightweight wrapper around SGD_Genes for position-level classification.
    """
    
    def __init__(self, gene_json_path: str):
        """
        Initialize using SGD_Genes.
        
        Args:
            gene_json_path: Path to yeast_genes_with_info.json
        """
        self.sgd_genes = SGD_Genes(gene_list_with_info=gene_json_path)
        self._build_chromosome_index()
    
    @staticmethod
    def _normalize_chromosome_name(chrom_name: str) -> str:
        """
        Normalize chromosome names to standard format (ChrI, ChrII, etc.).
        
        Handles:
        - Chromosome_I -> ChrI
        - Chromosome_V -> ChrV  
        - ChrI -> ChrI
        """
        if chrom_name.startswith("Chromosome_"):
            return "Chr" + chrom_name.split("_")[1]
        return chrom_name
    
    def _build_chromosome_index(self):
        """Build an index of genes organized by chromosome for fast lookup."""
        self.genes_by_chromosome: Dict[str, List[dict]] = {}
        
        all_genes_info = self.sgd_genes.list_all_gene_info()
        
        for gene_name, info in all_genes_info.items():
            location = info.get('location')
            if not location:
                continue
            
            chromosome = self._normalize_chromosome_name(location['chromosome'])
            
            gene_data = {
                'name': gene_name,
                'start': location['start'],
                'end': location['end'],
                'is_essential': info.get('essentiality', False)
            }
            
            if chromosome not in self.genes_by_chromosome:
                self.genes_by_chromosome[chromosome] = []
            self.genes_by_chromosome[chromosome].append(gene_data)
        
        # Sort by start position for efficient lookup
        for chrom in self.genes_by_chromosome:
            self.genes_by_chromosome[chrom].sort(key=lambda g: g['start'])
    
    def classify_position(self, chromosome: str, position: int) -> str:
        """
        Classify a single genomic position.
        
        Args:
            chromosome: Chromosome name (e.g., 'ChrI')
            position: Genomic position
        
        Returns:
            Classification: 'essential_gene', 'non_essential_gene', or 'outside_gene'
        """
        if chromosome not in self.genes_by_chromosome:
            return 'outside_gene'
        
        genes = self.genes_by_chromosome[chromosome]
        
        # Find overlapping genes
        overlapping_genes = [
            g for g in genes 
            if g['start'] <= position <= g['end']
        ]
        
        if not overlapping_genes:
            return 'outside_gene'
        
        # Priority: essential > non-essential
        for gene in overlapping_genes:
            if gene['is_essential']:
                return 'essential_gene'
        
        return 'non_essential_gene'
    
    def classify_positions_batch(self, chromosome: str, positions: np.ndarray) -> np.ndarray:
        """
        Classify multiple positions efficiently.
        
        Args:
            chromosome: Chromosome name
            positions: Array of genomic positions
        
        Returns:
            Array of classifications (same order as positions)
        """
        return np.array([
            self.classify_position(chromosome, int(pos)) 
            for pos in positions
        ])
    
    def get_chromosome_genes(self, chromosome: str) -> List[dict]:
        """Get all genes on a specific chromosome."""
        return self.genes_by_chromosome.get(chromosome, [])
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        total_genes = sum(len(genes) for genes in self.genes_by_chromosome.values())
        essential_count = sum(
            sum(1 for g in genes if g['is_essential'])
            for genes in self.genes_by_chromosome.values()
        )
        
        return {
            'total_genes': total_genes,
            'essential_genes': essential_count,
            'non_essential_genes': total_genes - essential_count,
            'chromosomes': len(self.genes_by_chromosome)
        }


def main():
    """Test the position classifier."""
    import os
    
    # Test with the actual gene database
    gene_db_path = os.path.join(
        os.path.dirname(__file__),
        '../Utils/SGD_API/architecture_info/yeast_genes_with_info.json'
    )
    
    print("Loading gene database...")
    classifier = PositionClassifier(gene_db_path)
    
    stats = classifier.get_statistics()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # # Test some positions
    # print("\nTesting position classification:")
    # test_cases = [
    #     ('ChrI', 1000),
    #     ('ChrI', 50000),
    #     ('ChrI', 100000),
    # ]
    
    # for chrom, pos in test_cases:
    #     classification = classifier.classify_position(chrom, pos)
    #     print(f"  {chrom}:{pos} -> {classification}")
    
    # # Test batch classification
    # print("\nTesting batch classification on ChrI positions 1000-1100:")
    # positions = np.arange(1000, 1100)
    # classifications = classifier.classify_positions_batch('ChrI', positions)
    # unique, counts = np.unique(classifications, return_counts=True)
    # for cls, count in zip(unique, counts):
    #     print(f"  {cls}: {count} positions")


if __name__ == '__main__':
    main()
