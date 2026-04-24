import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Add Utils to path
sys.path.append(str(Path(__file__).parent.parent / "Utils"))
from SGD_API.yeast_genes import SGD_Genes


class EssentialityAnalysis:
    def __init__(self, base_path, gene_info_file, output_dir):
        """
        Initialize the essentiality analysis.
        
        Parameters:
        base_path (str): Path to Signal_processing/strains directory
        gene_info_file (str): Path to yeast_genes_with_info.json
        output_dir (str): Directory to save intermediate results
        """
        self.base_path = Path(base_path)
        self.gene_info_file = gene_info_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for data files
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load gene information
        print("Loading gene information...")
        self.sgd_genes = SGD_Genes(gene_list_with_info=gene_info_file)
        self.essential_genes = set(self.sgd_genes.list_essential_genes())
        self.nonessential_genes = set(self.sgd_genes.list_nonessential_genes())
        
        print(f"Loaded {len(self.essential_genes)} essential genes")
        print(f"Loaded {len(self.nonessential_genes)} non-essential genes")
        
        # Store results
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # results[strain][threshold][gene] = weighted_mean_mu_z
        
    def chromosome_name_mapping(self, chrom_name):
        """Map chromosome names between formats (e.g., Chromosome_I -> ChrI)"""
        if chrom_name.startswith("Chromosome_"):
            num = chrom_name.replace("Chromosome_", "")
            return f"Chr{num}"
        return chrom_name
    
    def calculate_weighted_mean_mu_z(self, gene_info, segments_df):
        """
        Calculate position-weighted mean mu_z_score for a gene.
        
        Each position in the gene contributes to the mean based on which segment it falls in.
        
        Parameters:
        gene_info (dict): Gene information with location
        segments_df (pd.DataFrame): Segments for the chromosome
        
        Returns:
        float: Weighted mean mu_z_score, or None if no overlap
        """
        gene_start = gene_info["location"]["start"]
        gene_end = gene_info["location"]["end"]
        gene_length = gene_end - gene_start
        
        if gene_length == 0:
            return None
        
        # Find overlapping segments
        weighted_sum = 0
        total_overlap = 0
        
        for _, segment in segments_df.iterrows():
            # Segment positions (assuming position units match gene positions)
            # start_index and end_index_exclusive are in window units
            # We need to convert to genomic coordinates
            # For now, assuming they're indexed in a way that matches genomic positions
            seg_start = segment['start_index']
            seg_end = segment['end_index_exclusive']
            
            # Calculate overlap
            overlap_start = max(gene_start, seg_start)
            overlap_end = min(gene_end, seg_end)
            overlap_length = max(0, overlap_end - overlap_start)
            
            if overlap_length > 0:
                weighted_sum += overlap_length * segment['mu_z_score']
                total_overlap += overlap_length
        
        if total_overlap == 0:
            return None
        
        return weighted_sum / total_overlap
    
    def save_strain_threshold_results(self, strain, threshold):
        """
        Save results for a specific strain/threshold combination to CSV.
        
        Parameters:
        strain (str): Strain name
        threshold (str): Threshold value
        """
        if threshold not in self.results[strain]:
            return
        
        # Prepare data for saving
        data_rows = []
        for gene_id, mu_z in self.results[strain][threshold].items():
            gene_info = self.sgd_genes.retrieve_gene(gene_id)
            data_rows.append({
                'gene_id': gene_id,
                'gene_name': gene_info['gene_name'],
                'chromosome': gene_info['location']['chromosome'],
                'start': gene_info['location']['start'],
                'end': gene_info['location']['end'],
                'essentiality': gene_info['essentiality'],
                'weighted_mean_mu_z': mu_z
            })
        
        # Save to CSV
        df = pd.DataFrame(data_rows)
        output_file = self.data_dir / f"{strain}_th{threshold}_gene_mu_z.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved results: {output_file}")
    
    def process_strain_threshold(self, strain, threshold, mu_z=0.25):
        """
        Process all chromosomes for a given strain and threshold.
        
        Parameters:
        strain (str): Strain name (e.g., 'strain_dnrp')
        threshold (str): Threshold value (e.g., '2.00')
        mu_z (float): Mu Z-score threshold for merged segments (e.g., 0.25)
        """
        print(f"\nProcessing {strain} with threshold {threshold}...")
        
        strain_path = self.base_path / strain
        if not strain_path.exists():
            print(f"  Strain path not found: {strain_path}")
            return
        
        # Iterate through all chromosomes
        for chrom_dir in strain_path.iterdir():
            if not chrom_dir.is_dir() or not chrom_dir.name.startswith("Chr"):
                continue
            
            chrom_name = chrom_dir.name
            merged_segments_path = chrom_dir / f"{chrom_name}_distances" / "window100" / "merged_segments"
            
            if not merged_segments_path.exists():
                continue
            
            # Find the CSV file for this threshold
            csv_pattern = f"{chrom_name}_th{threshold}_merged_segments_muZ{mu_z}.csv"
            csv_file = merged_segments_path / csv_pattern
            
            if not csv_file.exists():
                continue
            
            # Load segment data
            segments_df = pd.read_csv(csv_file)
            print(f"  Processing {chrom_name}: {len(segments_df)} segments")
            
            # Map chromosome name for gene lookup
            chrom_gene_format = f"Chromosome_{chrom_name.replace('Chr', '')}"
            
            # Process all genes on this chromosome
            for gene_id, gene_data in self.sgd_genes.list_all_gene_info().items():
                if gene_data["location"]["chromosome"] != chrom_gene_format:
                    continue
                
                # Calculate weighted mean mu_z for this gene
                weighted_mean = self.calculate_weighted_mean_mu_z(gene_data, segments_df)
                
                if weighted_mean is not None:
                    self.results[strain][threshold][gene_id] = weighted_mean
        
        # Save results after processing this strain/threshold combination
        self.save_strain_threshold_results(strain, threshold)
        
        # Generate and save individual plot immediately
        self.create_individual_plot(strain, threshold)
    
    def collect_all_data(self, mu_z=0.25):
        """Process all strains and thresholds, generating plots as we go."""
        # Define thresholds to analyze
        thresholds = ['0.50', '1.00', '1.50', '2.00', '2.50', '3.00', '3.50', '4.00', '4.50', '5.00']
        
        # Get all strain directories
        strains = [d.name for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith("strain_")]
        strains.sort()
        
        print(f"Found {len(strains)} strains: {', '.join(strains)}")
        
        for threshold in thresholds:
            print(f"\n{'='*60}")
            print(f"Processing threshold {threshold}")
            print(f"{'='*60}")
            
            for strain in strains:
                self.process_strain_threshold(strain, threshold, mu_z)
            
            # After all strains for this threshold, create combined plot
            self.create_combined_plot(threshold)
    
    def remove_top_outliers(self, values, percentile=95):
        """
        Remove the top outliers from a list of values.
        
        Parameters:
        values (list): List of numeric values
        percentile (int): Percentile threshold (default 95 for top 5%)
        
        Returns:
        list: Values with top outliers removed
        """
        if not values:
            return values
        
        threshold = np.percentile(values, percentile)
        filtered_values = [v for v in values if v <= threshold]
        
        removed_count = len(values) - len(filtered_values)
        if removed_count > 0:
            print(f"    Removed {removed_count} outliers (>{threshold:.3f})")
        
        return filtered_values
    
    def prepare_plotting_data(self, threshold):
        """
        Prepare data for plotting for a specific threshold.
        
        Returns:
        dict: Dictionary with strain -> {essential: [values], nonessential: [values]}
        """
        plotting_data = {}
        
        for strain in self.results.keys():
            if threshold not in self.results[strain]:
                continue
            
            essential_values = []
            nonessential_values = []
            
            for gene_id, mu_z in self.results[strain][threshold].items():
                if gene_id in self.essential_genes:
                    essential_values.append(mu_z)
                elif gene_id in self.nonessential_genes:
                    nonessential_values.append(mu_z)
            
            if essential_values or nonessential_values:
                plotting_data[strain] = {
                    'essential': essential_values,
                    'nonessential': nonessential_values
                }
        
        return plotting_data
    
    def create_individual_plot(self, strain, threshold):
        """Create individual box plot for a specific strain and threshold."""
        if threshold not in self.results[strain]:
            return
        
        # Prepare data for this strain
        essential_values = []
        nonessential_values = []
        
        for gene_id, mu_z in self.results[strain][threshold].items():
            if gene_id in self.essential_genes:
                essential_values.append(mu_z)
            elif gene_id in self.nonessential_genes:
                nonessential_values.append(mu_z)
        
        if not essential_values or not nonessential_values:
            print(f"  Insufficient data for plot (essential: {len(essential_values)}, non-essential: {len(nonessential_values)})")
            return
        
        # Remove top 5% outliers
        print(f"  Filtering outliers for {strain}:")
        print(f"    Essential: {len(essential_values)} genes")
        essential_values = self.remove_top_outliers(essential_values)
        print(f"    Non-essential: {len(nonessential_values)} genes")
        nonessential_values = self.remove_top_outliers(nonessential_values)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        plot_data = [essential_values, nonessential_values]
        labels = ['Essential', 'Non-essential']
        
        # Create box plot
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['#ff7f0e', '#1f77b4']  # Orange for essential, blue for non-essential
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Statistical test
        stat, pval = stats.mannwhitneyu(essential_values, nonessential_values, alternative='two-sided')
        ax.text(0.5, 0.98, f'Mann-Whitney U p-value: {pval:.2e}', 
               transform=ax.transAxes, ha='center', va='top', fontsize=10)
        
        ax.set_ylabel('Weighted Mean μ z-score', fontsize=12)
        ax.set_title(f'{strain} - Threshold {threshold}\n(Top 5% outliers removed)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Save plot
        output_file = self.output_dir / f'{strain}_th{threshold}_essentiality.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved plot: {output_file}")
    
    def create_combined_plot(self, threshold):
        """Create combined plot with two subfigures (essential vs nonessential across strains)."""
        plotting_data = self.prepare_plotting_data(threshold)
        
        if not plotting_data:
            print(f"\nNo data for combined plot at threshold {threshold}")
            return
        
        print(f"\nCreating combined plot for threshold {threshold}...")
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        strains = sorted(plotting_data.keys())
        
        # Left subplot: Essential genes across strains
        # Remove top 5% outliers from each strain's data
        essential_data = []
        essential_labels = []
        for strain in strains:
            if plotting_data[strain]['essential']:
                filtered = self.remove_top_outliers(plotting_data[strain]['essential'])
                if filtered:  # Only add if data remains after filtering
                    essential_data.append(filtered)
                    essential_labels.append(strain.replace('strain_', ''))
        
        if essential_data:
            bp1 = axes[0].boxplot(essential_data, labels=essential_labels, patch_artist=True)
            for patch in bp1['boxes']:
                patch.set_facecolor('#ff7f0e')
                patch.set_alpha(0.6)
            
            axes[0].set_ylabel('Weighted Mean μ z-score', fontsize=12)
            axes[0].set_xlabel('Strain', fontsize=12)
            axes[0].set_title('Essential Genes', fontsize=14, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
        
        # Right subplot: Non-essential genes across strains
        # Remove top 5% outliers from each strain's data
        nonessential_data = []
        nonessential_labels = []
        for strain in strains:
            if plotting_data[strain]['nonessential']:
                filtered = self.remove_top_outliers(plotting_data[strain]['nonessential'])
                if filtered:  # Only add if data remains after filtering
                    nonessential_data.append(filtered)
                    nonessential_labels.append(strain.replace('strain_', ''))
        
        if nonessential_data:
            bp2 = axes[1].boxplot(nonessential_data, labels=nonessential_labels, patch_artist=True)
            for patch in bp2['boxes']:
                patch.set_facecolor('#1f77b4')
                patch.set_alpha(0.6)
            
            axes[1].set_ylabel('Weighted Mean μ z-score', fontsize=12)
            axes[1].set_xlabel('Strain', fontsize=12)
            axes[1].set_title('Non-essential Genes', fontsize=14, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
        
        # Save combined plot
        plt.suptitle(f'μ z-score Distribution - Threshold {threshold}\n(Top 5% outliers removed)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save combined plot
        output_file = self.output_dir / f'combined_all_strains_th{threshold}_essentiality.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved combined plot: {output_file}\n")
    
    def generate_summary_report(self):
        report_file = self.output_dir / "summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ESSENTIALITY ANALYSIS SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write("Note: Statistics below include all data.\n")
            f.write("      Plots have top 5% outliers removed for better visualization.\n\n")
            
            for strain in sorted(self.results.keys()):
                f.write(f"\n{strain}:\n")
                f.write("-" * 60 + "\n")
                
                for threshold in sorted(self.results[strain].keys()):
                    essential_values = []
                    nonessential_values = []
                    
                    for gene_id, mu_z in self.results[strain][threshold].items():
                        if gene_id in self.essential_genes:
                            essential_values.append(mu_z)
                        elif gene_id in self.nonessential_genes:
                            nonessential_values.append(mu_z)
                    
                    if essential_values and nonessential_values:
                        stat, pval = stats.mannwhitneyu(essential_values, nonessential_values, alternative='two-sided')
                        
                        f.write(f"\n  Threshold {threshold}:\n")
                        f.write(f"    Essential genes: n={len(essential_values)}, "
                              f"mean={np.mean(essential_values):.3f}, median={np.median(essential_values):.3f}\n")
                        f.write(f"    Non-essential genes: n={len(nonessential_values)}, "
                              f"mean={np.mean(nonessential_values):.3f}, median={np.median(nonessential_values):.3f}\n")
                        f.write(f"    Mann-Whitney U test: p={pval:.2e}\n")
        
        print(f"\nSummary report saved: {report_file}")
    
    def print_summary_statistics(self):
        """Print summary statistics for all strains and thresholds."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        for strain in sorted(self.results.keys()):
            print(f"\n{strain}:")
            for threshold in sorted(self.results[strain].keys()):
                essential_values = []
                nonessential_values = []
                
                for gene_id, mu_z in self.results[strain][threshold].items():
                    if gene_id in self.essential_genes:
                        essential_values.append(mu_z)
                    elif gene_id in self.nonessential_genes:
                        nonessential_values.append(mu_z)
                
                if essential_values and nonessential_values:
                    stat, pval = stats.mannwhitneyu(essential_values, nonessential_values, alternative='two-sided')
                    
                    print(f"  Threshold {threshold}:")
                    print(f"    Essential genes: n={len(essential_values)}, "
                          f"mean={np.mean(essential_values):.3f}, median={np.median(essential_values):.3f}")
                    print(f"    Non-essential genes: n={len(nonessential_values)}, "
                          f"mean={np.mean(nonessential_values):.3f}, median={np.median(nonessential_values):.3f}")
                    print(f"    Mann-Whitney U test: p={pval:.2e}")


def main():
    # Setup paths
    base_path = Path(__file__).parent.parent / "Signal_processing" / "strains"
    gene_info_file = Path(__file__).parent.parent / "Utils" / "SGD_API" / "architecture_info" / "yeast_genes_with_info.json"
    output_dir = Path(__file__).parent / "essentiality_plots"
    
    # Create analysis object
    analysis = EssentialityAnalysis(str(base_path), str(gene_info_file), str(output_dir))
    
    # Collect all data and generate plots as we go
    analysis.collect_all_data()
    
    # Print summary statistics
    analysis.print_summary_statistics()
    
    # Generate summary report
    analysis.generate_summary_report()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Plots: {output_dir}/*.png")
    print(f"  - Data: {output_dir}/data/*.csv")
    print(f"  - Report: {output_dir}/summary_report.txt")
    print("="*80)
    
    # Generate all plots
    analysis.generate_all_plots(str(output_dir))
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
