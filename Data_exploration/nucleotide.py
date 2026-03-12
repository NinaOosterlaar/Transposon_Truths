import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.plot_config import setup_plot_style, COLORS
from Utils.SGD_API.genome import Genome
from Utils.reader import read_wig, label_from_filename
import math
from collections import defaultdict, Counter

# Set up standardized plot style
setup_plot_style()
from statistics import mean, stdev

def _revcomp(s: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return s.translate(comp)[::-1]

def _canonical_kmer(s: str) -> str:
    rc = _revcomp(s)
    return s if s <= rc else rc  # lexicographically smallest of (kmer, revcomp)

def _collapse_rc_counts(counts: dict) -> dict:
    """Collapse a dict{kmer->count} by summing reverse-complement pairs."""
    out = {}
    for kmer, c in counts.items():
        key = _canonical_kmer(kmer)
        out[key] = out.get(key, 0) + c
    return out


def document_sequences(input_file, output_dir="Data_exploration/results/sequences", bin = 5):
    """ Document nucleotide sequences surrounding a transposon position with a given bin size.
    
    Args:
        genome: Genome object to retrieve sequences from
        input_file: Path to input file with transposon positions
        output_dir: Directory to save the sequences 
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label = label_from_filename(input_file)
    genome = Genome()
    chrom_df = read_wig(input_file)
    sequences = {}
    for chrom in chrom_df:
        sequences[chrom] = []
        df = chrom_df[chrom]
        chrom_sequence = genome.get_sequence(chrom)
        for index, row in df.iterrows():
            position = int(row['Position'])
            start = max(0, position - bin)
            end = position + bin
            sequence = chrom_sequence[start:end+1]
            sequences[chrom].append({
                'position': position,
                'sequence': sequence
            })
    # Save sequences to output files
    file_path = os.path.join(output_dir, f"{label}_sequences.json")
    with open(file_path, 'w') as f:
        json.dump(sequences, f, indent=4)

def process_all_data(input_folder, bin=5, sequences=False, kmers = False):
    """ Process all WIG files in the input folder to document sequences.
    
    Args:
        input_folder: Folder containing WIG files
        bin: Number of nucleotides to include on each side of the position
    """
    if sequences and kmers:
        raise ValueError("Please choose either sequences or kmers processing, not both.")
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if sequences:
                if file.endswith(".wig"):
                    input_file = os.path.join(root, file)
                    output_dir = os.path.join("Data_exploration/results/sequences", os.path.relpath(root, input_folder))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    document_sequences(input_file, output_dir=output_dir, bin=bin)
            if kmers:
                if file.endswith("_sequences.json"):
                    input_file = os.path.join(root, file)
                    print(f"Processing k-mers for {input_file}")
                    with open(input_file, 'r') as f:
                        sequences = json.load(f)
                    output_dir = os.path.join("Data_exploration/results/sequences_data/kmer_counts", os.path.relpath(root, input_folder))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_file = os.path.join(output_dir, file.replace("_sequences.json", "_kmer_counts.json"))
                    calculate_kmer_occurrences(sequences, output_file, k_sizes=[3, 5, 7, 9, 11])

def calculate_kmer_occurrences(sequences, output_file, k_sizes=[3, 5, 7, 9, 11]):
    """ Calculate the k-mer occurrence surrounding transposon positions.
    The middle point of the k-mer in the sequence corresponds to the transposon position.
    
    Args:
        sequences: Dict of sequences per chromosome
        k_sizes: List of k-mer sizes to consider
    """
    kmer_counts = {k: {} for k in k_sizes}
    for chrom in sequences:
        for entry in sequences[chrom]:
            seq = entry['sequence']
            seq_length = len(seq)
            for k in k_sizes:
                if seq_length != 11:
                    continue
                mid = seq_length // 2
                start = max(0, mid - k // 2)
                end = start + k
                kmer = seq[start:end]
                if len(kmer) == k:
                    if kmer not in kmer_counts[k]:
                        kmer_counts[k][kmer] = 0
                    kmer_counts[k][kmer] += 1
    print(output_file)
    with open(output_file, 'w') as f:
        json.dump(kmer_counts, f, indent=4)
        
def compute_log_ratio(output_file, sequences, k=5, collapse_rc=True):
    genome = Genome()
    genome_counts_all = genome.compute_kmer_count(input_file="SGD_API/architecture_info/genome_kmer_counts.json")
    if str(k) not in genome_counts_all:
        raise KeyError(f"No genome counts for k={k} in file")

    g_counts = genome_counts_all[str(k)]  # dict: kmer -> genome count
    obs_counts = sequences.get(str(k), {})  # dict: kmer -> observed count

    if collapse_rc:
        g_counts = _collapse_rc_counts(g_counts)
        obs_counts = _collapse_rc_counts(obs_counts)

    total_k_positions = sum(g_counts.values())
    if total_k_positions == 0:
        raise ValueError(f"total_k_positions is zero for k={k}")

    N_obs = sum(obs_counts.values())

    log_values = {}
    for kmer, gcount in g_counts.items():
        freq = gcount / total_k_positions
        expected = freq * N_obs
        observed = float(obs_counts.get(kmer, 0))
        log_values[kmer] = np.log2((observed + 1e-10) / (expected + 1e-10))

    summary = {
        "k": k,
        "collapsed_rc": bool(collapse_rc),
        "total_observed": N_obs,
        "total_expected": float(sum((g_counts[s] / total_k_positions) * N_obs for s in g_counts)),
        "max_log_value": max(log_values.values(), default=0.0),
        "min_log_value": min(log_values.values(), default=0.0),
        "mean_log_value": float(np.mean(list(log_values.values()))) if log_values else 0.0,
        "std_log_value": float(np.std(list(log_values.values()))) if log_values else 0.0,
    }

    # append-or-create JSON (your existing logic)
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}
        existing.setdefault("results", []).append({"summary": summary, "log_values": log_values})
        with open(output_file, "w") as f:
            json.dump(existing, f, indent=4)
    else:
        with open(output_file, "w") as f:
            json.dump({"results": [{"summary": summary, "log_values": log_values}]}, f, indent=4)
        
def compute_all_log_values(input_folder, k_size = [5]):
    for k in k_size:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith("_kmer_counts.json"):
                    input_file = os.path.join(root, file)
                print(f"Computing log values for {input_file}")
                with open(input_file, 'r') as f:
                    sequences = json.load(f)
                output_dir = os.path.join("Data_exploration/results/sequences_data/log_values", os.path.relpath(root, input_folder))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_file = os.path.join(output_dir, file.replace("_kmer_counts.json", "_log_values.json"))
                compute_log_ratio(output_file, sequences, k=k)

def summarize_top_kmers(
    root_dir="Data_exploration/results/sequences_data/log_values",
    out_csv="Data_exploration/results/top_kmer_summary.csv",
    topN=10,
    stats_on_all_datasets=True,
):
    """
    Scan *_log_values.json under root_dir and summarize, for each k:
      - The k-mers that most often appear in the per-dataset Top-10 lists.
      - How many datasets they appear in as Top-10 (top10_occurrences).
      - Mean and std of log2 across datasets ("according to the datasets").

    Args:
        root_dir: folder containing per-dataset *_log_values.json files (recursively).
        out_csv: output CSV path.
        topN: how many k-mers to report per k.
        stats_on_all_datasets: True = mean/std over ALL datasets' log2 values.
                               False = mean/std only over datasets where k-mer was Top-10.
    Output CSV columns:
        k,kmer,top10_occurrences,total_datasets,mean_log2,std_log2
    """
    # k -> {dataset: set(top10 kmers)}
    per_k_top10_sets = defaultdict(dict)
    # k -> kmer -> list of log2 values across datasets (or only where top, if stats_on_all_datasets=False)
    per_k_kmer_vals  = defaultdict(lambda: defaultdict(list))
    # k -> number of datasets that had results for this k
    per_k_total_ds   = Counter()

    # Walk tree and load *_log_values.json
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith("_log_values.json"):
                continue
            path = os.path.join(subdir, file)
            dataset = os.path.splitext(file)[0]
            try:
                with open(path, "r") as f:
                    js = json.load(f)
            except Exception as e:
                print(f"[WARN] Skipping unreadable JSON: {path} ({e})")
                continue

            results = js.get("results", [])
            for r in results:
                summ = r.get("summary", {})
                k = summ.get("k", None)
                logvals = r.get("log_values", {})
                if k is None or not isinstance(logvals, dict) or not logvals:
                    continue
                k = int(k)

                # Determine this dataset's top-10 by log2 value
                top10 = sorted(logvals.items(), key=lambda kv: kv[1], reverse=True)[:10]
                top10_set = set(k for k, _ in top10)
                per_k_top10_sets[k][dataset] = top10_set
                per_k_total_ds[k] += 1

                # Collect values for stats
                if stats_on_all_datasets:
                    # add log2 for ALL kmers in this dataset
                    for kmer, val in logvals.items():
                        try:
                            per_k_kmer_vals[k][kmer].append(float(val))
                        except Exception:
                            pass
                else:
                    # add ONLY for kmers that are top10 in this dataset
                    for kmer in top10_set:
                        try:
                            per_k_kmer_vals[k][kmer].append(float(logvals[kmer]))
                        except Exception:
                            pass

    if not per_k_top10_sets:
        raise RuntimeError(f"No *_log_values.json files found under {root_dir}")

    # Build CSV lines
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    lines = ["k,kmer,top10_occurrences,total_datasets,mean_log2,std_log2"]

    for k in sorted(per_k_top10_sets.keys()):
        # frequency of appearing in per-dataset top-10
        freq = Counter()
        for ds, kmerset in per_k_top10_sets[k].items():
            for km in kmerset:
                freq[km] += 1

        # mean/std for each kmer (based on per_k_kmer_vals)
        stats = {}
        for kmer, values in per_k_kmer_vals[k].items():
            if not values:
                continue
            mu = mean(values)
            sd = stdev(values) if len(values) > 1 else 0.0
            stats[kmer] = (mu, sd)

        # rank by freq desc, then by mean desc (break ties by effect size)
        ranked = sorted(
            freq.items(),
            key=lambda kv: (kv[1], stats.get(kv[0], (float("-inf"), 0.0))[0]),
            reverse=True
        )[:topN]

        total_ds = per_k_total_ds[k]
        for kmer, occ in ranked:
            mu, sd = stats.get(kmer, (math.nan, math.nan))
            lines.append(f"{k},{kmer},{occ},{total_ds},{mu:.6f},{sd:.6f}")

    with open(out_csv, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Wrote {out_csv}")

if __name__ == "__main__":
    # input_file = "Data/wiggle_format/strain_FD/FD7_1_FDDP210435821-2a_HTWL7DSX2_L4_trimmed_forward_notrimmed.sorted.bam.wig"
    # document_sequences(input_file)
    # process_all_data(input_folder="Data/wiggle_format", bin=5, sequences=True)
    # input_file = "Data_exploration/results/sequences/strain_FD/FD7_1_sequences.json"
    # with open(input_file, 'r') as f:
    #     sequences = json.load(f)
    # output_file = "Data_exploration/results/sequences/strain_FD/FD7_1_kmer_counts.json"
    # calculate_kmer_occurrences(sequences, output_file, k_sizes=[1,2,3,4,5])
    # process_all_data("Data_exploration/results/sequences_data/sequences", kmers=True)
    # compute_all_log_values("Data_exploration/results/sequences_data/kmer_counts", k_size=[3, 5, 7, 9])
    summarize_top_kmers(
        root_dir="Data_exploration/results/sequences_data/log_values",
        out_csv="Data_exploration/results/top_kmer_summary.csv",
        topN=10,
        stats_on_all_datasets=True,   # set False to compute mean/std only where a k-mer is Top-10
    )
