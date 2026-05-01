"""
Shared utilities for imputation analysis (mu and pi).
Centralizes common functions for gene data processing, position mapping, and data loading.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Set

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Utils.SGD_API.yeast_genes import SGD_Genes

# Chromosome list (S288C reference)
CHROMOSOMES = [
    "ChrI", "ChrII", "ChrIII", "ChrIV", "ChrV", "ChrVI", "ChrVII", 
    "ChrVIII", "ChrIX", "ChrX", "ChrXI", "ChrXII", "ChrXIII", "ChrXIV", 
    "ChrXV", "ChrXVI"
]

SPLITS = ["train", "val", "test"]
WINDOW_SIZE = 20
STEP_SIZE = 1


def normalize_chromosome_name(chr_name: Optional[str]) -> Optional[str]:
    """Normalize chromosome labels from SGD metadata to reconstruction labels."""
    if not chr_name or chr_name in CHROMOSOMES:
        return chr_name
    if chr_name.startswith("Chromosome_"):
        return f"Chr{chr_name.split('_', 1)[1]}"
    if chr_name.startswith("chr"):
        normalized = f"Chr{chr_name[3:]}"
        return normalized if normalized in CHROMOSOMES else chr_name
    return chr_name


def concatenate_nonempty(arrays: List[np.ndarray]) -> np.ndarray:
    """Concatenate non-empty arrays and return an empty array when none exist."""
    nonempty_arrays = [np.asarray(array, dtype=float) for array in arrays if len(array) > 0]
    return np.concatenate(nonempty_arrays) if nonempty_arrays else np.array([], dtype=float)


def load_zero_positions_from_original_data(original_csv_file: str) -> Set[int]:
    """Load 1-based genomic positions where the original value equals zero."""
    if not os.path.exists(original_csv_file):
        print(f"Warning: Original data file not found: {original_csv_file}")
        return set()

    df = pd.read_csv(original_csv_file, usecols=["Position", "Value"], low_memory=False,
                     dtype={"Position": "string", "Value": "string"})
    
    positions = pd.to_numeric(df["Position"], errors="coerce")
    values = pd.to_numeric(df["Value"], errors="coerce")
    valid_mask = ~(positions.isna() | values.isna())
    
    if (~valid_mask).sum() > 0:
        print(f"Warning: Found {(~valid_mask).sum()} invalid rows in {original_csv_file}")
    
    return set(positions.loc[valid_mask & (values == 0)].astype(np.int64).tolist())


def load_split_metadata(split_dir: str) -> List[Dict[str, Any]]:
    """Load split metadata JSON from a reconstruction split folder."""
    metadata_path = os.path.join(split_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return []
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata if isinstance(metadata, list) else []
    except Exception as exc:
        print(f"Warning: Could not load metadata from {metadata_path}: {exc}")
        return []


def infer_window_size_from_metadata(metadata: List[Dict[str, Any]]) -> int:
    """Infer preprocessing window size from metadata.bin_size."""    
    bin_sizes = set()
    for meta in metadata:
        if (val := meta.get("bin_size")) is not None:
            try:
                if (parsed := int(val)) > 0:
                    bin_sizes.add(parsed)
            except (TypeError, ValueError):
                continue
    
    if not bin_sizes:
        return WINDOW_SIZE
    
    sorted_sizes = sorted(bin_sizes)
    if len(sorted_sizes) > 1:
        print(f"Warning: Multiple bin_size values found: {sorted_sizes}. Using {sorted_sizes[0]}")
    
    return sorted_sizes[0]


def resolve_window_size_for_split(split_dir: str) -> int:
    """Resolve effective window size for one split."""
    metadata = load_split_metadata(split_dir)
    window_size = infer_window_size_from_metadata(metadata)
    print(f"Split {os.path.basename(split_dir)}: window_size={window_size}")
    return window_size


def map_position_to_window(position: int, window_size: int = WINDOW_SIZE, 
                          step_size: int = STEP_SIZE) -> Tuple[int, int]:
    """Map a 0-based reconstruction index to its 1-based genomic window."""
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be > 0")
    
    window_start = max(1, position * step_size + 1)
    window_end = window_start + window_size - 1
    return window_start, window_end


def is_position_all_zero(position: int, zero_positions: Set[int], 
                         window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE) -> bool:
    """Check if a reconstruction position maps to an all-zero original window."""
    window_start, window_end = map_position_to_window(position, window_size, step_size)
    return all(pos in zero_positions for pos in range(window_start, window_end + 1))


def load_gene_data() -> Dict:
    """Load gene data from SGD and organize by chromosome."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    
    # Try to find existing gene JSON file
    gene_files = [
        "Utils/SGD_API/architecture_info/yeast_genes_with_info.json",
        "Utils/SGD_API/S288C/genes_info.json",
        "genes_info.json",
    ]
    
    genes_dict = {}
    for gene_file in gene_files:
        if os.path.exists(gene_file):
            print(f"Loading genes from {gene_file}")
            with open(gene_file, 'r') as f:
                genes_dict = json.load(f)
                break
    
    if not genes_dict:
        print("No pre-cached gene file found. Loading from SGD API...")
        try:
            sgd = SGD_Genes(gene_list_with_info="Utils/SGD_API/architecture_info/yeast_genes_with_info.json")
            genes_dict = sgd.list_all_gene_info()
        except Exception as e:
            print(f"Warning: Could not load from SGD: {e}")
            return defaultdict(list)
    
    # Organize genes by chromosome
    genes_by_chr = defaultdict(list)
    for gene_name, info in genes_dict.items():
        if "location" not in info:
            continue
        loc = info["location"]
        chr_name = normalize_chromosome_name(loc.get("chromosome"))
        start, end = loc.get("start"), loc.get("end")
        if chr_name and start is not None and end is not None:
            genes_by_chr[chr_name].append({
                "gene": gene_name,
                "start": int(start),
                "end": int(end),
                "essentiality": info.get("essentiality", None)
            })
    
    # Sort by start position
    for chr_name in genes_by_chr:
        genes_by_chr[chr_name].sort(key=lambda x: x["start"])
    
    return genes_by_chr


def get_overlapping_genes(genes_by_chr: Dict, chr_name: str, position: int,
                         window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE) -> List[Dict]:
    """Find all genes overlapping with a reconstructed position's window."""
    if chr_name not in genes_by_chr:
        return []
    
    window_start, window_end = map_position_to_window(position, window_size, step_size)
    
    return [gene for gene in genes_by_chr[chr_name] 
            if gene["start"] <= window_end and gene["end"] >= window_start]


def assign_values_to_genes(csv_file: str, genes_by_chr: Dict, chr_name: str, 
                           value_column: str, filter_zeros: bool = True,
                           zero_positions: Optional[Set[int]] = None,
                           window_size: int = WINDOW_SIZE, 
                           step_size: int = STEP_SIZE) -> Dict[str, List[float]]:
    """
    Generic function to assign values (mu or pi) to genes based on position overlap.
    
    Args:
        csv_file: Path to CSV file
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name
        value_column: Column name to extract ("mu" or "pi")
        filter_zeros: If True, only keep values where original window is all zero
        zero_positions: Set of 1-based positions where original Value == 0
        window_size: Moving average window size
        step_size: Moving average step size
        
    Returns:
        Dictionary mapping gene_name -> [list of values]
    """
    df = pd.read_csv(csv_file, usecols=["position", value_column], low_memory=False,
                     dtype={"position": "string", value_column: "string"})
    
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    
    valid_mask = ~(df["position"].isna() | df[value_column].isna())
    if (~valid_mask).sum() > 0:
        print(f"Warning: Found {(~valid_mask).sum()} invalid rows in {csv_file}")
    
    df = df.loc[valid_mask, ["position", value_column]]
    if df.empty:
        return defaultdict(list)
    
    genes_values = defaultdict(list)
    
    for position, value in df.itertuples(index=False, name=None):
        position = int(position)
        value = float(value)
        
        # Filter by zero positions if requested
        if filter_zeros and zero_positions is not None:
            if not is_position_all_zero(position, zero_positions, window_size, step_size):
                continue
        
        # Find overlapping genes and assign value
        for gene in get_overlapping_genes(genes_by_chr, chr_name, position, window_size, step_size):
            genes_values[gene['gene']].append(value)
    
    return genes_values


def aggregate_by_essentiality(genes_values: Dict[str, List[float]], 
                              genes_by_chr: Dict, chr_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate values by essentiality status."""
    essentiality_lookup = {gene['gene']: gene['essentiality'] 
                          for gene in genes_by_chr.get(chr_name, [])}
    
    essential, nonessential = [], []
    for gene_name, values in genes_values.items():
        essentiality = essentiality_lookup.get(gene_name)
        if essentiality is None:
            continue
        (essential if essentiality else nonessential).extend(values)
    
    return np.array(essential), np.array(nonessential)
