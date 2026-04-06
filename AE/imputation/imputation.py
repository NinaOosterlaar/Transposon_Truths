import csv
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Set

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Utils.SGD_API.yeast_genes import SGD_Genes
from Utils.plot_config import setup_plot_style, COLORS

setup_plot_style()

# Chromosome list (S288C reference)
CHROMOSOMES = [
    "ChrI", "ChrII", "ChrIII", "ChrIV", "ChrV", "ChrVI", "ChrVII", 
    "ChrVIII", "ChrIX", "ChrX", "ChrXI", "ChrXII", "ChrXIII", "ChrXIV", 
    "ChrXV", "ChrXVI"
]

# Mu offset values
MU_OFFSETS = ["muoff0", "muoff1", "muoff2", "muoff4", "muoff5"]

# Splits to analyze
SPLITS = ["train", "val", "test"]

# Window parameters for position mapping
WINDOW_SIZE = 19
STEP_SIZE = 1
POSITION_MODE_AUTO = "auto"
POSITION_MODE_INDEX = "index"
POSITION_MODE_CENTER = "center"
VALID_POSITION_MODES = {POSITION_MODE_AUTO, POSITION_MODE_INDEX, POSITION_MODE_CENTER}


def normalize_chromosome_name(chr_name: Optional[str]) -> Optional[str]:
    """Normalize chromosome labels from SGD metadata to reconstruction labels."""
    if not chr_name:
        return None

    if chr_name in CHROMOSOMES:
        return chr_name

    if chr_name.startswith("Chromosome_"):
        return f"Chr{chr_name.split('_', 1)[1]}"

    if chr_name.startswith("chr"):
        suffix = chr_name[3:]
        normalized = f"Chr{suffix}"
        if normalized in CHROMOSOMES:
            return normalized

    return chr_name


def concatenate_nonempty(arrays: List[np.ndarray]) -> np.ndarray:
    """Concatenate non-empty arrays and return an empty array when none exist."""
    nonempty_arrays = [np.asarray(array, dtype=float) for array in arrays if len(array) > 0]
    if not nonempty_arrays:
        return np.array([], dtype=float)
    return np.concatenate(nonempty_arrays)


def parse_mu_offset_value(mu_offset_name: str) -> Optional[float]:
    """Parse numeric mu_offset value from a muoff* folder name."""
    if not mu_offset_name:
        return None

    value_str = str(mu_offset_name).replace("muoff", "")
    try:
        return float(value_str)
    except ValueError:
        return None


def collect_essential_nonessential_arrays(
    results: Dict[str, Dict],
    strain: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate essential/non-essential arrays across chromosomes."""
    if strain is None:
        combined = results.get('combined', {})
        essential = concatenate_nonempty([v['essential'] for v in combined.values()])
        nonessential = concatenate_nonempty([v['nonessential'] for v in combined.values()])
        return essential, nonessential

    per_strain = results.get('per_strain', {})
    strain_results = per_strain.get(strain, {})
    essential = concatenate_nonempty([v['essential'] for v in strain_results.values()])
    nonessential = concatenate_nonempty([v['nonessential'] for v in strain_results.values()])
    return essential, nonessential


def write_pi_rows(
    writer: Any,
    split_label: str,
    mu_offset_name: str,
    mu_offset_value: Optional[float],
    strain: str,
    essentiality: str,
    values: np.ndarray,
) -> int:
    """Write pi values to CSV."""
    row_count = 0
    mu_offset_cell = "" if mu_offset_value is None else mu_offset_value

    for value in values:
        writer.writerow([
            split_label,
            mu_offset_name,
            mu_offset_cell,
            strain,
            essentiality,
            float(value),
        ])
        row_count += 1

    return row_count


def save_plot_data(
    output_dir: str,
    file_tag: str,
    split_label: str,
    results_by_muoffset: Dict[str, Dict],
    strains: Optional[List[str]] = None,
    mu_offset_folders: Optional[List[str]] = None,
) -> None:
    """Save the raw pi values used for plotting to CSV."""
    data_dir = os.path.join(output_dir, "plot_data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{file_tag}_pi_values.csv")

    # Use provided mu_offset_folders or fall back to results keys
    folders_to_use = mu_offset_folders if mu_offset_folders is not None else list(results_by_muoffset.keys())

    total_rows = 0
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "split",
            "mu_offset",
            "mu_offset_value",
            "strain",
            "essentiality",
            "pi",
        ])

        for muoff_name in folders_to_use:
            if muoff_name not in results_by_muoffset:
                continue

            results = results_by_muoffset[muoff_name]
            mu_offset_value = parse_mu_offset_value(muoff_name)

            essential, nonessential = collect_essential_nonessential_arrays(results)
            total_rows += write_pi_rows(
                writer,
                split_label,
                muoff_name,
                mu_offset_value,
                "combined",
                "essential",
                essential,
            )
            total_rows += write_pi_rows(
                writer,
                split_label,
                muoff_name,
                mu_offset_value,
                "combined",
                "nonessential",
                nonessential,
            )

            if strains:
                for strain in strains:
                    strain_essential, strain_nonessential = collect_essential_nonessential_arrays(
                        results,
                        strain=strain,
                    )
                    if len(strain_essential) == 0 and len(strain_nonessential) == 0:
                        continue

                    total_rows += write_pi_rows(
                        writer,
                        split_label,
                        muoff_name,
                        mu_offset_value,
                        strain,
                        "essential",
                        strain_essential,
                    )
                    total_rows += write_pi_rows(
                        writer,
                        split_label,
                        muoff_name,
                        mu_offset_value,
                        strain,
                        "nonessential",
                        strain_nonessential,
                    )

    print(f"    Saved plot data: {file_path} ({total_rows} rows)")


def compute_mu_offset_difference_summary(
    results_by_muoffset: Dict[str, Dict],
    split_label: str,
    mu_offset_folders: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Summarize mean essential vs non-essential pi differences per mu_offset."""
    rows = []

    # Use provided mu_offset_folders or fall back to results keys
    folders_to_use = mu_offset_folders if mu_offset_folders is not None else list(results_by_muoffset.keys())

    for muoff_name in folders_to_use:
        if muoff_name not in results_by_muoffset:
            continue

        results = results_by_muoffset[muoff_name]
        mu_offset_value = parse_mu_offset_value(muoff_name)
        essential, nonessential = collect_essential_nonessential_arrays(results)
        n_essential = int(len(essential))
        n_nonessential = int(len(nonessential))

        if n_essential > 0 and n_nonessential > 0:
            mean_essential = float(np.mean(essential))
            mean_nonessential = float(np.mean(nonessential))
            difference = mean_essential - mean_nonessential
        else:
            mean_essential = np.nan
            mean_nonessential = np.nan
            difference = np.nan

        rows.append({
            "split": split_label,
            "mu_offset": muoff_name,
            "mu_offset_value": mu_offset_value,
            "n_essential": n_essential,
            "n_nonessential": n_nonessential,
            "mean_essential": mean_essential,
            "mean_nonessential": mean_nonessential,
            "difference": difference,
        })

    return pd.DataFrame(rows)


def create_mu_offset_difference_plot(
    diff_df: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """Plot mean essential vs non-essential difference across mu_offset values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if diff_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    plot_df = diff_df.dropna(subset=["difference", "mu_offset_value"]).sort_values("mu_offset_value")
    if plot_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    x_values = plot_df["mu_offset_value"].to_numpy(dtype=float)
    y_values = plot_df["difference"].to_numpy(dtype=float)

    line_color = COLORS.get("blue", "#0072B2")
    marker_color = COLORS.get("orange", "#E69600")
    baseline_color = COLORS.get("black", "#000000")

    ax.plot(x_values, y_values, color=line_color, linewidth=2)
    ax.scatter(x_values, y_values, color=marker_color, s=60, zorder=3)
    ax.axhline(0, color=baseline_color, linewidth=1, alpha=0.6)

    ax.set_xlabel("mu_offset")
    ax.set_ylabel("Mean pi difference (essential - non-essential)")
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x_values)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if len(y_values) > 0:
        y_min = float(np.nanmin(y_values))
        ax.set_ylim(bottom=min(0.0, y_min))
    else:
        ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    return fig


def load_zero_positions_from_original_data(original_csv_file: str) -> Set[int]:
    """
    Load 1-based genomic positions where the original value equals zero.

    Args:
        original_csv_file: Path to combined strain CSV with columns [Position, Value]

    Returns:
        Set of 1-based genomic positions where Value == 0
    """
    if not os.path.exists(original_csv_file):
        print(f"Warning: Original data file not found: {original_csv_file}. No positions will pass the zero filter.")
        return set()

    raw_df = pd.read_csv(
        original_csv_file,
        usecols=["Position", "Value"],
        low_memory=False,
        dtype={"Position": "string", "Value": "string"}
    )

    positions = pd.to_numeric(raw_df["Position"], errors="coerce")
    values = pd.to_numeric(raw_df["Value"], errors="coerce")

    invalid_mask = positions.isna() | values.isna()
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(
            f"Warning: Found {invalid_count} invalid rows in {original_csv_file} "
            f"(non-numeric Position or Value). These rows will be skipped."
        )

    zero_positions = positions.loc[~invalid_mask & (values == 0)].astype(np.int64)
    return set(zero_positions.tolist())


def load_split_metadata(split_dir: str) -> List[Dict[str, Any]]:
    """Load split metadata JSON (if present) from a reconstruction split folder."""
    metadata_path = os.path.join(split_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return []

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as exc:
        print(f"Warning: Could not load split metadata from {metadata_path}: {exc}")
        return []

    if not isinstance(metadata, list):
        print(f"Warning: Unexpected metadata format in {metadata_path}; expected a list.")
        return []

    return metadata


def infer_window_size_from_metadata(metadata: List[Dict[str, Any]], fallback_window_size: int) -> int:
    """Infer preprocessing window size from metadata.bin_size with fallback."""
    if fallback_window_size <= 0:
        raise ValueError("fallback_window_size must be > 0")

    bin_sizes_set = set()
    for meta in metadata:
        bin_size_value = meta.get("bin_size")
        if bin_size_value is None:
            continue
        try:
            parsed_bin_size = int(bin_size_value)
        except (TypeError, ValueError):
            continue
        if parsed_bin_size > 0:
            bin_sizes_set.add(parsed_bin_size)

    bin_sizes = sorted(bin_sizes_set)

    if not bin_sizes:
        return fallback_window_size

    if len(bin_sizes) > 1:
        print(
            f"Warning: Multiple bin_size values found in metadata ({bin_sizes}). "
            f"Using {bin_sizes[0]}."
        )

    return int(bin_sizes[0])


def infer_position_mode_from_metadata(
    metadata: List[Dict[str, Any]],
    fallback_mode: str = POSITION_MODE_INDEX,
) -> str:
    """
    Infer reconstruction position convention from metadata.

    Priority:
    1) Explicit `position_mode` in metadata (newer preprocessing output)
    2) Heuristic on `start_pos`: 0-based starts imply index mode, otherwise center mode
    3) Fallback to provided mode
    """
    explicit_modes = {
        str(meta.get("position_mode"))
        for meta in metadata
        if str(meta.get("position_mode")) in {POSITION_MODE_INDEX, POSITION_MODE_CENTER}
    }

    if len(explicit_modes) == 1:
        return explicit_modes.pop()

    if len(explicit_modes) > 1:
        print(
            f"Warning: Conflicting position_mode values in metadata ({sorted(explicit_modes)}). "
            f"Falling back to heuristic."
        )

    starts: List[int] = []
    for meta in metadata:
        start_pos = meta.get("start_pos")
        if start_pos is None:
            continue
        try:
            starts.append(int(start_pos))
        except (TypeError, ValueError):
            continue

    if not starts:
        return fallback_mode

    return POSITION_MODE_INDEX if min(starts) <= 0 else POSITION_MODE_CENTER


def resolve_position_mapping_for_split(
    split_dir: str,
    requested_position_mode: str,
    fallback_window_size: int,
) -> Tuple[str, int]:
    """Resolve effective position mode and window size for one split."""
    if requested_position_mode not in VALID_POSITION_MODES:
        raise ValueError(
            f"Unknown position_mode={requested_position_mode}. "
            f"Expected one of {sorted(VALID_POSITION_MODES)}"
        )

    metadata = load_split_metadata(split_dir)
    effective_window_size = infer_window_size_from_metadata(metadata, fallback_window_size)

    if requested_position_mode == POSITION_MODE_AUTO:
        effective_position_mode = infer_position_mode_from_metadata(
            metadata,
            fallback_mode=POSITION_MODE_INDEX,
        )
        source = "metadata" if metadata else "fallback"
        print(
            f"Split {os.path.basename(split_dir)}: position_mode={effective_position_mode} "
            f"(source={source}), window_size={effective_window_size}"
        )
    else:
        effective_position_mode = requested_position_mode
        print(
            f"Split {os.path.basename(split_dir)}: position_mode={effective_position_mode} "
            f"(manual), window_size={effective_window_size}"
        )

    return effective_position_mode, effective_window_size


def map_reconstruction_position_to_original_window(
    position: int,
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_INDEX,
) -> Tuple[int, int]:
    """
    Map one reconstructed position to its original 1-based genomic window.

    Modes:
    - index: position is a 0-based moving-average index i
      window = [i * step + 1, i * step + window_size]
    - center: position is a genomic center coordinate of the moving-average window
      window_start = position - floor((window_size - 1)/2)
    """
    if moving_average_window_size <= 0:
        raise ValueError("moving_average_window_size must be > 0")
    if moving_average_step_size <= 0:
        raise ValueError("moving_average_step_size must be > 0")
    if position_mode not in {POSITION_MODE_INDEX, POSITION_MODE_CENTER}:
        raise ValueError(
            f"position_mode must be '{POSITION_MODE_INDEX}' or '{POSITION_MODE_CENTER}', got: {position_mode}"
        )

    position = int(position)

    if position_mode == POSITION_MODE_INDEX:
        window_start = position * moving_average_step_size + 1
    else:
        left_half = (moving_average_window_size - 1) // 2
        window_start = position - left_half

    window_start = max(1, int(window_start))
    window_end = int(window_start + moving_average_window_size - 1)
    return window_start, window_end


def is_pi_position_associated_with_original_zero(
    position: int,
    original_zero_positions: Set[int],
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_INDEX,
) -> bool:
    """
    Determine whether a reconstruction position maps to an all-zero original window.

    The `position` value can be interpreted either as a moving-average index
    (`position_mode='index'`) or as a genomic center coordinate
    (`position_mode='center'`).
    """
    window_start, window_end = map_reconstruction_position_to_original_window(
        position=position,
        moving_average_window_size=moving_average_window_size,
        moving_average_step_size=moving_average_step_size,
        position_mode=position_mode,
    )

    return all(pos in original_zero_positions for pos in range(window_start, window_end + 1))


def load_gene_data(mu_offset_root_dir: str) -> Dict:
    """
    Load gene data from SGD_Genes class and filter to genes in reconstruction.
    
    Args:
        mu_offset_root_dir: Root directory containing mu_offset folders
        
    Returns:
        Dictionary mapping gene_location strings to gene info
        {chr_name -> {start -> {end -> {essentiality, gene_name}}}}
    """
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
                genes_raw = json.load(f)
                genes_dict = genes_raw
                break
    
    if not genes_dict:
        print("No pre-cached gene file found. Attempting to load from SGD API...")
        try:
            sgd = SGD_Genes(gene_list_with_info="Utils/SGD_API/architecture_info/yeast_genes_with_info.json")
            genes_dict = sgd.list_all_gene_info()
        except Exception as e:
            print(f"Warning: Could not load from SGD: {e}")
            genes_dict = {}
    
    # Organize genes by chromosome and position for efficient overlap queries
    genes_by_chr = defaultdict(list)
    for gene_name, info in genes_dict.items():
        if "location" in info:
            loc = info["location"]
            chr_name = normalize_chromosome_name(loc.get("chromosome"))
            start = loc.get("start")
            end = loc.get("end")
            if chr_name and start is not None and end is not None:
                genes_by_chr[chr_name].append({
                    "gene": gene_name,
                    "start": int(start),
                    "end": int(end),
                    "essentiality": info.get("essentiality", None)
                })
    
    # Sort by start position for efficient overlap detection
    for chr_name in genes_by_chr:
        genes_by_chr[chr_name].sort(key=lambda x: x["start"])
    
    return genes_by_chr


def get_genes_overlapping_position(
    genes_by_chr: Dict,
    chr_name: str,
    position: int,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_INDEX,
) -> List[Dict]:
    """
    Find all genes overlapping with a reconstructed position's window.
    
    Args:
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name (e.g., "ChrIII")
        position: Position value in reconstruction CSV
        window_size: Window size (19)
        step_size: Step size (1)
        position_mode: Position convention ('index' or 'center')
        
    Returns:
        List of genes overlapping this position
    """
    if chr_name not in genes_by_chr:
        return []

    window_start, window_end = map_reconstruction_position_to_original_window(
        position=position,
        moving_average_window_size=window_size,
        moving_average_step_size=step_size,
        position_mode=position_mode,
    )
    
    # Binary search to find genes in range (chromosomes sorted by start)
    genes = genes_by_chr[chr_name]
    overlapping = []
    
    for gene in genes:
        gene_start = gene["start"]
        gene_end = gene["end"]
        
        # Check overlap: [window_start, window_end] intersects [gene_start, gene_end]
        if gene_start <= window_end and gene_end >= window_start:
            overlapping.append(gene)
    
    return overlapping


def assign_pi_to_genes(
    csv_file: str,
    genes_by_chr: Dict,
    chr_name: str,
    filter_pi_by_original_zeros: bool = True,
    original_zero_positions: Optional[Set[int]] = None,
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_INDEX,
) -> Dict[str, List[float]]:
    """
    Parse CSV and assign pi values to genes based on position-gene overlap.
    
    Args:
        csv_file: Path to CSV file with columns [position, reconstruction, mu, pi, theta]
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name
        filter_pi_by_original_zeros:
            If True, only keep pi values whose mapped original moving-average
            window is entirely zero in Data/combined_strains
        original_zero_positions: Set of 1-based positions where original Value == 0
        moving_average_window_size:
            Moving average window size used in preprocessing (default 19)
        moving_average_step_size:
            Moving average step size used in preprocessing (default 1)
        position_mode:
            Position convention used in reconstruction CSV ('index' or 'center')
        
    Returns:
        Dictionary mapping gene_name -> [list of pi values]
    """
    raw_df = pd.read_csv(
        csv_file,
        usecols=["position", "pi"],
        low_memory=False,
        dtype={"position": "string", "pi": "string"}
    )
    df = raw_df.copy()
    df["position"] = pd.to_numeric(raw_df["position"], errors="coerce")
    df["pi"] = pd.to_numeric(raw_df["pi"], errors="coerce")

    invalid_mask = df["position"].isna() | df["pi"].isna()
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(f"Warning: Found {invalid_count} invalid rows in {csv_file} (non-numeric position or pi). These rows will be skipped.")

    df = df.loc[~invalid_mask, ["position", "pi"]]
    if df.empty:
        return defaultdict(list)
    
    genes_pi = defaultdict(list)
    
    if filter_pi_by_original_zeros and original_zero_positions is None:
        print(
            f"Warning: zero filtering enabled but no original position set provided for {csv_file}. "
            f"Falling back to unfiltered pi values."
        )

    for position, pi_value in df.itertuples(index=False, name=None):
        position = int(position)
        pi_value = float(pi_value)

        if filter_pi_by_original_zeros and original_zero_positions is not None:
            if not is_pi_position_associated_with_original_zero(
                position=position,
                original_zero_positions=original_zero_positions,
                moving_average_window_size=moving_average_window_size,
                moving_average_step_size=moving_average_step_size,
                position_mode=position_mode,
            ):
                continue
        
        # Find overlapping genes
        overlapping_genes = get_genes_overlapping_position(
            genes_by_chr,
            chr_name,
            position,
            window_size=moving_average_window_size,
            step_size=moving_average_step_size,
            position_mode=position_mode,
        )
        
        # Assign pi value to each overlapping gene
        for gene in overlapping_genes:
            genes_pi[gene['gene']].append(pi_value)
    
    return genes_pi


def aggregate_pi_by_essentiality(
    genes_pi: Dict[str, List[float]],
    genes_by_chr: Dict,
    chr_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate all pi values by essentiality status.
    
    Args:
        genes_pi: Dictionary mapping gene name to list of pi values
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name
        
    Returns:
        Tuple of (essential_pi_values, nonessential_pi_values) as numpy arrays
    """
    # Create lookup for essentiality
    essentiality_lookup = {}
    for gene in genes_by_chr.get(chr_name, []):
        essentiality_lookup[gene['gene']] = gene['essentiality']
    
    essential_pi = []
    nonessential_pi = []
    
    for gene_name, pi_values in genes_pi.items():
        essentiality = essentiality_lookup.get(gene_name)
        
        if essentiality is None:
            continue
        
        if essentiality:  # Essential
            essential_pi.extend(pi_values)
        else:  # Non-essential
            nonessential_pi.extend(pi_values)
    
    return np.array(essential_pi), np.array(nonessential_pi)


def process_single_chromosomal_split(
    mu_offset_dir: str,
    split: str,
    genes_by_chr: Dict,
    strains: Optional[List[str]] = None,
    filter_pi_by_original_zeros: bool = True,
    original_data_root_dir: str = "Data/combined_strains",
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    original_zero_positions_cache: Optional[Dict[Tuple[str, str], Set[int]]] = None,
    position_mode: str = POSITION_MODE_AUTO,
) -> Dict[str, Dict]:
    """
    Process all chromosomes for a single mu_offset and split (e.g., train).
    Collect results separately per strain and combined.
    
    Args:
        mu_offset_dir: Path to specific muoff* directory
        split: "train", "val", or "test"
        genes_by_chr: Gene data
        strains: List of strains to process, or None for all found
        filter_pi_by_original_zeros: If True, only include pi where original data Value == 0
        original_data_root_dir: Root directory containing original strain files
        moving_average_window_size: Moving average window size used in preprocessing
        moving_average_step_size: Moving average step size used in preprocessing
        original_zero_positions_cache:
            Optional cache {(strain, chromosome): set_of_zero_positions} to avoid
            re-reading large original files
        position_mode:
            Position convention to use ('auto', 'index', 'center').
            In 'auto' mode, this is inferred from split metadata.
        
    Returns:
        Dictionary with structure:
        {
            'per_strain': {strain_name: {chr_name: {'essential': [...], 'nonessential': [...]}}},
            'combined': {chr_name: {'essential': [...], 'nonessential': [...]}}
        }
    """
    split_dir = os.path.join(mu_offset_dir, split)
    
    if not os.path.exists(split_dir):
        return {'per_strain': {}, 'combined': {}}
    
    # Find all strain directories
    if strains is None:
        strains = sorted([d for d in os.listdir(split_dir) 
                         if os.path.isdir(os.path.join(split_dir, d)) and d.startswith('strain_')])

    effective_position_mode, effective_window_size = resolve_position_mapping_for_split(
        split_dir=split_dir,
        requested_position_mode=position_mode,
        fallback_window_size=moving_average_window_size,
    )
    
    results = {
        'per_strain': defaultdict(lambda: defaultdict(lambda: {'essential': [], 'nonessential': []})),
        'combined': defaultdict(lambda: {'essential': [], 'nonessential': []})
    }
    
    # Process each strain
    for strain in strains:
        strain_dir = os.path.join(split_dir, strain)
        
        for chr_name in CHROMOSOMES:
            csv_file = os.path.join(strain_dir, f"{chr_name}.csv")
            
            if not os.path.exists(csv_file):
                continue

            zero_positions = None
            if filter_pi_by_original_zeros:
                if original_zero_positions_cache is None:
                    original_zero_positions_cache = {}

                cache_key = (strain, chr_name)
                if cache_key not in original_zero_positions_cache:
                    original_csv_file = os.path.join(
                        original_data_root_dir,
                        strain,
                        f"{chr_name}_distances.csv"
                    )
                    original_zero_positions_cache[cache_key] = load_zero_positions_from_original_data(original_csv_file)

                zero_positions = original_zero_positions_cache[cache_key]
            
            # Assign pi to genes
            genes_pi = assign_pi_to_genes(
                csv_file,
                genes_by_chr,
                chr_name,
                filter_pi_by_original_zeros=filter_pi_by_original_zeros,
                original_zero_positions=zero_positions,
                moving_average_window_size=effective_window_size,
                moving_average_step_size=moving_average_step_size,
                position_mode=effective_position_mode,
            )
            
            # Aggregate by essentiality
            essential, nonessential = aggregate_pi_by_essentiality(
                genes_pi, genes_by_chr, chr_name
            )
            
            # Store per-strain results
            results['per_strain'][strain][chr_name]['essential'] = essential
            results['per_strain'][strain][chr_name]['nonessential'] = nonessential
            
            # Accumulate for combined results
            results['combined'][chr_name]['essential'].extend(essential)
            results['combined'][chr_name]['nonessential'].extend(nonessential)
    
    # Convert combined lists to arrays
    for chr_name in results['combined']:
        results['combined'][chr_name]['essential'] = np.array(results['combined'][chr_name]['essential'])
        results['combined'][chr_name]['nonessential'] = np.array(results['combined'][chr_name]['nonessential'])
    
    return results


def create_comparison_boxplot(
    ax,
    essential_pi: np.ndarray,
    nonessential_pi: np.ndarray,
    title: str
) -> None:
    """
    Create a boxplot comparing essential vs non-essential gene pi values.
    
    Args:
        ax: Matplotlib axis to plot on
        essential_pi: Array of pi values for essential genes
        nonessential_pi: Array of pi values for non-essential genes
        title: Title for the subplot
    """
    data_to_plot = [essential_pi, nonessential_pi]
    
    bp = ax.boxplot(
        data_to_plot,
        labels=['Essential', 'Non-Essential'],
        patch_artist=True,
        widths=0.6
    )
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('π value', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # # Add sample size info
    # n_ess = len(essential_pi)
    # n_noness = len(nonessential_pi)
    # ax.text(0.98, 0.97, f'n={n_ess}/{n_noness}', 
    #         transform=ax.transAxes, ha='right', va='top',
    #         fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_split_figure_combined(
    results_by_muoffset: Dict[str, Dict],
    split: str,
    figsize: Tuple = (16, 10),
    mu_offset_folders: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create one figure with subplots for all mu_offsets.
    Each subplot compares essential vs non-essential across all strains (combined).
    
    Args:
        results_by_muoffset: {muoff_name: results_dict}
        split: "train", "val", or "test"
        figsize: Figure size
        mu_offset_folders: List of mu_offset folder names to process
        
    Returns:
        Matplotlib figure
    """
    # Use provided mu_offset_folders or fall back to results keys
    folders_to_use = mu_offset_folders if mu_offset_folders is not None else list(results_by_muoffset.keys())
    
    # Determine grid size based on number of folders
    n_folders = len(folders_to_use)
    if n_folders == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    else:
        ncols = min(3, n_folders)
        nrows = (n_folders + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, muoff_name in enumerate(folders_to_use):
        ax = axes[idx]
        
        if muoff_name not in results_by_muoffset:
            ax.text(0.5, 0.5, f'{muoff_name}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        results = results_by_muoffset[muoff_name]
        combined = results.get('combined', {})
        
        # Aggregate across all chromosomes
        all_essential = concatenate_nonempty([
            v['essential'] for v in combined.values()
        ])
        all_nonessential = concatenate_nonempty([
            v['nonessential'] for v in combined.values()
        ])
        
        if len(all_essential) > 0 and len(all_nonessential) > 0:
            create_comparison_boxplot(
                ax, all_essential, all_nonessential,
                f'{muoff_name} ({split})'
            )
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(len(folders_to_use), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Essential vs Non-Essential Genes - {split.upper()}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def create_split_figure_per_strain(
    results_by_muoffset: Dict[str, Dict],
    strain: str,
    split: str,
    figsize: Tuple = (16, 10),
    mu_offset_folders: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create one figure with subplots for all mu_offsets for a specific strain.
    
    Args:
        results_by_muoffset: {muoff_name: results_dict}
        strain: Strain name
        split: "train", "val", or "test"
        figsize: Figure size
        mu_offset_folders: List of mu_offset folder names to process
        
    Returns:
        Matplotlib figure
    """
    # Use provided mu_offset_folders or fall back to results keys
    folders_to_use = mu_offset_folders if mu_offset_folders is not None else list(results_by_muoffset.keys())
    
    # Determine grid size based on number of folders
    n_folders = len(folders_to_use)
    if n_folders == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    else:
        ncols = min(3, n_folders)
        nrows = (n_folders + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, muoff_name in enumerate(folders_to_use):
        ax = axes[idx]
        
        if muoff_name not in results_by_muoffset:
            ax.text(0.5, 0.5, f'{muoff_name}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        results = results_by_muoffset[muoff_name]
        per_strain = results.get('per_strain', {})
        
        if strain not in per_strain:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        strain_results = per_strain[strain]
        
        # Aggregate across all chromosomes for this strain
        all_essential = concatenate_nonempty([
            v['essential'] for v in strain_results.values()
        ])
        all_nonessential = concatenate_nonempty([
            v['nonessential'] for v in strain_results.values()
        ])
        
        if len(all_essential) > 0 and len(all_nonessential) > 0:
            create_comparison_boxplot(
                ax, all_essential, all_nonessential,
                f'{muoff_name} ({split})'
            )
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(len(folders_to_use), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{strain} - Essential vs Non-Essential Genes - {split.upper()}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def run_full_analysis(
    mu_offset_root_dir: str = "Data/reconstruction/mu_offset",
    output_dir: str = "AE/results/pi_analysis",
    filter_pi_by_original_zeros: bool = True,
    original_data_root_dir: str = "Data/combined_strains",
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_AUTO,
    mu_offset_folders: Optional[List[str]] = None,
) -> None:
    """
    Run complete analysis: load genes, process all splits and mu_offsets,
    generate visualizations for combined and per-strain analyses.
    
    Args:
        mu_offset_root_dir: Root directory containing mu_offset folders
        output_dir: Output directory for figures
        filter_pi_by_original_zeros:
            If True (default), only use pi values at positions where original
            moving-average window in Data/combined_strains is entirely zero
        original_data_root_dir: Root directory containing original strain CSV files
        moving_average_window_size: Moving average window size used in preprocessing
        moving_average_step_size: Moving average step size used in preprocessing
        position_mode:
            Position convention to use ('auto', 'index', 'center').
            In 'auto' mode, inferred from each split metadata.json.
        mu_offset_folders: Optional list of mu_offset folder names to process.
            If None, uses MU_OFFSETS constant. Use this to specify custom folder names.
    """

    if position_mode not in VALID_POSITION_MODES:
        raise ValueError(
            f"Unknown position_mode={position_mode}. Expected one of {sorted(VALID_POSITION_MODES)}"
        )

    if filter_pi_by_original_zeros and not output_dir.endswith("_original_zero_filter"):
        output_dir = f"{output_dir}_original_zero_filter"

    os.makedirs(output_dir, exist_ok=True)
    
    # Use custom mu_offset folders if provided, otherwise use the default MU_OFFSETS
    folders_to_process = mu_offset_folders if mu_offset_folders is not None else MU_OFFSETS
    
    print("Loading gene data...")
    genes_by_chr = load_gene_data(mu_offset_root_dir)

    if filter_pi_by_original_zeros:
        print(
            f"Filtering pi values to original zeros from {original_data_root_dir} "
            f"(keep only all-zero windows; window_size={moving_average_window_size}, "
            f"step_size={moving_average_step_size})."
        )
        print(f"Using filtered output directory: {output_dir}")
    else:
        print("Using all pi values (original zero filter disabled).")
    
    # Find all strains from first mu_offset/train
    sample_dir = os.path.join(mu_offset_root_dir, folders_to_process[0], "train")
    all_strains = sorted([d for d in os.listdir(sample_dir) 
                         if os.path.isdir(os.path.join(sample_dir, d)) and d.startswith('strain_')])
    print(f"Found strains: {all_strains}")

    # Aggregate pi values across all splits and all strains for one final figure
    overall_by_muoffset = defaultdict(lambda: {'essential': [], 'nonessential': []})
    # Aggregate pi values across all splits for each strain and mu_offset
    overall_by_strain_muoffset = defaultdict(
        lambda: defaultdict(lambda: {'essential': [], 'nonessential': []})
    )
    original_zero_positions_cache: Dict[Tuple[str, str], Set[int]] = {}
    
    # Process each split
    for split in SPLITS:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        
        # Results organized by mu_offset
        results_by_muoffset = {}
        
        for muoff_name in folders_to_process:
            mu_offset_dir = os.path.join(mu_offset_root_dir, muoff_name)
            
            if not os.path.exists(mu_offset_dir):
                print(f"  Skipping {muoff_name} (not found)")
                continue
            
            print(f"  Processing {muoff_name}...")
            results = process_single_chromosomal_split(
                mu_offset_dir,
                split,
                genes_by_chr,
                all_strains,
                filter_pi_by_original_zeros=filter_pi_by_original_zeros,
                original_data_root_dir=original_data_root_dir,
                moving_average_window_size=moving_average_window_size,
                moving_average_step_size=moving_average_step_size,
                original_zero_positions_cache=original_zero_positions_cache,
                position_mode=position_mode,
            )
            results_by_muoffset[muoff_name] = results

            # Accumulate combined chromosome results for all-splits summary
            for chr_values in results.get('combined', {}).values():
                overall_by_muoffset[muoff_name]['essential'].append(chr_values['essential'])
                overall_by_muoffset[muoff_name]['nonessential'].append(chr_values['nonessential'])

            # Accumulate per-strain chromosome results for all-splits per-strain summary
            for strain_name, strain_results in results.get('per_strain', {}).items():
                for chr_values in strain_results.values():
                    overall_by_strain_muoffset[strain_name][muoff_name]['essential'].append(chr_values['essential'])
                    overall_by_strain_muoffset[strain_name][muoff_name]['nonessential'].append(chr_values['nonessential'])

        print("  Saving plot data...")
        save_plot_data(
            output_dir=output_dir,
            file_tag=split,
            split_label=split,
            results_by_muoffset=results_by_muoffset,
            strains=all_strains,
            mu_offset_folders=folders_to_process,
        )

        diff_df = compute_mu_offset_difference_summary(
            results_by_muoffset, 
            split_label=split,
            mu_offset_folders=folders_to_process,
        )
        if not diff_df.empty:
            diff_data_dir = os.path.join(output_dir, "plot_data")
            os.makedirs(diff_data_dir, exist_ok=True)
            diff_data_path = os.path.join(diff_data_dir, f"{split}_mu_offset_diff.csv")
            diff_df.to_csv(diff_data_path, index=False)
            print(f"    Saved mu_offset difference data: {diff_data_path}")

            fig_diff = create_mu_offset_difference_plot(
                diff_df,
                title=f"mu_offset difference (essential - non-essential) - {split.upper()}",
            )
            fig_diff_path = os.path.join(output_dir, f"{split}_mu_offset_difference.png")
            fig_diff.savefig(fig_diff_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {fig_diff_path}")
            plt.close(fig_diff)
        else:
            print("    Skipping mu_offset difference plot: no data found.")
        
        # Generate figures for combined analysis
        print(f"  Creating combined analysis figure...")
        fig_combined = create_split_figure_combined(
            results_by_muoffset, 
            split, 
            mu_offset_folders=folders_to_process,
        )
        fig_path_combined = os.path.join(output_dir, f"{split}_combined.png")
        fig_combined.savefig(fig_path_combined, dpi=150, bbox_inches='tight')
        print(f"    Saved: {fig_path_combined}")
        plt.close(fig_combined)
        
        # Generate figures for each strain
        print(f"  Creating per-strain analysis figures...")
        for strain in all_strains:
            fig_strain = create_split_figure_per_strain(
                results_by_muoffset, 
                strain, 
                split,
                mu_offset_folders=folders_to_process,
            )
            fig_path_strain = os.path.join(output_dir, f"{split}_{strain}.png")
            fig_strain.savefig(fig_path_strain, dpi=150, bbox_inches='tight')
            print(f"    Saved: {fig_path_strain}")
            plt.close(fig_strain)

    print(f"\n{'='*60}")
    print("Creating overall analysis figure (all strains + train/val/test)...")
    print(f"{'='*60}")

    all_sets_results = {}
    for muoff_name in folders_to_process:
        if muoff_name not in overall_by_muoffset:
            continue

        all_essential = concatenate_nonempty(overall_by_muoffset[muoff_name]['essential'])
        all_nonessential = concatenate_nonempty(overall_by_muoffset[muoff_name]['nonessential'])

        all_sets_results[muoff_name] = {
            'combined': {
                'all_splits': {
                    'essential': all_essential,
                    'nonessential': all_nonessential
                }
            }
        }

    if all_sets_results:
        print("  Saving overall plot data...")
        save_plot_data(
            output_dir=output_dir,
            file_tag="all_sets_combined",
            split_label="all_sets",
            results_by_muoffset=all_sets_results,
            mu_offset_folders=folders_to_process,
        )

        diff_df = compute_mu_offset_difference_summary(
            all_sets_results, 
            split_label="all_sets",
            mu_offset_folders=folders_to_process,
        )
        if not diff_df.empty:
            diff_data_dir = os.path.join(output_dir, "plot_data")
            os.makedirs(diff_data_dir, exist_ok=True)
            diff_data_path = os.path.join(diff_data_dir, "all_sets_mu_offset_diff.csv")
            diff_df.to_csv(diff_data_path, index=False)
            print(f"    Saved mu_offset difference data: {diff_data_path}")

            fig_diff = create_mu_offset_difference_plot(
                diff_df,
                title="mu_offset difference (essential - non-essential) - ALL SETS",
            )
            fig_diff_path = os.path.join(output_dir, "all_sets_mu_offset_difference.png")
            fig_diff.savefig(fig_diff_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {fig_diff_path}")
            plt.close(fig_diff)
        else:
            print("    Skipping overall mu_offset difference plot: no data found.")

        fig_all_sets = create_split_figure_combined(
            all_sets_results, 
            split="all_sets",
            mu_offset_folders=folders_to_process,
        )
        fig_path_all_sets = os.path.join(output_dir, "all_sets_all_strains_combined.png")
        fig_all_sets.savefig(fig_path_all_sets, dpi=150, bbox_inches='tight')
        print(f"    Saved: {fig_path_all_sets}")
        plt.close(fig_all_sets)
    else:
        print("    Skipping overall figure: no data found across splits.")

    print(f"\n{'='*60}")
    print("Creating per-strain overall analysis figures (all splits)...")
    print(f"{'='*60}")

    for strain in all_strains:
        strain_all_sets_results = {}

        for muoff_name in folders_to_process:
            if muoff_name not in overall_by_strain_muoffset.get(strain, {}):
                continue

            all_essential = concatenate_nonempty(
                overall_by_strain_muoffset[strain][muoff_name]['essential']
            )
            all_nonessential = concatenate_nonempty(
                overall_by_strain_muoffset[strain][muoff_name]['nonessential']
            )

            strain_all_sets_results[muoff_name] = {
                'per_strain': {
                    strain: {
                        'all_splits': {
                            'essential': all_essential,
                            'nonessential': all_nonessential
                        }
                    }
                }
            }

        if strain_all_sets_results:
            save_plot_data(
                output_dir=output_dir,
                file_tag=f"all_sets_{strain}",
                split_label="all_sets",
                results_by_muoffset=strain_all_sets_results,
                strains=[strain],
                mu_offset_folders=folders_to_process,
            )
            fig_strain_all_sets = create_split_figure_per_strain(
                strain_all_sets_results, 
                strain, 
                split="all_sets",
                mu_offset_folders=folders_to_process,
            )
            fig_path_strain_all_sets = os.path.join(output_dir, f"all_sets_{strain}.png")
            fig_strain_all_sets.savefig(fig_path_strain_all_sets, dpi=150, bbox_inches='tight')
            print(f"    Saved: {fig_path_strain_all_sets}")
            plt.close(fig_strain_all_sets)
        else:
            print(f"    Skipping {strain}: no data found across splits.")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Run analysis on specific ZINBAE reconstruction folder
    run_full_analysis(
        mu_offset_root_dir="Data/reconstruction",
        output_dir="AE/results/pi_analysis_ZINBAE_layers752",
        mu_offset_folders=["ZINBAE_layers752_ep93_noise0.150_muoff0.000"],
    )
