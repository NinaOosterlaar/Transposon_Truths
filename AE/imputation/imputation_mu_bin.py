import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Set

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Utils.SGD_API.yeast_genes import SGD_Genes
from Utils.plot_config import setup_plot_style

setup_plot_style()

# Chromosome list (S288C reference)
CHROMOSOMES = [
    "ChrI", "ChrII", "ChrIII", "ChrIV", "ChrV", "ChrVI", "ChrVII", 
    "ChrVIII", "ChrIX", "ChrX", "ChrXI", "ChrXII", "ChrXIII", "ChrXIV", 
    "ChrXV", "ChrXVI"
]

# Splits to analyze
SPLITS = ["train", "val", "test"]

# Bin parameters for position mapping fallback
WINDOW_SIZE = 17
STEP_SIZE = 1

# Position mapping conventions
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


def trim_top_percent(values: np.ndarray, top_percent: float = 5.0) -> np.ndarray:
    """Remove the highest `top_percent` values from a numeric array."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return arr

    if top_percent <= 0:
        return arr

    if top_percent >= 100:
        return np.array([], dtype=float)

    cutoff = np.percentile(arr, 100 - top_percent)
    return arr[arr <= cutoff]


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
    """Infer preprocessing bin size from metadata.bin_size with fallback."""
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
            f"(source={source}), bin_size={effective_window_size}"
        )
    else:
        effective_position_mode = requested_position_mode
        print(
            f"Split {os.path.basename(split_dir)}: position_mode={effective_position_mode} "
            f"(manual), bin_size={effective_window_size}"
        )

    return effective_position_mode, effective_window_size


def map_reconstruction_position_to_original_window(
    position: int,
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_INDEX,
) -> Tuple[int, int]:
    """
    Map one reconstructed position to its original 1-based genomic bin range.

    Modes:
    - index: position is a 0-based bin index i
      bin_range = [i * bin_size + 1, (i + 1) * bin_size]
    - center: position is a genomic center coordinate of a bin
      bin_start = position - floor((bin_size - 1)/2)
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
        window_start = position * moving_average_window_size + 1
    else:
        # center mode: metadata/reconstruction stores center-like genomic bin positions
        left_half = (moving_average_window_size - 1) // 2
        window_start = position - left_half

    window_start = max(1, int(window_start))
    window_end = int(window_start + moving_average_window_size - 1)
    return window_start, window_end


def is_mu_position_associated_with_original_zero(
    position: int,
    original_zero_positions: Set[int],
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_INDEX,
) -> bool:
    """
    Determine whether a reconstruction position maps to an all-zero original bin.

    The `position` value can be interpreted either as a bin index
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


def load_gene_data(reconstruction_root_dir: str) -> Dict:
    """
    Load gene data from SGD_Genes class and filter to genes in reconstruction.
    
    Args:
        reconstruction_root_dir: Root directory containing reconstruction outputs
        
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
    Find all genes overlapping with a reconstructed position's original bin range.
    
    Args:
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name (e.g., "ChrIII")
        position: Position value from reconstruction CSV
        window_size: Bin size in bp
        step_size: Deprecated for bin mapping (kept for API compatibility)
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


def assign_mu_to_genes(
    csv_file: str,
    genes_by_chr: Dict,
    chr_name: str,
    filter_mu_by_original_zeros: bool = True,
    original_zero_positions: Optional[Set[int]] = None,
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_INDEX,
) -> Dict[str, List[float]]:
    """
    Parse CSV and assign mu values to genes based on position-gene overlap.
    
    Args:
        csv_file: Path to CSV file with columns [position, reconstruction, mu, pi, theta]
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name
        filter_mu_by_original_zeros:
            If True, only keep mu values whose mapped original bin range
            is entirely zero in Data/combined_strains
        original_zero_positions: Set of 1-based positions where original Value == 0
        moving_average_window_size:
            Bin size used in preprocessing
        moving_average_step_size:
            Deprecated for bin mapping (kept for API compatibility)
        position_mode:
            Position convention used in reconstruction CSV ('index' or 'center')
        
    Returns:
        Dictionary mapping gene_name -> [list of mu values]
    """
    raw_df = pd.read_csv(
        csv_file,
        usecols=["position", "mu"],
        low_memory=False,
        dtype={"position": "string", "mu": "string"}
    )
    df = raw_df.copy()
    df["position"] = pd.to_numeric(raw_df["position"], errors="coerce")
    df["mu"] = pd.to_numeric(raw_df["mu"], errors="coerce")

    invalid_mask = df["position"].isna() | df["mu"].isna()
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(f"Warning: Found {invalid_count} invalid rows in {csv_file} (non-numeric position or mu). These rows will be skipped.")

    df = df.loc[~invalid_mask, ["position", "mu"]]
    if df.empty:
        return defaultdict(list)
    
    genes_mu = defaultdict(list)
    
    if filter_mu_by_original_zeros and original_zero_positions is None:
        print(
            f"Warning: zero filtering enabled but no original position set provided for {csv_file}. "
            f"Falling back to unfiltered mu values."
        )

    for position, mu_value in df.itertuples(index=False, name=None):
        position = int(position)
        mu_value = float(mu_value)

        if filter_mu_by_original_zeros and original_zero_positions is not None:
            if not is_mu_position_associated_with_original_zero(
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
        
        # Assign mu value to each overlapping gene
        for gene in overlapping_genes:
            genes_mu[gene['gene']].append(mu_value)
    
    return genes_mu


def aggregate_mu_by_essentiality(
    genes_mu: Dict[str, List[float]],
    genes_by_chr: Dict,
    chr_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate all mu values by essentiality status.
    
    Args:
        genes_mu: Dictionary mapping gene name to list of mu values
        genes_by_chr: Gene data organized by chromosome
        chr_name: Chromosome name
        
    Returns:
        Tuple of (essential_mu_values, nonessential_mu_values) as numpy arrays
    """
    # Create lookup for essentiality
    essentiality_lookup = {}
    for gene in genes_by_chr.get(chr_name, []):
        essentiality_lookup[gene['gene']] = gene['essentiality']
    
    essential_mu = []
    nonessential_mu = []
    
    for gene_name, mu_values in genes_mu.items():
        essentiality = essentiality_lookup.get(gene_name)
        
        if essentiality is None:
            continue
        
        if essentiality:  # Essential
            essential_mu.extend(mu_values)
        else:  # Non-essential
            nonessential_mu.extend(mu_values)
    
    return np.array(essential_mu), np.array(nonessential_mu)


def process_single_split(
    split_dir: str,
    genes_by_chr: Dict,
    strains: Optional[List[str]] = None,
    filter_mu_by_original_zeros: bool = True,
    original_data_root_dir: str = "Data/combined_strains",
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    original_zero_positions_cache: Optional[Dict[Tuple[str, str], Set[int]]] = None,
    position_mode: str = POSITION_MODE_AUTO,
) -> Dict[str, Dict]:
    """
    Process all chromosomes for a single split (e.g., train).
    Collect results separately per strain and combined.
    
    Args:
        split_dir: Path to split directory containing strain_* folders
        genes_by_chr: Gene data
        strains: List of strains to process, or None for all found
        filter_mu_by_original_zeros: If True, only include mu where original data Value == 0
        original_data_root_dir: Root directory containing original strain files
        moving_average_window_size: Bin size used in preprocessing
        moving_average_step_size: Deprecated for bin mapping (kept for API compatibility)
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
            if filter_mu_by_original_zeros:
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
            
            # Assign mu to genes
            genes_mu = assign_mu_to_genes(
                csv_file,
                genes_by_chr,
                chr_name,
                filter_mu_by_original_zeros=filter_mu_by_original_zeros,
                original_zero_positions=zero_positions,
                moving_average_window_size=effective_window_size,
                moving_average_step_size=moving_average_step_size,
                position_mode=effective_position_mode,
            )
            
            # Aggregate by essentiality
            essential, nonessential = aggregate_mu_by_essentiality(
                genes_mu, genes_by_chr, chr_name
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
    essential_mu: np.ndarray,
    nonessential_mu: np.ndarray,
    title: str
) -> None:
    """
    Create a boxplot comparing essential vs non-essential gene mu values.
    
    Args:
        ax: Matplotlib axis to plot on
        essential_mu: Array of mu values for essential genes
        nonessential_mu: Array of mu values for non-essential genes
        title: Title for the subplot
    """
    # Remove the highest 5% in each group to keep the boxplot readable.
    essential_mu = trim_top_percent(essential_mu, top_percent=5.0)
    nonessential_mu = trim_top_percent(nonessential_mu, top_percent=5.0)
    data_to_plot = [essential_mu, nonessential_mu]
    
    bp = ax.boxplot(
        data_to_plot,
        tick_labels=['Essential', 'Non-Essential'],
        patch_artist=True,
        widths=0.6
    )
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('mu value', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # # Add sample size info
    # n_ess = len(essential_mu)
    # n_noness = len(nonessential_mu)
    # ax.text(0.98, 0.97, f'n={n_ess}/{n_noness}', 
    #         transform=ax.transAxes, ha='right', va='top',
    #         fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_split_figure_combined(
    results: Dict[str, Dict],
    split: str,
    figsize: Tuple = (7, 6)
) -> plt.Figure:
    """
    Create one figure comparing essential vs non-essential genes for a split.

    Args:
        results: Split results dictionary with keys ['combined', 'per_strain']
        split: Split label for title and output naming
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    combined = results.get('combined', {})
    all_essential = concatenate_nonempty([
        chr_values['essential'] for chr_values in combined.values()
    ])
    all_nonessential = concatenate_nonempty([
        chr_values['nonessential'] for chr_values in combined.values()
    ])

    if len(all_essential) > 0 and len(all_nonessential) > 0:
        create_comparison_boxplot(
            ax,
            all_essential,
            all_nonessential,
            title=f"All chromosomes ({split})"
        )
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f'Essential vs Non-Essential Genes - {split.upper()}',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )
    plt.tight_layout()

    return fig


def create_split_figure_per_strain(
    results: Dict[str, Dict],
    strain: str,
    split: str,
    figsize: Tuple = (7, 6)
) -> plt.Figure:
    """
    Create one figure comparing essential vs non-essential genes for one strain.

    Args:
        results: Split results dictionary with keys ['combined', 'per_strain']
        strain: Strain name
        split: Split label for title and output naming
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    per_strain = results.get('per_strain', {})
    strain_results = per_strain.get(strain, {})

    all_essential = concatenate_nonempty([
        chr_values['essential'] for chr_values in strain_results.values()
    ])
    all_nonessential = concatenate_nonempty([
        chr_values['nonessential'] for chr_values in strain_results.values()
    ])

    if len(all_essential) > 0 and len(all_nonessential) > 0:
        create_comparison_boxplot(
            ax,
            all_essential,
            all_nonessential,
            title=f"{strain} ({split})"
        )
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f'{strain} - Essential vs Non-Essential Genes - {split.upper()}',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )
    plt.tight_layout()

    return fig


def discover_available_splits(reconstruction_root_dir: str) -> List[Tuple[str, str]]:
    """Return available split labels and directories under reconstruction root."""
    split_aliases = {
        "train": ["train"],
        "val": ["val", "validation"],
        "test": ["test"],
    }

    available_splits: List[Tuple[str, str]] = []
    for split in SPLITS:
        for split_folder in split_aliases.get(split, [split]):
            split_dir = os.path.join(reconstruction_root_dir, split_folder)
            if os.path.isdir(split_dir):
                available_splits.append((split, split_dir))
                break

    return available_splits


def run_full_analysis(
    reconstruction_root_dir: str = "Data/reconstruction/ZINBAE_layers1168_ep116_noise0.150_muoff0.000",
    output_dir: str = "AE/results/mu_analysis_bin",
    filter_mu_by_original_zeros: bool = True,
    original_data_root_dir: str = "Data/combined_strains",
    moving_average_window_size: int = WINDOW_SIZE,
    moving_average_step_size: int = STEP_SIZE,
    position_mode: str = POSITION_MODE_AUTO,
) -> None:
    """
    Run complete analysis: load genes, process split folders, and generate
    combined/per-strain visualizations without mu_offset-specific grouping.

    Args:
        reconstruction_root_dir: Root directory containing split folders
            (e.g., train, validation/val, test)
        output_dir: Output directory for figures
        filter_mu_by_original_zeros:
            If True (default), only use mu values at positions where original
            bin range in Data/combined_strains is entirely zero
        original_data_root_dir: Root directory containing original strain CSV files
        moving_average_window_size: Bin size used in preprocessing
        moving_average_step_size: Deprecated for bin mapping (kept for API compatibility)
        position_mode:
            Position convention to use ('auto', 'index', 'center').
            In 'auto' mode, inferred from each split metadata.json.
    """

    if position_mode not in VALID_POSITION_MODES:
        raise ValueError(
            f"Unknown position_mode={position_mode}. Expected one of {sorted(VALID_POSITION_MODES)}"
        )

    if filter_mu_by_original_zeros and not output_dir.endswith("_original_zero_filter"):
        output_dir = f"{output_dir}_original_zero_filter"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading gene data...")
    genes_by_chr = load_gene_data(reconstruction_root_dir)

    if filter_mu_by_original_zeros:
        print(
            f"Filtering mu values to original zeros from {original_data_root_dir} "
            f"(keep only all-zero bins; bin_size={moving_average_window_size})."
        )
        print(f"Using filtered output directory: {output_dir}")
    else:
        print("Using all mu values (original zero filter disabled).")

    available_splits = discover_available_splits(reconstruction_root_dir)
    if not available_splits:
        raise FileNotFoundError(
            f"No split directories found under {reconstruction_root_dir}. "
            "Expected at least one of: train, val/validation, test."
        )

    print(
        "Using split directories: "
        + ", ".join([os.path.basename(split_dir) for _, split_dir in available_splits])
    )

    sample_dir = available_splits[0][1]
    all_strains = sorted([
        d for d in os.listdir(sample_dir)
        if os.path.isdir(os.path.join(sample_dir, d)) and d.startswith('strain_')
    ])
    print(f"Found strains: {all_strains}")

    overall_combined = {'essential': [], 'nonessential': []}
    overall_by_strain = defaultdict(lambda: {'essential': [], 'nonessential': []})
    original_zero_positions_cache: Dict[Tuple[str, str], Set[int]] = {}

    for split, split_dir in available_splits:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")

        results = process_single_split(
            split_dir,
            genes_by_chr,
            all_strains,
            filter_mu_by_original_zeros=filter_mu_by_original_zeros,
            original_data_root_dir=original_data_root_dir,
            moving_average_window_size=moving_average_window_size,
            moving_average_step_size=moving_average_step_size,
            original_zero_positions_cache=original_zero_positions_cache,
            position_mode=position_mode,
        )

        for chr_values in results.get('combined', {}).values():
            overall_combined['essential'].append(chr_values['essential'])
            overall_combined['nonessential'].append(chr_values['nonessential'])

        for strain_name, strain_results in results.get('per_strain', {}).items():
            for chr_values in strain_results.values():
                overall_by_strain[strain_name]['essential'].append(chr_values['essential'])
                overall_by_strain[strain_name]['nonessential'].append(chr_values['nonessential'])

        print("  Creating combined analysis figure...")
        fig_combined = create_split_figure_combined(results, split)
        fig_path_combined = os.path.join(output_dir, f"{split}_combined.png")
        fig_combined.savefig(fig_path_combined, dpi=150, bbox_inches='tight')
        print(f"    Saved: {fig_path_combined}")
        plt.close(fig_combined)

        print("  Creating per-strain analysis figures...")
        for strain in all_strains:
            fig_strain = create_split_figure_per_strain(results, strain, split)
            fig_path_strain = os.path.join(output_dir, f"{split}_{strain}.png")
            fig_strain.savefig(fig_path_strain, dpi=150, bbox_inches='tight')
            print(f"    Saved: {fig_path_strain}")
            plt.close(fig_strain)

    print(f"\n{'='*60}")
    print("Creating overall analysis figure (all splits combined)...")
    print(f"{'='*60}")

    all_essential = concatenate_nonempty(overall_combined['essential'])
    all_nonessential = concatenate_nonempty(overall_combined['nonessential'])

    if len(all_essential) > 0 and len(all_nonessential) > 0:
        all_splits_results = {
            'combined': {
                'all_splits': {
                    'essential': all_essential,
                    'nonessential': all_nonessential,
                }
            }
        }
        fig_all_splits = create_split_figure_combined(all_splits_results, split="all_splits")
        fig_path_all_splits = os.path.join(output_dir, "all_splits_combined.png")
        fig_all_splits.savefig(fig_path_all_splits, dpi=150, bbox_inches='tight')
        print(f"    Saved: {fig_path_all_splits}")
        plt.close(fig_all_splits)
    else:
        print("    Skipping overall figure: no data found across splits.")

    print(f"\n{'='*60}")
    print("Creating per-strain overall analysis figures (all splits)...")
    print(f"{'='*60}")

    for strain in all_strains:
        all_essential = concatenate_nonempty(overall_by_strain[strain]['essential'])
        all_nonessential = concatenate_nonempty(overall_by_strain[strain]['nonessential'])

        if len(all_essential) == 0 or len(all_nonessential) == 0:
            print(f"    Skipping {strain}: no data found across splits.")
            continue

        strain_all_splits_results = {
            'per_strain': {
                strain: {
                    'all_splits': {
                        'essential': all_essential,
                        'nonessential': all_nonessential,
                    }
                }
            }
        }

        fig_strain_all_splits = create_split_figure_per_strain(
            strain_all_splits_results,
            strain,
            split="all_splits",
        )
        fig_path_strain_all_splits = os.path.join(output_dir, f"all_splits_{strain}.png")
        fig_strain_all_splits.savefig(fig_path_strain_all_splits, dpi=150, bbox_inches='tight')
        print(f"    Saved: {fig_path_strain_all_splits}")
        plt.close(fig_strain_all_splits)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_full_analysis()
