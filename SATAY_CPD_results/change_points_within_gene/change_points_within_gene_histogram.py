import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from Utils.plot_config import setup_plot_style, COLORS


class Config:
    """Configuration for counting change points within gene ranges."""

    BASE_DIR = Path(__file__).parent.parent
    GENE_INFO_PATH = BASE_DIR / "Utils" / "SGD_API" / "architecture_info" / "yeast_genes_with_info.json"
    STRAINS_DATA_PATH = BASE_DIR / "SATAY_CPD_results" / "CPD_SATAY_results"
    OUTPUT_DIR = BASE_DIR / "results" / "change_points_within_gene"

    # Easy to modify later
    STRAINS = ["FD", "yEK19", "yEK23"]
    THRESHOLD = 3.0
    MERGED_SEGMENTS_THRESHOLD = 0.25
    WINDOW_SIZE = 100
    OVERLAP = 50
    GENE_EXTENSION_BP = 100

    # Chromosome lengths from Utils/SGD_API/yeast_architecture.py
    CHROMOSOME_LENGTH = {
        "ChrI": 230218,
        "ChrII": 813184,
        "ChrIII": 316620,
        "ChrIV": 1531933,
        "ChrV": 576874,
        "ChrVI": 270161,
        "ChrVII": 1090940,
        "ChrVIII": 562643,
        "ChrIX": 439888,
        "ChrX": 745751,
        "ChrXI": 666816,
        "ChrXII": 1078171,
        "ChrXIII": 924431,
        "ChrXIV": 784333,
        "ChrXV": 1091291,
        "ChrXVI": 948066,
        "ChrM": 85779,
        "2micron": 6318,
    }

    # Requested: exclude mitochondrial and plasmid chromosomes
    ANALYSIS_CHROMOSOMES = [
        "ChrI", "ChrII", "ChrIII", "ChrIV", "ChrV", "ChrVI", "ChrVII", "ChrVIII",
        "ChrIX", "ChrX", "ChrXI", "ChrXII", "ChrXIII", "ChrXIV", "ChrXV", "ChrXVI",
    ]

    # Plotting controls for normalized change-point histogram
    NORMALIZED_CP_TOP_PERCENTILE_TO_KEEP = 99
    NORMALIZED_CP_BIN_COUNT = 25


def convert_chromosome_name(chrom_name: str) -> str:
    """Convert chromosome names to segment file format (e.g., Chromosome_V -> ChrV)."""
    if chrom_name.startswith("Chromosome_"):
        return "Chr" + chrom_name.split("_", 1)[1]
    return chrom_name


def load_genes(gene_info_path: Path) -> pd.DataFrame:
    """Load genes and essentiality from the JSON gene database."""
    with gene_info_path.open("r") as f:
        gene_data = json.load(f)

    rows = []
    for gene_id, gene_info in gene_data.items():
        location = gene_info.get("location", {})
        chromosome = convert_chromosome_name(location.get("chromosome", ""))
        start = location.get("start")
        end = location.get("end")

        if chromosome == "" or start is None or end is None:
            continue

        rows.append(
            {
                "gene_id": gene_id,
                "gene_name": gene_info.get("gene_name", gene_id),
                "chromosome": chromosome,
                "start": int(start),
                "end": int(end),
                "is_essential": bool(gene_info.get("essentiality", False)),
            }
        )

    return pd.DataFrame(rows)

def load_changepoints_from_result_txt(
    strains_data_path: Path,
    strain: str,
    chromosomes: list,
    threshold: float,
    window_size: int,
    overlap: int,
) -> pd.DataFrame:
    """Load change-point positions from window result txt files."""
    rows = []
    strain_folder = strains_data_path / f"strain_{strain}"
    threshold_candidates = [f"{threshold:.2f}", f"{threshold}", f"{threshold:.3f}".rstrip("0").rstrip(".")]
    threshold_candidates = list(dict.fromkeys(threshold_candidates))

    for chrom in chromosomes:
        base_dir = strain_folder / chrom / f"{chrom}_distances" / f"window{window_size}"
        if not base_dir.exists():
            continue

        file_path = None
        for t in threshold_candidates:
            candidate = base_dir / f"{chrom}_distances_ws{window_size}_ov{overlap}_th{t}.txt"
            if candidate.exists():
                file_path = candidate
                break

        if file_path is None:
            continue

        try:
            with file_path.open("r") as f:
                for line in f:
                    value = line.strip()
                    if value == "":
                        continue
                    rows.append(
                        {
                            "chromosome": chrom,
                            "position": int(float(value)),
                            "strain": strain,
                            "threshold": threshold,
                            "source": "txt_results",
                        }
                    )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["chromosome", "position", "strain"], keep="first").reset_index(drop=True)


def load_changepoints_from_merged_segments(
    strains_data_path: Path,
    strain: str,
    chromosomes: list,
    threshold: float,
    window_size: int,
) -> pd.DataFrame:
    """Load change-point boundaries from merged_segments files."""
    rows = []
    strain_folder = strains_data_path / f"strain_{strain}"

    threshold_candidates = [f"{threshold:.2f}", f"{threshold}", f"{threshold:.3f}".rstrip("0").rstrip(".")]
    threshold_candidates = list(dict.fromkeys(threshold_candidates))

    for chrom in chromosomes:
        merged_dir = strain_folder / chrom / f"{chrom}_distances" / f"window{window_size}" / "merged_segments"
        if not merged_dir.exists():
            continue

        file_path = None
        for t in threshold_candidates:
            candidate = merged_dir / f"{chrom}_merged_segments_muZ{t}.csv"
            if candidate.exists():
                file_path = candidate
                break

        if file_path is None:
            continue

        try:
            df = pd.read_csv(file_path)
            if "start_index" not in df.columns or "end_index_exclusive" not in df.columns:
                continue

            for _, row in df.iterrows():
                rows.append(
                    {
                        "chromosome": chrom,
                        "position": int(row["start_index"]),
                        "strain": strain,
                        "threshold": threshold,
                        "source": "merged_segments",
                    }
                )
                rows.append(
                    {
                        "chromosome": chrom,
                        "position": int(row["end_index_exclusive"]),
                        "strain": strain,
                        "threshold": threshold,
                        "source": "merged_segments",
                    }
                )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["chromosome", "position", "strain"], keep="first").reset_index(drop=True)


def load_merged_segment_lengths(
    strains_data_path: Path,
    strains: list,
    chromosomes: list,
    threshold: float,
    window_size: int,
) -> pd.DataFrame:
    """Load segment lengths from merged_segments files for multiple strains."""
    rows = []

    threshold_candidates = [f"{threshold:.2f}", f"{threshold}", f"{threshold:.3f}".rstrip("0").rstrip(".")]
    threshold_candidates = list(dict.fromkeys(threshold_candidates))

    for strain in strains:
        strain_folder = strains_data_path / f"strain_{strain}"

        for chrom in chromosomes:
            merged_dir = strain_folder / chrom / f"{chrom}_distances" / f"window{window_size}" / "merged_segments"
            if not merged_dir.exists():
                continue

            file_path = None
            for t in threshold_candidates:
                candidate = merged_dir / f"{chrom}_merged_segments_muZ{t}.csv"
                if candidate.exists():
                    file_path = candidate
                    break

            if file_path is None:
                continue

            try:
                df = pd.read_csv(file_path)
                if "start_index" not in df.columns or "end_index_exclusive" not in df.columns:
                    continue

                for _, row in df.iterrows():
                    segment_length = int(row["end_index_exclusive"]) - int(row["start_index"])
                    rows.append(
                        {
                            "strain": strain,
                            "chromosome": chrom,
                            "segment_length": segment_length,
                            "threshold": threshold,
                        }
                    )
            except Exception:
                continue

    return pd.DataFrame(rows)


def count_change_points_per_gene(
    genes_df: pd.DataFrame,
    changepoints_df: pd.DataFrame,
    extension_bp: int,
) -> pd.DataFrame:
    """
    Count change points for each gene within [start-extension, end+extension], inclusive.
    """
    cp_by_chr = {}
    for chrom, group in changepoints_df.groupby("chromosome"):
        cp_by_chr[chrom] = np.sort(group["position"].to_numpy(dtype=np.int64))

    counts = []
    for row in genes_df.itertuples(index=False):
        cp_positions = cp_by_chr.get(row.chromosome)
        if cp_positions is None or cp_positions.size == 0:
            cp_count = 0
        else:
            left = row.start - extension_bp
            right = row.end + extension_bp
            # Inclusive interval count using binary search
            left_idx = np.searchsorted(cp_positions, left, side="left")
            right_idx = np.searchsorted(cp_positions, right, side="right")
            cp_count = int(right_idx - left_idx)

        counts.append(
            {
                "gene_id": row.gene_id,
                "gene_name": row.gene_name,
                "chromosome": row.chromosome,
                "start": int(row.start),
                "end": int(row.end),
                "is_essential": bool(row.is_essential),
                "change_point_count": cp_count,
            }
        )

    return pd.DataFrame(counts)


def make_histograms(counts_df: pd.DataFrame, output_path: Path, threshold: float, extension_bp: int) -> None:
    """Create two separate histograms for essential and non-essential genes."""
    essential = counts_df[counts_df["is_essential"]]["change_point_count"].to_numpy()
    nonessential = counts_df[~counts_df["is_essential"]]["change_point_count"].to_numpy()

    max_count = int(counts_df["change_point_count"].max()) if len(counts_df) > 0 else 0
    bins = np.arange(-0.5, max_count + 1.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    axes[0].hist(essential, bins=bins, color=COLORS["red"], alpha=0.8, edgecolor="black")
    axes[0].set_title("Essential Genes")
    axes[0].set_xlabel(f"Change points within gene")
    axes[0].set_ylabel("Number of genes")

    axes[1].hist(nonessential, bins=bins, color=COLORS["blue"], alpha=0.8, edgecolor="black")
    axes[1].set_title("Non-essential Genes")
    axes[1].set_xlabel(f"Change points within gene  ")

    fig.suptitle(f"Pooled strains: change points per gene (threshold={threshold:.2f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_overlapping_histogram(
    counts_df: pd.DataFrame,
    output_path: Path,
    threshold: float,
    extension_bp: int,
) -> None:
    """Create one overlapping histogram with essential genes drawn on top."""
    essential = counts_df[counts_df["is_essential"]]["change_point_count"].to_numpy()
    nonessential = counts_df[~counts_df["is_essential"]]["change_point_count"].to_numpy()

    max_count = int(counts_df["change_point_count"].max()) if len(counts_df) > 0 else 0
    bins = np.arange(-0.5, max_count + 1.5, 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw non-essential first, then essential on top.
    ax.hist(
        nonessential,
        bins=bins,
        color=COLORS["blue"],
        alpha=0.45,
        edgecolor="black",
        linewidth=0.7,
        label="Non-essential genes",
        zorder=1,
    )
    ax.hist(
        essential,
        bins=bins,
        color=COLORS["red"],
        alpha=0.65,
        edgecolor="black",
        linewidth=0.7,
        label="Essential genes",
        zorder=2,
    )

    ax.set_xlabel(f"Change points within gene +/- {extension_bp} bp")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"Overlapping histogram (pooled strains, threshold={threshold:.2f})")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_merged_change_point_count_histogram(
    counts_df: pd.DataFrame,
    output_path: Path,
    threshold_cp: float,
    threshold_ms: float,
    extension_bp: int,
) -> None:
    """Create one merged overlapping histogram for essential and non-essential genes."""
    essential = counts_df[counts_df["is_essential"]]["change_point_count"].to_numpy(dtype=np.int64)
    nonessential = counts_df[~counts_df["is_essential"]]["change_point_count"].to_numpy(dtype=np.int64)
    if essential.size == 0 and nonessential.size == 0:
        return

    max_count = int(counts_df["change_point_count"].max())
    bins = np.arange(-0.5, max_count + 1.5, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        nonessential,
        bins=bins,
        color=COLORS["blue"],
        alpha=0.45,
        edgecolor="black",
        linewidth=0.7,
        label="Non-essential genes",
        zorder=1,
    )
    ax.hist(
        essential,
        bins=bins,
        color=COLORS["red"],
        alpha=0.65,
        edgecolor="black",
        linewidth=0.7,
        label="Essential genes",
        zorder=2,
    )
    ax.set_xlabel(f"Change points within gene +/- {extension_bp} bp")
    ax.set_ylabel("Number of genes")
    ax.set_title(
        f"Merged overlapping change-point distribution (pooled strains, thCP={threshold_cp:.2f}, thMS={threshold_ms:.2f})"
    )
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_length_normalized_change_point_histogram(
    counts_df: pd.DataFrame,
    output_path: Path,
    threshold_cp: float,
    threshold_ms: float | None = None,
) -> None:
    """Create overlap histogram for gene-length-normalized change points (CP per kb)."""
    essential = counts_df[counts_df["is_essential"]]["change_points_per_kb"].to_numpy(dtype=float)
    nonessential = counts_df[~counts_df["is_essential"]]["change_points_per_kb"].to_numpy(dtype=float)

    if essential.size == 0 and nonessential.size == 0:
        return

    all_values = counts_df["change_points_per_kb"].to_numpy(dtype=float)
    upper = float(np.percentile(all_values, Config.NORMALIZED_CP_TOP_PERCENTILE_TO_KEEP)) if all_values.size > 0 else 1.0
    if upper <= 0:
        upper = 1.0
    bins = np.linspace(0.0, upper, Config.NORMALIZED_CP_BIN_COUNT)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        nonessential,
        bins=bins,
        color=COLORS["blue"],
        alpha=0.45,
        edgecolor="black",
        linewidth=0.7,
        label="Non-essential genes",
        zorder=1,
    )
    ax.hist(
        essential,
        bins=bins,
        color=COLORS["red"],
        alpha=0.65,
        edgecolor="black",
        linewidth=0.7,
        label="Essential genes",
        zorder=2,
    )

    ax.set_xlabel("Change points per kb gene length")
    ax.set_ylabel("Number of genes")
    if threshold_ms is None:
        title = f"Length-normalized change-point distribution (pooled strains, thCP={threshold_cp:.2f})"
    else:
        title = (
            "Length-normalized change-point distribution "
            f"(pooled strains, thCP={threshold_cp:.2f}, thMS={threshold_ms:.2f})"
        )
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    config = Config()
    setup_plot_style()

    if not config.GENE_INFO_PATH.exists():
        print(f"Gene info file not found: {config.GENE_INFO_PATH}", file=sys.stderr)
        sys.exit(1)

    if not config.STRAINS_DATA_PATH.exists():
        print(f"Strains data path not found: {config.STRAINS_DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    genes_df = load_genes(config.GENE_INFO_PATH)
    if genes_df.empty:
        print("No genes loaded from gene info file.", file=sys.stderr)
        sys.exit(1)

    pooled_counts = []
    pooled_counts_merged = []
    pooled_genome_segments = []

    genes_df = genes_df[genes_df["chromosome"].isin(config.ANALYSIS_CHROMOSOMES)].copy()

    for strain in config.STRAINS:
        changepoints_df = load_changepoints_from_result_txt(
            strains_data_path=config.STRAINS_DATA_PATH,
            strain=strain,
            chromosomes=config.ANALYSIS_CHROMOSOMES,
            threshold=config.THRESHOLD,
            window_size=config.WINDOW_SIZE,
            overlap=config.OVERLAP,
        )

        if changepoints_df.empty:
            print(f"No change points found for strain {strain} at threshold {config.THRESHOLD:.2f}.")
            continue

        strain_counts = count_change_points_per_gene(
            genes_df=genes_df,
            changepoints_df=changepoints_df,
            extension_bp=config.GENE_EXTENSION_BP,
        )
        strain_counts["strain"] = strain
        strain_counts["threshold"] = config.THRESHOLD
        pooled_counts.append(strain_counts)

        merged_changepoints_df = load_changepoints_from_merged_segments(
            strains_data_path=config.STRAINS_DATA_PATH,
            strain=strain,
            chromosomes=config.ANALYSIS_CHROMOSOMES,
            threshold=config.MERGED_SEGMENTS_THRESHOLD,
            window_size=config.WINDOW_SIZE,
        )
        if not merged_changepoints_df.empty:
            strain_counts_merged = count_change_points_per_gene(
                genes_df=genes_df,
                changepoints_df=merged_changepoints_df,
                extension_bp=config.GENE_EXTENSION_BP,
            )
            strain_counts_merged["strain"] = strain
            strain_counts_merged["threshold_cp"] = config.THRESHOLD
            strain_counts_merged["threshold_ms"] = config.MERGED_SEGMENTS_THRESHOLD
            pooled_counts_merged.append(strain_counts_merged)

    if not pooled_counts:
        print("No pooled results generated. Check threshold and input files.", file=sys.stderr)
        sys.exit(1)

    pooled_df = pd.concat(pooled_counts, ignore_index=True)
    pooled_df["gene_length_bp"] = (pooled_df["end"] - pooled_df["start"]).clip(lower=1)
    pooled_df["change_points_per_kb"] = pooled_df["change_point_count"] / (pooled_df["gene_length_bp"] / 1000.0)

    pooled_merged_cp_df = pd.concat(pooled_counts_merged, ignore_index=True) if pooled_counts_merged else pd.DataFrame()
    if not pooled_merged_cp_df.empty:
        pooled_merged_cp_df["gene_length_bp"] = (pooled_merged_cp_df["end"] - pooled_merged_cp_df["start"]).clip(lower=1)
        pooled_merged_cp_df["change_points_per_kb"] = (
            pooled_merged_cp_df["change_point_count"] / (pooled_merged_cp_df["gene_length_bp"] / 1000.0)
        )

    pooled_genome_segments_df = pd.concat(pooled_genome_segments, ignore_index=True) if pooled_genome_segments else pd.DataFrame()

    pooled_merged_segments_df = load_merged_segment_lengths(
        strains_data_path=config.STRAINS_DATA_PATH,
        strains=config.STRAINS,
        chromosomes=config.ANALYSIS_CHROMOSOMES,
        threshold=config.MERGED_SEGMENTS_THRESHOLD,
        window_size=config.WINDOW_SIZE,
    )

    csv_path = config.OUTPUT_DIR / f"gene_change_point_counts_th{config.THRESHOLD:.2f}.csv"
    pooled_df.to_csv(csv_path, index=False)

    hist_path = config.OUTPUT_DIR / f"gene_change_point_histograms_th{config.THRESHOLD:.2f}.png"
    make_histograms(
        counts_df=pooled_df,
        output_path=hist_path,
        threshold=config.THRESHOLD,
        extension_bp=config.GENE_EXTENSION_BP,
    )

    overlap_hist_path = config.OUTPUT_DIR / f"gene_change_point_histogram_overlap_th{config.THRESHOLD:.2f}.png"
    make_overlapping_histogram(
        counts_df=pooled_df,
        output_path=overlap_hist_path,
        threshold=config.THRESHOLD,
        extension_bp=config.GENE_EXTENSION_BP,
    )

    merged_cp_hist_path = config.OUTPUT_DIR / (
        f"gene_change_point_histogram_merged_overlap_thCP{config.THRESHOLD:.2f}_thMS{config.MERGED_SEGMENTS_THRESHOLD:.2f}.png"
    )
    if not pooled_merged_cp_df.empty:
        make_merged_change_point_count_histogram(
            counts_df=pooled_merged_cp_df,
            output_path=merged_cp_hist_path,
            threshold_cp=config.THRESHOLD,
            threshold_ms=config.MERGED_SEGMENTS_THRESHOLD,
            extension_bp=config.GENE_EXTENSION_BP,
        )

    normalized_cp_hist_path = config.OUTPUT_DIR / f"gene_change_point_histogram_normalized_by_length_thCP{config.THRESHOLD:.2f}.png"
    make_length_normalized_change_point_histogram(
        counts_df=pooled_df,
        output_path=normalized_cp_hist_path,
        threshold_cp=config.THRESHOLD,
    )

    merged_normalized_cp_hist_path = config.OUTPUT_DIR / (
        f"gene_change_point_histogram_normalized_by_length_merged_thCP{config.THRESHOLD:.2f}_thMS{config.MERGED_SEGMENTS_THRESHOLD:.2f}.png"
    )
    if not pooled_merged_cp_df.empty:
        make_length_normalized_change_point_histogram(
            counts_df=pooled_merged_cp_df,
            output_path=merged_normalized_cp_hist_path,
            threshold_cp=config.THRESHOLD,
            threshold_ms=config.MERGED_SEGMENTS_THRESHOLD,
        )

    if not pooled_genome_segments_df.empty:
        genome_segments_csv = config.OUTPUT_DIR / f"segment_lengths_genome_th{config.THRESHOLD:.2f}.csv"
        pooled_genome_segments_df.to_csv(genome_segments_csv, index=False)
    else:
        genome_segments_csv = None

    if not pooled_merged_segments_df.empty:
        merged_segments_csv = config.OUTPUT_DIR / (
            f"segment_lengths_merged_segments_thMS{config.MERGED_SEGMENTS_THRESHOLD:.2f}.csv"
        )
        pooled_merged_segments_df.to_csv(merged_segments_csv, index=False)
    else:
        merged_segments_csv = None

    essential_count = int(pooled_df["is_essential"].sum())
    nonessential_count = int((~pooled_df["is_essential"]).sum())

    print(f"Saved per-gene counts to: {csv_path}")
    print(f"Saved histograms to: {hist_path}")
    print(f"Saved overlapping histogram to: {overlap_hist_path}")
    if not pooled_merged_cp_df.empty:
        print(f"Saved merged change-point histogram to: {merged_cp_hist_path}")
    else:
        print("Merged change-point histogram not generated: no merged-segment change points found.")
    print(f"Saved length-normalized change-point histogram to: {normalized_cp_hist_path}")
    if not pooled_merged_cp_df.empty:
        print(f"Saved merged length-normalized change-point histogram to: {merged_normalized_cp_hist_path}")
    if genome_segments_csv is not None:
        print(f"Saved genome-wide segment lengths to: {genome_segments_csv}")
    if merged_segments_csv is not None:
        print(f"Saved merged-segment lengths to: {merged_segments_csv}")

    print(f"Rows (pooled across strains): {len(pooled_df)}")
    print(f"Essential rows: {essential_count}")
    print(f"Non-essential rows: {nonessential_count}")

    if not pooled_genome_segments_df.empty:
        genome_lengths = pooled_genome_segments_df["segment_length"]
        print(
            "Genome-wide segments: "
            f"n={len(genome_lengths)}, mean={genome_lengths.mean():.2f}, "
            f"median={genome_lengths.median():.2f}, max={genome_lengths.max()}"
        )

    if not pooled_merged_segments_df.empty:
        merged_lengths = pooled_merged_segments_df["segment_length"]
        print(
            "Merged-segment lengths (pooled strains): "
            f"n={len(merged_lengths)}, mean={merged_lengths.mean():.2f}, "
            f"median={merged_lengths.median():.2f}, max={merged_lengths.max()}"
        )


if __name__ == "__main__":
    main()
