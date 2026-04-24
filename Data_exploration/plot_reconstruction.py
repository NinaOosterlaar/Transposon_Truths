import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

CHROMS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]

cpd = {
    "I": [-910, -780, -500, -350, -200, -70, 80, 760],
    "II": [-740, -580, -70, 65, 800],
    "III": [-900, -200, -80, 80, 750],
    "IV": [-900, -800, -740, -500, -380, -60, 60],
    "V": [-520, -280, -70, 90, 200, 430],
    "VI": [-960, -860, 460],
    "VII": [-440, -110, 70, 260, 320, 730],
    "VIII": [-860, -510, -80, 80, 200, 370, 580],
    "IX": [-400, -60, 80, 200, 300, 760],
    "X": [-920, -480, -80, 60, 200],
    "XI": [-660, -380, -70, 60, 180, 280, 470],
    "XII": [-180, -75, 70, 460],
    "XIII": [-770, -420, -300, -140, -65, 80, 780],
    "XIV": [-920, -140, -65, 80],
    "XV": [-740, -120, 60, 380, 930],
    "XVI": [-580, -170, -70, 80],
}


def retrieve_gaussian_pred(window_size, threshold, base_dir):
    """Read Gaussian CPD prediction files and return chromosome -> change-point list."""
    pred = {}

    for chrom in CHROMS:
        filepath = (
            base_dir
            / f"Chr{chrom}"
            / f"Chr{chrom}_centromere_window"
            / f"window{window_size}"
            / f"Chr{chrom}_centromere_window_ws{window_size}_ov50_th{threshold:.2f}.txt"
        )

        if not filepath.exists():
            print(f"Warning: File not found for Chr{chrom}: {filepath}")
            pred[chrom] = []
            continue

        change_points = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    change_points.append(int(text))
                except ValueError:
                    # Means/sigmas/metadata lines are intentionally ignored.
                    pass

        pred[chrom] = change_points
        print(f"Chr{chrom}: Found {len(change_points)} Gaussian change points (th={threshold})")

    return pred


def _normalize_columns(df):
    out = df.copy()
    out.columns = out.columns.str.lower().str.replace(" ", "_")
    return out


def _get_x_column(df):
    if "centromere_distance" in df.columns:
        return "centromere_distance"
    if "position" in df.columns:
        return "position"
    raise KeyError(f"Expected one of ['centromere_distance', 'position'] columns, got {list(df.columns)}")


def plot_reconstruction_with_gaussian(
    chrom,
    filepath,
    outpath,
    gaussian_pred,
    threshold,
    n=1000,
    major_tick_step=100,
    minor_tick_step=50,
    dpi=200,
    value_max=None,
    pred_window=100,
):
    df = _normalize_columns(pd.read_csv(filepath))

    if "value" not in df.columns:
        raise KeyError(f"{filepath}: expected 'value' column, got {list(df.columns)}")

    x_col = _get_x_column(df)

    # Keep only the centromere window requested for plotting.
    win = df.loc[df[x_col].between(-n, n)].copy()
    if win.empty:
        raise ValueError(f"{filepath}: no rows in requested range [-{n}, {n}]")

    if value_max is not None:
        win.loc[win["value"] > value_max, "value"] = float("nan")

    win = win.sort_values(x_col)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(win[x_col], win["value"], linewidth=1, label="Reconstruction")

    ax.axvline(0, linestyle="--", linewidth=2)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_step))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_step))
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.15)

    x_min, x_max = win[x_col].min(), win[x_col].max()

    cpd_x_all = cpd.get(chrom, [])
    cpd_x = [x for x in cpd_x_all if x_min <= x <= x_max]

    pred_x_all = gaussian_pred.get(chrom, [])
    pred_x_shifted = [x - 1000 for x in pred_x_all]
    pred_x = [x for x in pred_x_shifted if x_min <= x <= x_max]

    y_min, y_max = ax.get_ylim()
    offset = 0.03 * (y_max - y_min)
    y_cpd = y_min - offset

    if len(cpd_x) > 0:
        ax.scatter(
            cpd_x,
            [y_cpd] * len(cpd_x),
            s=28,
            marker="o",
            color="red",
            zorder=5,
            label="CPD",
        )

    for x in pred_x:
        ax.axvspan(
            x - pred_window / 2,
            x + pred_window / 2,
            alpha=0.12,
            color="tab:blue",
            zorder=1,
        )

    if len(pred_x) > 0:
        ax.scatter(
            pred_x,
            [y_cpd] * len(pred_x),
            s=16,
            marker="o",
            color="tab:blue",
            zorder=6,
            label=f"Gaussian Pred (th={threshold:g})",
        )

    # Expand bottom axis limit so CPD/pred dots remain visible.
    ax.set_ylim(y_cpd - 2.0 * offset, y_max)

    title_name = os.path.basename(filepath).replace("_centromere_window.csv", "")
    ax.set_title(f"{title_name} reconstruction with Gaussian overlay (th={threshold:g})")
    ax.set_xlabel("Centromere distance")
    ax.set_ylabel("Value")

    if (len(cpd_x) > 0) or (len(pred_x) > 0):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    ax.tick_params(axis="x", labelrotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_all_chromosomes(input_dir, gaussian_dir, output_dir, threshold=3.0, window_size=100):
    input_dir = Path(input_dir)
    gaussian_dir = Path(gaussian_dir)
    output_dir = Path(output_dir)

    gaussian_pred = retrieve_gaussian_pred(window_size=window_size, threshold=threshold, base_dir=gaussian_dir)

    for chrom in CHROMS:
        infile = input_dir / f"Chr{chrom}_centromere_window.csv"
        outfile = output_dir / f"Chr{chrom}_centromere_window_reconstruction.png"

        if not infile.exists():
            print(f"Skipping missing input: {infile}")
            continue

        plot_reconstruction_with_gaussian(
            chrom=chrom,
            filepath=infile,
            outpath=outfile,
            gaussian_pred=gaussian_pred,
            threshold=threshold,
            n=1000,
            major_tick_step=100,
            minor_tick_step=50,
            value_max=None,
            pred_window=window_size,
            dpi=200,
        )
        print(f"Saved: {outfile}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reconstructed centromere windows with Gaussian CPD overlay.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Data/reconstruction_cpd_test_all_chrom/centromere_window/7/yEK23_1"),
        help="Folder containing chromosome CSV files like ChrVII_centromere_window.csv",
    )
    parser.add_argument(
        "--gaussian-dir",
        type=Path,
        default=Path("Signal_processing/Results2/reconstruction_cpd/Gaussian/7/yEK23_1"),
        help="Gaussian CPD root folder containing ChrI..ChrXVI subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Signal_processing/Results2/reconstruction_cpd/reconstruction"),
        help="Output folder for generated comparison plots",
    )
    parser.add_argument("--threshold", type=float, default=7.0, help="Gaussian threshold to use")
    parser.add_argument("--window-size", type=int, default=100, help="CPD window size used in Gaussian files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_chromosomes(
        input_dir=args.input_dir,
        gaussian_dir=args.gaussian_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        window_size=args.window_size,
    )
