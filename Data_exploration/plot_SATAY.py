import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# Add Signal_processing directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "Signal_processing"))
from ZINB_MLE.estimate_ZINB import estimate_zinb
from retrieve_pred_from_cpd import retrieve_pred_from_cpd

CHROMS = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI"]
cpd = {
    "I": [-910, -780, -500, -350, -200, -70, 80, 760],
    "II": [-740, -580, -70, 65, 800],
    "III": [-900, -200, -80, 80, 750],
    "IV": [-900, -800, -740, -500, -380, -60, 60 ],
    "V": [-520, -280, -70, 90, 200, 430],
    "VI": [-960, -860, 460],    
    "VII": [-440, -110, 70, 260, 320, 730],
    "VIII": [-860, -510, -80, 80, 200, 370, 580, ],
    "IX": [-400, -60, 80, 200, 300, 760],
    "X": [-920, -480, -80, 60, 200 ],
    "XI": [-660, -380 ,-70, 60, 180, 280, 470],
    "XII": [-180, -75, 70, 460 ],
    "XIII": [-770, -420, -300, -140, -65, 80, 780],
    "XIV": [-920, -140, -65, 80, ],
    "XV": [-740, -120, 60, 380, 930],
    "XVI": [-580, -170, -70, 80,],
}

# pred = {
#     "I": [640, 920, 1080, 1720], 
#     "II": [240, 400, 600, 680, 920, 1080, 1200, 1280, 1400, 1480, 1560, 1640, 1800, 1920 ],
#     "III": [960, 1080],
#     "IV": [920, 1080],
#     "V": [160, 280, 480, 680, 840, 960, 1080, 1200, 1400 ],
#     "VI": [880, 1480],    
#     "VII": [560, 1040, 1240],
#     "VIII": [920, 1080, 1680, 1760],
#     "IX": [960, 1080, 1200],
#     "X": [520, 920, 1040, 1200],
#     "XI": [840, 920, 1040, 1160, 1480],
#     "XII": [800, 920, 1040, 1440],
#     "XIII": [680, 840, 960, 1080, 1160, 1240],
#     "XIV": [840, 960, 1080],
#     "XV": [720, 880, 1080, 1160],
#     "XVI": [960, 1080],
# }

# pred = {
#     "I": [240, 640, 800, 920, 1080, 1440, 1520, 1720, 1800],
#     "II": [960, 180 ],
#     "III": [960, 180],
#     "IV": [920, 1080],
#     "V": [480, 680, 760, 840, 1200 ],
#     "VI": [],
#     "VII": [560],
#     "VIII": [920, 1080, 1360, 1680, 1760],
#     "IX": [960, 1080, 1200],
#     "X": [920],
#     "XI": [840, 920, 1040, 1160, 1480],
#     "XII": [800],
#     "XIII": [840, 960, 1080, 1160, 1240],
#     "XIV": [840, 960, 1080],
#     "XV": [720, 1040, 1160],
#     "XVI": [960, 1080],
# }

    
    

window = 80

def plot_around_centromere_x_is_centdist(
    chrom,
    filepath,
    outpath,
    n=1000,
    centromere_target=0,
    strict_centromere_match=True,
    major_tick_step=100,
    minor_tick_step=50,
    dpi=200,
    value_max=500,
    window_size=100,
    show_zinb_params=True,
    show_cpd=True,
):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Find centromere midpoint index
    if strict_centromere_match:
        idxs = df.index[df["centromere_distance"] == centromere_target].to_list()
        if not idxs:
            raise ValueError(
                f"{filepath}: No rows found with centromere_distance == {centromere_target}. "
                f"Set strict_centromere_match=False to use closest row instead."
            )
        cent_idx = idxs[0]
    else:
        cent_idx = (df["centromere_distance"] - centromere_target).abs().idxmin()

    # Window around centromere (±n rows)
    start = max(0, cent_idx - n)
    end = min(len(df) - 1, cent_idx + n)
    win = df.iloc[start:end + 1].copy()

    # Filter out extreme values (breaks the line at those points)
    if "value" not in win.columns:
        raise KeyError(f"{filepath}: expected a 'value' column after normalization, got {list(win.columns)}")
    win.loc[win["value"] > value_max, "value"] = float("nan")

    # Nucleosome locations in centromere_distance coords
    nuc0_x = win.loc[win["nucleosome_distance"] == 0, "centromere_distance"].to_numpy()

    # Sort by x for a clean line plot
    win = win.sort_values("centromere_distance")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(win["centromere_distance"], win["value"], linewidth=1)

    for x in nuc0_x:
        ax.axvline(x, alpha=0.35, linewidth=1)

    ax.axvline(0, linestyle="--", linewidth=2)

    # Ticks
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_step))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_step))
    ax.minorticks_on()

    # Grid (major + faint minor)
    ax.grid(True, which="major", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.15)

    ax.set_title(os.path.basename(filepath).replace("_distances.csv", ""))
    ax.set_xlabel("Centromere distance")
    ax.set_ylabel("Value")

    # ---- CPD red dots + PRED dots + transparent window bands ----
    if show_cpd:
        x_min, x_max = win["centromere_distance"].min(), win["centromere_distance"].max()

        cpd_x_all = cpd.get(chrom, [])
        cpd_x = [x for x in cpd_x_all if x_min <= x <= x_max]

        pred_x_all = pred.get(chrom, [])
        pred_x_shifted = [x - 1000 for x in pred_x_all]  # shift into centromere_distance coords
        pred_x = [x for x in pred_x_shifted if x_min <= x <= x_max]
        print("pred_x_shifted:", pred_x_shifted)
        print("pred_x_in_range:", pred_x)
       
        # place dots slightly below the x-axis (y=0) based on current y-range
        y_min, y_max = ax.get_ylim()
        offset = 0.03 * (y_max - y_min)
        y_cpd = -offset

        # Plot CPD dots (red)
        if len(cpd_x) > 0:
            ax.scatter(
                cpd_x,
                [y_cpd] * len(cpd_x),
                s=28,
                marker="o",
                color="red",
                zorder=5,
                label="CPD"
            )

        # Transparent windows centered on each pred (width = global `window`)
        for x in pred_x:
            ax.axvspan(
                x - window / 2,
                x + window / 2,
                alpha=0.12,
                zorder=1
            )

        # Plot Pred dots (smaller, same y as CPD)
        if len(pred_x) > 0:
            ax.scatter(
                pred_x,
                [y_cpd] * len(pred_x),
                s=14,
                marker="o",
                color="tab:blue",
                zorder=6,
                label="Pred"
            )

        # Only show legend if something was actually plotted
        if (len(cpd_x) > 0) or (len(pred_x) > 0):
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # ---- ZINB parameter estimation annotations ----
    if show_zinb_params:
        win_sorted = win.sort_values("centromere_distance").reset_index(drop=True)
        values_array = win_sorted["value"].values
        positions_array = win_sorted["centromere_distance"].values

        valid_mask = ~np.isnan(values_array)
        n_points = np.sum(valid_mask)
        n_windows = int(np.ceil(n_points / window_size))

        window_annotations = []
        valid_values = values_array[valid_mask]
        valid_positions = positions_array[valid_mask]

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, n_points)
            window_data = valid_values[start_idx:end_idx]
            window_positions = valid_positions[start_idx:end_idx]

            if len(window_data) < 10:
                continue

            window_center = np.mean(window_positions)

            rounded_data = np.round(window_data).astype(int)
            threshold = np.percentile(rounded_data, 95)
            filtered_data = rounded_data[rounded_data <= threshold]

            if len(filtered_data) < 10:
                continue

            try:
                estimates = estimate_zinb(filtered_data, max_iter=1000)
                if estimates.get("converged", False):
                    pi_val = estimates["pi"]
                    mu_val = estimates["mu"]
                    theta_val = estimates["theta"]
                    annot_text = f"π={pi_val:.2f}\nμ={mu_val:.1f}\nθ={theta_val:.1f}"

                    window_annotations.append({
                        "position": window_center,
                        "text": annot_text,
                        "end_pos": window_positions[-1],
                    })
            except Exception as e:
                print(f"Warning: Failed to estimate ZINB for {chrom} window {i+1}: {e}")
                continue

        y_max = ax.get_ylim()[1]
        annotation_heights = [0.85 * y_max, 0.70 * y_max, 0.55 * y_max]

        for idx, annot in enumerate(window_annotations):
            y_pos = annotation_heights[idx % len(annotation_heights)]
            ax.text(
                annot["position"],
                y_pos,
                annot["text"],
                fontsize=6,
                ha="center",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="gray",
                    linewidth=0.5
                )
            )
            if idx < len(window_annotations) - 1:
                ax.axvline(annot["end_pos"], color="gray", alpha=0.2, linewidth=0.5, linestyle=":")

    ax.tick_params(axis="x", labelrotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
def save_data(chrom, data, out_dir, flank_bp=1000,
              centdist_col="centromere_distance", position_col="Position"):
    os.makedirs(out_dir, exist_ok=True)

    df = data.copy()

    # normalize internal column names for access
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    centdist_col_l = centdist_col.lower()
    position_col_l = position_col.lower()  # "position"

    if centdist_col_l not in df.columns:
        raise KeyError(
            f"Expected '{centdist_col}' in input data. Got columns: {list(df.columns)}"
        )

    # keep only ±flank_bp around centromere
    df_out = df.loc[df[centdist_col_l].between(-flank_bp, flank_bp)].copy()

    # Put centromere_distance into the Position column (overwrite or create)
    df_out[position_col_l] = df_out[centdist_col_l].astype(float)

    # Also keep centromere_distance as a separate column with nice casing
    df_out["centromere_distance_copy"] = df_out[centdist_col_l].astype(float)

    # Now rename columns to requested output casing
    df_out.rename(columns={
        position_col_l: "Position",
        "centromere_distance_copy": "Centromere_Distance",
    }, inplace=True)

    # (Optional) keep the original centromere_distance too, but usually you don't need 3 copies.
    # If you DO want to keep it as well, comment out the next line.
    df_out.drop(columns=[centdist_col_l], inplace=True)

    # sort by Position for readability
    df_out = df_out.sort_values("Position")

    out_path = os.path.join(out_dir, f"Chr{chrom}_centromere_window.csv")
    df_out.to_csv(out_path, index=False)


if __name__ == "__main__":
    threshold = 7
    window_size = 80
    base_dir = Path("Signal_processing/results/sliding_mean_SATAY/sliding_ZINB_CPD_v2")
    pred = retrieve_pred_from_cpd(window_size, threshold, base_dir)
    in_dir = "Data/combined_strains/strain_yEK23"
    # out_dir = "Data_exploration/plot_SATAY"
    out_dir_figures = "Data_exploration/plot_SATAY/version2"
    out_dir_samples = "Signal_processing/sample_data/Centromere_region"

    for chrom in CHROMS:
        infile = os.path.join(in_dir, f"Chr{chrom}_distances.csv")
        outfile = os.path.join(out_dir_figures, f"Chr{chrom}_centromere_window.png")

        if not os.path.exists(infile):
            print(f"Skipping (file not found): {infile}")
            continue
        
        save_data(chrom, pd.read_csv(infile), out_dir_samples)

        plot_around_centromere_x_is_centdist(
            chrom=chrom,                # <-- pass chrom
            filepath=infile,
            outpath=outfile,
            n=1000,
            major_tick_step=100,
            minor_tick_step=50,
            strict_centromere_match=True,
            value_max=500,
            window_size=100,
            show_zinb_params=False,
            show_cpd=True,
        )

        print(f"Saved: {outfile}")