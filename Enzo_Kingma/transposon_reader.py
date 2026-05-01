import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def read_strain_data(strain_path):
    """
    Read transposon data from a strain directory.
    
    Parameters
    ----------
    strain_path : str or Path
        Path to the strain directory containing chromosome CSV files
        
    Returns
    -------
    dict
        Dictionary with chromosome names as keys (e.g., 'ChrI', 'ChrII') and DataFrames as values.
        Each DataFrame contains columns: Position, Value (transposon count), Centromere_Distance
    """
    strain_path = Path(strain_path)
    
    # Get all chromosome distance files
    chr_files = sorted(strain_path.glob("Chr*_distances.csv"))
    
    if not chr_files:
        raise FileNotFoundError(f"No chromosome files found in {strain_path}")
    
    # Read each chromosome file into a dictionary
    chr_data = {}
    for chr_file in chr_files:
        # Extract chromosome name (e.g., 'ChrI' from 'ChrI_distances.csv')
        chr_name = chr_file.stem.replace('_distances', '')
        
        df = pd.read_csv(chr_file)
        # Select only the columns we need
        df = df[['Position', 'Value', 'Centromere_Distance']]
        chr_data[chr_name] = df
    
    return chr_data


def fit_centromere_bias_from_rates(
    chr_data,
    distance_col="Centromere_Distance",
    value_col="Value",
    max_fit_distance=400000,
    flatten_distance=200000,
    degree=3,
    bin_size=5000,
):
    """
    Estimate insertion rate as occupied insertion sites per available bp
    at each absolute distance from the centromere.

    This version aggregates across all chromosome arms directly.
    """

    rows = []

    bin_edges = np.arange(0, max_fit_distance + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2

    for chr_name, chr_df in chr_data.items():
        df = chr_df.copy()
        df = df[np.isfinite(df[distance_col])].copy()

        df["abs_distance"] = df[distance_col].abs()
        df["occupied"] = (df[value_col] > 0).astype(int)

        df = df[df["abs_distance"] <= max_fit_distance].copy()

        df["distance_bin"] = pd.cut(
            df["abs_distance"],
            bins=bin_edges,
            labels=bin_centers,
            include_lowest=True,
        )

        rows.append(df[["distance_bin", "occupied"]])

    all_df = pd.concat(rows, ignore_index=True)

    # For each distance bin:
    # occupied = number of insertion sites
    # available_bp = number of genomic positions represented in that bin
    binned = (
        all_df
        .groupby("distance_bin", observed=True)
        .agg(
            occupied_insertions=("occupied", "sum"),
            available_bp=("occupied", "size"),
        )
        .reset_index()
    )

    binned["rc"] = binned["distance_bin"].astype(float)
    binned["insertion_rate"] = (
        binned["occupied_insertions"] / binned["available_bp"]
    )

    binned = binned.sort_values("rc")

    # Fit local insertion rate directly
    fit_part = binned[binned["rc"] <= max_fit_distance].copy()

    x = fit_part["rc"].to_numpy()
    y = fit_part["insertion_rate"].to_numpy()

    coeffs_rate = np.polyfit(x, y, deg=degree)
    rate_poly = np.poly1d(coeffs_rate)

    lambda_flat = rate_poly(flatten_distance)

    def insertion_rate(rc):
        rc = np.abs(np.asarray(rc))
        rate = rate_poly(np.minimum(rc, flatten_distance))
        rate = np.where(rc >= flatten_distance, lambda_flat, rate)
        return np.maximum(rate, 0)

    return coeffs_rate, rate_poly, insertion_rate, binned


if __name__ == "__main__":
    strain_path = "Data/combined_strains/strain_yEK19"
    chr_data = read_strain_data(strain_path)

    print(f"Loaded {len(chr_data)} chromosomes:")
    for chr_name, df in chr_data.items():
        print(f"  {chr_name}: {len(df)} positions")

    coeffs_rate, rate_poly, insertion_rate, rate_df = fit_centromere_bias_from_rates(
        chr_data,
        max_fit_distance=400000,
        flatten_distance=200000,
        degree=3,
        bin_size=1000,
    )

    print("\nRate polynomial coefficients:")
    print(coeffs_rate)

    print("\nInsertion rate at 0 bp:")
    print(insertion_rate(0))

    print("\nInsertion rate at 200 kb:")
    print(insertion_rate(200_000))

    print("\nInsertion rate at -200 kb:")
    print(insertion_rate(-200_000))

    print("\nInsertion rate at 400 kb, flattened:")
    print(insertion_rate(400_000))

    print("\nFirst rows of binned insertion-rate table:")
    print(rate_df.head())

    rc_plot = np.linspace(0, 400_000, 500)

    plt.figure()
    plt.scatter(
        rate_df["rc"],
        rate_df["insertion_rate"],
        s=10,
        label="Empirical binned insertion rate",
    )
    plt.plot(
        rc_plot,
        insertion_rate(rc_plot),
        label="Polynomial fit",
    )
    plt.axvline(
        200_000,
        linestyle="--",
        label="Flattening distance",
    )
    plt.xlabel("Absolute distance to centromere (bp)")
    plt.ylabel("Occupied insertion sites per bp")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: save the binned rates and fitted coefficients
    output_dir = Path("Data/centromere_bias_fits")
    output_dir.mkdir(parents=True, exist_ok=True)

    rate_df.to_csv(output_dir / "strain_yEK19_binned_insertion_rates.csv", index=False)

    coeff_df = pd.DataFrame({
        "coefficient": coeffs_rate,
        "power": list(range(len(coeffs_rate) - 1, -1, -1)),
    })
    coeff_df.to_csv(output_dir / "strain_yEK19_rate_polynomial_coefficients.csv", index=False)