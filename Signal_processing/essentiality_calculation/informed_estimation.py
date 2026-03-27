import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)


RESULT_FILENAME_RE = re.compile(
	r"^(?P<dataset_name>.+)_ws(?P<window_size>\d+)_ov(?P<overlap_pct>\d+)_th(?P<threshold>\d+(?:\.\d+)?)\.txt$"
)


def resolve_path(path):
	if os.path.isabs(path):
		return path
	return os.path.join(PROJECT_ROOT, path)


def parse_result_file(result_file):
	"""Parse change points and theta from a version3 result file."""
	change_points = []
	theta_global = None
	in_cp_section = True

	with open(result_file, "r") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line:
				continue

			if line.startswith("scores:"):
				in_cp_section = False
				continue

			if line.startswith("theta_global:"):
				theta_global = float(line.split(":", 1)[1].strip())
				continue

			if line.startswith("window_size:"):
				continue

			if in_cp_section:
				try:
					cp = int(float(line))
					change_points.append(cp)
				except ValueError:
					continue

	if theta_global is None:
		raise ValueError(f"Could not find theta_global in {result_file}")

	return change_points, theta_global


def parse_result_filename(filename):
	match = RESULT_FILENAME_RE.match(filename)
	if match is None:
		return {
			"dataset_name": None,
			"window_size": None,
			"overlap_pct": None,
			"threshold": None,
		}

	return {
		"dataset_name": match.group("dataset_name"),
		"window_size": int(match.group("window_size")),
		"overlap_pct": int(match.group("overlap_pct")),
		"threshold": float(match.group("threshold")),
	}


def _pick_column_by_keyword(df, keyword, fallback_index):
	keyword = keyword.lower()
	for col in df.columns:
		if keyword in col.lower():
			return col

	if 0 <= fallback_index < len(df.columns):
		return df.columns[fallback_index]

	raise ValueError(
		f"Could not find a '{keyword}' column and fallback index {fallback_index} is out of range."
	)


def read_dataset_with_distances(csv_file):
	df = pd.read_csv(csv_file)
	if df.empty:
		raise ValueError(f"Input file is empty: {csv_file}")

	value_col = _pick_column_by_keyword(df, "value", 1)
	centromere_col = _pick_column_by_keyword(df, "centromere", 2)
	nucleosome_col = _pick_column_by_keyword(df, "nucleosome", 3)

	data = df[value_col].astype(float).to_numpy()
	centromere_distances = df[centromere_col].astype(int).to_numpy()
	nucleosome_distances = df[nucleosome_col].astype(int).to_numpy()
	return data, centromere_distances, nucleosome_distances


def load_density_lookup_tables(nucleosome_file, centromere_file):
	nucleosome_df = pd.read_csv(nucleosome_file)
	centromere_df = pd.read_csv(centromere_file)
	return nucleosome_df, centromere_df


def interpolate_density(distance, lookup_df, distance_col, density_col="NonZero_Density"):
	distances = lookup_df[distance_col].values
	densities = lookup_df[density_col].values
	return float(np.interp(distance, distances, densities))


def build_density_lookups(nucleosome_df, centromere_df, nucleosome_distances):
	max_nucl_distance = int(np.max(nucleosome_distances))

	distance_to_density = nucleosome_df.set_index("Nucleosome_Distance_Bin")["NonZero_Density"]
	distance_to_density = distance_to_density.reindex(range(max_nucl_distance + 1), fill_value=0)

	centromere_distance_to_density = (
		centromere_df.set_index("Centromere_Distance_Bin")["NonZero_Density"].sort_index()
	)

	return distance_to_density, centromere_distance_to_density


def estimate_segments_informed(
	data,
	centromere_distances,
	nucleosome_distances,
	change_points,
	distance_to_density,
	centromere_distance_to_density,
	eps=1e-10,
):
	"""Estimate segment mu using the same pi-informed formula as in v3."""
	n = len(data)
	valid_cps = sorted({int(cp) for cp in change_points if 0 < int(cp) < n})
	boundaries = [0] + valid_cps + [n]

	cmin = int(centromere_distance_to_density.index.min())
	cmax = int(centromere_distance_to_density.index.max())
	centromere_lookup_df = centromere_distance_to_density.reset_index()

	global_nucl_density = float(distance_to_density.loc[nucleosome_distances].mean())
	global_nucl_density = max(global_nucl_density, eps)

	segment_rows = []
	for idx in range(len(boundaries) - 1):
		start = boundaries[idx]
		end = boundaries[idx + 1]
		segment = data[start:end]
		seg_nucl_dist = nucleosome_distances[start:end]

		if len(segment) == 0:
			segment_rows.append(
				{
					"segment_id": idx + 1,
					"start_index": start,
					"end_index_exclusive": end,
					"length": 0,
					"raw_mean": np.nan,
					"centromere_mid_distance": np.nan,
					"centromere_density": np.nan,
					"segment_nucleosome_density": np.nan,
					"global_nucleosome_density": global_nucl_density,
					"pi_informed": np.nan,
					"mu_informed": np.nan,
				}
			)
			continue

		middle = start + (end - start) // 2
		middle = min(max(middle, 0), n - 1)

		centr_dist_middle = int(centromere_distances[middle])
		centr_dist_middle = min(max(centr_dist_middle, cmin), cmax)

		centromere_density = interpolate_density(
			centr_dist_middle,
			centromere_lookup_df,
			"Centromere_Distance_Bin",
			"NonZero_Density",
		)

		seg_nucl_density = float(distance_to_density.loc[seg_nucl_dist].mean())
		pi_informed = float(
			np.clip(centromere_density * (seg_nucl_density / global_nucl_density), eps, 1.0 - eps)
		)

		pi_informed = 1 - pi_informed

		raw_mean = float(np.mean(segment))
		mu_informed = float(np.clip(raw_mean / max(1.0 - pi_informed, eps), eps, None))

		segment_rows.append(
			{
				"segment_id": idx + 1,
				"start_index": start,
				"end_index_exclusive": end,
				"length": end - start,
				"raw_mean": raw_mean,
				"centromere_mid_distance": centr_dist_middle,
				"centromere_density": centromere_density,
				"segment_nucleosome_density": seg_nucl_density,
				"global_nucleosome_density": global_nucl_density,
				"pi_informed": pi_informed,
				"mu_informed": mu_informed,
			}
		)

	return segment_rows


def process_dataset(dataset_num, base_data_folder, base_results_folder, output_subdir, eps):
	dataset_folder = os.path.join(base_data_folder, str(dataset_num))
	dataset_data_file = os.path.join(dataset_folder, "SATAY_with_pi.csv")
	nucleosome_lookup_file = os.path.join(dataset_folder, "density_vs_distance_nucleosome_density.csv")
	centromere_lookup_file = os.path.join(dataset_folder, "density_vs_distance_centromere_density.csv")
	dataset_results_folder = os.path.join(base_results_folder, str(dataset_num))

	missing = [
		path
		for path in [
			dataset_data_file,
			nucleosome_lookup_file,
			centromere_lookup_file,
			dataset_results_folder,
		]
		if not os.path.exists(path)
	]
	if missing:
		print(f"Skipping dataset {dataset_num}: missing required path(s):")
		for path in missing:
			print(f"  - {path}")
		return 0

	data, centromere_distances, nucleosome_distances = read_dataset_with_distances(dataset_data_file)
	nucleosome_df, centromere_df = load_density_lookup_tables(
		nucleosome_lookup_file, centromere_lookup_file
	)
	distance_to_density, centromere_distance_to_density = build_density_lookups(
		nucleosome_df, centromere_df, nucleosome_distances
	)

	processed_files = 0

	window_folders = [
		os.path.join(dataset_results_folder, name)
		for name in sorted(os.listdir(dataset_results_folder))
		if os.path.isdir(os.path.join(dataset_results_folder, name)) and name.startswith("window")
	]

	for window_folder in window_folders:
		result_files = [
			name
			for name in sorted(os.listdir(window_folder))
			if name.endswith(".txt")
		]
		if not result_files:
			continue

		output_folder = os.path.join(window_folder, output_subdir)
		os.makedirs(output_folder, exist_ok=True)

		for result_name in result_files:
			result_path = os.path.join(window_folder, result_name)
			change_points, theta_global = parse_result_file(result_path)
			file_meta = parse_result_filename(result_name)

			segment_rows = estimate_segments_informed(
				data=data,
				centromere_distances=centromere_distances,
				nucleosome_distances=nucleosome_distances,
				change_points=change_points,
				distance_to_density=distance_to_density,
				centromere_distance_to_density=centromere_distance_to_density,
				eps=eps,
			)

			segment_df = pd.DataFrame(segment_rows)
			segment_df.insert(0, "source_result_file", result_name)
			segment_df.insert(1, "dataset_num", int(dataset_num))
			segment_df.insert(2, "theta_global", float(theta_global))
			segment_df.insert(3, "num_change_points", int(len(change_points)))
			segment_df.insert(4, "dataset_name", file_meta["dataset_name"])
			segment_df.insert(5, "window_size", file_meta["window_size"])
			segment_df.insert(6, "overlap_pct", file_meta["overlap_pct"])
			segment_df.insert(7, "threshold", file_meta["threshold"])

			output_name = result_name.replace(".txt", "_segment_mu_informed.csv")
			output_path = os.path.join(output_folder, output_name)
			segment_df.to_csv(output_path, index=False)
			processed_files += 1

	print(
		f"Dataset {dataset_num}: wrote informed segment estimates for {processed_files} threshold files"
	)
	return processed_files


def parse_arguments():
	parser = argparse.ArgumentParser(
		description=(
			"Estimate segment-level mu from version3 change point files using "
			"centromere/nucleosome-informed pi (v3 style)."
		)
	)
	parser.add_argument(
		"--base_data_folder",
		type=str,
		default="Signal_processing/tests/SATAY_synthetic",
		help="Folder containing SATAY synthetic datasets (1..10).",
	)
	parser.add_argument(
		"--base_results_folder",
		type=str,
		default="Signal_processing/results/version3",
		help="Folder containing version3 CPD outputs.",
	)
	parser.add_argument(
		"--datasets",
		type=int,
		nargs="+",
		default=list(range(1, 11)),
		help="Dataset numbers to process.",
	)
	parser.add_argument(
		"--output_subdir",
		type=str,
		default="segment_mu_informed",
		help="Subfolder created inside each window folder for informed segment outputs.",
	)
	parser.add_argument("--eps", type=float, default=1e-10, help="Numerical epsilon.")
	return parser.parse_args()


def main():
	args = parse_arguments()
	base_data_folder = resolve_path(args.base_data_folder)
	base_results_folder = resolve_path(args.base_results_folder)

	total_files = 0
	for dataset_num in args.datasets:
		total_files += process_dataset(
			dataset_num=dataset_num,
			base_data_folder=base_data_folder,
			base_results_folder=base_results_folder,
			output_subdir=args.output_subdir,
			eps=args.eps,
		)

	print(f"Done. Generated informed segment-mu outputs for {total_files} result files.")


if __name__ == "__main__":
	main()
