import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Signal_processing.sliding_mean.sliding_ZINB_CPD import sliding_ZINB_CPD
from Signal_processing.sliding_mean.sliding_ZINB_CPD_ref import sliding_ZINB_CPD_ref
from Signal_processing.sliding_mean.sliding_ZINB_CPD_v2 import sliding_ZINB_CPD_v2
from Signal_processing.sliding_mean.sliding_ZINB_CPD_v3 import sliding_ZINB_CPD_v3
from Utils.plot_config import COLORS, setup_plot_style


setup_plot_style()


def precision_recall_one_to_one(detected_cps, true_cps, tol):
	"""Calculate precision and recall with one-to-one greedy matching."""
	detected_cps = np.array(detected_cps)
	true_cps = np.array(true_cps)

	if len(detected_cps) == 0:
		return 0.0, 0.0
	if len(true_cps) == 0:
		return 0.0, 0.0

	matched_true = set()
	matched_detected = set()

	pairs = []
	for i, det_cp in enumerate(detected_cps):
		for j, true_cp in enumerate(true_cps):
			dist = abs(det_cp - true_cp)
			if dist <= tol:
				pairs.append((i, j, dist))

	pairs.sort(key=lambda x: x[2])

	true_positives = 0
	for det_idx, true_idx, _ in pairs:
		if det_idx not in matched_detected and true_idx not in matched_true:
			matched_detected.add(det_idx)
			matched_true.add(true_idx)
			true_positives += 1

	precision = true_positives / len(detected_cps) if len(detected_cps) > 0 else 0.0
	recall = true_positives / len(true_cps) if len(true_cps) > 0 else 0.0
	return precision, recall


def read_data_and_distances(data_file):
	"""Read SATAY_with_pi.csv with count signal and distance columns."""
	df = pd.read_csv(data_file)
	required_cols = {
		"Value",
		"Centromere_distance",
		"Nucleosome_distance",
	}
	missing = required_cols.difference(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in {data_file}: {sorted(missing)}")

	data = df["Value"].values.astype(int)
	centromere_distances = df["Centromere_distance"].values.astype(int)
	nucleosome_distances = df["Nucleosome_distance"].values.astype(int)
	return data, centromere_distances, nucleosome_distances


def read_true_change_points(param_file):
	"""Read true CPs from SATAY_without_pi_params.csv."""
	df = pd.read_csv(param_file)
	if "region_start" not in df.columns:
		raise ValueError(f"Missing 'region_start' column in {param_file}")
	return df["region_start"].values[1:].astype(int).tolist()


def run_detector(
	method,
	data,
	window_size,
	overlap,
	threshold,
	theta,
	pi_file,
	nucleosome_distances,
	centromere_distances,
	nucleosome_file,
	centromere_file,
):
	"""Dispatch to the selected detector implementation.

	Mapping used here follows the requested names:
	- ref -> sliding_ZINB_CPD_ref
	- v0  -> sliding_ZINB_CPD (file without version suffix)
	- v1  -> sliding_ZINB_CPD_v2
	- v2  -> sliding_ZINB_CPD_v3
	"""
	if method == "ref":
		return sliding_ZINB_CPD_ref(
			data,
			window_size,
			overlap,
			threshold,
			theta_global=0.5,
			pi_file=pi_file,
		)

	if method == "v0":
		return sliding_ZINB_CPD(
			data,
			window_size,
			overlap,
			threshold,
			theta_global=theta,
		)

	if method == "v1":
		return sliding_ZINB_CPD_v2(
			data,
			nucleosome_distances,
			window_size,
			overlap,
			threshold,
			theta_global=theta,
			nucleosome_file=nucleosome_file,
		)

	if method == "v2":
		return sliding_ZINB_CPD_v3(
			data,
			nucleosome_distances,
			centromere_distances,
			window_size,
			overlap,
			threshold,
			theta_global=theta,
			nucleosome_file=nucleosome_file,
			centromere_file=centromere_file,
		)

	raise ValueError(f"Unknown method: {method}")


def evaluate_method_on_dataset(
	method,
	dataset_id,
	data,
	true_cps,
	window_size,
	overlap,
	thresholds,
	theta,
	pi_file,
	nucleosome_distances,
	centromere_distances,
	nucleosome_file,
	centromere_file,
):
	"""Run one method for all thresholds on one dataset."""
	rows = []

	for threshold in thresholds:
		change_points, _ = run_detector(
			method,
			data,
			window_size,
			overlap,
			threshold,
			theta,
			pi_file,
			nucleosome_distances,
			centromere_distances,
			nucleosome_file,
			centromere_file,
		)
		precision, recall = precision_recall_one_to_one(change_points, true_cps, window_size)

		rows.append(
			{
				"dataset_id": dataset_id,
				"method": method,
				"threshold": float(threshold),
				"precision": float(precision),
				"recall": float(recall),
				"num_detected": int(len(change_points)),
				"num_true": int(len(true_cps)),
			}
		)

	return rows


def calculate_auc_per_dataset(results_df):
	"""Compute PR AUC for each (dataset, method)."""
	auc_rows = []
	grouped = results_df.groupby(["dataset_id", "method"])  # each curve

	for (dataset_id, method), df_group in grouped:
		curve = df_group.sort_values("threshold")
		recalls = curve["recall"].values
		precisions = curve["precision"].values

		sort_idx = np.argsort(recalls)
		try:
			pr_auc = auc(recalls[sort_idx], precisions[sort_idx])
		except Exception:
			pr_auc = np.nan

		auc_rows.append(
			{
				"dataset_id": int(dataset_id),
				"method": method,
				"auc": float(pr_auc) if not np.isnan(pr_auc) else np.nan,
			}
		)

	return pd.DataFrame(auc_rows)


def aggregate_curves(results_df):
	"""Aggregate mean/std precision and recall per method and threshold."""
	grouped = results_df.groupby(["method", "threshold"])
	agg = grouped.agg(
		precision_mean=("precision", "mean"),
		precision_std=("precision", "std"),
		recall_mean=("recall", "mean"),
		recall_std=("recall", "std"),
		n_datasets=("dataset_id", "nunique"),
	).reset_index()

	agg["precision_std"] = agg["precision_std"].fillna(0.0)
	agg["recall_std"] = agg["recall_std"].fillna(0.0)
	return agg


def aggregate_auc(auc_df):
	"""Aggregate mean/std AUC per method over datasets."""
	grouped = auc_df.groupby("method")
	agg_auc = grouped.agg(
		auc_mean=("auc", "mean"),
		auc_std=("auc", "std"),
		n_datasets=("dataset_id", "count"),
	).reset_index()
	agg_auc["auc_std"] = agg_auc["auc_std"].fillna(0.0)
	return agg_auc


def plot_precision_recall_with_std(agg_curve_df, agg_auc_df, output_path):
	"""Plot all methods on one PR curve"""
	method_meta = {
		"ref": {"label": "ref", "color": COLORS["blue"]},
		"v0": {"label": "v0", "color": COLORS["orange"]},
		"v1": {"label": "v1", "color": COLORS["green"]},
		"v2": {"label": "v2", "color": COLORS["red"]},
	}

	fig, ax = plt.subplots(figsize=(12, 8))

	for method in ["ref", "v0", "v1", "v2"]:
		method_curve = agg_curve_df[agg_curve_df["method"] == method].sort_values("threshold")
		if method_curve.empty:
			continue

		auc_row = agg_auc_df[agg_auc_df["method"] == method]
		if not auc_row.empty:
			auc_mean = auc_row["auc_mean"].iloc[0]
			auc_std = auc_row["auc_std"].iloc[0]
			legend_label = f"{method_meta[method]['label']})"
		else:
			legend_label = method_meta[method]["label"]

		ax.plot(
			method_curve["recall_mean"].values,
			method_curve["precision_mean"].values,
			fmt="o-",
			linewidth=1.8,
			markersize=4,
			capsize=2,
			alpha=0.85,
			color=method_meta[method]["color"],
			label=legend_label,
		)

	ax.set_xlabel("Recall")
	ax.set_ylabel("Precision")
	ax.set_title("Precision-Recall Comparison Across Versions\n(mean ± std over 10 datasets)")
	ax.grid(True, alpha=0.3)
	ax.set_xlim(-0.05, 1.05)
	ax.set_ylim(-0.05, 1.05)
	ax.legend(loc="best")

	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()


def parse_args():
	parser = argparse.ArgumentParser(description="Compare ZINB CPD versions on synthetic SATAY datasets.")
	parser.add_argument(
		"--base_folder",
		type=str,
		default="Signal_processing/final/SATAY_synthetic",
		help="Folder with dataset subfolders 1..10",
	)
	parser.add_argument(
		"--output_folder",
		type=str,
		default="Signal_processing/final/results/compare_versions_theta0_ws100",
		help="Where to save metrics and plots",
	)
	parser.add_argument("--window_size", type=int, default=100)
	parser.add_argument("--overlap", type=float, default=0.5)
	parser.add_argument("--theta", type=float, default=0.0)
	parser.add_argument("--threshold_min", type=float, default=0.0)
	parser.add_argument("--threshold_max", type=float, default=40.0)
	parser.add_argument("--threshold_step", type=float, default=1.0)
	parser.add_argument("--dataset_start", type=int, default=1)
	parser.add_argument("--dataset_end", type=int, default=10)
	return parser.parse_args()


def main():
	args = parse_args()

	os.makedirs(args.output_folder, exist_ok=True)

	thresholds = np.arange(
		args.threshold_min,
		args.threshold_max + 0.5 * args.threshold_step,
		args.threshold_step,
	)
	dataset_ids = list(range(args.dataset_start, args.dataset_end + 1))
	methods = ["ref", "v0", "v1", "v2"]

	print("Configuration:")
	print(f"  Base folder: {args.base_folder}")
	print(f"  Output folder: {args.output_folder}")
	print(f"  Datasets: {dataset_ids}")
	print(f"  Methods: {methods}")
	print(f"  Window size: {args.window_size}")
	print(f"  Overlap: {args.overlap}")
	print(f"  Theta: {args.theta}")
	print(f"  Threshold count: {len(thresholds)}")

	all_rows = []

	for dataset_id in dataset_ids:
		dataset_folder = os.path.join(args.base_folder, str(dataset_id))
		data_file = os.path.join(dataset_folder, "SATAY_with_pi.csv")
		param_file = os.path.join(dataset_folder, "SATAY_without_pi_params.csv")
		pi_file = os.path.join(dataset_folder, "pi_values.csv")
		nucleosome_file = os.path.join(dataset_folder, "density_vs_distance_nucleosome_density.csv")
		centromere_file = os.path.join(dataset_folder, "density_vs_distance_centromere_density.csv")

		required_files = [
			data_file,
			param_file,
			pi_file,
			nucleosome_file,
			centromere_file,
		]
		missing_files = [p for p in required_files if not os.path.exists(p)]
		if missing_files:
			raise FileNotFoundError(
				f"Dataset {dataset_id} is missing required files: {missing_files}"
			)

		print(f"\nProcessing dataset {dataset_id}")
		data, centromere_distances, nucleosome_distances = read_data_and_distances(data_file)
		true_cps = read_true_change_points(param_file)
		print(f"  Points: {len(data)} | True CPs: {len(true_cps)}")

		for method in methods:
			print(f"  Method {method} ...")
			rows = evaluate_method_on_dataset(
				method,
				dataset_id,
				data,
				true_cps,
				args.window_size,
				args.overlap,
				thresholds,
				args.theta,
				pi_file,
				nucleosome_distances,
				centromere_distances,
				nucleosome_file,
				centromere_file,
			)
			all_rows.extend(rows)
		# break # Remove this break to process all datasets

	results_df = pd.DataFrame(all_rows)
	results_csv = os.path.join(args.output_folder, "all_results.csv")
	results_df.to_csv(results_csv, index=False)
	print(f"\nSaved detailed results to {results_csv}")

	auc_df = calculate_auc_per_dataset(results_df)
	auc_csv = os.path.join(args.output_folder, "auc_per_dataset.csv")
	auc_df.to_csv(auc_csv, index=False)
	print(f"Saved per-dataset AUC to {auc_csv}")

	agg_curve_df = aggregate_curves(results_df)
	agg_curve_csv = os.path.join(args.output_folder, "precision_recall_aggregated.csv")
	agg_curve_df.to_csv(agg_curve_csv, index=False)
	print(f"Saved aggregated PR stats to {agg_curve_csv}")

	agg_auc_df = aggregate_auc(auc_df)
	agg_auc_csv = os.path.join(args.output_folder, "auc_aggregated.csv")
	agg_auc_df.to_csv(agg_auc_csv, index=False)
	print(f"Saved aggregated AUC stats to {agg_auc_csv}")

	plot_path = os.path.join(args.output_folder, "precision_recall_compare_versions.png")
	plot_precision_recall_with_std(agg_curve_df, agg_auc_df, plot_path)
	print(f"Saved PR comparison plot to {plot_path}")

	print("\nAUC summary (mean ± std):")
	for _, row in agg_auc_df.sort_values("method").iterrows():
		print(
			f"  {row['method']}: "
			f"{row['auc_mean']:.4f} ± {row['auc_std']:.4f} "
			f"(n={int(row['n_datasets'])})"
		)


if __name__ == "__main__":
	main()
