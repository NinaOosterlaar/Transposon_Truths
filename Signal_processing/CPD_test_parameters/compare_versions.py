from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Signal_processing.CPD_algorithms.sliding_ZINB.sliding_ZINB_CPD import sliding_ZINB_CPD
from Signal_processing.CPD_algorithms.sliding_ZINB.sliding_ZINB_CPD_ref import sliding_ZINB_CPD_ref
from Signal_processing.CPD_algorithms.sliding_ZINB.sliding_ZINB_CPD_v2 import sliding_ZINB_CPD_v2		
from Signal_processing.CPD_algorithms.sliding_ZINB.sliding_ZINB_CPD_v3 import sliding_ZINB_CPD_v3
from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Utils.plot_config import COLORS, setup_plot_style


setup_plot_style()


# Edit this block to run directly without command-line argparse flags.
RUN_CONFIG = {
	"base_folder": "Data/SATAY_synthetic",
	"output_folder": "Signal_processing/results_new/compare_versions_ws100",
	"window_size": 100,
	"overlap": 0.5,
	"theta": 0,
	"threshold_min": 0.0,
	"threshold_max": 40.0,
	"threshold_step": 1.0,
	"dataset_start": 1,
	"dataset_end": 10,
	"n_workers": 8,
}


def get_tp_fp_fn_one_to_one(detected_cps, true_cps, tol):
	"""Calculate TP, FP, FN with one-to-one greedy matching.
	
	Returns:
		Tuple of (TP, FP, FN) counts
	"""
	detected_cps = np.array(detected_cps)
	true_cps = np.array(true_cps)

	if len(detected_cps) == 0 and len(true_cps) == 0:
		return 0, 0, 0
	if len(detected_cps) == 0:
		return 0, 0, len(true_cps)  # All true CPs are FN
	if len(true_cps) == 0:
		return 0, len(detected_cps), 0  # All detected CPs are FP

	matched_true = set()
	matched_detected = set()

	pairs = []
	for i, det_cp in enumerate(detected_cps):
		for j, true_cp in enumerate(true_cps):
			dist = abs(det_cp - true_cp)
			if dist <= tol:
				pairs.append((i, j, dist))

	pairs.sort(key=lambda x: x[2])

	for det_idx, true_idx, _ in pairs:
		if det_idx not in matched_detected and true_idx not in matched_true:
			matched_detected.add(det_idx)
			matched_true.add(true_idx)

	tp = len(matched_detected)  # Same as len(matched_true)
	fp = len(detected_cps) - tp
	fn = len(true_cps) - tp
	
	return tp, fp, fn


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
	- v0  -> sliding_ZINB_CPD 
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
			nucleosome_distance_col="Nucleosome_Distance_Bin",
			centromere_distance_col="Centromere_Distance_Bin",
			density_col="NonZero_Density",
		)

	raise ValueError(f"Unknown method: {method}")


def apply_threshold_to_scores(scores, threshold, window_size, overlap, step_size):
	"""Apply threshold filtering to LRT scores to extract change points.
	
	This replicates the threshold logic from the detector functions.
	"""
	change_points = []
	last_cp, last_score = -np.inf, 0.0
	
	for idx, score in enumerate(scores):
		if score > threshold:
			# Position calculation: idx corresponds to window index
			# Change point is at start + window_size
			cp = idx * step_size + window_size
			
			if (cp - last_cp) >= window_size:
				change_points.append(cp)
				last_cp, last_score = cp, score
			elif score > last_score:
				change_points[-1] = cp
				last_cp, last_score = cp, score
	
	return change_points


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
	"""Run one method once to get LRT scores, then apply all thresholds."""
	# Run detector once with threshold=0 to get all LRT scores
	_, scores = run_detector(
		method,
		data,
		window_size,
		overlap,
		threshold=0.0,  # Use 0 to capture all scores
		theta=theta,
		pi_file=pi_file,
		nucleosome_distances=nucleosome_distances,
		centromere_distances=centromere_distances,
		nucleosome_file=nucleosome_file,
		centromere_file=centromere_file,
	)
	
	# Calculate step size for position mapping
	step_size = max(1, int(window_size * (1 - overlap)))
	
	# Now apply each threshold to the saved scores
	rows = []
	for threshold in thresholds:
		change_points = apply_threshold_to_scores(scores, threshold, window_size, overlap, step_size)
		tp, fp, fn = get_tp_fp_fn_one_to_one(change_points, true_cps, window_size)

		rows.append(
			{
				"dataset_id": dataset_id,
				"method": method,
				"theta": float(theta),
				"threshold": float(threshold),
				"TP": int(tp),
				"FP": int(fp),
				"FN": int(fn),
				"num_detected": int(len(change_points)),
				"num_true": int(len(true_cps)),
			}
		)

	return rows


def aggregate_curves(results_df):
	"""Aggregate TP/FP/FN across datasets, then calculate precision and recall.
	
	This aggregates counts before calculating metrics, avoiding the issue where
	a dataset with 0 detections contributes precision=0 to the average.
	"""
	grouped = results_df.groupby(["method", "threshold"])
	
	# Sum TP, FP, FN across all datasets for each (method, threshold)
	agg = grouped.agg(
		TP_sum=("TP", "sum"),
		FP_sum=("FP", "sum"),
		FN_sum=("FN", "sum"),
		n_datasets=("dataset_id", "nunique"),
	).reset_index()
	
	# Calculate precision and recall from aggregated counts
	agg["precision"] = agg["TP_sum"] / (agg["TP_sum"] + agg["FP_sum"])
	agg["recall"] = agg["TP_sum"] / (agg["TP_sum"] + agg["FN_sum"])
	
	# Handle division by zero
	agg["precision"] = agg["precision"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
	agg["recall"] = agg["recall"].fillna(0.0)
	
	return agg


def plot_precision_recall_with_std(agg_curve_df, output_path):
	"""Plot all methods on one PR curve"""
	method_meta = {
		"ref": {"label": "ref", "color": COLORS["blue"]},
		"v0": {"label": "version 1", "color": COLORS["orange"]},
		"v1": {"label": "version 2", "color": COLORS["green"]},
		"v2": {"label": "version 3", "color": COLORS["red"]},
	}

	fig, ax = plt.subplots(figsize=(10, 10))

	for method in ["ref", "v0", "v1", "v2"]:
		method_curve = agg_curve_df[agg_curve_df["method"] == method].sort_values("threshold")
		if method_curve.empty:
			continue



		legend_label = method_meta[method]["label"]

		ax.plot(
			method_curve["recall"].values,
			method_curve["precision"].values,
			"o-",
			linewidth=3.0,
			markersize=6,
			alpha=0.85,
			color=method_meta[method]["color"],
			label=legend_label,
		)

	ax.set_xlabel("Recall")
	ax.set_ylabel("Precision")
	ax.set_title("Precision-Recall Comparison Across Versions\n(aggregated over 10 datasets)")
	ax.grid(True, alpha=0.3)
	ax.set_xlim(-0.05, 1.05)
	ax.set_ylim(-0.05, 1.05)
	ax.set_aspect('equal')
	ax.legend(loc="best")

	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()


def load_run_args():
	"""Return run settings from the in-file RUN_CONFIG block."""
	return SimpleNamespace(**RUN_CONFIG)


def resolve_theta_for_dataset(data, requested_theta, eps=1e-10, theta_max=1000):
	"""Return theta and metadata for one dataset.

	If requested_theta is None or non-positive, estimate theta from data.
	"""
	if requested_theta is not None and requested_theta > 0:
		return float(requested_theta), "fixed", None

	results = estimate_zinb(data, eps=eps)
	theta = float(results["theta"])
	if theta >= theta_max:
		raise ValueError(
			f"Estimated theta is very large ({theta:.4f}), indicating estimation failure."
		)
	return theta, "estimated", results


def resolve_worker_count(requested_workers, task_count):
	"""Resolve the number of worker processes to use."""
	if task_count < 1:
		return 1

	available_cpus = os.cpu_count() or 1
	if requested_workers is None:
		return max(1, min(available_cpus, task_count))
	if requested_workers < 1:
		raise ValueError("--n_workers must be at least 1")
	return min(requested_workers, task_count)


def main():
	args = load_run_args()

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
	if args.theta is not None and args.theta > 0:
		print(f"  Theta mode: fixed ({args.theta})")
	else:
		print("  Theta mode: estimated per dataset")
	print(f"  Threshold count: {len(thresholds)}")
	available_cpus = os.cpu_count() or 1
	method_workers = resolve_worker_count(args.n_workers, len(methods))
	print(f"  CPUs available: {available_cpus}")
	print(f"  Method workers: {method_workers}")

	all_rows = []
	executor = None
	if method_workers > 1:
		executor = ProcessPoolExecutor(max_workers=method_workers)

	try:
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
			theta_value, theta_mode, theta_info = resolve_theta_for_dataset(data, args.theta)
			print(f"  Points: {len(data)} | True CPs: {len(true_cps)}")
			if theta_mode == "fixed":
				print(f"  Theta used: {theta_value:.4f} (fixed)")
			else:
				print(
					f"  Theta used: {theta_value:.4f} (estimated; "
					f"pi={theta_info['pi']:.4f}, mu={theta_info['mu']:.4f})"
				)

			if executor is None:
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
						theta_value,
						pi_file,
						nucleosome_distances,
						centromere_distances,
						nucleosome_file,
						centromere_file,
					)
					all_rows.extend(rows)
				continue

			future_to_method = {}
			for method in methods:
				print(f"  Method {method} ...")
				future = executor.submit(
					evaluate_method_on_dataset,
					method,
					dataset_id,
					data,
					true_cps,
					args.window_size,
					args.overlap,
					thresholds,
					theta_value,
					pi_file,
					nucleosome_distances,
					centromere_distances,
					nucleosome_file,
					centromere_file,
				)
				future_to_method[future] = method

			dataset_rows = {}
			for future in as_completed(future_to_method):
				method = future_to_method[future]
				try:
					dataset_rows[method] = future.result()
				except Exception as exc:
					raise RuntimeError(
						f"Method {method} failed while processing dataset {dataset_id}"
					) from exc
				print(f"  Method {method} done")

			for method in methods:
				all_rows.extend(dataset_rows[method])
	finally:
		if executor is not None:
			executor.shutdown()

	results_df = pd.DataFrame(all_rows).sort_values(
		["dataset_id", "method", "threshold"]
	).reset_index(drop=True)
	results_csv = os.path.join(args.output_folder, "all_results.csv")
	results_df.to_csv(results_csv, index=False)
	print(f"\nSaved detailed results to {results_csv}")

	agg_curve_df = aggregate_curves(results_df)
	agg_curve_csv = os.path.join(args.output_folder, "precision_recall_aggregated.csv")
	agg_curve_df.to_csv(agg_curve_csv, index=False)
	print(f"Saved aggregated PR stats to {agg_curve_csv}")

	plot_path = os.path.join(args.output_folder, "precision_recall_compare_versions.png")
	plot_precision_recall_with_std(agg_curve_df, plot_path)
	print(f"Saved PR comparison plot to {plot_path}")


if __name__ == "__main__":
	main()
