import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)


DEFAULT_ESTIMATOR_SUBDIRS = ["segment_mu", "segment_mu_informed"]

ESTIMATOR_METRICS_TO_PLOT = [
	("mean_abs_relative_error", "Relative Mean Absolute Error"),
	("pairwise_order_accuracy", "Order Performance"),
]

THRESHOLD_METRICS_TO_PLOT = [
	("mean_abs_relative_error", "Mean Absolute Relative Error"),
	("pairwise_order_accuracy", "Pairwise Order Accuracy"),
]

THRESHOLD_FROM_FILENAME_RE = re.compile(r"_th(?P<threshold>-?\d+(?:\.\d+)?)")


def resolve_path(path):
	if os.path.isabs(path):
		return path
	return os.path.join(PROJECT_ROOT, path)


def read_true_region_params(params_file):
	required_columns = {"region_start", "region_end", "region_mean"}
	params_df = pd.read_csv(params_file)

	missing_columns = sorted(required_columns - set(params_df.columns))
	if missing_columns:
		raise ValueError(
			f"Missing required column(s) in {params_file}: {', '.join(missing_columns)}"
		)

	params_df = params_df[["region_start", "region_end", "region_mean"]].copy()
	params_df["region_start"] = params_df["region_start"].astype(int)
	params_df["region_end"] = params_df["region_end"].astype(int)
	params_df["region_mean"] = params_df["region_mean"].astype(float)
	params_df = params_df[params_df["region_end"] > params_df["region_start"]]
	params_df = params_df.sort_values(["region_start", "region_end"]).reset_index(drop=True)

	if params_df.empty:
		raise ValueError(f"No valid regions found in params file: {params_file}")

	return params_df


def read_true_pi_values(pi_file):
	"""Read true pi values from pi_values.csv file."""
	try:
		pi_df = pd.read_csv(pi_file)
		if "pi_value" not in pi_df.columns:
			raise ValueError(f"Missing 'pi_value' column in {pi_file}")
		return pi_df["pi_value"].to_numpy(dtype=np.float64)
	except FileNotFoundError:
		return None


def detect_prediction_column(segment_df):
	candidates = ["mu_estimate", "mu_informed"]
	for column in candidates:
		if column in segment_df.columns:
			return column

	raise ValueError(
		"Could not detect a prediction column. Expected one of: "
		+ ", ".join(candidates)
	)


def compute_overlap_weighted_true_mu(segment_df, params_df):
	if "start_index" not in segment_df.columns or "end_index_exclusive" not in segment_df.columns:
		raise ValueError(
			"Segment file must contain 'start_index' and 'end_index_exclusive' columns."
		)

	seg_starts = segment_df["start_index"].astype(int).to_numpy()
	seg_ends = segment_df["end_index_exclusive"].astype(int).to_numpy()

	reg_starts = params_df["region_start"].to_numpy(dtype=np.int64)
	reg_ends = params_df["region_end"].to_numpy(dtype=np.int64)
	reg_means = params_df["region_mean"].to_numpy(dtype=np.float64)

	n_segments = len(segment_df)
	true_mu = np.full(n_segments, np.nan, dtype=np.float64)
	overlap_bp = np.zeros(n_segments, dtype=np.int64)

	order = np.argsort(seg_starts)
	region_idx = 0
	n_regions = len(params_df)

	for ordered_idx in order:
		start = int(seg_starts[ordered_idx])
		end = int(seg_ends[ordered_idx])

		if end <= start:
			continue

		while region_idx < n_regions and reg_ends[region_idx] <= start:
			region_idx += 1

		j = region_idx
		weighted_sum = 0.0
		covered = 0

		while j < n_regions and reg_starts[j] < end:
			overlap_start = max(start, int(reg_starts[j]))
			overlap_end = min(end, int(reg_ends[j]))
			overlap = overlap_end - overlap_start

			if overlap > 0:
				weighted_sum += overlap * float(reg_means[j])
				covered += overlap

			j += 1

		overlap_bp[ordered_idx] = covered
		if covered > 0:
			true_mu[ordered_idx] = weighted_sum / covered

	return true_mu, overlap_bp


def compute_empirical_mu_from_observed(segment_df, pi_values, eps=1e-10):
	"""Compute empirical mu = observed_mean / (1 - true_pi_mean) for each segment."""
	if "start_index" not in segment_df.columns or "end_index_exclusive" not in segment_df.columns:
		raise ValueError("Segment file must contain 'start_index' and 'end_index_exclusive' columns.")
	
	if pi_values is None:
		return np.full(len(segment_df), np.nan, dtype=np.float64), np.full(len(segment_df), np.nan, dtype=np.float64)
	
	seg_starts = segment_df["start_index"].astype(int).to_numpy()
	seg_ends = segment_df["end_index_exclusive"].astype(int).to_numpy()
	
	n_segments = len(segment_df)
	empirical_mu = np.full(n_segments, np.nan, dtype=np.float64)
	mean_pi = np.full(n_segments, np.nan, dtype=np.float64)
	
	for idx in range(n_segments):
		start = int(seg_starts[idx])
		end = int(seg_ends[idx])
		
		if end <= start or end > len(pi_values):
			continue
		
		# Get mean pi for this segment
		segment_pi = pi_values[start:end]
		pi_mean = float(np.mean(segment_pi))
		mean_pi[idx] = pi_mean
		
		# Get observed mean from raw_mean column if available
		if "raw_mean" in segment_df.columns:
			obs_mean = float(segment_df.iloc[idx]["raw_mean"])
		else:
			continue
		
		# Compute empirical mu: obs_mean = (1 - pi) * mu => mu = obs_mean / (1 - pi)
		if pi_mean < 1.0 - eps:
			empirical_mu[idx] = obs_mean / (1.0 - pi_mean)
		else:
			empirical_mu[idx] = np.nan
	
	return empirical_mu, mean_pi


def add_error_columns(segment_df, prediction_column, relative_eps):
	output_df = segment_df.copy()

	if "length" in output_df.columns:
		lengths = output_df["length"].astype(float).to_numpy()
	else:
		lengths = (
			output_df["end_index_exclusive"].astype(float).to_numpy()
			- output_df["start_index"].astype(float).to_numpy()
		)
		output_df["length"] = lengths

	pred = output_df[prediction_column].astype(float).to_numpy()
	true = output_df["true_mu_overlap"].astype(float).to_numpy()
	overlap_bp = output_df["truth_overlap_bp"].astype(float).to_numpy()

	valid = np.isfinite(pred) & np.isfinite(true) & (overlap_bp > 0)
	abs_error = np.full(len(output_df), np.nan, dtype=np.float64)
	abs_error[valid] = np.abs(pred[valid] - true[valid])

	rel_error = np.full(len(output_df), np.nan, dtype=np.float64)
	rel_mask = valid & (np.abs(true) > relative_eps)
	rel_error[rel_mask] = np.abs((pred[rel_mask] - true[rel_mask]) / true[rel_mask])

	output_df["truth_overlap_fraction"] = np.divide(
		overlap_bp,
		np.maximum(lengths, 1.0),
		out=np.zeros_like(overlap_bp, dtype=np.float64),
		where=np.maximum(lengths, 1.0) > 0,
	)
	output_df["absolute_error"] = abs_error
	output_df["relative_error"] = rel_error

	return output_df


def add_empirical_error_columns(segment_df, prediction_column, relative_eps):
	"""Add error columns based on empirical mu (from observed data)."""
	output_df = segment_df.copy()
	
	if "empirical_mu" not in output_df.columns:
		return output_df
	
	pred = output_df[prediction_column].astype(float).to_numpy()
	empirical_mu = output_df["empirical_mu"].astype(float).to_numpy()
	
	valid = np.isfinite(pred) & np.isfinite(empirical_mu)
	abs_error_emp = np.full(len(output_df), np.nan, dtype=np.float64)
	abs_error_emp[valid] = np.abs(pred[valid] - empirical_mu[valid])
	
	rel_error_emp = np.full(len(output_df), np.nan, dtype=np.float64)
	rel_mask = valid & (np.abs(empirical_mu) > relative_eps)
	rel_error_emp[rel_mask] = np.abs((pred[rel_mask] - empirical_mu[rel_mask]) / empirical_mu[rel_mask])
	
	output_df["absolute_error_empirical"] = abs_error_emp
	output_df["relative_error_empirical"] = rel_error_emp
	
	return output_df


def pairwise_order_accuracy(true_values, pred_values, true_diff_tol, max_pairs, random_seed):
	n = len(true_values)
	if n < 2:
		return np.nan, 0, 0, "none"

	total_pairs = n * (n - 1) // 2
	print(f"Total pairs: {total_pairs}")

	if total_pairs <= max_pairs:
		i, j = np.triu_indices(n, k=1)
		mode = "all_pairs"
		evaluated_pairs = len(i)
	else:
		rng = np.random.default_rng(random_seed)
		i = np.empty(0, dtype=np.int64)
		j = np.empty(0, dtype=np.int64)

		while len(i) < max_pairs:
			needed = max_pairs - len(i)
			sample_i = rng.integers(0, n, size=needed * 3)
			sample_j = rng.integers(0, n, size=needed * 3)
			valid = sample_i < sample_j
			sample_i = sample_i[valid][:needed]
			sample_j = sample_j[valid][:needed]
			i = np.concatenate([i, sample_i])
			j = np.concatenate([j, sample_j])

		mode = "sampled_pairs"
		evaluated_pairs = len(i)

	diff_true = true_values[i] - true_values[j]
	comparable = np.abs(diff_true) > true_diff_tol
	comparable_pairs = int(np.sum(comparable))

	if comparable_pairs == 0:
		return np.nan, comparable_pairs, evaluated_pairs, mode

	diff_pred = pred_values[i] - pred_values[j]
	correct = np.sign(diff_true[comparable]) == np.sign(diff_pred[comparable])
	return float(np.mean(correct)), comparable_pairs, evaluated_pairs, mode


def compute_summary_metrics(
	segment_df,
	prediction_column,
	relative_eps,
	pairwise_true_diff_tol,
	pairwise_max_pairs,
	pairwise_seed,
):
	pred = segment_df[prediction_column].astype(float).to_numpy()
	true = segment_df["true_mu_overlap"].astype(float).to_numpy()
	overlap_bp = segment_df["truth_overlap_bp"].astype(float).to_numpy()
	lengths = segment_df["length"].astype(float).to_numpy()

	valid = np.isfinite(pred) & np.isfinite(true) & (overlap_bp > 0)
	n_segments_total = int(len(segment_df))
	n_segments_valid = int(np.sum(valid))
	coverage_mean = float(np.nanmean(segment_df["truth_overlap_fraction"].to_numpy(dtype=float)))

	metrics = {
		"segments_total": n_segments_total,
		"segments_with_truth": n_segments_valid,
		"mean_truth_overlap_fraction": coverage_mean,
		"mae": np.nan,
		"rmse": np.nan,
		"weighted_mae": np.nan,
		"weighted_rmse": np.nan,
		"mean_abs_relative_error": np.nan,
		"median_abs_relative_error": np.nan,
		"weighted_mean_abs_relative_error": np.nan,
		"segments_nonzero_truth": 0,
		"segments_zero_truth": 0,
		"zero_truth_mae": np.nan,
		"spearman_rho": np.nan,
		"pairwise_order_accuracy": np.nan,
		"pairwise_comparable_pairs": 0,
		"pairwise_pairs_evaluated": 0,
		"pairwise_mode": "none",
	}

	if n_segments_valid == 0:
		return metrics

	err = pred[valid] - true[valid]
	abs_err = np.abs(err)
	w = lengths[valid]

	metrics["mae"] = float(np.mean(abs_err))
	metrics["rmse"] = float(np.sqrt(np.mean(err ** 2)))

	if np.sum(w) > 0:
		metrics["weighted_mae"] = float(np.average(abs_err, weights=w))
		metrics["weighted_rmse"] = float(np.sqrt(np.average(err ** 2, weights=w)))

	rel_mask = valid & (np.abs(true) > relative_eps)
	metrics["segments_nonzero_truth"] = int(np.sum(rel_mask))
	if np.any(rel_mask):
		rel_abs = np.abs((pred[rel_mask] - true[rel_mask]) / true[rel_mask])
		rel_w = lengths[rel_mask]
		metrics["mean_abs_relative_error"] = float(np.mean(rel_abs))
		metrics["median_abs_relative_error"] = float(np.median(rel_abs))
		if np.sum(rel_w) > 0:
			metrics["weighted_mean_abs_relative_error"] = float(np.average(rel_abs, weights=rel_w))

	zero_mask = valid & (np.abs(true) <= relative_eps)
	metrics["segments_zero_truth"] = int(np.sum(zero_mask))
	if np.any(zero_mask):
		metrics["zero_truth_mae"] = float(np.mean(np.abs(pred[zero_mask])))

	true_valid = true[valid]
	pred_valid = pred[valid]
	metrics["spearman_rho"] = float(
		pd.Series(true_valid).corr(pd.Series(pred_valid), method="spearman")
	)

	order_acc, comparable_pairs, evaluated_pairs, mode = pairwise_order_accuracy(
		true_values=true_valid,
		pred_values=pred_valid,
		true_diff_tol=pairwise_true_diff_tol,
		max_pairs=pairwise_max_pairs,
		random_seed=pairwise_seed,
	)
	metrics["pairwise_order_accuracy"] = order_acc
	metrics["pairwise_comparable_pairs"] = comparable_pairs
	metrics["pairwise_pairs_evaluated"] = int(evaluated_pairs)
	metrics["pairwise_mode"] = mode

	return metrics


def compute_empirical_summary_metrics(
	segment_df,
	prediction_column,
	relative_eps,
	pairwise_true_diff_tol,
	pairwise_max_pairs,
	pairwise_seed,
):
	"""Compute summary metrics using empirical mu as ground truth."""
	if "empirical_mu" not in segment_df.columns:
		return {}
	
	pred = segment_df[prediction_column].astype(float).to_numpy()
	empirical_mu = segment_df["empirical_mu"].astype(float).to_numpy()
	lengths = segment_df["length"].astype(float).to_numpy()

	valid = np.isfinite(pred) & np.isfinite(empirical_mu)
	n_segments_total = int(len(segment_df))
	n_segments_valid = int(np.sum(valid))

	metrics = {
		"segments_total_empirical": n_segments_total,
		"segments_with_empirical": n_segments_valid,
		"mae_empirical": np.nan,
		"rmse_empirical": np.nan,
		"weighted_mae_empirical": np.nan,
		"weighted_rmse_empirical": np.nan,
		"mean_abs_relative_error_empirical": np.nan,
		"median_abs_relative_error_empirical": np.nan,
		"weighted_mean_abs_relative_error_empirical": np.nan,
		"spearman_rho_empirical": np.nan,
		"pairwise_order_accuracy_empirical": np.nan,
		"pairwise_comparable_pairs_empirical": 0,
		"pairwise_pairs_evaluated_empirical": 0,
		"pairwise_mode_empirical": "none",
	}

	if n_segments_valid == 0:
		return metrics

	err = pred[valid] - empirical_mu[valid]
	abs_err = np.abs(err)
	w = lengths[valid]

	metrics["mae_empirical"] = float(np.mean(abs_err))
	metrics["rmse_empirical"] = float(np.sqrt(np.mean(err ** 2)))

	if np.sum(w) > 0:
		metrics["weighted_mae_empirical"] = float(np.average(abs_err, weights=w))
		metrics["weighted_rmse_empirical"] = float(np.sqrt(np.average(err ** 2, weights=w)))

	rel_mask = valid & (np.abs(empirical_mu) > relative_eps)
	if np.any(rel_mask):
		rel_abs = np.abs((pred[rel_mask] - empirical_mu[rel_mask]) / empirical_mu[rel_mask])
		rel_w = lengths[rel_mask]
		metrics["mean_abs_relative_error_empirical"] = float(np.mean(rel_abs))
		metrics["median_abs_relative_error_empirical"] = float(np.median(rel_abs))
		if np.sum(rel_w) > 0:
			metrics["weighted_mean_abs_relative_error_empirical"] = float(np.average(rel_abs, weights=rel_w))

	empirical_valid = empirical_mu[valid]
	pred_valid = pred[valid]
	metrics["spearman_rho_empirical"] = float(
		pd.Series(empirical_valid).corr(pd.Series(pred_valid), method="spearman")
	)

	order_acc, comparable_pairs, evaluated_pairs, mode = pairwise_order_accuracy(
		true_values=empirical_valid,
		pred_values=pred_valid,
		true_diff_tol=pairwise_true_diff_tol,
		max_pairs=pairwise_max_pairs,
		random_seed=pairwise_seed,
	)
	metrics["pairwise_order_accuracy_empirical"] = order_acc
	metrics["pairwise_comparable_pairs_empirical"] = comparable_pairs
	metrics["pairwise_pairs_evaluated_empirical"] = int(evaluated_pairs)
	metrics["pairwise_mode_empirical"] = mode

	return metrics


def extract_metadata(segment_df, dataset_num, estimator_subdir, segment_file_name, window_name):
	metadata = {
		"dataset_num": int(dataset_num),
		"window_name": window_name,
		"estimator_type": estimator_subdir,
		"segment_file": segment_file_name,
	}

	if segment_df.empty:
		return metadata

	first_row = segment_df.iloc[0]
	passthrough_columns = [
		"source_result_file",
		"dataset_name",
		"window_size",
		"overlap_pct",
		"threshold",
		"theta_global",
		"num_change_points",
	]
	for column in passthrough_columns:
		if column in segment_df.columns:
			metadata[column] = first_row[column]

	return metadata


def evaluate_segment_file(
	segment_file,
	params_df,
	pi_values,
	output_dir,
	relative_eps,
	pairwise_true_diff_tol,
	pairwise_max_pairs,
	pairwise_seed,
):
	segment_df = pd.read_csv(segment_file)
	prediction_column = detect_prediction_column(segment_df)

	if "length" not in segment_df.columns and {
		"start_index",
		"end_index_exclusive",
	}.issubset(segment_df.columns):
		segment_df["length"] = (
			segment_df["end_index_exclusive"].astype(float)
			- segment_df["start_index"].astype(float)
		)

	# Compute population-based ground truth (from params file)
	true_mu, overlap_bp = compute_overlap_weighted_true_mu(segment_df, params_df)
	segment_df["true_mu_overlap"] = true_mu
	segment_df["truth_overlap_bp"] = overlap_bp
	segment_df = add_error_columns(
		segment_df,
		prediction_column=prediction_column,
		relative_eps=relative_eps,
	)

	# Compute empirical ground truth (from observed data and true pi)
	empirical_mu, mean_pi = compute_empirical_mu_from_observed(segment_df, pi_values)
	segment_df["empirical_mu"] = empirical_mu
	segment_df["true_pi_mean"] = mean_pi
	segment_df = add_empirical_error_columns(
		segment_df,
		prediction_column=prediction_column,
		relative_eps=relative_eps,
	)

	# Compute both sets of metrics
	metrics = compute_summary_metrics(
		segment_df,
		prediction_column=prediction_column,
		relative_eps=relative_eps,
		pairwise_true_diff_tol=pairwise_true_diff_tol,
		pairwise_max_pairs=pairwise_max_pairs,
		pairwise_seed=pairwise_seed,
	)
	
	empirical_metrics = compute_empirical_summary_metrics(
		segment_df,
		prediction_column=prediction_column,
		relative_eps=relative_eps,
		pairwise_true_diff_tol=pairwise_true_diff_tol,
		pairwise_max_pairs=pairwise_max_pairs,
		pairwise_seed=pairwise_seed,
	)
	
	# Merge both metric sets
	metrics.update(empirical_metrics)
	metrics["prediction_column"] = prediction_column

	os.makedirs(output_dir, exist_ok=True)
	file_stem = os.path.splitext(os.path.basename(segment_file))[0]
	detail_path = os.path.join(output_dir, f"{file_stem}_performance.csv")
	segment_df.to_csv(detail_path, index=False)

	return metrics, detail_path


def build_aggregated_estimator_summary(summary_df):
	if summary_df.empty:
		return pd.DataFrame()

	grouped = summary_df.groupby("estimator_type", dropna=False)
	agg_df = grouped.agg(
		files_evaluated=("segment_file", "count"),
		mean_rmse=("rmse", "mean"),
		mean_mae=("mae", "mean"),
		mean_mare=("mean_abs_relative_error", "mean"),
		mean_spearman=("spearman_rho", "mean"),
		mean_pairwise_order_accuracy=("pairwise_order_accuracy", "mean"),
	).reset_index()
	return agg_df.sort_values("estimator_type").reset_index(drop=True)


def _prepare_metric_stats(df, group_columns, metric_column):
	required_columns = set(group_columns + [metric_column])
	if not required_columns.issubset(df.columns):
		return pd.DataFrame()

	plot_df = df[group_columns + [metric_column]].copy()
	plot_df[metric_column] = pd.to_numeric(plot_df[metric_column], errors="coerce")
	plot_df = plot_df.dropna(subset=[metric_column])
	if plot_df.empty:
		return pd.DataFrame()

	stats_df = (
		plot_df.groupby(group_columns, dropna=False)[metric_column]
		.agg(["mean", "std", "count"])
		.reset_index()
	)
	stats_df["std"] = stats_df["std"].fillna(0.0)
	stats_df["count"] = stats_df["count"].astype(int)
	return stats_df


def _extract_error_bars(stats_df, errorbar_type):
	if errorbar_type == "sem":
		counts = stats_df["count"].clip(lower=1).to_numpy(dtype=float)
		errors = stats_df["std"].to_numpy(dtype=float) / np.sqrt(counts)
	else:
		errors = stats_df["std"].to_numpy(dtype=float)

	return np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)


def _format_threshold_label(value):
	if float(value).is_integer():
		return str(int(value))
	return f"{float(value):.2f}".rstrip("0").rstrip(".")


def normalize_threshold_bounds(threshold_min, threshold_max):
	if threshold_min is not None and threshold_max is not None and threshold_min > threshold_max:
		return threshold_max, threshold_min
	return threshold_min, threshold_max


def extract_threshold_value(segment_preview_df, segment_filename):
	if not segment_preview_df.empty and "threshold" in segment_preview_df.columns:
		threshold_value = pd.to_numeric(segment_preview_df.iloc[0]["threshold"], errors="coerce")
		if pd.notna(threshold_value):
			return float(threshold_value)

	match = THRESHOLD_FROM_FILENAME_RE.search(segment_filename)
	if match:
		return float(match.group("threshold"))

	return np.nan


def threshold_is_selected(threshold_value, threshold_min, threshold_max):
	if threshold_min is None and threshold_max is None:
		return True

	if pd.isna(threshold_value):
		return False

	if threshold_min is not None and threshold_value < threshold_min:
		return False

	if threshold_max is not None and threshold_value > threshold_max:
		return False

	return True


def plot_metrics_by_estimator_bar(summary_df, output_path, errorbar_type):
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

	subplot_labels = ['a', 'b']
	
	# Mapping for display names
	estimator_display_names = {
		"segment_mu": "EM algorithm",
		"segment_mu_informed": "pi informed"
	}

	for idx, (metric_column, metric_label) in enumerate(ESTIMATOR_METRICS_TO_PLOT):
		ax = axes[idx]
		stats_df = _prepare_metric_stats(summary_df, ["estimator_type"], metric_column)

		if stats_df.empty:
			ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
			ax.set_axis_off()
			continue

		errors = _extract_error_bars(stats_df, errorbar_type)
		colors = plt.cm.Set2(np.linspace(0.0, 1.0, len(stats_df)))
		
		# Apply display name mapping
		display_labels = [estimator_display_names.get(str(x), str(x)) for x in stats_df["estimator_type"]]
		
		ax.bar(
			display_labels,
			stats_df["mean"].to_numpy(dtype=float),
			yerr=errors,
			capsize=6,
			alpha=0.9,
			color=colors,
			edgecolor="black",
			linewidth=0.6,
		)
		ax.set_title(metric_label)
		ax.set_xlabel("Estimator")
		ax.set_ylabel(f"Mean ± {errorbar_type.upper()}")
		ax.grid(axis="y", linestyle="--", alpha=0.35)

		if metric_column in {"pairwise_order_accuracy", "spearman_rho"}:
			ax.set_ylim(0.0, 1.0)

		# Add subplot label
		ax.text(-0.1, 1.05, subplot_labels[idx], transform=ax.transAxes,
		        fontsize=14, fontweight='bold', va='top', ha='right')

	# fig.suptitle("Estimator Comparison (bar plot with error bars)", fontsize=14, y=1.02)
	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)
	return output_path


def plot_method_comparison_averaged(
	summary_df,
	threshold_min,
	threshold_max,
	output_path,
	errorbar_type,
	metrics_to_plot,
):
	"""
	Create a side-by-side comparison plot averaging performance across all thresholds
	in the range [threshold_min, threshold_max].
	"""
	if summary_df.empty or "estimator_type" not in summary_df.columns:
		print(f"Warning: Cannot create averaged comparison plot - missing required data")
		return None

	# Filter for threshold range if specified
	plot_df = summary_df.copy()
	if "threshold" in plot_df.columns and (threshold_min is not None or threshold_max is not None):
		plot_df["threshold"] = pd.to_numeric(plot_df["threshold"], errors="coerce")
		if threshold_min is not None:
			plot_df = plot_df[plot_df["threshold"] >= threshold_min]
		if threshold_max is not None:
			plot_df = plot_df[plot_df["threshold"] <= threshold_max]
	
	if plot_df.empty:
		print(f"Warning: No data found in threshold range [{threshold_min}, {threshold_max}]")
		return None

	# Mapping for display names
	estimator_display_names = {
		"segment_mu": "EM algorithm",
		"segment_mu_informed": "pi informed"
	}
	
	fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
	colors = {"segment_mu": plt.cm.Set2(0), "segment_mu_informed": "#808080"}  # Second bar is gray
	
	for idx, (metric_column, metric_label) in enumerate(metrics_to_plot):
		ax = axes[idx]
		
		# Check if metric exists
		if metric_column not in plot_df.columns:
			ax.text(0.5, 0.5, f"No data for {metric_label}", 
			        ha='center', va='center', transform=ax.transAxes)
			ax.set_title(metric_label, fontsize=13, fontweight='bold')
			continue
		
		# Calculate statistics for each estimator (averaged across all thresholds in range)
		stats_df = _prepare_metric_stats(plot_df, ["estimator_type"], metric_column)
		
		if stats_df.empty:
			ax.text(0.5, 0.5, f"No valid data for {metric_label}", 
			        ha='center', va='center', transform=ax.transAxes)
			ax.set_title(metric_label, fontsize=13, fontweight='bold')
			continue
		
		stats_df = stats_df.copy()
		stats_df["error"] = _extract_error_bars(stats_df, errorbar_type)
		
		estimator_values = sorted(stats_df["estimator_type"].astype(str).unique())
		x_positions = np.arange(len(estimator_values))
		
		# Create bars for each estimator
		bars = []
		for i, estimator in enumerate(estimator_values):
			estimator_stats = stats_df[stats_df["estimator_type"].astype(str) == estimator]
			if estimator_stats.empty:
				continue
			
			mean_val = float(estimator_stats["mean"].iloc[0])
			error_val = float(estimator_stats["error"].iloc[0]) if pd.notna(estimator_stats["error"].iloc[0]) else 0.0
			display_label = estimator_display_names.get(estimator, estimator)
			
			bar = ax.bar(
				i,
				mean_val,
				width=0.75,
				yerr=error_val,
				capsize=6,
				label=display_label,
				alpha=0.85,
				color=colors.get(estimator, plt.cm.Set2(i)),
				edgecolor="black",
				linewidth=1.0,
			)
			bars.append(bar)
		
		ax.set_title(metric_label, fontsize=13, fontweight='bold', pad=10)
		ax.set_ylabel(f"Mean ± {errorbar_type.upper()}", fontsize=11)
		ax.set_xticks(x_positions)
		ax.set_xticklabels([estimator_display_names.get(e, e) for e in estimator_values], fontsize=11)
		ax.grid(axis="y", linestyle="--", alpha=0.35)
		ax.tick_params(axis='x', which='major', labelsize=11)
		
		# Set appropriate y-limits
		if metric_column in {"pairwise_order_accuracy", "spearman_rho"}:
			ax.set_ylim(0.0, 1.05)
		
		# Add subplot label (A, B)
		subplot_labels = ['A', 'B']
		ax.text(-0.15, 1.08, subplot_labels[idx], transform=ax.transAxes,
		        fontsize=15, fontweight='bold', va='top', ha='right')
	
	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)
	print(f"Created averaged method comparison plot: {output_path}")
	return output_path


def plot_metric_by_threshold_bar(
	summary_df,
	metric_column,
	metric_label,
	output_path,
	errorbar_type,
):
	if "threshold" not in summary_df.columns or "estimator_type" not in summary_df.columns:
		return None

	plot_df = summary_df.copy()
	plot_df["threshold"] = pd.to_numeric(plot_df["threshold"], errors="coerce")
	plot_df = plot_df.dropna(subset=["threshold"])

	stats_df = _prepare_metric_stats(plot_df, ["threshold", "estimator_type"], metric_column)
	if stats_df.empty:
		return None

	stats_df = stats_df.copy()
	stats_df["error"] = _extract_error_bars(stats_df, errorbar_type)

	threshold_values = sorted(stats_df["threshold"].unique())
	estimator_values = sorted(stats_df["estimator_type"].astype(str).unique())
	
	# Mapping for display names
	estimator_display_names = {
		"segment_mu": "EM algorithm",
		"segment_mu_informed": "pi informed"
	}

	x = np.arange(len(threshold_values), dtype=float)
	n_estimators = max(1, len(estimator_values))
	width = 0.8 / n_estimators

	fig_width = max(10.0, 0.7 * len(threshold_values))
	fig, ax = plt.subplots(figsize=(fig_width, 6.5), constrained_layout=True)
	colors = plt.cm.Set2(np.linspace(0.0, 1.0, n_estimators))

	for idx, estimator in enumerate(estimator_values):
		estimator_stats = stats_df[stats_df["estimator_type"].astype(str) == estimator]
		mean_map = {
			float(row["threshold"]): float(row["mean"])
			for _, row in estimator_stats.iterrows()
			if pd.notna(row["mean"])
		}
		error_map = {
			float(row["threshold"]): float(row["error"])
			for _, row in estimator_stats.iterrows()
			if pd.notna(row["error"])
		}

		means = np.array([mean_map.get(float(t), np.nan) for t in threshold_values], dtype=float)
		errors = np.array([error_map.get(float(t), 0.0) for t in threshold_values], dtype=float)
		offset = (idx - (n_estimators - 1) / 2.0) * width
		
		# Use display name for legend
		display_label = estimator_display_names.get(estimator, estimator)

		ax.bar(
			x + offset,
			means,
			width=width * 0.95,
			yerr=errors,
			capsize=4,
			label=display_label,
			alpha=0.9,
			color=colors[idx],
			edgecolor="black",
			linewidth=0.5,
		)

	ax.set_title(f"{metric_label} by threshold")
	ax.set_xlabel("Threshold")
	ax.set_ylabel(f"Mean ± {errorbar_type.upper()}")
	ax.set_xticks(x)
	ax.set_xticklabels([_format_threshold_label(t) for t in threshold_values], rotation=45, ha="right")
	ax.grid(axis="y", linestyle="--", alpha=0.35)
	ax.legend(title="Estimator", frameon=False)

	if metric_column in {"pairwise_order_accuracy", "spearman_rho"}:
		ax.set_ylim(0.0, 1.0)

	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)
	return output_path


def plot_method_comparison_at_threshold(
	summary_df,
	threshold_value,
	output_path,
	errorbar_type,
):
	"""
	Create a side-by-side comparison plot of RMSE and order performance
	for both estimation methods at a specific threshold value.
	
	Args:
		summary_df: DataFrame with performance metrics
		threshold_value: Specific threshold to filter for (e.g., 3.0)
		output_path: Path to save the plot
		errorbar_type: 'std' or 'sem' for error bars
	"""
	if summary_df.empty or "threshold" not in summary_df.columns or "estimator_type" not in summary_df.columns:
		print(f"Warning: Cannot create method comparison plot - missing required data")
		return None

	# Filter for specific threshold
	plot_df = summary_df.copy()
	plot_df["threshold"] = pd.to_numeric(plot_df["threshold"], errors="coerce")
	plot_df = plot_df[plot_df["threshold"] == threshold_value]
	
	if plot_df.empty:
		print(f"Warning: No data found for threshold {threshold_value}")
		return None

	# Mapping for display names
	estimator_display_names = {
		"segment_mu": "EM algorithm",
		"segment_mu_informed": "pi informed"
	}
	
	# Metrics to plot
	metrics = [
		("mean_abs_relative_error", "Mean Absolute Relative Error"),
		("pairwise_order_accuracy", "Pairwise Order Accuracy")
	]
	
	fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
	colors = {"segment_mu": plt.cm.Set2(0), "segment_mu_informed": "#808080"}  # Second bar is gray
	
	for idx, (metric_column, metric_label) in enumerate(metrics):
		ax = axes[idx]
		
		# Check if metric exists
		if metric_column not in plot_df.columns:
			ax.text(0.5, 0.5, f"No data for {metric_label}", 
			        ha='center', va='center', transform=ax.transAxes)
			ax.set_title(metric_label, fontsize=13, fontweight='bold')
			continue
		
		# Calculate statistics for each estimator
		stats_df = _prepare_metric_stats(plot_df, ["estimator_type"], metric_column)
		
		if stats_df.empty:
			ax.text(0.5, 0.5, f"No valid data for {metric_label}", 
			        ha='center', va='center', transform=ax.transAxes)
			ax.set_title(metric_label, fontsize=13, fontweight='bold')
			continue
		
		stats_df = stats_df.copy()
		stats_df["error"] = _extract_error_bars(stats_df, errorbar_type)
		
		estimator_values = sorted(stats_df["estimator_type"].astype(str).unique())
		x_positions = np.arange(len(estimator_values))
		
		# Create bars for each estimator
		bars = []
		for i, estimator in enumerate(estimator_values):
			estimator_stats = stats_df[stats_df["estimator_type"].astype(str) == estimator]
			if estimator_stats.empty:
				continue
			
			mean_val = float(estimator_stats["mean"].iloc[0])
			error_val = float(estimator_stats["error"].iloc[0]) if pd.notna(estimator_stats["error"].iloc[0]) else 0.0
			display_label = estimator_display_names.get(estimator, estimator)
			
			bar = ax.bar(
				i,
				mean_val,
				width=0.75,  # Increased width from 0.65 to 0.75
				yerr=error_val,
				capsize=6,
				label=display_label,
				alpha=0.85,
				color=colors.get(estimator, plt.cm.Set2(i)),
				edgecolor="black",
				linewidth=1.0,
			)
			bars.append(bar)
		
		ax.set_title(metric_label, fontsize=13, fontweight='bold', pad=10)
		ax.set_ylabel(f"Mean ± {errorbar_type.upper()}", fontsize=11)
		ax.set_xticks(x_positions)
		ax.set_xticklabels([estimator_display_names.get(e, e) for e in estimator_values], fontsize=11)
		ax.grid(axis="y", linestyle="--", alpha=0.35)
		ax.tick_params(axis='x', which='major', labelsize=11)
		
		# Set appropriate y-limits
		if metric_column in {"pairwise_order_accuracy", "spearman_rho"}:
			ax.set_ylim(0.0, 1.05)
		
		# Add subplot label (A, B)
		subplot_labels = ['A', 'B']
		ax.text(-0.15, 1.08, subplot_labels[idx], transform=ax.transAxes,
		        fontsize=15, fontweight='bold', va='top', ha='right')
	
	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)
	print(f"Created method comparison plot: {output_path}")
	return output_path


def plot_method_comparison_empirical(
	summary_df,
	threshold_value,
	output_path,
	errorbar_type,
):
	"""
	Create a side-by-side comparison plot of RMSE and order performance
	using empirical mu as ground truth at a specific threshold value.
	"""
	if summary_df.empty or "threshold" not in summary_df.columns or "estimator_type" not in summary_df.columns:
		print(f"Warning: Cannot create empirical comparison plot - missing required data")
		return None

	# Filter for specific threshold
	plot_df = summary_df.copy()
	plot_df["threshold"] = pd.to_numeric(plot_df["threshold"], errors="coerce")
	plot_df = plot_df[plot_df["threshold"] == threshold_value]
	
	if plot_df.empty:
		print(f"Warning: No data found for threshold {threshold_value}")
		return None

	# Mapping for display names
	estimator_display_names = {
		"segment_mu": "EM algorithm",
		"segment_mu_informed": "pi informed"
	}
	
	# Metrics to plot (empirical versions)
	metrics = [
		("mean_abs_relative_error_empirical", "Mean Absolute Relative Error (Empirical)"),
		("pairwise_order_accuracy_empirical", "Pairwise Order Accuracy (Empirical)")
	]
	
	fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
	colors = {"segment_mu": plt.cm.Set2(0), "segment_mu_informed": "#808080"}  # Second bar is gray
	
	for idx, (metric_column, metric_label) in enumerate(metrics):
		ax = axes[idx]
		
		# Check if metric exists
		if metric_column not in plot_df.columns:
			ax.text(0.5, 0.5, f"No data for {metric_label}", 
			        ha='center', va='center', transform=ax.transAxes)
			ax.set_title(metric_label, fontsize=13, fontweight='bold')
			continue
		
		# Calculate statistics for each estimator
		stats_df = _prepare_metric_stats(plot_df, ["estimator_type"], metric_column)
		
		if stats_df.empty:
			ax.text(0.5, 0.5, f"No valid data for {metric_label}", 
			        ha='center', va='center', transform=ax.transAxes)
			ax.set_title(metric_label, fontsize=13, fontweight='bold')
			continue
		
		stats_df = stats_df.copy()
		stats_df["error"] = _extract_error_bars(stats_df, errorbar_type)
		
		estimator_values = sorted(stats_df["estimator_type"].astype(str).unique())
		x_positions = np.arange(len(estimator_values))
		
		# Create bars for each estimator
		bars = []
		for i, estimator in enumerate(estimator_values):
			estimator_stats = stats_df[stats_df["estimator_type"].astype(str) == estimator]
			if estimator_stats.empty:
				continue
			
			mean_val = float(estimator_stats["mean"].iloc[0])
			error_val = float(estimator_stats["error"].iloc[0]) if pd.notna(estimator_stats["error"].iloc[0]) else 0.0
			display_label = estimator_display_names.get(estimator, estimator)
			
			bar = ax.bar(
				i,
				mean_val,
				width=0.75,
				yerr=error_val,
				capsize=6,
				label=display_label,
				alpha=0.85,
				color=colors.get(estimator, plt.cm.Set2(i)),
				edgecolor="black",
				linewidth=1.0,
			)
			bars.append(bar)
		
		ax.set_title(metric_label, fontsize=13, fontweight='bold', pad=10)
		ax.set_ylabel(f"Mean ± {errorbar_type.upper()}", fontsize=11)
		ax.set_xticks(x_positions)
		ax.set_xticklabels([estimator_display_names.get(e, e) for e in estimator_values], fontsize=11)
		ax.grid(axis="y", linestyle="--", alpha=0.35)
		ax.tick_params(axis='x', which='major', labelsize=11)
		
		# Set appropriate y-limits
		if "order_accuracy" in metric_column or "spearman" in metric_column:
			ax.set_ylim(0.0, 1.05)
		
		# Add subplot label (A, B)
		subplot_labels = ['A', 'B']
		ax.text(-0.15, 1.08, subplot_labels[idx], transform=ax.transAxes,
		        fontsize=15, fontweight='bold', va='top', ha='right')
	
	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)
	print(f"Created empirical method comparison plot: {output_path}")
	return output_path


def build_summary_plots(summary_df, output_dir, errorbar_type, comparison_threshold=3.0, threshold_min=None, threshold_max=None):
	if summary_df.empty:
		return []

	os.makedirs(output_dir, exist_ok=True)
	plot_paths = []

	# Create method comparison plot at specific threshold (if specified)
	if comparison_threshold is not None:
		# Population-based plot
		comparison_plot_path = os.path.join(output_dir, f"method_comparison_threshold_{comparison_threshold:.1f}_population.png")
		try:
			generated_path = plot_method_comparison_at_threshold(
				summary_df=summary_df,
				threshold_value=comparison_threshold,
				output_path=comparison_plot_path,
				errorbar_type=errorbar_type,
			)
			if generated_path is not None:
				plot_paths.append(generated_path)
		except Exception as exc:
			print(f"Warning: failed to create population-based method comparison plot: {exc}")

		# Empirical-based plot
		empirical_plot_path = os.path.join(output_dir, f"method_comparison_threshold_{comparison_threshold:.1f}_empirical.png")
		try:
			generated_path = plot_method_comparison_empirical(
				summary_df=summary_df,
				threshold_value=comparison_threshold,
				output_path=empirical_plot_path,
				errorbar_type=errorbar_type,
			)
			if generated_path is not None:
				plot_paths.append(generated_path)
		except Exception as exc:
			print(f"Warning: failed to create empirical-based method comparison plot: {exc}")

	# Create averaged method comparison plots across threshold range
	if threshold_min is not None or threshold_max is not None:
		# Population-based averaged plot
		population_metrics = [
			("mean_abs_relative_error", "Mean Absolute Relative Error"),
			("pairwise_order_accuracy", "Pairwise Order Accuracy")
		]
		averaged_population_path = os.path.join(output_dir, "method_comparison_threshold_averaged_population.png")
		try:
			generated_path = plot_method_comparison_averaged(
				summary_df=summary_df,
				threshold_min=threshold_min,
				threshold_max=threshold_max,
				output_path=averaged_population_path,
				errorbar_type=errorbar_type,
				metrics_to_plot=population_metrics,
			)
			if generated_path is not None:
				plot_paths.append(generated_path)
		except Exception as exc:
			print(f"Warning: failed to create averaged population-based method comparison plot: {exc}")

		# Empirical-based averaged plot
		empirical_metrics = [
			("mean_abs_relative_error_empirical", "Mean Absolute Relative Error (Empirical)"),
			("pairwise_order_accuracy_empirical", "Pairwise Order Accuracy (Empirical)")
		]
		averaged_empirical_path = os.path.join(output_dir, "method_comparison_threshold_averaged_empirical.png")
		try:
			generated_path = plot_method_comparison_averaged(
				summary_df=summary_df,
				threshold_min=threshold_min,
				threshold_max=threshold_max,
				output_path=averaged_empirical_path,
				errorbar_type=errorbar_type,
				metrics_to_plot=empirical_metrics,
			)
			if generated_path is not None:
				plot_paths.append(generated_path)
		except Exception as exc:
			print(f"Warning: failed to create averaged empirical-based method comparison plot: {exc}")

	estimator_plot_path = os.path.join(output_dir, "metrics_by_estimator_bar.png")
	try:
		plot_paths.append(
			plot_metrics_by_estimator_bar(
				summary_df=summary_df,
				output_path=estimator_plot_path,
				errorbar_type=errorbar_type,
			)
		)
	except Exception as exc:
		print(f"Warning: failed to create estimator comparison plot: {exc}")

	for metric_column, metric_label in THRESHOLD_METRICS_TO_PLOT:
		metric_plot_path = os.path.join(output_dir, f"{metric_column}_by_threshold_bar.png")
		try:
			generated_path = plot_metric_by_threshold_bar(
				summary_df=summary_df,
				metric_column=metric_column,
				metric_label=metric_label,
				output_path=metric_plot_path,
				errorbar_type=errorbar_type,
			)
			if generated_path is not None:
				plot_paths.append(generated_path)
		except Exception as exc:
			print(f"Warning: failed to create threshold plot for {metric_column}: {exc}")

	return plot_paths


def process_dataset(
	dataset_num,
	base_data_folder,
	base_results_folder,
	estimator_subdirs,
	output_subdir,
	relative_eps,
	pairwise_true_diff_tol,
	pairwise_max_pairs,
	pairwise_seed,
	threshold_min,
	threshold_max,
):
	dataset_data_folder = os.path.join(base_data_folder, str(dataset_num))
	params_file = os.path.join(dataset_data_folder, "SATAY_without_pi_params.csv")
	pi_file = os.path.join(dataset_data_folder, "pi_values.csv")
	
	# Try both naming conventions: "dataset_X" and "X"
	dataset_results_folder = os.path.join(base_results_folder, str(dataset_num))
	if not os.path.isdir(dataset_results_folder):
		dataset_results_folder = os.path.join(base_results_folder, f"dataset_{dataset_num}")

	if not os.path.isfile(params_file):
		print(f"Skipping dataset {dataset_num}: missing params file {params_file}")
		return []

	if not os.path.isdir(dataset_results_folder):
		print(f"Skipping dataset {dataset_num}: missing result folder {dataset_results_folder}")
		return []

	params_df = read_true_region_params(params_file)
	pi_values = read_true_pi_values(pi_file)
	if pi_values is None:
		print(f"Warning: dataset {dataset_num} missing pi_values.csv - empirical metrics will be unavailable")
	
	summary_rows = []
	detail_files_written = 0
	filtered_out_files = 0

	window_folders = [
		name
		for name in sorted(os.listdir(dataset_results_folder))
		if os.path.isdir(os.path.join(dataset_results_folder, name)) and name.startswith("window")
	]

	for window_name in window_folders:
		window_folder = os.path.join(dataset_results_folder, window_name)

		for estimator_subdir in estimator_subdirs:
			estimator_folder = os.path.join(window_folder, estimator_subdir)
			if not os.path.isdir(estimator_folder):
				continue

			segment_files = [
				name
				for name in sorted(os.listdir(estimator_folder))
				if name.endswith(".csv")
			]

			for segment_name in segment_files:
				segment_file = os.path.join(estimator_folder, segment_name)

				try:
					segment_preview_df = pd.read_csv(segment_file, nrows=1)
				except Exception as exc:
					print(f"Skipping file due to error while reading header row: {segment_file}")
					print(f"  Error: {exc}")
					continue

				threshold_value = extract_threshold_value(segment_preview_df, segment_name)
				if not threshold_is_selected(
					threshold_value=threshold_value,
					threshold_min=threshold_min,
					threshold_max=threshold_max,
				):
					filtered_out_files += 1
					continue

				detail_output_dir = os.path.join(
					window_folder,
					output_subdir,
					estimator_subdir,
				)

				try:
					metrics, detail_path = evaluate_segment_file(
						segment_file=segment_file,
						params_df=params_df,
						pi_values=pi_values,
						output_dir=detail_output_dir,
						relative_eps=relative_eps,
						pairwise_true_diff_tol=pairwise_true_diff_tol,
						pairwise_max_pairs=pairwise_max_pairs,
						pairwise_seed=pairwise_seed,
					)
				except Exception as exc:
					print(f"Skipping file due to error: {segment_file}")
					print(f"  Error: {exc}")
					continue

				metadata = extract_metadata(
					segment_df=segment_preview_df,
					dataset_num=dataset_num,
					estimator_subdir=estimator_subdir,
					segment_file_name=segment_name,
					window_name=window_name,
				)

				row = {}
				row.update(metadata)
				row.update(metrics)
				row["detail_file"] = detail_path
				summary_rows.append(row)
				detail_files_written += 1

	print(
		f"Dataset {dataset_num}: evaluated {len(summary_rows)} files, "
		f"wrote {detail_files_written} detailed performance files, "
		f"filtered out {filtered_out_files} files by threshold range"
	)
	return summary_rows


def parse_arguments():
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate segment-level mu estimates against true synthetic SATAY region means "
			"using overlap-weighted ground truth and ordering metrics."
		)
	)
	parser.add_argument(
		"--base_data_folder",
		type=str,
		default="Data/SATAY_synthetic",
		help="Folder containing SATAY synthetic datasets (1..10).",
	)
	parser.add_argument(
		"--base_results_folder",
		type=str,
		default="Signal_processing/results_new/essentiality_score",
		help="Folder containing segment_mu outputs from pure_estimation/informed_estimation.",
	)
	parser.add_argument(
		"--datasets",
		type=int,
		nargs="+",
		default=list(range(1, 11)),
		help="Dataset numbers to evaluate.",
	)
	parser.add_argument(
		"--estimator_subdirs",
		type=str,
		nargs="+",
		default=DEFAULT_ESTIMATOR_SUBDIRS,
		help="Subdirectories inside each window folder containing segment estimates.",
	)
	parser.add_argument(
		"--output_subdir",
		type=str,
		default="performance",
		help="Subfolder inside each window folder where detailed performance files are written.",
	)
	parser.add_argument(
		"--summary_filename",
		type=str,
		default="performance_summary.csv",
		help="Filename for the global summary CSV in <base_results_folder>/<output_subdir>/.",
	)
	parser.add_argument(
		"--aggregated_summary_filename",
		type=str,
		default="performance_summary_by_estimator.csv",
		help="Filename for estimator-level aggregated summary CSV.",
	)
	parser.add_argument(
		"--relative_eps",
		type=float,
		default=1e-4,
		help="Minimum |true_mu| used for relative error calculations.",
	)
	parser.add_argument(
		"--pairwise_true_diff_tol",
		type=float,
		default=1e-2,
		help="Tolerance under which true-mu differences are treated as ties for order accuracy.",
	)
	parser.add_argument(
		"--pairwise_max_pairs",
		type=int,
		default=1000000,
		help="Maximum pair count for pairwise order accuracy (sampled if exceeded).",
	)
	parser.add_argument(
		"--pairwise_seed",
		type=int,
		default=0,
		help="Random seed used when pairwise order accuracy sampling is required.",
	)
	parser.add_argument(
		"--threshold_min",
		type=float,
		default=0,
		help="Inclusive lower bound for threshold filtering (optional).",
	)
	parser.add_argument(
		"--threshold_max",
		type=float,
		default=10,
		help="Inclusive upper bound for threshold filtering (optional).",
	)
	parser.add_argument(
		"--skip_plots",
		action="store_true",
		help="Skip generation of summary bar plots.",
	)
	parser.add_argument(
		"--plot_output_subdir",
		type=str,
		default="plots",
		help="Subfolder under output_subdir where plot images are written.",
	)
	parser.add_argument(
		"--errorbar_type",
		type=str,
		choices=["std", "sem"],
		default="std",
		help="Error-bar type used in plots: standard deviation (std) or standard error (sem).",
	)
	parser.add_argument(
		"--comparison_threshold",
		type=float,
		default=3,
		help="Threshold value for method comparison plot (default: 3.0). Set to None to skip.",
	)
	return parser.parse_args()


def main():
	args = parse_arguments()

	base_data_folder = resolve_path(args.base_data_folder)
	base_results_folder = resolve_path(args.base_results_folder)
	threshold_min, threshold_max = normalize_threshold_bounds(
		args.threshold_min,
		args.threshold_max,
	)

	if threshold_min is not None or threshold_max is not None:
		low_text = "-inf" if threshold_min is None else f"{threshold_min:g}"
		high_text = "inf" if threshold_max is None else f"{threshold_max:g}"
		print(f"Applying threshold filter (inclusive): [{low_text}, {high_text}]")

	all_rows = []
	for dataset_num in args.datasets:
		all_rows.extend(
			process_dataset(
				dataset_num=dataset_num,
				base_data_folder=base_data_folder,
				base_results_folder=base_results_folder,
				estimator_subdirs=args.estimator_subdirs,
				output_subdir=args.output_subdir,
				relative_eps=args.relative_eps,
				pairwise_true_diff_tol=args.pairwise_true_diff_tol,
				pairwise_max_pairs=args.pairwise_max_pairs,
				pairwise_seed=args.pairwise_seed,
				threshold_min=threshold_min,
				threshold_max=threshold_max,
			)
		)

	summary_output_dir = os.path.join(base_results_folder, args.output_subdir)
	os.makedirs(summary_output_dir, exist_ok=True)

	if not all_rows:
		print("No performance rows were generated. Check folder paths and input files.")
		return

	summary_df = pd.DataFrame(all_rows)
	sort_columns = [
		column
		for column in ["dataset_num", "window_name", "estimator_type", "threshold", "segment_file"]
		if column in summary_df.columns
	]
	if sort_columns:
		summary_df = summary_df.sort_values(sort_columns).reset_index(drop=True)

	summary_path = os.path.join(summary_output_dir, args.summary_filename)
	summary_df.to_csv(summary_path, index=False)

	aggregated_df = build_aggregated_estimator_summary(summary_df)
	aggregated_path = os.path.join(summary_output_dir, args.aggregated_summary_filename)
	aggregated_df.to_csv(aggregated_path, index=False)

	plot_paths = []
	if not args.skip_plots:
		plot_output_dir = os.path.join(summary_output_dir, args.plot_output_subdir)
		plot_paths = build_summary_plots(
			summary_df=summary_df,
			output_dir=plot_output_dir,
			errorbar_type=args.errorbar_type,
			comparison_threshold=args.comparison_threshold,
			threshold_min=threshold_min,
			threshold_max=threshold_max,
		)

	print(f"Done. Wrote global performance summary to {summary_path}")
	print(f"Done. Wrote aggregated estimator summary to {aggregated_path}")
	if plot_paths:
		print("Done. Wrote summary plots:")
		for path in plot_paths:
			print(f"  - {path}")


if __name__ == "__main__":
	main()
