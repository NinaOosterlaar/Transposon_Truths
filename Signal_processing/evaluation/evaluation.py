import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS

# Set up standardized plot style
setup_plot_style()

def calculate_jaccard_index(set_a, set_b):
    """Calculate the Jaccard index between two sets."""
    intersection = 0
    union = 0
    for chrom in set_a.keys():
        intersection += len(set_a[chrom].intersection(set_b.get(chrom, set())))
        union += len(set_a[chrom].union(set_b.get(chrom, set())))
    if union == 0:
        return 0.0
    return intersection / union 

def precision(detected_cps, true_cps, tol):
    """Calculate precision of detected change points."""
    true_positives = 0
    for cp in detected_cps:
        if any(abs(cp - true_cp) <= tol for true_cp in true_cps):
            true_positives += 1
    precision_value = true_positives / len(detected_cps) if detected_cps else 0
    return precision_value

def recall(detected_cps, true_cps, tol):
    """Calculate recall of detected change points."""
    true_positives = 0
    for true_cp in true_cps:
        if any(abs(cp - true_cp) <= tol for cp in detected_cps):
            true_positives += 1
    recall_value = true_positives / len(true_cps) if true_cps else 0
    return recall_value

def F1_score(precision_value, recall_value):
    """Calculate F1 score from precision and recall."""
    if precision_value + recall_value == 0:
        return 0
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)

def annotation_error(pred_cps, true_cps):
    """
    Annotation error = |#predicted change points - #true change points|.

    Parameters
    ----------
    pred_cps : array-like
        Predicted change point locations (ints).
    true_cps : array-like
        Ground-truth change point locations (ints).

    Returns
    -------
    int
        Absolute difference in cardinalities.
    """
    pred_cps = np.asarray(pred_cps, dtype=int).ravel()
    true_cps = np.asarray(true_cps, dtype=int).ravel()
    return int(abs(len(pred_cps) - len(true_cps)))


def hausdorff_distance(true_cps, pred_cps):
    """
    Hausdorff distance between two sets of change points (1D time indices).

    Hausdorff(T*, T^) := max{
        max_{t^ in T^} min_{t* in T*} |t^ - t*|,
        max_{t* in T*} min_{t^ in T^} |t^ - t*|
    }

    Notes
    -----
    - If both sets are empty, returns 0.0.
    - If exactly one set is empty, returns np.inf (distance undefined/inf).

    Parameters
    ----------
    true_cps : array-like
        Ground-truth change point locations (ints).
    pred_cps : array-like
        Predicted change point locations (ints).

    Returns
    -------
    float
        Hausdorff distance.
    """
    true_cps = np.asarray(true_cps, dtype=float).ravel()
    pred_cps = np.asarray(pred_cps, dtype=float).ravel()

    if true_cps.size == 0 and pred_cps.size == 0:
        return 0.0
    if true_cps.size == 0 or pred_cps.size == 0:
        return float(np.inf)

    # Directed distances
    # d(pred -> true)
    d_pt = np.max([np.min(np.abs(t_hat - true_cps)) for t_hat in pred_cps])
    # d(true -> pred)
    d_tp = np.max([np.min(np.abs(t_star - pred_cps)) for t_star in true_cps])

    return float(max(d_pt, d_tp))

def rand_index(true_cps, pred_cps, n_points):
    """
    Rand Index between two segmentations defined by breakpoint (change-point) sets.

    A pair (i, j) is an agreement if:
      - i and j are in the same segment under both segmentations, OR
      - i and j are in different segments under both segmentations.

    RandIndex = (#agreements) / (n_points * (n_points - 1) / 2)

    This implementation is O(n_points) using a contingency table, not O(n_points^2).

    Parameters
    ----------
    true_cps : array-like
        Ground-truth change point locations (ints). Assumed in [1, n_points-1] if 0-based indexing of points.
    pred_cps : array-like
        Predicted change point locations (ints). Same convention as true_cps.
    n_points : int
        Number of indices/positions in the signal (e.g., length of the time series).

    Returns
    -------
    float
        Rand Index in [0, 1]. If n_points < 2, returns 1.0.
    """
    if n_points < 2:
        return 1.0

    true_labels = _segmentation_labels_from_cps(true_cps, n_points)
    pred_labels = _segmentation_labels_from_cps(pred_cps, n_points)

    # Build contingency table between predicted segments and true segments
    pred_ids, pred_inv = np.unique(pred_labels, return_inverse=True)
    true_ids, true_inv = np.unique(true_labels, return_inverse=True)

    k = pred_ids.size
    m = true_ids.size
    contingency = np.zeros((k, m), dtype=np.int64)
    np.add.at(contingency, (pred_inv, true_inv), 1)

    # Helper: nC2
    def comb2(x):
        x = np.asarray(x, dtype=np.int64)
        return (x * (x - 1)) // 2

    total_pairs = comb2(n_points)

    # Agreements in "same segment" for both = sum over cells of C(n_ij, 2)
    tp = comb2(contingency).sum()

    # Pairs in same segment in predicted = sum over rows C(n_i., 2)
    pred_same = comb2(contingency.sum(axis=1)).sum()

    # Pairs in same segment in true = sum over cols C(n_.j, 2)
    true_same = comb2(contingency.sum(axis=0)).sum()

    fp = pred_same - tp
    fn = true_same - tp
    tn = total_pairs - tp - fp - fn

    return float((tp + tn) / total_pairs)


def adjusted_rand_index(true_cps, pred_cps, n_points):
    """
    Adjusted Rand Index between two segmentations defined by breakpoint sets.
    
    The Adjusted Rand Index (ARI) corrects the Rand Index for chance agreement.
    It ranges from -1 to 1, where:
      - 1.0 = perfect agreement
      - 0.0 = random labeling
      - negative values = worse than random
    
    This is preferred over Rand Index as it doesn't suffer from high values
    when there are many small segments.

    Parameters
    ----------
    true_cps : array-like
        Ground-truth change point locations (ints).
    pred_cps : array-like
        Predicted change point locations (ints).
    n_points : int
        Number of indices/positions in the signal.

    Returns
    -------
    float
        Adjusted Rand Index in [-1, 1].
    """
    if n_points < 2:
        return 1.0

    true_labels = _segmentation_labels_from_cps(true_cps, n_points)
    pred_labels = _segmentation_labels_from_cps(pred_cps, n_points)

    # Build contingency table
    pred_ids, pred_inv = np.unique(pred_labels, return_inverse=True)
    true_ids, true_inv = np.unique(true_labels, return_inverse=True)

    k = pred_ids.size
    m = true_ids.size
    contingency = np.zeros((k, m), dtype=np.int64)
    np.add.at(contingency, (pred_inv, true_inv), 1)

    # Helper: nC2
    def comb2(x):
        x = np.asarray(x, dtype=np.int64)
        return (x * (x - 1)) // 2

    # Sum over cells
    sum_comb_contingency = comb2(contingency).sum()

    # Sum over rows and columns
    sum_comb_pred = comb2(contingency.sum(axis=1)).sum()
    sum_comb_true = comb2(contingency.sum(axis=0)).sum()

    total_pairs = comb2(n_points)
    
    # Expected index (for chance agreement)
    expected_index = (sum_comb_pred * sum_comb_true) / total_pairs if total_pairs > 0 else 0

    # Max index
    max_index = (sum_comb_pred + sum_comb_true) / 2

    # Adjusted Rand Index
    if max_index == expected_index:
        if sum_comb_contingency == expected_index:
            return 1.0
        else:
            return 0.0
    
    ari = (sum_comb_contingency - expected_index) / (max_index - expected_index)
    return float(ari)


def _segmentation_labels_from_cps(cps, n_points):
    """
    Convert change points to a per-index segment label array of length n_points.

    Convention:
    - Points are indexed 0..n_points-1.
    - A change point at position c means a boundary between c-1 and c
      (i.e., segment changes starting at index c).
      So cps should be integers in [1, n_points-1].

    Returns
    -------
    labels : np.ndarray, shape (n_points,)
        Segment id for each index.
    """
    cps = np.asarray(cps, dtype=int).ravel()
    cps = cps[(cps >= 1) & (cps <= n_points - 1)]
    cps = np.unique(cps)
    cps.sort()

    boundaries = np.concatenate(([0], cps, [n_points]))
    labels = np.empty(n_points, dtype=np.int64)

    seg_id = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        labels[start:end] = seg_id
        seg_id += 1
    return labels

def match_cps_one_to_one(true_cps, pred_cps, tol):
    """
    One-to-one matching between predicted and true CPs within tolerance.

    Strategy:
    - Build all candidate matches (pred, true) with |pred-true| <= tol
    - Sort candidates by distance (then by indices for stability)
    - Greedily assign pairs so each pred and each true is used at most once

    Returns
    -------
    matches : list of (pred_cp, true_cp)
    unmatched_pred : list of pred_cp
    unmatched_true : list of true_cp
    """
    true_cps = np.unique(np.asarray(true_cps, dtype=int).ravel())
    pred_cps = np.unique(np.asarray(pred_cps, dtype=int).ravel())

    # Edge cases
    if true_cps.size == 0 and pred_cps.size == 0:
        return [], [], []
    if true_cps.size == 0:
        return [], pred_cps.tolist(), []
    if pred_cps.size == 0:
        return [], [], true_cps.tolist()

    candidates = []
    for p in pred_cps:
        # Consider only true cps in [p-tol, p+tol]
        lo, hi = p - tol, p + tol
        for t in true_cps[(true_cps >= lo) & (true_cps <= hi)]:
            candidates.append((abs(p - t), p, t))

    # No possible matches
    if not candidates:
        return [], pred_cps.tolist(), true_cps.tolist()

    candidates.sort()  # primarily by distance
    used_p = set()
    used_t = set()
    matches = []

    for _, p, t in candidates:
        if p in used_p or t in used_t:
            continue
        used_p.add(p)
        used_t.add(t)
        matches.append((p, t))

    unmatched_pred = [p for p in pred_cps.tolist() if p not in used_p]
    unmatched_true = [t for t in true_cps.tolist() if t not in used_t]
    return matches, unmatched_pred, unmatched_true


def tp_fp_fn_from_cps(true_cps, pred_cps, tol):
    """
    Compute TP/FP/FN using one-to-one matching within tolerance.
    """
    matches, unmatched_pred, unmatched_true = match_cps_one_to_one(true_cps, pred_cps, tol)
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_true)
    return tp, fp, fn


def _num_positive_boundary_positions(true_cps, n_points, tol):
    """
    Number of boundary locations (between indices) that are considered 'positive'
    given tolerance windows around true change points.

    Boundary positions are assumed to be integers in [1, n_points-1].
    (i.e., a boundary between i-1 and i is represented by i)
    """
    if n_points < 2:
        return 0

    true_cps = np.unique(np.asarray(true_cps, dtype=int).ravel())
    # Clip to valid boundary positions
    true_cps = true_cps[(true_cps >= 1) & (true_cps <= n_points - 1)]

    if true_cps.size == 0:
        return 0

    mask = np.zeros(n_points - 1, dtype=bool)  # indices 1..n_points-1 mapped to 0..n_points-2
    for cp in true_cps:
        lo = max(1, cp - tol)
        hi = min(n_points - 1, cp + tol)
        mask[(lo - 1):hi] = True

    return int(mask.sum())


def roc_curve_from_cps_by_threshold(results, true_cps, n_points, tol):
    """
    Event-level ROC curve from predicted CP sets at different thresholds.

    Parameters
    ----------
    results : list of (threshold, pred_cps)
        threshold: float
        pred_cps: array-like of predicted change points for that threshold
    true_cps : array-like
        Ground-truth change points (boundary positions).
    n_points : int
        Length of the signal (number of indices/positions).
        Candidate boundary positions are 1..n_points-1.
    tol : int
        Matching tolerance (in boundary index units).

    Returns
    -------
    fpr : np.ndarray
    tpr : np.ndarray
    thresholds : np.ndarray (sorted descending)
    """
    true_cps = np.unique(np.asarray(true_cps, dtype=int).ravel())

    # Sort thresholds high -> low (typical ROC convention)
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)

    P = len(true_cps)  # number of true events (CPs)
    pos_boundaries = _num_positive_boundary_positions(true_cps, n_points, tol)
    total_boundaries = max(0, n_points - 1)
    N = total_boundaries - pos_boundaries  # truly negative boundary positions

    thresholds = np.array([thr for thr, _ in results_sorted], dtype=float)
    tpr = np.zeros_like(thresholds, dtype=float)
    fpr = np.zeros_like(thresholds, dtype=float)

    for i, (thr, pred_cps) in enumerate(results_sorted):
        tp, fp, fn = tp_fp_fn_from_cps(true_cps, pred_cps, tol)

        # TPR: event recall
        tpr[i] = tp / P if P > 0 else 0.0

        # FPR: false events / negative boundary opportunities
        fpr[i] = fp / N if N > 0 else 0.0

    return fpr, tpr, thresholds


def auc_trapezoid(fpr, tpr):
    """
    Compute AUC using trapezoidal rule.
    Ensure fpr is sorted ascending for a conventional AUC.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def plot_roc_curve(fpr, tpr, ax=None, label=None):
    """
    Plot ROC curve given fpr and tpr arrays.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    if label is not None:
        ax.legend()
    return ax


def _binary_labels_from_true_cps(n_points, true_cps, tol=0):
    """
    Create boolean array y_true where y_true[i]=True if i is within tol of any true cp.
    """
    y_true = np.zeros(n_points, dtype=bool)
    true_cps = np.unique(true_cps[(true_cps >= 0) & (true_cps < n_points)])
    for cp in true_cps:
        lo = max(0, cp - int(tol))
        hi = min(n_points - 1, cp + int(tol))
        y_true[lo : hi + 1] = True
    return y_true

def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE) between two numeric arrays.

    MAE = mean_i |y_true[i] - y_pred[i]|

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes must match. Got {y_true.shape} vs {y_pred.shape}.")

    return float(np.mean(np.abs(y_true - y_pred)))


def precision_recall_curve(detected_cps_list, precisions, recalls, thresholds=None, 
                           label=None, ax=None, color=None, marker='o', markersize=4, 
                           linewidth=2, alpha=1.0):
    """
    Create a precision-recall curve from lists of detected change points at different thresholds.
    
    This function is designed to work with the output from change point detection algorithms
    where you have precision and recall values computed at different threshold values.
    
    Parameters
    ----------
    detected_cps_list : list of array-like or None
        List of detected change points at each threshold. Can be None if precisions and recalls are provided.
    precisions : array-like
        Precision values corresponding to each threshold.
    recalls : array-like
        Recall values corresponding to each threshold.
    thresholds : array-like or None
        Threshold values used. If None, will be inferred from order.
    label : str or None
        Label for the curve (for legend).
    ax : matplotlib.axes.Axes or None
        Axes object to plot on. If None, will return data without plotting.
    color : str or None
        Color for the curve.
    marker : str
        Marker style for points.
    markersize : float
        Size of markers.
    linewidth : float
        Width of the line.
    alpha : float
        Transparency of the curve.
    
    Returns
    -------
    dict
        Dictionary containing 'precisions', 'recalls', 'thresholds' arrays
    """
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    
    if thresholds is not None:
        thresholds = np.asarray(thresholds)
    
    result = {
        'precisions': precisions,
        'recalls': recalls,
        'thresholds': thresholds
    }
    
    if ax is not None:
        plot_kwargs = {
            'marker': marker,
            'markersize': markersize,
            'linewidth': linewidth,
            'alpha': alpha
        }
        if label is not None:
            plot_kwargs['label'] = label
        if color is not None:
            plot_kwargs['color'] = color
            
        ax.plot(recalls, precisions, **plot_kwargs)
    
    return result


def plot_precision_recall_curves(pr_curves_data, output_path=None, title='Precision-Recall Curve',
                                 xlabel='Recall', ylabel='Precision', figsize=(10, 8)):
    """
    Plot multiple precision-recall curves on the same figure.
    
    Parameters
    ----------
    pr_curves_data : list of dict
        List of dictionaries, each containing:
        - 'recalls': array of recall values
        - 'precisions': array of precision values
        - 'label': str, label for the curve
        - 'color' (optional): color for the curve
    output_path : str or None
        Path to save the figure. If None, figure is not saved.
    title : str
        Title for the plot.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    figsize : tuple
        Figure size (width, height).
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for curve_data in pr_curves_data:
        recalls = curve_data['recalls']
        precisions = curve_data['precisions']
        label = curve_data.get('label', None)
        color = curve_data.get('color', None)
        marker = curve_data.get('marker', 'o')
        markersize = curve_data.get('markersize', 4)
        linewidth = curve_data.get('linewidth', 2)
        
        plot_kwargs = {
            'marker': marker,
            'markersize': markersize,
            'linewidth': linewidth,
        }
        if label is not None:
            plot_kwargs['label'] = label
        if color is not None:
            plot_kwargs['color'] = color
            
        ax.plot(recalls, precisions, **plot_kwargs)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, ax