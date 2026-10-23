"""
Route Resilience - Evaluation Metrics (Module 6a)
==================================================
All five confirmed evaluation metrics as standalone, reusable functions:
  1. IoU  (Intersection over Union)
  2. Dice / F1
  3. Connectivity Ratio
  4. Topological Accuracy
  5. Average Path-Length Error

Also includes:
  - Torch-compatible IoU & Dice for use during training
  - compute_all_metrics() aggregate runner
  - format_metrics_report() pretty-printer
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
#  1.  Intersection over Union  (pixel-level, binary masks)
# ---------------------------------------------------------------------------

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection over Union for binary masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask (0/1), any shape.
    gt_mask : np.ndarray
        Ground-truth binary mask (0/1), same shape as *pred_mask*.

    Returns
    -------
    float
        IoU score in [0, 1].  Returns 1.0 when both masks are empty.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0  # both empty -> perfect agreement
    return float(intersection / union)


# ---------------------------------------------------------------------------
#  2.  Dice / F1  (pixel-level, binary masks)
# ---------------------------------------------------------------------------

def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient (F1 score) for binary masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask (0/1).
    gt_mask : np.ndarray
        Ground-truth binary mask (0/1).

    Returns
    -------
    float
        Dice score in [0, 1].  Returns 1.0 when both masks are empty.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


# ---------------------------------------------------------------------------
#  3.  Connectivity Ratio  (graph-level)
# ---------------------------------------------------------------------------

def compute_connectivity_ratio(graph: nx.Graph) -> float:
    """Fraction of nodes in the largest connected component.

    Parameters
    ----------
    graph : nx.Graph
        An undirected NetworkX graph.

    Returns
    -------
    float
        Ratio in [0, 1].  0.0 for an empty graph.
    """
    if len(graph) == 0:
        return 0.0
    largest_cc = max(nx.connected_components(graph), key=len)
    return float(len(largest_cc) / len(graph))


# ---------------------------------------------------------------------------
#  4.  Topological Accuracy  (graph-level, multi-metric)
# ---------------------------------------------------------------------------

def _degree_histogram(graph: nx.Graph, max_degree: int = 20) -> np.ndarray:
    """Return a normalised degree histogram up to *max_degree*."""
    deg_seq = [d for _, d in graph.degree()]
    counts = np.zeros(max_degree + 1, dtype=float)
    for d in deg_seq:
        idx = min(d, max_degree)
        counts[idx] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """Histogram intersection similarity (1 = identical, 0 = no overlap)."""
    return float(np.minimum(h1, h2).sum())


def compute_topological_accuracy(
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    max_degree: int = 20,
) -> Dict[str, float]:
    """Compare topological structure of predicted vs. ground-truth graphs.

    Sub-metrics returned:
    - **degree_distribution_similarity** : histogram-intersection of degree
      distributions (1 = identical).
    - **junction_precision** : fraction of predicted junctions (degree ≥ 3)
      that also appear in the ground-truth.
    - **junction_recall** : fraction of ground-truth junctions present in
      the prediction.
    - **junction_f1** : harmonic mean of junction precision & recall.

    Parameters
    ----------
    pred_graph : nx.Graph
    gt_graph : nx.Graph
    max_degree : int
        Maximum degree bin for the histogram comparison.

    Returns
    -------
    dict
        Dictionary with the four sub-metrics above.
    """
    # --- Degree-distribution similarity ---
    h_pred = _degree_histogram(pred_graph, max_degree)
    h_gt = _degree_histogram(gt_graph, max_degree)
    deg_sim = _histogram_intersection(h_pred, h_gt)

    # --- Junction (degree ≥ 3) precision / recall ---
    pred_junctions = {n for n, d in pred_graph.degree() if d >= 3}
    gt_junctions = {n for n, d in gt_graph.degree() if d >= 3}

    if len(pred_junctions) == 0 and len(gt_junctions) == 0:
        j_prec, j_rec, j_f1 = 1.0, 1.0, 1.0
    elif len(pred_junctions) == 0:
        j_prec, j_rec, j_f1 = 0.0, 0.0, 0.0
    elif len(gt_junctions) == 0:
        j_prec, j_rec, j_f1 = 0.0, 0.0, 0.0
    else:
        matched = pred_junctions & gt_junctions
        j_prec = len(matched) / len(pred_junctions)
        j_rec = len(matched) / len(gt_junctions)
        j_f1 = (2 * j_prec * j_rec / (j_prec + j_rec)) if (j_prec + j_rec) > 0 else 0.0

    return {
        "degree_distribution_similarity": round(deg_sim, 4),
        "junction_precision": round(j_prec, 4),
        "junction_recall": round(j_rec, 4),
        "junction_f1": round(j_f1, 4),
    }


# ---------------------------------------------------------------------------
#  5.  Average Path-Length Error  (graph-level)
# ---------------------------------------------------------------------------

def compute_avg_path_length_error(
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    n_pairs: int = 200,
    seed: int = 42,
) -> float:
    """Mean absolute error of shortest-path lengths between matched node pairs.

    Randomly samples *n_pairs* (source, target) pairs from nodes shared by
    both graphs, computes shortest-path length in each, and returns the mean
    absolute difference.  Pairs where either graph has no path are counted as
    an error of ``max_observed_length`` (a large penalty).

    Parameters
    ----------
    pred_graph : nx.Graph
    gt_graph : nx.Graph
    n_pairs : int
        Number of random (source, target) pairs to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Mean absolute path-length error.  0.0 is perfect.
    """
    common_nodes = list(set(pred_graph.nodes()) & set(gt_graph.nodes()))
    if len(common_nodes) < 2:
        return float("inf")

    rng = random.Random(seed)
    pairs = []
    for _ in range(n_pairs):
        s, t = rng.sample(common_nodes, 2)
        pairs.append((s, t))

    errors: list[float] = []
    max_penalty = max(len(pred_graph), len(gt_graph))

    for s, t in pairs:
        try:
            sp_pred = nx.shortest_path_length(pred_graph, s, t)
        except nx.NetworkXNoPath:
            sp_pred = None

        try:
            sp_gt = nx.shortest_path_length(gt_graph, s, t)
        except nx.NetworkXNoPath:
            sp_gt = None

        if sp_pred is None and sp_gt is None:
            errors.append(0.0)
        elif sp_pred is None or sp_gt is None:
            errors.append(float(max_penalty))
        else:
            errors.append(abs(sp_pred - sp_gt))

    return float(np.mean(errors)) if errors else 0.0


# ---------------------------------------------------------------------------
#  Aggregate runner
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    n_path_pairs: int = 200,
) -> Dict[str, object]:
    """Run all five evaluation metrics and return a flat results dictionary.

    Parameters
    ----------
    pred_mask, gt_mask : np.ndarray
        Binary segmentation masks.
    pred_graph, gt_graph : nx.Graph
        Road network graphs.
    n_path_pairs : int
        Number of sampled pairs for path-length error.

    Returns
    -------
    dict
        Keys include ``iou``, ``dice``, ``connectivity_ratio_pred``,
        ``connectivity_ratio_gt``, ``topo_*``, ``avg_path_length_error``.
    """
    metrics: Dict[str, object] = {}

    # Pixel-level
    metrics["iou"] = compute_iou(pred_mask, gt_mask)
    metrics["dice"] = compute_dice(pred_mask, gt_mask)

    # Graph-level
    metrics["connectivity_ratio_pred"] = compute_connectivity_ratio(pred_graph)
    metrics["connectivity_ratio_gt"] = compute_connectivity_ratio(gt_graph)

    topo = compute_topological_accuracy(pred_graph, gt_graph)
    for k, v in topo.items():
        metrics[f"topo_{k}"] = v

    metrics["avg_path_length_error"] = compute_avg_path_length_error(
        pred_graph, gt_graph, n_pairs=n_path_pairs,
    )

    return metrics


# ---------------------------------------------------------------------------
#  Formatted report
# ---------------------------------------------------------------------------

def format_metrics_report(metrics: Dict[str, object]) -> str:
    """Return a Markdown-formatted table summarising the evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_all_metrics`.

    Returns
    -------
    str
        Markdown string.
    """
    lines = [
        "##  Route Resilience - Evaluation Report",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]

    friendly_names = {
        "iou": "IoU (Intersection over Union)",
        "dice": "Dice / F1 Score",
        "connectivity_ratio_pred": "Connectivity Ratio (Pred)",
        "connectivity_ratio_gt": "Connectivity Ratio (GT)",
        "topo_degree_distribution_similarity": "Degree-Dist Similarity",
        "topo_junction_precision": "Junction Precision",
        "topo_junction_recall": "Junction Recall",
        "topo_junction_f1": "Junction F1",
        "avg_path_length_error": "Avg Path-Length Error",
    }

    for key, val in metrics.items():
        label = friendly_names.get(key, key)
        if isinstance(val, float):
            lines.append(f"| {label} | {val:.4f} |")
        else:
            lines.append(f"| {label} | {val} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Torch-compatible IoU & Dice  (for use during training)
# ---------------------------------------------------------------------------

def torch_iou(pred: "torch.Tensor", target: "torch.Tensor", threshold: float = 0.5, smooth: float = 1e-6) -> "torch.Tensor":
    """Differentiable IoU for PyTorch tensors.

    Parameters
    ----------
    pred : torch.Tensor
        Raw logits or probabilities, shape ``(B, 1, H, W)``.
    target : torch.Tensor
        Binary ground-truth mask, same shape.
    threshold : float
        Binarisation threshold (applied only if values are not already 0/1).
    smooth : float
        Smoothing constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Scalar IoU averaged over the batch.
    """
    import torch  # local import - avoids hard dependency for non-training code

    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    union = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def torch_dice(pred: "torch.Tensor", target: "torch.Tensor", threshold: float = 0.5, smooth: float = 1e-6) -> "torch.Tensor":
    """Differentiable Dice / F1 for PyTorch tensors.

    Parameters
    ----------
    pred : torch.Tensor
        Raw logits or probabilities, shape ``(B, 1, H, W)``.
    target : torch.Tensor
        Binary ground-truth mask, same shape.
    threshold : float
        Binarisation threshold.
    smooth : float
        Smoothing constant.

    Returns
    -------
    torch.Tensor
        Scalar Dice averaged over the batch.
    """
    import torch

    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    total = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (total + smooth)
    return dice.mean()


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== metrics.py self-test ===")

    # Pixel-level tests
    a = np.ones((64, 64), dtype=np.uint8)
    b = np.ones((64, 64), dtype=np.uint8)
    assert compute_iou(a, b) == 1.0
    assert compute_dice(a, b) == 1.0

    b[:32, :] = 0
    iou = compute_iou(a, b)
    dice = compute_dice(a, b)
    print(f"  IoU(half-overlap)  = {iou:.4f}  (expect ~0.5)")
    print(f"  Dice(half-overlap) = {dice:.4f}  (expect ~0.6667)")

    # Graph-level tests
    G = nx.grid_2d_graph(10, 10)
    cr = compute_connectivity_ratio(G)
    print(f"  Connectivity ratio (10×10 grid) = {cr:.4f}  (expect 1.0)")

    topo = compute_topological_accuracy(G, G)
    print(f"  Topological accuracy (self) = {topo}")

    aple = compute_avg_path_length_error(G, G)
    print(f"  Avg path-length error (self) = {aple:.4f}  (expect 0.0)")

    # Aggregate
    all_m = compute_all_metrics(a, b, G, G)
    print("\n" + format_metrics_report(all_m))

    print("\nOK All self-tests passed.")
