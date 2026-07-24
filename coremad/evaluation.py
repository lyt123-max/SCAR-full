from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

try:
    from .temporal_metrics import HAS_TEMPORAL_METRICS, compute_temporal_metrics
except ImportError:
    _temporal_path = Path(__file__).with_name("temporal_metrics.py")
    _temporal_spec = importlib.util.spec_from_file_location(
        "_coremad_temporal_metrics", _temporal_path
    )
    if _temporal_spec is None or _temporal_spec.loader is None:
        raise ImportError(f"Cannot load temporal metrics from {_temporal_path}.")
    _temporal_module = importlib.util.module_from_spec(_temporal_spec)
    _temporal_spec.loader.exec_module(_temporal_module)
    HAS_TEMPORAL_METRICS = _temporal_module.HAS_TEMPORAL_METRICS
    compute_temporal_metrics = _temporal_module.compute_temporal_metrics


def _anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(labels, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _point_adjusted_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    segments = _anomaly_segments(labels)
    if not segments:
        return {
            "pa_best_f1": float("nan"),
            "pa_best_threshold": float("nan"),
        }
    normal_sorted = np.sort(scores[labels == 0])
    segment_max = np.asarray([scores[start:end].max() for start, end in segments])
    segment_length = np.asarray([end - start for start, end in segments], dtype=np.float64)
    positive_points = float(labels.sum())
    thresholds = np.unique(scores.astype(np.float64))
    false_positive = normal_sorted.size - np.searchsorted(
        normal_sorted, thresholds, side="left"
    )

    segment_order = np.argsort(segment_max, kind="mergesort")
    sorted_segment_max = segment_max[segment_order]
    sorted_segment_length = segment_length[segment_order]
    prefix_length = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(sorted_segment_length)]
    )
    first_detected = np.searchsorted(sorted_segment_max, thresholds, side="left")
    true_positive = positive_points - prefix_length[first_detected]

    precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
    recall = true_positive / max(positive_points, 1e-12)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.nanargmax(f1))
    return {
        "pa_best_f1": float(f1[best_index]),
        "pa_best_threshold": float(thresholds[best_index]),
    }


def _temporal_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray | None = None,
) -> dict[str, float]:
    empty = {
        "aff_precision": float("nan"),
        "aff_recall": float("nan"),
        "aff_f1": float("nan"),
        "vus_roc": float("nan"),
        "vus_pr": float("nan"),
        "vus_window": float("nan"),
    }
    if not HAS_TEMPORAL_METRICS:
        return empty
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return empty
    minimum = float(finite.min())
    maximum = float(finite.max())
    normalized = (
        np.zeros_like(scores, dtype=np.float64)
        if maximum - minimum < 1e-12
        else np.clip((scores - minimum) / (maximum - minimum), 0.0, 1.0)
    )
    lengths = [end - start for start, end in _anomaly_segments(labels)]
    vus_window = max(1, int(np.median(lengths))) if lengths else 1
    try:
        result = compute_temporal_metrics(
            labels,
            normalized,
            sliding_window=vus_window,
            predictions=predictions,
        )
    except Exception:
        return empty
    return {
        "aff_precision": result["aff_precision"],
        "aff_recall": result["aff_recall"],
        "aff_f1": result["aff_f1"],
        "vus_roc": result["vus_roc"],
        "vus_pr": result["vus_pr"],
        "vus_window": float(vus_window),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def binary_point_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    labels_array = np.asarray(labels).reshape(-1)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have the same shape.")
    mask = np.isfinite(scores_array)
    if evaluation_mask is not None:
        requested_mask = np.asarray(evaluation_mask, dtype=bool).reshape(-1)
        if requested_mask.shape != mask.shape:
            raise ValueError("evaluation_mask must align with labels.")
        mask &= requested_mask
    y = (labels_array[mask] > 0).astype(np.int8)
    score = scores_array[mask]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires at least one positive and one negative sample.")

    ranks = _average_ranks(score)
    roc_auc = (
        float(ranks[y == 1].sum()) - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order]
    sorted_score = score[order]
    true_positives = np.cumsum(sorted_y)
    distinct_ends = np.concatenate(
        [np.flatnonzero(np.diff(sorted_score) != 0), np.asarray([len(y) - 1])]
    )
    true_positives = true_positives[distinct_ends]
    precision = true_positives / (distinct_ends + 1)
    recall = true_positives / positives
    pr_auc = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1))
    result = {
        "roc_auc": float(roc_auc),
        "pr_auc": pr_auc,
        "best_f1": float(f1[best_index]),
        "point_best_f1": float(f1[best_index]),
        "precision_at_best_f1": float(precision[best_index]),
        "recall_at_best_f1": float(recall[best_index]),
        "threshold_at_best_f1": float(sorted_score[distinct_ends[best_index]]),
        "n_evaluated": int(len(y)),
        "n_positive": positives,
        "n_negative": negatives,
    }
    threshold = float(sorted_score[distinct_ends[best_index]])
    predictions = (score >= threshold).astype(np.int8)
    result.update(_point_adjusted_metrics(y, score))
    result.update(_temporal_metrics(y, score, predictions=predictions))
    return result
