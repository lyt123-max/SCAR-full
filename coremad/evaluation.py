from __future__ import annotations

import numpy as np


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
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": pr_auc,
        "best_f1": float(f1[best_index]),
        "precision_at_best_f1": float(precision[best_index]),
        "recall_at_best_f1": float(recall[best_index]),
        "threshold_at_best_f1": float(sorted_score[distinct_ends[best_index]]),
        "n_evaluated": int(len(y)),
        "n_positive": positives,
        "n_negative": negatives,
    }
