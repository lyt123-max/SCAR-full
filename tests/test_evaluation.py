from __future__ import annotations

import numpy as np

from coremad.evaluation import binary_point_metrics


def test_binary_point_metrics_perfect_ranking() -> None:
    result = binary_point_metrics(
        labels=np.array([0, 0, 1, 1]),
        scores=np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert result["roc_auc"] == 1.0
    assert result["pr_auc"] == 1.0
    assert result["best_f1"] == 1.0


def test_binary_point_metrics_honors_evaluation_mask() -> None:
    result = binary_point_metrics(
        labels=np.array([0, 1, 0, 1]),
        scores=np.array([0.1, 0.2, 0.9, 0.8]),
        evaluation_mask=np.array([1, 1, 0, 0], dtype=bool),
    )
    assert result["roc_auc"] == 1.0
    assert result["n_evaluated"] == 2


def test_binary_point_metrics_groups_tied_thresholds() -> None:
    result = binary_point_metrics(
        labels=np.array([1, 0]),
        scores=np.array([0.5, 0.5]),
    )
    assert result["roc_auc"] == 0.5
    assert result["pr_auc"] == 0.5
