from __future__ import annotations

import inspect
from collections.abc import Mapping

import numpy as np

try:
    from TSB_AD.evaluation.affiliation.generics import convert_vector_to_events
    from TSB_AD.evaluation.affiliation.metrics import pr_from_events
    from TSB_AD.evaluation.metrics import get_metrics as temporal_get_metrics

    HAS_TEMPORAL_METRICS = True
    HAS_AFFILIATION_COMPONENTS = True
except Exception:
    convert_vector_to_events = None
    pr_from_events = None
    HAS_AFFILIATION_COMPONENTS = False
    try:
        from vus.metrics import get_metrics as temporal_get_metrics

        HAS_TEMPORAL_METRICS = True
    except Exception:
        temporal_get_metrics = None
        HAS_TEMPORAL_METRICS = False


def _value(result: Mapping[str, object], *keys: str) -> float:
    for key in keys:
        if key in result:
            return float(result[key])
    return float("nan")


def _affiliation_precision_recall(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    if not HAS_AFFILIATION_COMPONENTS:
        return float("nan"), float("nan")
    events_pred = convert_vector_to_events(predictions.astype(np.int8))
    events_gt = convert_vector_to_events(labels.astype(np.int8))
    result = pr_from_events(events_pred, events_gt, (0, len(predictions)))
    return (
        float(result["Affiliation_Precision"]),
        float(result["Affiliation_Recall"]),
    )


def compute_temporal_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    sliding_window: int,
    predictions: np.ndarray | None = None,
) -> dict[str, float]:
    empty = {
        "aff_precision": float("nan"),
        "aff_recall": float("nan"),
        "aff_f1": float("nan"),
        "r_auc_roc": float("nan"),
        "r_auc_pr": float("nan"),
        "vus_roc": float("nan"),
        "vus_pr": float("nan"),
        "vus_window": float("nan"),
    }
    if not HAS_TEMPORAL_METRICS:
        return empty

    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape.")
    predictions_array = (
        None
        if predictions is None
        else np.asarray(predictions, dtype=np.int32).reshape(-1)
    )
    if predictions_array is not None and predictions_array.shape != labels.shape:
        raise ValueError("predictions must align with labels.")

    parameters = inspect.signature(temporal_get_metrics).parameters
    kwargs: dict[str, object] = {"slidingWindow": max(1, int(sliding_window))}
    if "metric" in parameters:
        kwargs["metric"] = "all"
    if predictions_array is not None and "pred" in parameters:
        kwargs["pred"] = predictions_array
    result = temporal_get_metrics(scores, labels, **kwargs)

    aff_precision = _value(
        result, "Affiliation_Precision", "Affiliation-Precision"
    )
    aff_recall = _value(result, "Affiliation_Recall", "Affiliation-Recall")
    if (
        predictions_array is not None
        and (not np.isfinite(aff_precision) or not np.isfinite(aff_recall))
    ):
        aff_precision, aff_recall = _affiliation_precision_recall(
            labels, predictions_array
        )
    aff_f1 = (
        2.0 * aff_precision * aff_recall
        / max(aff_precision + aff_recall, 1e-12)
        if np.isfinite(aff_precision) and np.isfinite(aff_recall)
        else _value(result, "Affiliation_F", "Affiliation-F")
    )
    return {
        "aff_precision": aff_precision,
        "aff_recall": aff_recall,
        "aff_f1": float(aff_f1),
        "r_auc_roc": _value(result, "R_AUC_ROC", "R-AUC-ROC"),
        "r_auc_pr": _value(result, "R_AUC_PR", "R-AUC-PR"),
        "vus_roc": _value(result, "VUS_ROC", "VUS-ROC"),
        "vus_pr": _value(result, "VUS_PR", "VUS-PR"),
        "vus_window": float(max(1, int(sliding_window))),
    }
