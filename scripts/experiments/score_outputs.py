from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


CORE_FUSION_SCORE_KEYS = ("raw_max", "zscore_mean", "cdf_mean", "cdf_max")
EXTRA_FUSION_SCORE_KEYS = ("cdf_mean_soft_support", "cdf_softmax")
PRIMARY_SCORE_KEY = "cdf_mean"
REQUIRED_REPORT_METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "point_best_f1",
    "pa_best_f1",
    "aff_precision",
    "aff_recall",
    "aff_f1",
    "vus_roc",
    "vus_pr",
)

_NON_SCORE_KEYS = {
    "dataset_metadata",
    "evaluation_protocol",
    "point_coverage_ratio",
    "runtime_seconds",
    "schema_version",
    "score_files",
    "sequence_coverage",
    "sequence_evaluation_mode",
    "sequence_score_aggregation",
    "strategy",
    "use_context_retrieval",
    "use_state_filtering",
    "weak_pointwise",
}


def _is_metric_group(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        key in value for key in ("roc_auc", "pr_auc", "vus_roc", "vus_pr", "best_f1")
    )


def score_metric_groups(
    payload: Mapping[str, Any],
    *,
    require_core: bool = True,
    require_report_metrics: bool = False,
    require_finite_report_metrics: bool = False,
) -> dict[str, Mapping[str, Any]]:
    """Return fusion and diagnostic metric groups from trainer or strategy JSON."""
    root = payload.get("scores")
    if not isinstance(root, Mapping):
        root = payload

    groups: dict[str, Mapping[str, Any]] = {}
    for key in (*CORE_FUSION_SCORE_KEYS, *EXTRA_FUSION_SCORE_KEYS):
        value = root.get(key)
        if _is_metric_group(value):
            groups[key] = value

    subscores = root.get("subscores")
    if isinstance(subscores, Mapping):
        for key, value in subscores.items():
            if _is_metric_group(value):
                groups[str(key)] = value

    # Strategy runs expose diagnostic scores as peers inside ``scores``.
    reserved = {
        *CORE_FUSION_SCORE_KEYS,
        *EXTRA_FUSION_SCORE_KEYS,
        "selected",
        "subscores",
        *_NON_SCORE_KEYS,
    }
    for key, value in root.items():
        if key not in reserved and _is_metric_group(value):
            groups[str(key)] = value

    if require_core:
        missing = [key for key in CORE_FUSION_SCORE_KEYS if key not in groups]
        if missing:
            raise KeyError(f"missing required SCAR score metrics: {missing}")
    if require_report_metrics:
        incomplete = {
            score_key: [
                metric_key
                for metric_key in REQUIRED_REPORT_METRIC_KEYS
                if metric_key not in metrics
            ]
            for score_key, metrics in groups.items()
        }
        incomplete = {key: value for key, value in incomplete.items() if value}
        if incomplete:
            raise KeyError(f"incomplete SCAR score metrics: {incomplete}")
    if require_finite_report_metrics:
        nonfinite = {
            score_key: [
                metric_key
                for metric_key in REQUIRED_REPORT_METRIC_KEYS
                if not _is_finite_number(metrics.get(metric_key))
            ]
            for score_key, metrics in groups.items()
        }
        nonfinite = {key: value for key, value in nonfinite.items() if value}
        if nonfinite:
            raise ValueError(f"non-finite SCAR score metrics: {nonfinite}")
    return groups


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def require_finite_report_metrics(
    metrics: Mapping[str, Any], *, context: str = "metrics"
) -> None:
    invalid = [
        key
        for key in REQUIRED_REPORT_METRIC_KEYS
        if not _is_finite_number(metrics.get(key))
    ]
    if invalid:
        raise ValueError(f"{context} has missing or non-finite report metrics: {invalid}")


def flatten_score_metrics(
    payload: Mapping[str, Any],
    *,
    require_core: bool = True,
    require_report_metrics: bool = True,
) -> dict[str, Any]:
    groups = score_metric_groups(
        payload,
        require_core=require_core,
        require_report_metrics=require_report_metrics,
    )
    selected = payload.get("selected")
    if not _is_metric_group(selected):
        root = payload.get("scores")
        selected = root.get("selected") if isinstance(root, Mapping) else None
    if not _is_metric_group(selected) and _is_metric_group(payload):
        selected = payload

    if require_report_metrics and _is_metric_group(selected):
        missing_selected = [
            key for key in REQUIRED_REPORT_METRIC_KEYS if key not in selected
        ]
        if missing_selected:
            raise KeyError(
                f"incomplete selected SCAR score metrics: {missing_selected}"
            )

    result: dict[str, Any] = {
        "selected_score_key": payload.get("selected_score_key", PRIMARY_SCORE_KEY)
    }
    if _is_metric_group(selected):
        for key, value in selected.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[str(key)] = value
    for score_key, metrics in groups.items():
        for metric_key, value in metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[f"{score_key}_{metric_key}"] = value
    return result
