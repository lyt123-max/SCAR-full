from __future__ import annotations

import math
from typing import Any, Iterable


AXES = ("parameters", "bank_bytes", "latency_ms")


def _relative_error(target: float, actual: float) -> float:
    if target <= 0.0 or actual <= 0.0:
        raise ValueError("Budget values must be positive.")
    return abs(actual - target) / target


def _log_error(target: float, actual: float) -> float:
    if target <= 0.0 or actual <= 0.0:
        raise ValueError("Budget values must be positive.")
    return abs(math.log(actual / target))


def select_budget_candidate(
    target: dict[str, float],
    candidates: Iterable[dict[str, Any]],
    *,
    tolerance: float = 0.15,
) -> dict[str, Any]:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        relative = {
            axis: _relative_error(float(target[axis]), float(candidate[axis]))
            for axis in AXES
        }
        log_error = sum(
            _log_error(float(target[axis]), float(candidate[axis])) for axis in AXES
        )
        rows.append(
            {
                "candidate": dict(candidate),
                "relative_error": relative,
                "log_error_sum": log_error,
                "within_tolerance": all(value <= tolerance for value in relative.values()),
            }
        )
    if not rows:
        raise ValueError("At least one budget candidate is required.")

    eligible = [row for row in rows if row["within_tolerance"]]
    pool = eligible or rows
    selected = min(
        pool,
        key=lambda row: (
            round(float(row["log_error_sum"]), 12),
            float(row["candidate"]["latency_ms"]),
            float(row["candidate"]["bank_bytes"]),
            int(row["candidate"].get("config_size", 0)),
            str(row["candidate"].get("name", "")),
        ),
    )
    return {
        **selected,
        "tolerance": float(tolerance),
        "target": {axis: float(target[axis]) for axis in AXES},
        "candidate_count": len(rows),
    }
