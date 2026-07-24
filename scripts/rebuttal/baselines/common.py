from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.score_outputs import require_finite_report_metrics


def load_common_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = data_dir / "common"
    train = np.asarray(np.load(common / "train.npy"), dtype=np.float32)
    test = np.asarray(np.load(common / "test.npy"), dtype=np.float32)
    labels = (np.asarray(np.load(common / "labels.npy")).reshape(-1) > 0).astype(
        np.int8
    )
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("Prepared train and test arrays must be two-dimensional.")
    if train.shape[1] != test.shape[1] or len(test) != len(labels):
        raise ValueError("Prepared baseline arrays are not aligned.")
    return train, test, labels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def measure_inference(
    function: Callable[[], np.ndarray],
    *,
    warmups: int = 1,
    repeats: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    if warmups != 1 or repeats != 3:
        raise ValueError("Formal timing protocol requires one warmup and three repeats.")
    for _ in range(warmups):
        function()
    durations: list[float] = []
    result: np.ndarray | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        current = np.asarray(function(), dtype=np.float64).reshape(-1)
        durations.append(time.perf_counter() - started)
        if result is None:
            result = current
        elif not np.allclose(result, current, equal_nan=True):
            raise ValueError("Repeated inference produced non-deterministic scores.")
    assert result is not None
    return result, {
        "schema_version": 1,
        "warmup_runs": warmups,
        "timed_runs": repeats,
        "durations_seconds": durations,
        "mean_seconds": float(np.mean(durations)),
        "std_seconds": float(np.std(durations)),
        "median_seconds": float(np.median(durations)),
    }


def _load_binary_metrics():
    path = REPO_ROOT / "coremad" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("scar_baseline_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import point evaluator from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.binary_point_metrics


def save_standard_outputs(
    *,
    output_dir: Path,
    method: str,
    dataset: str,
    seed: int,
    scores: np.ndarray,
    labels: np.ndarray,
    timing: dict[str, Any],
    implementation: dict[str, Any],
) -> None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = (np.asarray(labels).reshape(-1) > 0).astype(np.int8)
    if len(scores) != len(labels):
        raise ValueError(
            f"{method} score/label length mismatch: {len(scores)} vs {len(labels)}."
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{method} produced NaN or infinite scores.")
    metrics = _load_binary_metrics()(labels, scores)
    require_finite_report_metrics(metrics, context=f"{method}/{dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "scores.npy", scores)
    np.save(output_dir / "labels.npy", labels)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {"schema_version": 1, "method": method, "dataset": dataset, **metrics},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    score_hash = hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "third_party_modified": False,
        "score_length": int(len(scores)),
        "score_sha256": score_hash,
        "implementation": implementation,
        "timing_protocol": {"warmup_runs": 1, "timed_runs": 3},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def overlap_average(window_scores: np.ndarray, series_length: int) -> np.ndarray:
    values = np.asarray(window_scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("window_scores must have shape [windows, window_length].")
    totals = np.zeros(series_length, dtype=np.float64)
    counts = np.zeros(series_length, dtype=np.int64)
    for start, row in enumerate(values):
        end = min(start + len(row), series_length)
        width = end - start
        if width <= 0:
            break
        totals[start:end] += row[:width]
        counts[start:end] += 1
    observed = counts > 0
    if not observed.any():
        raise ValueError("No window scores overlap the target series.")
    totals[observed] /= counts[observed]
    first = int(np.flatnonzero(observed)[0])
    last = int(np.flatnonzero(observed)[-1])
    totals[:first] = totals[first]
    totals[last + 1 :] = totals[last]
    return totals
