from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from .common import (
        load_common_data,
        measure_inference,
        save_standard_outputs,
        set_seed,
    )
except ImportError:
    try:
        from scripts.rebuttal.baselines.common import (
            load_common_data,
            measure_inference,
            save_standard_outputs,
            set_seed,
        )
    except ImportError:
        from common import (
            load_common_data,
            measure_inference,
            save_standard_outputs,
            set_seed,
        )


REFERENCE_RATIOS = (0.10, 0.25, 0.50, 1.00)
QUERY_RATIOS = (0.25, 0.50, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run auditable point-feature KNN or LOF baselines."
    )
    parser.add_argument("--method", choices=("KNN", "LOF"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neighbors", type=int, default=None)
    parser.add_argument("--max-reference-points", type=int, default=50_000)
    parser.add_argument("--scalability-query-points", type=int, default=50_000)
    parser.add_argument("--query-chunk-size", type=int, default=8_192)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def deterministic_subsample(array: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(array)
    if count <= 0:
        raise ValueError("Subsample count must be positive.")
    if len(values) <= count:
        return np.ascontiguousarray(values)
    indices = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    return np.ascontiguousarray(values[indices])


def fit_detector(
    method: str,
    reference: np.ndarray,
    *,
    neighbors: int,
    n_jobs: int,
) -> Any:
    if len(reference) <= neighbors:
        raise ValueError(
            f"{method} requires more reference points than neighbors: "
            f"{len(reference)} <= {neighbors}."
        )
    if method == "KNN":
        detector = NearestNeighbors(
            n_neighbors=neighbors,
            algorithm="auto",
            metric="euclidean",
            n_jobs=n_jobs,
        )
    elif method == "LOF":
        detector = LocalOutlierFactor(
            n_neighbors=neighbors,
            novelty=True,
            contamination="auto",
            metric="euclidean",
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unsupported classical baseline: {method}.")
    detector.fit(reference)
    return detector


def score_detector(
    method: str,
    detector: Any,
    query: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(query), chunk_size):
        batch = np.ascontiguousarray(query[start : start + chunk_size])
        if method == "KNN":
            distances, _ = detector.kneighbors(batch, return_distance=True)
            scores = distances[:, -1]
        else:
            scores = -detector.score_samples(batch)
        chunks.append(np.asarray(scores, dtype=np.float64))
    if not chunks:
        raise ValueError("Cannot score an empty query array.")
    return np.concatenate(chunks)


def timed_once(function: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    result = np.asarray(function(), dtype=np.float64)
    return result, float(time.perf_counter() - started)


def serialized_bytes(scaler: StandardScaler, detector: Any) -> int:
    return len(
        pickle.dumps(
            {"scaler": scaler, "detector": detector},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )


def write_scalability(
    *,
    output_dir: Path,
    method: str,
    scaler: StandardScaler,
    reference_pool: np.ndarray,
    test_scaled: np.ndarray,
    neighbors: int,
    n_jobs: int,
    chunk_size: int,
    query_cap: int,
) -> dict[str, Any]:
    fixed_query = deterministic_subsample(test_scaled, min(len(test_scaled), query_cap))
    reference_rows: list[dict[str, Any]] = []
    for ratio in REFERENCE_RATIOS:
        count = max(neighbors + 1, int(round(len(reference_pool) * ratio)))
        reference = deterministic_subsample(reference_pool, count)
        started = time.perf_counter()
        detector = fit_detector(
            method,
            reference,
            neighbors=neighbors,
            n_jobs=n_jobs,
        )
        fit_seconds = float(time.perf_counter() - started)
        _, inference_seconds = timed_once(
            lambda detector=detector: score_detector(
                method,
                detector,
                fixed_query,
                chunk_size=chunk_size,
            )
        )
        reference_rows.append(
            {
                "axis": "reference",
                "ratio": ratio,
                "reference_points": int(len(reference)),
                "query_points": int(len(fixed_query)),
                "fit_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
                "serialized_model_bytes": serialized_bytes(scaler, detector),
                "raw_reference_bytes": int(reference.nbytes),
            }
        )

    full_detector = fit_detector(
        method,
        reference_pool,
        neighbors=neighbors,
        n_jobs=n_jobs,
    )
    query_rows: list[dict[str, Any]] = []
    for ratio in QUERY_RATIOS:
        count = max(1, int(round(len(fixed_query) * ratio)))
        query = fixed_query[:count]
        _, inference_seconds = timed_once(
            lambda query=query: score_detector(
                method,
                full_detector,
                query,
                chunk_size=chunk_size,
            )
        )
        query_rows.append(
            {
                "axis": "query",
                "ratio": ratio,
                "reference_points": int(len(reference_pool)),
                "query_points": int(len(query)),
                "fit_seconds": 0.0,
                "inference_seconds": inference_seconds,
                "serialized_model_bytes": serialized_bytes(scaler, full_detector),
                "raw_reference_bytes": int(reference_pool.nbytes),
            }
        )

    rows = [*reference_rows, *query_rows]
    with (output_dir / "scalability.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "method": method,
        "reference_ratios": list(REFERENCE_RATIOS),
        "query_ratios": list(QUERY_RATIOS),
        "query_cap": int(query_cap),
        "rows": rows,
    }
    (output_dir / "scalability.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Formal classical baselines require seed 42.")
    if str(args.device).lower() != "cpu":
        raise ValueError("KNN and LOF formal adapters are registered as CPU methods.")
    set_seed(args.seed)
    train, test, labels = load_common_data(args.data_dir)
    if args.smoke:
        train = train[: min(len(train), 2_048)]
        test = test[: min(len(test), 2_048)]
        labels = labels[: len(test)]
    scaler = StandardScaler()
    train_scaled = np.asarray(scaler.fit_transform(train), dtype=np.float32)
    test_scaled = np.asarray(scaler.transform(test), dtype=np.float32)
    max_reference = min(
        len(train_scaled),
        2_048 if args.smoke else int(args.max_reference_points),
    )
    reference = deterministic_subsample(train_scaled, max_reference)
    neighbors = int(
        args.neighbors
        if args.neighbors is not None
        else (5 if args.method == "KNN" else 20)
    )
    started = time.perf_counter()
    detector = fit_detector(
        args.method,
        reference,
        neighbors=neighbors,
        n_jobs=args.n_jobs,
    )
    fit_seconds = float(time.perf_counter() - started)
    inference = lambda: score_detector(
        args.method,
        detector,
        test_scaled,
        chunk_size=args.query_chunk_size,
    )
    scores, timing = measure_inference(inference, warmups=1, repeats=3)
    timing["fit_seconds"] = fit_seconds
    timing["reference_points"] = int(len(reference))
    timing["test_points"] = int(len(test_scaled))
    timing["device"] = "cpu"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    model_path.write_bytes(
        pickle.dumps(
            {"scaler": scaler, "detector": detector},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    memory = {
        "schema_version": 1,
        "method": args.method,
        "reference_points": int(len(reference)),
        "feature_dim": int(reference.shape[1]),
        "raw_reference_bytes": int(reference.nbytes),
        "serialized_model_bytes": int(model_path.stat().st_size),
        "reference_cap": int(args.max_reference_points),
        "sampling": "deterministic_evenly_spaced_without_labels",
    }
    (output_dir / "memory_metrics.json").write_text(
        json.dumps(memory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_scalability(
        output_dir=output_dir,
        method=args.method,
        scaler=scaler,
        reference_pool=reference,
        test_scaled=test_scaled,
        neighbors=neighbors,
        n_jobs=args.n_jobs,
        chunk_size=args.query_chunk_size,
        query_cap=(
            min(2_048, len(test_scaled))
            if args.smoke
            else int(args.scalability_query_points)
        ),
    )
    save_standard_outputs(
        output_dir=output_dir,
        method=args.method,
        dataset=args.dataset,
        seed=args.seed,
        scores=scores,
        labels=labels,
        timing=timing,
        implementation={
            "type": "project_native_sklearn",
            "feature_protocol": "per_timestamp_multivariate_vector",
            "normalization": "train_fitted_standard_scaler",
            "neighbors": neighbors,
            "reference_cap": int(args.max_reference_points),
            "sampling": "deterministic_evenly_spaced_without_labels",
            "sklearn_estimator": type(detector).__name__,
        },
    )


if __name__ == "__main__":
    main()
