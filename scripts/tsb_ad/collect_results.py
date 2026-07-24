from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

try:
    from .common import (
        FORMAL_SEED,
        PROJECT_ROOT,
        normalize_edition,
        normalize_split,
        parse_file_name,
    )
except ImportError:
    from common import (
        FORMAL_SEED,
        PROJECT_ROOT,
        normalize_edition,
        normalize_split,
        parse_file_name,
    )


SCORE_KEYS = ("raw_max", "zscore_mean", "cdf_mean", "cdf_max")
METRIC_KEYS = ("vus_pr", "vus_roc", "auroc", "ap", "runtime_seconds", "coverage")
METRIC_SOURCE_KEYS = {
    "vus_pr": "vus_pr",
    "vus_roc": "vus_roc",
    "auroc": "roc_auc",
    "ap": "pr_auc",
}


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _mean(values: Iterable[Any]) -> float:
    finite_values = [_finite(value) for value in values]
    finite_values = [value for value in finite_values if math.isfinite(value)]
    return fmean(finite_values) if finite_values else float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _run_rows(
    root: Path,
    edition: str,
    split: str,
    seed: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pattern = root / edition / split
    if not pattern.exists():
        return rows, failures
    seed_pattern = "seed_*" if seed is None else f"seed_{int(seed)}"

    recorded_dirs: set[Path] = set()
    for record_path in sorted(pattern.glob(f"{seed_pattern}/*/run_record.json")):
        experiment_dir = record_path.parent
        recorded_dirs.add(experiment_dir)
        try:
            record = _read_json(record_path)
            if record.get("status") != "completed":
                failures.append(
                    {
                        "edition": edition,
                        "split": split,
                        "experiment_dir": str(experiment_dir),
                        "error": str(record.get("error") or f"run status is {record.get('status')!r}"),
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "edition": edition,
                    "split": split,
                    "experiment_dir": str(experiment_dir),
                    "error": f"invalid run record: {exc}",
                }
            )

    for metrics_path in sorted(pattern.glob(f"{seed_pattern}/*/test_metrics.json")):
        experiment_dir = metrics_path.parent
        seed_dir = experiment_dir.parent.name
        try:
            seed = int(seed_dir.removeprefix("seed_"))
            metrics = _read_json(metrics_path)
            metadata = metrics.get("dataset_metadata") or parse_file_name(f"{experiment_dir.name}.csv")
            file_name = str(metadata.get("file_name") or metadata.get("file") or f"{experiment_dir.name}.csv")
            source_dataset = str(metadata["source_dataset"])
            coverage = _finite(metrics.get("point_coverage_ratio"))
            record_path = experiment_dir / "run_record.json"
            record = _read_json(record_path) if record_path.exists() else {}
            runtime = _finite(record.get("runtime_seconds"))
            for score_key in SCORE_KEYS:
                score_metrics = metrics.get(score_key)
                if not isinstance(score_metrics, dict):
                    raise KeyError(f"missing score metrics for {score_key}")
                row: dict[str, Any] = {
                    "edition": edition,
                    "split": split,
                    "dataset": source_dataset,
                    "file": file_name,
                    "seed": seed,
                    "score_key": score_key,
                    "is_primary": int(score_key == "cdf_mean"),
                    "runtime_seconds": runtime,
                    "coverage": coverage,
                }
                for output_key, source_key in METRIC_SOURCE_KEYS.items():
                    row[output_key] = _finite(score_metrics.get(source_key))
                rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "edition": edition,
                    "split": split,
                    "experiment_dir": str(experiment_dir),
                    "error": str(exc),
                }
            )
    metric_dirs = {
        path.parent for path in pattern.glob(f"{seed_pattern}/*/test_metrics.json")
    }
    for experiment_dir in sorted(recorded_dirs - metric_dirs):
        if not any(item["experiment_dir"] == str(experiment_dir) for item in failures):
            failures.append(
                {
                    "edition": edition,
                    "split": split,
                    "experiment_dir": str(experiment_dir),
                    "error": "run record exists but test_metrics.json is missing",
                }
            )
    return rows, failures


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _wide_rows(
    long_rows: list[dict[str, Any]],
    identity_keys: tuple[str, ...],
    *,
    count_key: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        groups[tuple(row[key] for key in identity_keys)].append(row)

    output: list[dict[str, Any]] = []
    for identity, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        wide = dict(zip(identity_keys, identity))
        wide[count_key] = len({(row["file"], row["seed"]) for row in group})
        wide["n_files"] = len({row["file"] for row in group})
        wide["n_seeds"] = len({row["seed"] for row in group})
        for score_key in SCORE_KEYS:
            score_rows = [row for row in group if row["score_key"] == score_key]
            for metric_key in METRIC_KEYS:
                wide[f"{score_key}_{metric_key}"] = _mean(row[metric_key] for row in score_rows)
        output.append(wide)
    return output


def _macro_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        dataset_groups[(row["edition"], row["split"], row["dataset"], row["score_key"])].append(row)

    dataset_means: list[dict[str, Any]] = []
    for (edition, split, dataset, score_key), group in dataset_groups.items():
        item = {
            "edition": edition,
            "split": split,
            "dataset": dataset,
            "score_key": score_key,
        }
        for metric_key in METRIC_KEYS:
            item[metric_key] = _mean(row[metric_key] for row in group)
        dataset_means.append(item)

    macro_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_means:
        macro_groups[(row["edition"], row["split"])].append(row)

    output: list[dict[str, Any]] = []
    for (edition, split), group in sorted(macro_groups.items()):
        wide: dict[str, Any] = {
            "edition": edition,
            "split": split,
            "n_datasets": len({row["dataset"] for row in group}),
        }
        for score_key in SCORE_KEYS:
            score_rows = [row for row in group if row["score_key"] == score_key]
            for metric_key in METRIC_KEYS:
                wide[f"{score_key}_{metric_key}"] = _mean(row[metric_key] for row in score_rows)
        output.append(wide)
    return output


def collect(
    artifact_root: Path,
    edition: str,
    split: str,
    seed: int | None = FORMAL_SEED,
) -> dict[str, int]:
    edition = normalize_edition(edition)
    split = normalize_split(split)
    long_rows, failures = _run_rows(artifact_root, edition, split, seed)
    output_dir = artifact_root / edition / split
    identity_fields = ["edition", "split", "dataset", "file", "seed"]
    metric_fields = [f"{score}_{metric}" for score in SCORE_KEYS for metric in METRIC_KEYS]

    _write_csv(
        output_dir / "results_long.csv",
        long_rows,
        [
            "edition",
            "split",
            "dataset",
            "file",
            "seed",
            "score_key",
            "is_primary",
            "vus_pr",
            "vus_roc",
            "auroc",
            "ap",
            "runtime_seconds",
            "coverage",
        ],
    )
    per_series = _wide_rows(long_rows, tuple(identity_fields), count_key="n_runs")
    by_dataset = _wide_rows(long_rows, ("edition", "split", "dataset"), count_key="n_runs")
    official = _wide_rows(long_rows, ("edition", "split"), count_key="n_runs")
    macro = _macro_rows(long_rows)
    _write_csv(output_dir / "results_per_series_wide.csv", per_series, identity_fields + ["n_runs", "n_files", "n_seeds"] + metric_fields)
    _write_csv(output_dir / "results_by_dataset_wide.csv", by_dataset, ["edition", "split", "dataset", "n_runs", "n_files", "n_seeds"] + metric_fields)
    _write_csv(output_dir / "results_official_average_wide.csv", official, ["edition", "split", "n_runs", "n_files", "n_seeds"] + metric_fields)
    _write_csv(output_dir / "results_dataset_macro_average_wide.csv", macro, ["edition", "split", "n_datasets"] + metric_fields)
    _write_csv(output_dir / "collection_failures.csv", failures, ["edition", "split", "experiment_dir", "error"])
    return {
        "long_rows": len(long_rows),
        "series_runs": len({(row["file"], row["seed"]) for row in long_rows}),
        "failures": len(failures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fixed TSB-AD fusion results.")
    parser.add_argument("--edition", required=True, choices=["M", "U", "m", "u"])
    parser.add_argument("--split", default="eval", choices=["all", "tuning", "eval", "eval_full", "eval-full"])
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "artifacts" / "tsb_ad")
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Read historical seed_* directories instead of the formal seed-42 result only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect(
        args.artifact_root.resolve(),
        args.edition,
        args.split,
        seed=None if args.all_seeds else args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
