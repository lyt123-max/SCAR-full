from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.score_outputs import (
    REQUIRED_REPORT_METRIC_KEYS,
    flatten_score_metrics,
)


DATASETS = ("MSL", "PSM", "SMAP", "SMD", "SWAT")
WINDOWS = (64, 128, 256, 512)
PATCHES = (8, 16, 32, 64)
SYNTHETIC = (
    "synthetic_con0.0494",
    "synthetic_con0.072",
    "synthetic_glo0.048",
    "synthetic_glo0.0718",
    "synthetic_sea0.0482",
    "synthetic_sea0.0774",
    "synthetic_sha0.049",
    "synthetic_sha0.0742",
    "synthetic_sub_mix0.0574",
    "synthetic_sub_mix0.089",
    "synthetic_tre0.0482",
    "synthetic_tre0.0778",
)


def _metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return flatten_score_metrics(payload, require_core=True)


def _write(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    fieldnames.extend(
        sorted({key for row in rows for key in row}.difference(fieldnames))
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_type(name: str) -> str:
    return name.removeprefix("synthetic_").split("0.")[0].rstrip("_")


def _is_report_metric_column(key: str) -> bool:
    return key in REQUIRED_REPORT_METRIC_KEYS or any(
        key.endswith(f"_{metric_key}")
        for metric_key in REQUIRED_REPORT_METRIC_KEYS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect E29/E30/E31 table-only outputs.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    e29 = []
    for dataset in DATASETS:
        for length in WINDOWS:
            for strategy in ("global", "context_only", "full"):
                name = f"scar_e29_{dataset.lower()}_l{length}_{strategy}"
                metrics = _metrics(root / name / "metrics.json")
                e29.append(
                    {
                        "dataset": dataset,
                        "window_length": length,
                        "strategy": strategy,
                        **metrics,
                    }
                )

    e30 = []
    for dataset in (*DATASETS, *SYNTHETIC):
        for patch in PATCHES:
            for strategy in ("global", "full"):
                name = f"scar_e30_{dataset.lower()}_p{patch}_seed42_{strategy}"
                metrics = _metrics(root / name / "metrics.json")
                e30.append(
                    {
                        "dataset": dataset,
                        "anomaly_type": (
                            _synthetic_type(dataset)
                            if dataset.startswith("synthetic_")
                            else "real"
                        ),
                        "patch_size": str(patch),
                        "strategy": strategy,
                        **metrics,
                    }
                )
        name = f"scar_e30_{dataset.lower()}_multiscale_full"
        metrics = _metrics(root / name / "metrics.json")
        e30.append(
            {
                "dataset": dataset,
                "anomaly_type": (
                    _synthetic_type(dataset)
                    if dataset.startswith("synthetic_")
                    else "real"
                ),
                "patch_size": "8+32",
                "strategy": "full",
                **metrics,
            }
        )
    macro = []
    for anomaly_type in sorted({_synthetic_type(name) for name in SYNTHETIC}):
        for patch in (*map(str, PATCHES), "8+32"):
            for strategy in ("global", "full"):
                rows = [
                    row
                    for row in e30
                    if row["anomaly_type"] == anomaly_type
                    and row["patch_size"] == patch
                    and row["strategy"] == strategy
                ]
                if not rows:
                    continue
                metric_keys = sorted(
                    key
                    for key in rows[0]
                    if _is_report_metric_column(key)
                )
                macro.append(
                    {
                        "anomaly_type": anomaly_type,
                        "patch_size": patch,
                        "strategy": strategy,
                        "series_count": len(rows),
                        **{
                            f"macro_{key}": sum(float(row[key]) for row in rows) / len(rows)
                            for key in metric_keys
                        },
                    }
                )

    e31 = []
    for dataset in DATASETS:
        result_dir = root / f"scar_e31_{dataset.lower()}_timing_selection"
        path = result_dir / "budget_match.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = payload["candidate"]
        relative = payload["relative_error"]
        scar_metrics = _metrics(
            root / f"scar_main_{dataset.lower()}_seed42" / "test_metrics.json"
        )
        global_metrics = _metrics(result_dir / "metrics.json")
        e31.append(
            {
                "dataset": dataset,
                "d_z": candidate["d_z"],
                "coreset_keep_ratio": candidate["keep_ratio"],
                "top_K": candidate["top_K"],
                "parameters": candidate["parameters"],
                "bank_bytes": candidate["bank_bytes"],
                "latency_ms": candidate["latency_ms"],
                "parameter_error": relative["parameters"],
                "bank_error": relative["bank_bytes"],
                "latency_error": relative["latency_ms"],
                "within_15_percent": payload["within_tolerance"],
                "log_error_sum": payload["log_error_sum"],
                **{f"scar_{key}": value for key, value in scar_metrics.items()},
                **{f"global_{key}": value for key, value in global_metrics.items()},
            }
        )
    _write(args.output_dir / "e29_window_strategy.csv", e29)
    _write(args.output_dir / "e30_patch_strategy.csv", e30)
    _write(args.output_dir / "e30_synthetic_macro.csv", macro)
    _write(args.output_dir / "e31_budget_match.csv", e31)
    (args.output_dir / "p1_tables.json").write_text(
        json.dumps(
            {"schema_version": 1, "E29": e29, "E30": e30, "E30_macro": macro, "E31": e31},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
