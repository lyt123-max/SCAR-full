from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("MSL", "PSM", "SMAP", "SMD", "SWAT")
KEEP_RATIOS = (1.0, 0.5, 0.25, 0.10, 0.05)


def token(value: float) -> str:
    return str(value).replace(".", "p")


def retained_raw_starts(experiment: Path) -> set[int]:
    retained: set[int] = set()
    for path in (experiment / "memory_audit").glob("memory_audit_scale*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            starts = np.asarray(payload["raw_start"], dtype=np.int64)
            kept = np.asarray(payload["kept_after_coreset"], dtype=bool)
            retained.update(starts[kept].tolist())
    return retained


def candidate_recall(oracle_raw_starts: np.ndarray, retained: set[int]) -> float:
    oracle = np.asarray(oracle_raw_starts, dtype=np.int64)
    valid = oracle >= 0
    if oracle.ndim == 2:
        first = np.full(len(oracle), -1, dtype=np.int64)
        for row in range(len(oracle)):
            row_valid = oracle[row][valid[row]]
            if len(row_valid):
                first[row] = row_valid[0]
        oracle = first
        valid = oracle >= 0
    if not valid.any():
        return float("nan")
    return float(np.mean([int(value) in retained for value in oracle[valid]]))


def _selected_metrics(experiment: Path) -> dict:
    payload = json.loads((experiment / "test_metrics.json").read_text(encoding="utf-8"))
    selected = payload.get("selected", payload)
    return {
        "roc_auc": selected.get("roc_auc", selected.get("point_roc_auc")),
        "pr_auc": selected.get("pr_auc", selected.get("point_pr_auc")),
    }


def _resource(experiment: Path) -> dict:
    path = experiment / "resource_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    attempts = payload.get("attempts", [])
    peak_ram = max(
        (
            float(item.get("cpu", {}).get("peak_rss_bytes") or 0)
            for item in attempts
        ),
        default=0.0,
    )
    return {
        "inference_seconds": summary.get("inference_seconds"),
        "peak_ram_bytes": peak_ram,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect E38/E39 coreset evidence.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    rows = []
    for dataset in DATASETS:
        oracle_path = (
            root / f"scar_e1_e5_{dataset.lower()}_full" / "retrieval_full.npz"
        )
        with np.load(oracle_path, allow_pickle=False) as oracle:
            oracle_starts = np.asarray(oracle["neighbor_raw_starts"], dtype=np.int64)
        for ratio in KEEP_RATIOS:
            experiment = (
                root / f"scar_main_{dataset.lower()}_seed42"
                if ratio == 1.0
                else root / f"scar_e38_{dataset.lower()}_keep_{token(ratio)}"
            )
            memory_files = [experiment / "memory.pt", experiment / "memory_meta.json"]
            bank_bytes = sum(path.stat().st_size for path in memory_files if path.is_file())
            rows.append(
                {
                    "dataset": dataset,
                    "coreset_keep_ratio": ratio,
                    "candidate_recall_at_1": candidate_recall(
                        oracle_starts, retained_raw_starts(experiment)
                    ),
                    "bank_bytes": bank_bytes,
                    **_selected_metrics(experiment),
                    **_resource(experiment),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "e38_coreset.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "e39_resource_tradeoff.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
