from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def _metrics(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected", payload)
    if "selected" in selected and isinstance(selected["selected"], dict):
        selected = selected["selected"]
    roc = selected.get("roc_auc", selected.get("point_roc_auc"))
    ap = selected.get("pr_auc", selected.get("point_pr_auc"))
    if roc is None or ap is None:
        raise KeyError(f"Cannot locate AUROC/AP in {path}.")
    return float(roc), float(ap)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_type(name: str) -> str:
    return name.removeprefix("synthetic_").split("0.")[0].rstrip("_")


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
                roc, ap = _metrics(root / name / "metrics.json")
                e29.append(
                    {
                        "dataset": dataset,
                        "window_length": length,
                        "strategy": strategy,
                        "roc_auc": roc,
                        "pr_auc": ap,
                    }
                )

    e30 = []
    for dataset in (*DATASETS, *SYNTHETIC):
        for patch in PATCHES:
            for strategy in ("global", "full"):
                name = f"scar_e30_{dataset.lower()}_p{patch}_seed42_{strategy}"
                roc, ap = _metrics(root / name / "metrics.json")
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
                        "roc_auc": roc,
                        "pr_auc": ap,
                    }
                )
        name = f"scar_e30_{dataset.lower()}_multiscale_full"
        roc, ap = _metrics(root / name / "metrics.json")
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
                "roc_auc": roc,
                "pr_auc": ap,
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
                macro.append(
                    {
                        "anomaly_type": anomaly_type,
                        "patch_size": patch,
                        "strategy": strategy,
                        "series_count": len(rows),
                        "macro_roc_auc": sum(row["roc_auc"] for row in rows) / len(rows),
                        "macro_pr_auc": sum(row["pr_auc"] for row in rows) / len(rows),
                    }
                )

    e31 = []
    for dataset in DATASETS:
        path = root / f"scar_e31_{dataset.lower()}_timing_selection" / "budget_match.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = payload["candidate"]
        relative = payload["relative_error"]
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
