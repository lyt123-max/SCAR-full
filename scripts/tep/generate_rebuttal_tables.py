#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

T9_METRICS = [
    "SMC@K",
    "SFR",
    "SMR@K",
    "SMR@K-sequence-balanced",
    "delta_mem_mode",
    "fault-normal-gap-sequence-balanced",
    "Mode-FPR-Std",
    "EE95",
    "Tail@0.99_error",
    "Cross-mode-margin",
    "cross-mode-margin-sequence-balanced",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TEP rebuttal tables T2 and T9.")
    parser.add_argument("--full-experiment", type=Path, required=True)
    parser.add_argument("--selected-experiment", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_single_metrics(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "tep_mechanism" / "mechanism_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", {})
    if not experiments:
        raise ValueError(f"No experiment metrics found in {path}.")
    metrics = dict(next(iter(experiments.values())))
    missing = [key for key in T9_METRICS if key not in metrics]
    if missing:
        # Recompute missing schema-v3 additions in memory. The historical
        # experiment directory remains read-only.
        from compute_mechanism_metrics import _load_experiment_payload, compute_single_metrics

        experiment_payload = _load_experiment_payload(exp_dir, "tep_mechanism")
        recomputed, _ = compute_single_metrics(
            audit_logs=experiment_payload["audit_logs"],
            fault_logs=experiment_payload["fault_logs"],
            train_state_meta=experiment_payload["train_state_meta"],
            fault_sequence_records=experiment_payload["fault_sequence_records"],
            smc_k=10,
            smr_k=10,
            fpr_quantile=0.95,
        )
        for key in missing:
            if key in recomputed:
                metrics[key] = recomputed[key]
    return metrics


def discover_selected_experiment(full_experiment: Path) -> Path:
    candidates: list[Path] = []
    search_roots = {
        full_experiment.parent.resolve(),
        (REPO_ROOT / "artifacts").resolve(),
        (REPO_ROOT / "TEP-abalation").resolve(),
    }
    for root in search_roots:
        if not root.is_dir():
            continue
        for metrics_path in root.glob("*/tep_mechanism/mechanism_metrics.json"):
            exp_dir = metrics_path.parents[1]
            if exp_dir.resolve() == full_experiment.resolve():
                continue
            config_path = exp_dir / "config.json"
            if not config_path.is_file():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if str(config.get("dataset", "")).upper() != "TEP":
                continue
            if str(config.get("tep_protocol", "selected")).lower() != "selected":
                continue
            candidates.append(exp_dir)
    if not candidates:
        raise FileNotFoundError(
            "Cannot generate the selected/full T9 comparison because no selected TEP "
            "experiment with mechanism metrics was found. Pass --selected-experiment "
            "or set TEP_SELECTED_EXPERIMENT_DIR."
        )

    def rank(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        is_baseline = int(
            "tep_ablation_full" in name
            or "tep_baseline" in name
            or "selected" in name
        )
        metrics_mtime = int(
            (path / "tep_mechanism" / "mechanism_metrics.json").stat().st_mtime_ns
        )
        return is_baseline, metrics_mtime

    return max(candidates, key=rank)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_t2(full_experiment: Path) -> list[dict[str, Any]]:
    audit_path = full_experiment / "tep_data_audit" / "tep_data_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return [
        {
            "Mode": row["mode"],
            "Normal sequences": row["normal_sequences"],
            "Available faults": row["fault_sequences"],
            "Full length faults": row["full_length_faults"],
            "Short faults": row["short_faults"],
            "Min length": row["min_fault_length"],
            "Max length": row["max_fault_length"],
            "Usable faults": row["usable_faults"],
        }
        for row in audit["mode_summary"]
    ]


def build_t9(
    full_experiment: Path,
    selected_experiment: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiments = [
        ("selected", "M1/M3/M4", "8 file faults per mode", selected_experiment),
        ("full", "M1-M6", "IDV1-IDV28", full_experiment),
    ]
    for protocol, modes, faults, exp_dir in experiments:
        metrics = load_single_metrics(exp_dir)
        row: dict[str, Any] = {
            "Protocol": protocol,
            "Mode subset": modes,
            "Fault subset": faults,
        }
        for key in T9_METRICS:
            value = metrics.get(key, "")
            row[key] = float(value) if isinstance(value, (int, float)) else value
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_experiment = (
        args.selected_experiment
        if args.selected_experiment is not None
        else discover_selected_experiment(args.full_experiment)
    )
    t2_rows = build_t2(args.full_experiment)
    t9_rows = build_t9(args.full_experiment, selected_experiment)

    write_csv(args.output_dir / "table_t2_tep_availability.csv", t2_rows)
    write_csv(args.output_dir / "table_t9_tep_subset_robustness.csv", t9_rows)
    (args.output_dir / "table_t2_tep_availability.md").write_text(
        "# Table T2: TEP Mode-Fault Availability\n\n" + markdown_table(t2_rows) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "table_t9_tep_subset_robustness.md").write_text(
        "# Table T9: TEP Subset Robustness\n\n" + markdown_table(t9_rows) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "tep_rebuttal_tables.json").write_text(
        json.dumps({"T2": t2_rows, "T9": t9_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[TEP Tables] wrote T2/T9 to {args.output_dir}; "
        f"selected={selected_experiment}"
    )


if __name__ == "__main__":
    main()
