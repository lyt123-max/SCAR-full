from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ["GECCO", "GENESIS", "MSL", "SMAP", "PSM", "SWAT", "SMD"]
CORE_SCORE_ORDER = ["raw_max", "zscore_mean", "cdf_mean", "cdf_max"]
DEFAULT_METRIC_KEYS = [
    "roc_auc",
    "pr_auc",
    "point_best_f1",
    "pa_best_f1",
    "aff_precision",
    "aff_recall",
    "aff_f1",
    "range_precision",
    "range_recall",
    "range_f1",
    "vus_roc",
    "vus_pr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-dataset sensitivity summaries into combined CSV/JSON.")
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/sensitivity"))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--study-name-template", type=str, default="{dataset_lower}_sensitivity")
    parser.add_argument(
        "--score-keys",
        nargs="+",
        default=None,
        help="Defaults to every fusion and diagnostic subscore present in each summary.",
    )
    parser.add_argument("--metric-keys", nargs="+", default=DEFAULT_METRIC_KEYS)
    parser.add_argument("--output-prefix", type=Path, default=Path("./artifacts/sensitivity/sensitivity_summary"))
    return parser.parse_args()


def summary_json_path(artifact_root: Path, dataset: str, study_name_template: str) -> Path:
    dataset_upper = dataset.upper()
    study_name = study_name_template.format(dataset=dataset_upper, dataset_lower=dataset_upper.lower())
    return artifact_root / f"{dataset_upper.lower()}_sensitivity" / "summary" / f"{study_name}_summary.json"


def load_summary_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sensitivity summary payload: {path}")
    return payload


def latest_match(summary_dir: Path, pattern: str) -> Path | None:
    if not summary_dir.exists():
        return None
    candidates = sorted(
        (path for path in summary_dir.glob(pattern) if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0]


def resolve_summary_path(artifact_root: Path, dataset: str, study_name_template: str) -> Path | None:
    exact_path = summary_json_path(artifact_root, dataset, study_name_template)
    if exact_path.exists():
        return exact_path

    dataset_upper = dataset.upper()
    study_name = study_name_template.format(dataset=dataset_upper, dataset_lower=dataset_upper.lower())
    summary_dirs = [
        artifact_root / f"{dataset_upper.lower()}_sensitivity" / "summary",
        artifact_root / dataset_upper.lower() / "summary",
    ]
    for summary_dir in summary_dirs:
        candidate = latest_match(summary_dir, f"{study_name}_*_summary.json")
        if candidate is not None:
            return candidate
        candidate = latest_match(summary_dir, "*_summary.json")
        if candidate is not None:
            return candidate
    return None


def build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing_summaries: list[str] = []

    for dataset in (name.upper() for name in args.datasets):
        summary_path = resolve_summary_path(args.artifact_root, dataset, args.study_name_template)
        if summary_path is None:
            missing_summaries.append(str(summary_json_path(args.artifact_root, dataset, args.study_name_template)))
            continue
        payload = load_summary_payload(summary_path)
        if payload is None:
            missing_summaries.append(str(summary_path))
            continue

        specs = payload.get("specs", {})
        selected_params = payload.get("selected_params", [])
        dataset_display = str(payload.get("dataset_display", dataset))
        study_name = str(payload.get("study_name", f"{dataset.lower()}_sensitivity"))

        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            param_name = str(row.get("param", ""))
            spec = specs.get(param_name, {}) if isinstance(specs, dict) else {}
            score_metrics = row.get("score_metrics", {})
            if not isinstance(score_metrics, dict):
                score_metrics = {}
            discovered = set(str(key) for key in score_metrics)
            score_keys = (
                list(args.score_keys)
                if args.score_keys
                else [key for key in CORE_SCORE_ORDER if key in discovered]
                + sorted(discovered.difference(CORE_SCORE_ORDER))
            )
            for score_key in score_keys:
                metrics_for_score = score_metrics.get(score_key, {})
                if not isinstance(metrics_for_score, dict):
                    metrics_for_score = {}
                record: dict[str, Any] = {
                    "dataset": dataset,
                    "dataset_display": dataset_display,
                    "study_name": study_name,
                    "summary_json": str(summary_path),
                    "selected_params": ",".join(str(item) for item in selected_params),
                    "param": param_name,
                    "group": row.get("group", spec.get("group")),
                    "stage_mode": row.get("stage_mode", spec.get("stage_mode")),
                    "paper_panel": spec.get("paper_panel"),
                    "description": spec.get("description"),
                    "score_key": score_key,
                    "value": row.get("value"),
                    "value_label": row.get("value_label"),
                    "default": row.get("default", spec.get("default")),
                    "experiment_name": row.get("experiment_name"),
                    "experiment_dir": row.get("experiment_dir"),
                }
                for metric_key in args.metric_keys:
                    if metric_key in metrics_for_score:
                        record[metric_key] = metrics_for_score.get(metric_key)
                    elif score_key == "cdf_max":
                        record[metric_key] = row.get(metric_key)
                    else:
                        record[metric_key] = row.get(f"{score_key}_{metric_key}")
                records.append(record)

    return records, missing_summaries


def write_csv(path: Path, records: list[dict[str, Any]], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_display",
        "study_name",
        "summary_json",
        "selected_params",
        "param",
        "group",
        "stage_mode",
        "paper_panel",
        "description",
        "score_key",
        "value",
        "value_label",
        "default",
        "experiment_name",
        "experiment_dir",
        *metric_keys,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_json(path: Path, records: list[dict[str, Any]], missing_summaries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": records,
        "missing_summaries": missing_summaries,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = [dataset.upper() for dataset in args.datasets]
    records, missing_summaries = build_records(args)

    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    write_csv(csv_path, records, args.metric_keys)
    write_json(json_path, records, missing_summaries)

    print(f"[SensitivitySummary] wrote CSV: {csv_path}")
    print(f"[SensitivitySummary] wrote JSON: {json_path}")
    print("[SensitivitySummary] datasets: " + ", ".join(datasets))
    actual_score_keys = sorted({str(record["score_key"]) for record in records})
    print("[SensitivitySummary] score keys: " + ", ".join(actual_score_keys))
    print("[SensitivitySummary] metric keys: " + ", ".join(args.metric_keys))
    print(f"[SensitivitySummary] records={len(records)}")
    if missing_summaries:
        print("[SensitivitySummary] missing summaries:")
        for path in missing_summaries:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
