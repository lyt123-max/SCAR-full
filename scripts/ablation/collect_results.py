from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.score_outputs import (
    CORE_FUSION_SCORE_KEYS,
    score_metric_groups,
)


ABLATION_LABELS = {
    "full": "CoReM-AD (full)",
    "wo_decomposition": "(A1.1) w/o decomposition",
    "wo_state_aware_representation": "(A1.2) w/o state-guided modulation",
    "global_retrieval": "(A2.1) global retrieval",
    "state_only_retrieval": "(A2.2) state-only retrieval",
    "context_only_retrieval": "(A2.3) context-only retrieval",
    "dual_condition_retrieval": "(A2.4) dual-condition retrieval",
    "pit_fusion": "(A3.1) CDF/PIT mean fusion",
    "raw_max": "(A3.2) raw-max",
    "zscore_mean": "(A3.3) zscore-mean",
    "single_scale_short": "(A4.1) single-scale (short)",
    "single_scale_long": "(A4.2) single-scale (long)",
    "multi_scale": "(A4.3) multi-scale",
}

DEFAULT_ABLATIONS = [
    "full",
    "wo_decomposition",
    "wo_state_aware_representation",
    "global_retrieval",
    "state_only_retrieval",
    "context_only_retrieval",
    "dual_condition_retrieval",
    "raw_max",
    "zscore_mean",
    "single_scale_short",
    "single_scale_long",
    "multi_scale",
]

DEFAULT_DATASETS = ["MSL", "SMAP", "PSM", "SWAT", "SMD", "TEP"]

# Match the main experiment metrics already produced in test_metrics.json.
DEFAULT_METRIC_KEYS = [
    "roc_auc",
    "pr_auc",
    "best_f1",
    "precision_at_best_f1",
    "recall_at_best_f1",
    "pa_best_f1",
    "pa_precision_at_best_f1",
    "pa_recall_at_best_f1",
    "aff_precision",
    "aff_recall",
    "aff_f1",
    "range_precision",
    "range_recall",
    "range_f1",
    "r_auc_roc",
    "r_auc_pr",
    "vus_roc",
    "vus_pr",
]

MISSING_RESULT_SCORE_KEYS = ["selected", *CORE_FUSION_SCORE_KEYS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ablation results into CSV/JSON and optional Markdown tables.")
    parser.add_argument("--artifact_root", type=Path, default=Path("./artifacts"))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--ablations", nargs="+", default=DEFAULT_ABLATIONS)
    parser.add_argument("--experiment_name_template", type=str, default="{dataset_lower}_ablation_{ablation}")
    parser.add_argument(
        "--score_keys",
        nargs="+",
        default=None,
        help="Defaults to every fusion and diagnostic subscore present in test_metrics.json.",
    )
    parser.add_argument("--metric_keys", nargs="+", default=DEFAULT_METRIC_KEYS)
    parser.add_argument("--output_prefix", type=Path, default=Path("./artifacts/ablation_summary"))
    parser.add_argument(
        "--markdown_metric_keys",
        nargs="*",
        default=[],
        help="Metric keys to render as Markdown tables. If omitted, only CSV/JSON are written.",
    )
    parser.add_argument(
        "--markdown_score_key",
        type=str,
        default="cdf_mean",
        help="Score source used when rendering Markdown tables.",
    )
    parser.add_argument(
        "--f1_key",
        type=str,
        default=None,
        choices=["best_f1", "pa_best_f1"],
        help="Deprecated alias for rendering a single Markdown table via --markdown_metric_keys.",
    )
    return parser.parse_args()


def resolve_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def load_metrics_payload(artifact_root: Path, experiment_name: str) -> dict[str, Any] | None:
    experiment_dir = artifact_root / experiment_name
    if not experiment_dir.exists():
        candidates = sorted(
            (
                path
                for path in artifact_root.glob(f"{experiment_name}_*")
                if path.is_dir()
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        if not candidates:
            return None
        experiment_dir = candidates[0]
    metrics_path = experiment_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def format_metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = [dataset.upper() for dataset in args.datasets]
    records: list[dict[str, Any]] = []

    for ablation_key in args.ablations:
        ablation_label = ABLATION_LABELS.get(ablation_key, ablation_key)
        for dataset in datasets:
            experiment_name = args.experiment_name_template.format(
                dataset=dataset,
                dataset_lower=dataset.lower(),
                ablation=ablation_key,
            )
            payload = load_metrics_payload(args.artifact_root, experiment_name)

            score_payloads: dict[str, Any] = {}
            if payload is not None:
                selected = payload.get("selected")
                scores_root = payload.get("scores")
                if not isinstance(selected, dict) and isinstance(scores_root, dict):
                    selected = scores_root.get("selected")
                if isinstance(selected, dict):
                    score_payloads["selected"] = selected
                score_payloads.update(score_metric_groups(payload, require_core=False))
            score_keys = (
                list(args.score_keys)
                if args.score_keys
                else [key for key in MISSING_RESULT_SCORE_KEYS if key in score_payloads]
                + sorted(set(score_payloads).difference(MISSING_RESULT_SCORE_KEYS))
            )
            if not score_keys:
                score_keys = list(args.score_keys or MISSING_RESULT_SCORE_KEYS)
            for score_key in score_keys:
                record: dict[str, Any] = {
                    "dataset": dataset,
                    "ablation_key": ablation_key,
                    "ablation_label": ablation_label,
                    "experiment_name": experiment_name,
                    "score_key": score_key,
                }
                score_payload = score_payloads.get(score_key)
                for metric_key in args.metric_keys:
                    metric_value = resolve_nested(score_payload, metric_key) if isinstance(score_payload, dict) else None
                    record[metric_key] = metric_value
                records.append(record)

    return records


def write_csv(path: Path, records: list[dict[str, Any]], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "ablation_key", "ablation_label", "experiment_name", "score_key", *metric_keys]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def render_markdown_table(
    records: list[dict[str, Any]],
    ablations: list[str],
    datasets: list[str],
    score_key: str,
    metric_key: str,
) -> str:
    rows = [record for record in records if record["score_key"] == score_key]
    header = ["Ablation", *datasets]
    lines = [
        f"Table: Ablation study on `{metric_key}` from `{score_key}`.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for ablation_key in ablations:
        ablation_label = ABLATION_LABELS.get(ablation_key, ablation_key)
        row = [ablation_label]
        for dataset in datasets:
            match = next(
                (
                    record
                    for record in rows
                    if record["dataset"] == dataset and record["ablation_key"] == ablation_key
                ),
                None,
            )
            row.append(format_metric_value(None if match is None else match.get(metric_key)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.f1_key is not None and not args.markdown_metric_keys:
        args.markdown_metric_keys = [args.f1_key]
        print(
            "[AblationSummary] `--f1_key` is deprecated; "
            "prefer `--markdown_metric_keys` instead."
        )
    datasets = [dataset.upper() for dataset in args.datasets]
    records = build_records(args)

    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    write_csv(csv_path, records, args.metric_keys)
    write_json(json_path, records)

    print(f"[AblationSummary] wrote CSV: {csv_path}")
    print(f"[AblationSummary] wrote JSON: {json_path}")
    actual_score_keys = sorted({str(record["score_key"]) for record in records})
    print("[AblationSummary] included score keys: " + ", ".join(actual_score_keys))
    print(
        "[AblationSummary] included metric keys: "
        + ", ".join(args.metric_keys)
    )

    for metric_key in args.markdown_metric_keys:
        print()
        print(render_markdown_table(records, args.ablations, datasets, args.markdown_score_key, metric_key))


if __name__ == "__main__":
    main()
