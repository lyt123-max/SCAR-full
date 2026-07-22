from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from collect_results import ABLATION_LABELS, DEFAULT_ABLATIONS, format_metric_value


DEFAULT_DATASETS = ["MSL", "SMAP", "PSM", "SWAT", "SMD", "TEP", "GECCO", "GENESIS"]
PREFERRED_SUBSCORE_ORDER = [
    "completion_scale8",
    "completion_scale32",
    "knn_distance",
    "state_novelty",
    "soft_support_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a global ablation summary across multiple dataset-specific artifact roots. "
            "Outputs a wide table with main ROC-AUC / PR-AUC and all discovered subscore ROC-AUC / PR-AUC."
        )
    )
    parser.add_argument("--artifact_roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--ablations", nargs="*", default=DEFAULT_ABLATIONS)
    parser.add_argument(
        "--main_score_key",
        type=str,
        default="auto",
        help=(
            "Main score key used for the summary. "
            "Use `auto` to follow each experiment's selected_score_key; "
            "otherwise use a fixed key such as `cdf_mean`."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def unique_upper(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        name = item.upper()
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def infer_dataset(
    experiment_dir_name: str,
    experiment_name: str,
    config: dict[str, Any] | None,
    artifact_root: Path,
    dataset_candidates: list[str],
) -> str | None:
    if isinstance(config, dict):
        dataset = config.get("dataset")
        if dataset is not None:
            return str(dataset).upper()

    for candidate_name in (experiment_dir_name, experiment_name):
        exp_lower = candidate_name.lower()
        for dataset in dataset_candidates:
            if exp_lower.startswith(f"{dataset.lower()}_ablation_"):
                return dataset

    root_lower = artifact_root.name.lower()
    for dataset in dataset_candidates:
        if dataset.lower() in root_lower:
            return dataset
    return None


def infer_ablation_key(experiment_name: str, experiment_dir_name: str) -> str | None:
    for candidate_name in (experiment_dir_name, experiment_name):
        exp_lower = candidate_name.lower()
        for ablation_key in sorted(ABLATION_LABELS, key=len, reverse=True):
            marker = f"_ablation_{ablation_key}"
            if marker not in exp_lower:
                continue
            remainder = exp_lower.split(marker, 1)[1]
            if remainder == "" or remainder.startswith("_"):
                return ablation_key
    return None


def resolve_main_score_key(metrics_payload: dict[str, Any], requested_key: str) -> str:
    if requested_key != "auto":
        return requested_key
    selected_key = metrics_payload.get("selected_score_key")
    if isinstance(selected_key, str) and selected_key:
        return selected_key
    return "cdf_mean"


def preferred_subscore_sort_key(name: str) -> tuple[int, str]:
    if name in PREFERRED_SUBSCORE_ORDER:
        return (PREFERRED_SUBSCORE_ORDER.index(name), name)
    return (len(PREFERRED_SUBSCORE_ORDER) + 100, name)


def discover_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    requested_datasets = [item.upper() for item in args.datasets]
    dataset_filter = set(requested_datasets)
    dataset_candidates = unique_upper([*requested_datasets, *DEFAULT_DATASETS])
    ablation_filter = set(args.ablations)
    records: list[dict[str, Any]] = []
    discovered_subscores: set[str] = set()

    for artifact_root in args.artifact_roots:
        for metrics_path in sorted(artifact_root.rglob("test_metrics.json")):
            experiment_dir = metrics_path.parent
            metrics_payload = load_json(metrics_path)
            if not isinstance(metrics_payload, dict):
                continue

            config_payload = load_json(experiment_dir / "config.json")
            experiment_dir_name = experiment_dir.name
            experiment_name = (
                str(config_payload.get("experiment_name"))
                if isinstance(config_payload, dict) and config_payload.get("experiment_name")
                else experiment_dir_name
            )
            dataset = infer_dataset(
                experiment_dir_name,
                experiment_name,
                config_payload,
                artifact_root,
                dataset_candidates,
            )
            ablation_key = infer_ablation_key(experiment_name, experiment_dir_name)

            if dataset is None or ablation_key is None:
                continue
            if dataset_filter and dataset not in dataset_filter:
                continue
            if ablation_filter and ablation_key not in ablation_filter:
                continue

            resolved_main_score_key = resolve_main_score_key(metrics_payload, args.main_score_key)
            main_payload = metrics_payload.get(resolved_main_score_key, {})
            subscore_payload = metrics_payload.get("subscores", {})

            if not isinstance(subscore_payload, dict):
                subscore_payload = {}

            for key, value in subscore_payload.items():
                if isinstance(value, dict):
                    discovered_subscores.add(key)

            records.append(
                {
                    "dataset": dataset,
                    "ablation_key": ablation_key,
                    "ablation_label": ABLATION_LABELS.get(ablation_key, ablation_key),
                    "experiment_name": experiment_name,
                    "experiment_dir": str(experiment_dir),
                    "artifact_root": str(artifact_root),
                    "evaluation_protocol": metrics_payload.get("evaluation_protocol"),
                    "main_score_key": resolved_main_score_key,
                    "main": main_payload if isinstance(main_payload, dict) else {},
                    "subscores": subscore_payload,
                }
            )

    dataset_order = {name: idx for idx, name in enumerate([item.upper() for item in args.datasets])}
    ablation_order = {name: idx for idx, name in enumerate(args.ablations)}
    records.sort(
        key=lambda row: (
            dataset_order.get(row["dataset"], 10_000),
            ablation_order.get(row["ablation_key"], 10_000),
            row["experiment_name"],
        )
    )

    ordered_subscores = sorted(discovered_subscores, key=preferred_subscore_sort_key)
    return records, ordered_subscores


def build_wide_rows(records: list[dict[str, Any]], subscore_keys: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "dataset": record["dataset"],
            "ablation_key": record["ablation_key"],
            "ablation_label": record["ablation_label"],
            "experiment_name": record["experiment_name"],
            "experiment_dir": record["experiment_dir"],
            "artifact_root": record["artifact_root"],
            "evaluation_protocol": record["evaluation_protocol"],
            "main_score_key": record["main_score_key"],
            "main_roc_auc": record["main"].get("roc_auc"),
            "main_pr_auc": record["main"].get("pr_auc"),
            "main_vus_roc": record["main"].get("vus_roc"),
            "main_vus_pr": record["main"].get("vus_pr"),
        }
        for subscore_key in subscore_keys:
            payload = record["subscores"].get(subscore_key, {})
            row[f"{subscore_key}_roc_auc"] = payload.get("roc_auc") if isinstance(payload, dict) else None
            row[f"{subscore_key}_pr_auc"] = payload.get("pr_auc") if isinstance(payload, dict) else None
            row[f"{subscore_key}_vus_roc"] = payload.get("vus_roc") if isinstance(payload, dict) else None
            row[f"{subscore_key}_vus_pr"] = payload.get("vus_pr") if isinstance(payload, dict) else None
        rows.append(row)
    return rows


def build_long_rows(records: list[dict[str, Any]], subscore_keys: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "dataset": record["dataset"],
                "ablation_key": record["ablation_key"],
                "ablation_label": record["ablation_label"],
                "experiment_name": record["experiment_name"],
                "experiment_dir": record["experiment_dir"],
                "evaluation_protocol": record["evaluation_protocol"],
                "score_group": "main",
                "score_key": record["main_score_key"],
                "roc_auc": record["main"].get("roc_auc"),
                "pr_auc": record["main"].get("pr_auc"),
                "vus_roc": record["main"].get("vus_roc"),
                "vus_pr": record["main"].get("vus_pr"),
            }
        )
        for subscore_key in subscore_keys:
            payload = record["subscores"].get(subscore_key, {})
            rows.append(
                {
                    "dataset": record["dataset"],
                    "ablation_key": record["ablation_key"],
                    "ablation_label": record["ablation_label"],
                    "experiment_name": record["experiment_name"],
                    "experiment_dir": record["experiment_dir"],
                    "evaluation_protocol": record["evaluation_protocol"],
                    "score_group": "subscore",
                    "score_key": subscore_key,
                    "roc_auc": payload.get("roc_auc") if isinstance(payload, dict) else None,
                    "pr_auc": payload.get("pr_auc") if isinstance(payload, dict) else None,
                    "vus_roc": payload.get("vus_roc") if isinstance(payload, dict) else None,
                    "vus_pr": payload.get("vus_pr") if isinstance(payload, dict) else None,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, records: list[dict[str, Any]], subscore_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_order: list[str] = []
    seen: set[str] = set()
    for record in records:
        dataset = record["dataset"]
        if dataset not in seen:
            seen.add(dataset)
            dataset_order.append(dataset)

    lines: list[str] = [
        "# Global Ablation AUC Summary",
        "",
        f"Main score key: `{records[0]['main_score_key']}`" if records else "Main score key: `N/A`",
        "",
    ]

    for dataset in dataset_order:
        dataset_rows = [row for row in records if row["dataset"] == dataset]
        header = ["Ablation", "Main ROC-AUC", "Main PR-AUC", "Main VUS-ROC", "Main VUS-PR"]
        for subscore_key in subscore_keys:
            header.extend(
                [
                    f"{subscore_key} ROC",
                    f"{subscore_key} PR",
                    f"{subscore_key} VUS-ROC",
                    f"{subscore_key} VUS-PR",
                ]
            )

        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in dataset_rows:
            values = [
                str(row["ablation_label"]),
                format_metric_value(row["main"].get("roc_auc")),
                format_metric_value(row["main"].get("pr_auc")),
                format_metric_value(row["main"].get("vus_roc")),
                format_metric_value(row["main"].get("vus_pr")),
            ]
            for subscore_key in subscore_keys:
                payload = row["subscores"].get(subscore_key, {})
                values.append(format_metric_value(payload.get("roc_auc") if isinstance(payload, dict) else None))
                values.append(format_metric_value(payload.get("pr_auc") if isinstance(payload, dict) else None))
                values.append(format_metric_value(payload.get("vus_roc") if isinstance(payload, dict) else None))
                values.append(format_metric_value(payload.get("vus_pr") if isinstance(payload, dict) else None))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records, subscore_keys = discover_records(args)
    wide_rows = build_wide_rows(records, subscore_keys)
    long_rows = build_long_rows(records, subscore_keys)

    wide_fieldnames = [
        "dataset",
        "ablation_key",
        "ablation_label",
        "experiment_name",
        "experiment_dir",
        "artifact_root",
        "evaluation_protocol",
        "main_score_key",
        "main_roc_auc",
        "main_pr_auc",
        "main_vus_roc",
        "main_vus_pr",
    ]
    for subscore_key in subscore_keys:
        wide_fieldnames.extend(
            [
                f"{subscore_key}_roc_auc",
                f"{subscore_key}_pr_auc",
                f"{subscore_key}_vus_roc",
                f"{subscore_key}_vus_pr",
            ]
        )

    long_fieldnames = [
        "dataset",
        "ablation_key",
        "ablation_label",
        "experiment_name",
        "experiment_dir",
        "evaluation_protocol",
        "score_group",
        "score_key",
        "roc_auc",
        "pr_auc",
        "vus_roc",
        "vus_pr",
    ]

    wide_csv_path = args.output_prefix.with_name(f"{args.output_prefix.name}_wide").with_suffix(".csv")
    long_csv_path = args.output_prefix.with_name(f"{args.output_prefix.name}_long").with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")

    write_csv(wide_csv_path, wide_rows, wide_fieldnames)
    write_csv(long_csv_path, long_rows, long_fieldnames)
    write_json(
        json_path,
        {
            "main_score_key": args.main_score_key,
            "artifact_roots": [str(path) for path in args.artifact_roots],
            "subscore_keys": subscore_keys,
            "records": wide_rows,
        },
    )
    write_markdown(md_path, records, subscore_keys)

    print(f"[GlobalAUCTable] wrote wide CSV: {wide_csv_path}")
    print(f"[GlobalAUCTable] wrote long CSV: {long_csv_path}")
    print(f"[GlobalAUCTable] wrote JSON: {json_path}")
    print(f"[GlobalAUCTable] wrote Markdown: {md_path}")
    print(
        "[GlobalAUCTable] discovered subscore keys: "
        + (", ".join(subscore_keys) if subscore_keys else "<none>")
    )
    print(f"[GlobalAUCTable] collected experiments: {len(records)}")


if __name__ == "__main__":
    main()
