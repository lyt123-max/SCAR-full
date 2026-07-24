from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from collect_results import ABLATION_LABELS, format_metric_value, load_metrics_payload, resolve_nested
from coremad.config import CoReMADConfig
from coremad.data import (
    TEP_FAULT_FILES,
    TEP_NORMAL_FILES,
    TEP_NUM_INPUT_CHANNELS,
    _load_tep_raw_dataset_bundle,
    tep_file_fault_to_idv,
)


DATASET_LABELS = {
    "SMD": "SMD",
    "SWAT": "SWaT",
    "TEP": "TEP",
}

FUSION_SCORE_LABELS = {
    "raw_max": "Raw Max",
    "zscore_mean": "Z-Score Mean",
    "cdf_max": "CDF/PIT Max",
    "cdf_mean": "CDF/PIT Mean",
    "cdf_softmax": "CDF/PIT Softmax",
}

SUBSCORE_LABELS = {
    "knn_distance": "kNN Distance",
    "state_novelty": "State Novelty",
    "completion_scale8": "Completion (scale=8)",
    "completion_scale32": "Completion (scale=32)",
}

POINTWISE_MAIN_ABLATIONS = [
    "full",
    "wo_decomposition",
    "wo_state_aware_representation",
    "global_retrieval",
    "state_only_retrieval",
    "context_only_retrieval",
    "dual_condition_retrieval",
    "single_scale_short",
    "single_scale_long",
    "multi_scale",
]

TEP_ABLATIONS = [
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

MAIN_METRIC_KEYS = [
    "roc_auc",
    "pr_auc",
    "point_best_f1",
    "pa_best_f1",
    "aff_precision",
    "aff_recall",
    "aff_f1",
    "vus_roc",
    "vus_pr",
]
TEP_MAIN_METRIC_KEYS = list(MAIN_METRIC_KEYS)
STABILITY_SUBSCORES = ["knn_distance", "state_novelty", "completion_scale8", "completion_scale32"]
FUSION_FAMILY = ["raw_max", "zscore_mean", "cdf_max", "cdf_mean", "cdf_softmax"]
TEP_MECHANISM_METRICS = [
    "SMC@K",
    "SFR",
    "Mode-FPR-Std",
    "SMR@K",
    "delta_mem_mode",
    "EE95",
    "Tail@0.99_error",
    "Fault-Consistency-Std",
    "Evidence-Dom-Consistency",
    "Proto-Purity",
    "Proto-Entropy",
    "Cross-mode-margin",
]
TEP_FIGURES = [
    "state_embedding.png",
    "normal_final_violin_by_mode.png",
    "retrieval_mode_confusion_heatmap.png",
    "exceedance_plot.png",
    "fault_file_gain_heatmap.png",
    "fault_evidence_heatmap.png",
    "fault_triplet_consistency.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-oriented ablation/TEP tables.")
    parser.add_argument("--artifact_root", type=Path, default=Path("./artifacts"))
    parser.add_argument("--tep_data_root", type=Path, default=Path("."))
    parser.add_argument("--experiment_name_template", type=str, default="{dataset_lower}_ablation_{ablation}")
    parser.add_argument("--output_dir", type=Path, default=Path("./artifacts/paper_tables"))
    parser.add_argument("--pointwise_datasets", nargs="+", default=["SMD", "SWAT"])
    parser.add_argument("--pointwise_ablations", nargs="+", default=POINTWISE_MAIN_ABLATIONS)
    parser.add_argument("--pointwise_score_key", type=str, default="cdf_mean")
    parser.add_argument("--pointwise_metric_keys", nargs="+", default=MAIN_METRIC_KEYS)
    parser.add_argument("--fusion_score_keys", nargs="+", default=FUSION_FAMILY)
    parser.add_argument("--fusion_metric_keys", nargs="+", default=MAIN_METRIC_KEYS)
    parser.add_argument("--stability_datasets", nargs="+", default=["SMD", "SWAT"])
    parser.add_argument("--stability_metric_key", type=str, default="pr_auc")
    parser.add_argument("--stability_subscores", nargs="+", default=STABILITY_SUBSCORES)
    parser.add_argument("--fixed_fusion_key", type=str, default="cdf_mean")
    parser.add_argument("--tep_ablations", nargs="+", default=TEP_ABLATIONS)
    parser.add_argument("--tep_score_key", type=str, default="cdf_mean")
    return parser.parse_args()


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset.upper(), dataset.upper())


def resolve_experiment_dir(artifact_root: Path, experiment_name: str) -> Path | None:
    experiment_dir = artifact_root / experiment_name
    if experiment_dir.exists():
        return experiment_dir
    candidates = sorted(
        (path for path in artifact_root.glob(f"{experiment_name}_*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0]


def load_payload_and_dir(
    artifact_root: Path,
    experiment_name_template: str,
    dataset: str,
    ablation: str,
) -> tuple[dict[str, Any] | None, Path | None, str]:
    experiment_name = experiment_name_template.format(
        dataset=dataset.upper(),
        dataset_lower=dataset.lower(),
        ablation=ablation,
    )
    payload = load_metrics_payload(artifact_root, experiment_name)
    exp_dir = resolve_experiment_dir(artifact_root, experiment_name)
    return payload, exp_dir, experiment_name


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


def write_markdown_table(path: Path, title: str, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_tep_mode_fault(file_name: str) -> tuple[int, int]:
    stem = Path(file_name).stem
    if not stem.startswith("m") or "d" not in stem:
        raise ValueError(f"Unsupported TEP file name: {file_name}")
    mode_part, fault_part = stem.split("d", maxsplit=1)
    return int(mode_part[1:]), tep_file_fault_to_idv(int(fault_part))


def chunked_text(items: list[str], chunk_size: int) -> list[str]:
    return [", ".join(items[idx : idx + chunk_size]) for idx in range(0, len(items), chunk_size)]


def tex_shortstack(lines: list[str]) -> str:
    return r"\shortstack[l]{" + r"\\".join(lines) + "}"


def build_pointwise_main_table(args: argparse.Namespace) -> dict[str, Any]:
    datasets = [item.upper() for item in args.pointwise_datasets]
    rows: list[dict[str, Any]] = []
    for ablation in args.pointwise_ablations:
        row: dict[str, Any] = {
            "ablation_key": ablation,
            "ablation_label": ABLATION_LABELS.get(ablation, ablation),
        }
        for dataset in datasets:
            payload, _, experiment_name = load_payload_and_dir(
                args.artifact_root,
                args.experiment_name_template,
                dataset,
                ablation,
            )
            score_payload = resolve_nested(payload, args.pointwise_score_key) if payload is not None else None
            row[f"{dataset}_experiment_name"] = experiment_name
            for metric_key in args.pointwise_metric_keys:
                value = resolve_nested(score_payload, metric_key) if isinstance(score_payload, dict) else None
                row[f"{dataset}_{metric_key}"] = value
        rows.append(row)

    fieldnames = ["ablation_key", "ablation_label"]
    for dataset in datasets:
        fieldnames.append(f"{dataset}_experiment_name")
        for metric_key in args.pointwise_metric_keys:
            fieldnames.append(f"{dataset}_{metric_key}")

    write_csv(args.output_dir / "pointwise_main_ablation.csv", rows, fieldnames)
    write_json(args.output_dir / "pointwise_main_ablation.json", rows)

    md_header = ["Ablation"]
    for dataset in datasets:
        dataset_name = dataset_label(dataset)
        md_header.extend([f"{dataset_name} PR-AUC", f"{dataset_name} ROC-AUC", f"{dataset_name} F1"])
    md_rows = []
    for row in rows:
        md_row = [str(row["ablation_label"])]
        for dataset in datasets:
            for metric_key in args.pointwise_metric_keys:
                md_row.append(format_metric_value(row.get(f"{dataset}_{metric_key}")))
        md_rows.append(md_row)
    write_markdown_table(
        args.output_dir / "pointwise_main_ablation.md",
        f"Point-wise Main Ablation ({args.pointwise_score_key})",
        md_header,
        md_rows,
    )
    return {"datasets": datasets, "rows": rows}


def build_a3_fusion_table(args: argparse.Namespace) -> dict[str, Any]:
    datasets = [item.upper() for item in args.pointwise_datasets]
    rows: list[dict[str, Any]] = []
    for score_key in args.fusion_score_keys:
        row: dict[str, Any] = {
            "score_key": score_key,
            "score_label": FUSION_SCORE_LABELS.get(score_key, score_key),
        }
        for dataset in datasets:
            payload, _, experiment_name = load_payload_and_dir(
                args.artifact_root,
                args.experiment_name_template,
                dataset,
                "full",
            )
            score_payload = resolve_nested(payload, score_key) if payload is not None else None
            row[f"{dataset}_experiment_name"] = experiment_name
            for metric_key in args.fusion_metric_keys:
                value = resolve_nested(score_payload, metric_key) if isinstance(score_payload, dict) else None
                row[f"{dataset}_{metric_key}"] = value
        rows.append(row)

    fieldnames = ["score_key", "score_label"]
    for dataset in datasets:
        fieldnames.append(f"{dataset}_experiment_name")
        for metric_key in args.fusion_metric_keys:
            fieldnames.append(f"{dataset}_{metric_key}")

    write_csv(args.output_dir / "a3_fusion_family.csv", rows, fieldnames)
    write_json(args.output_dir / "a3_fusion_family.json", rows)

    md_header = ["Fusion"]
    for dataset in datasets:
        dataset_name = dataset_label(dataset)
        md_header.extend([f"{dataset_name} PR-AUC", f"{dataset_name} ROC-AUC", f"{dataset_name} F1"])
    md_rows = []
    for row in rows:
        md_row = [str(row["score_label"])]
        for dataset in datasets:
            for metric_key in args.fusion_metric_keys:
                md_row.append(format_metric_value(row.get(f"{dataset}_{metric_key}")))
        md_rows.append(md_row)
    write_markdown_table(
        args.output_dir / "a3_fusion_family.md",
        "A3 Fusion Family",
        md_header,
        md_rows,
    )
    return {"datasets": datasets, "rows": rows}


def compute_rank_map(score_map: dict[str, float]) -> dict[str, float]:
    ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while end < len(ordered) and ordered[end][1] == ordered[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for _, key in ordered[idx:end]:
            ranks[key] = avg_rank
        idx = end
    return ranks


def build_stability_table(args: argparse.Namespace) -> dict[str, Any]:
    datasets = [item.upper() for item in args.stability_datasets]
    candidate_methods = list(dict.fromkeys([*args.stability_subscores, *args.fusion_score_keys]))
    dataset_method_scores: dict[str, dict[str, float]] = {}
    oracle_scores: dict[str, float] = {}

    for dataset in datasets:
        payload, _, _ = load_payload_and_dir(
            args.artifact_root,
            args.experiment_name_template,
            dataset,
            "full",
        )
        if payload is None:
            continue
        method_scores: dict[str, float] = {}
        for method in candidate_methods:
            method_root = resolve_nested(payload, method)
            value = resolve_nested(method_root, args.stability_metric_key) if isinstance(method_root, dict) else None
            if value is not None:
                method_scores[method] = float(value)
        if not method_scores:
            continue
        dataset_method_scores[dataset] = method_scores
        raw_subscores = [
            score
            for name, score in method_scores.items()
            if name in args.stability_subscores
        ]
        if raw_subscores:
            oracle_scores[dataset] = max(raw_subscores)

    all_stats: list[dict[str, Any]] = []
    for method in candidate_methods:
        scores = [dataset_method_scores[dataset][method] for dataset in datasets if dataset in dataset_method_scores and method in dataset_method_scores[dataset]]
        if not scores:
            continue
        ranks = []
        wins = 0
        top2 = 0
        regrets = []
        for dataset in datasets:
            if dataset not in dataset_method_scores or method not in dataset_method_scores[dataset]:
                continue
            rank_map = compute_rank_map(dataset_method_scores[dataset])
            rank = rank_map[method]
            ranks.append(rank)
            if rank == 1:
                wins += 1
            if rank <= 2:
                top2 += 1
            if dataset in oracle_scores:
                regrets.append(oracle_scores[dataset] - dataset_method_scores[dataset][method])
        all_stats.append(
            {
                "method_key": method,
                "method_label": FUSION_SCORE_LABELS.get(method, SUBSCORE_LABELS.get(method, method)),
                "mean_score": mean(scores),
                "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
                "average_rank": mean(ranks) if ranks else None,
                "win_count": wins,
                "top2_count": top2,
                "mean_regret": mean(regrets) if regrets else None,
                "median_regret": median(regrets) if regrets else None,
                "max_regret": max(regrets) if regrets else None,
                "dataset_scores": {
                    dataset: dataset_method_scores[dataset][method]
                    for dataset in datasets
                    if dataset in dataset_method_scores and method in dataset_method_scores[dataset]
                },
            }
        )

    if not all_stats:
        payload = {"datasets": datasets, "metric_key": args.stability_metric_key, "all_methods": [], "focus_rows": []}
        write_json(args.output_dir / "stability_analysis.json", payload)
        write_csv(args.output_dir / "stability_analysis.csv", [], ["method_key"])
        write_markdown_table(
            args.output_dir / "stability_analysis.md",
            "Stability Analysis",
            ["Method", "Mean Score", "Std", "Avg Rank", "Wins", "Top-2", "Mean Regret", "Median Regret", "Max Regret"],
            [],
        )
        return payload

    best_fixed = max(
        (row for row in all_stats if row["method_key"] in args.stability_subscores),
        key=lambda row: row["mean_score"],
    )
    fixed_fusion = next(row for row in all_stats if row["method_key"] == args.fixed_fusion_key)
    oracle_focus = {
        "method_key": "oracle_best_subscore",
        "method_label": "Oracle Best Subscore",
        "mean_score": mean(oracle_scores.values()) if oracle_scores else None,
        "std_score": pstdev(list(oracle_scores.values())) if len(oracle_scores) > 1 else 0.0,
        "average_rank": 1.0 if oracle_scores else None,
        "win_count": len(oracle_scores),
        "top2_count": len(oracle_scores),
        "mean_regret": 0.0 if oracle_scores else None,
        "median_regret": 0.0 if oracle_scores else None,
        "max_regret": 0.0 if oracle_scores else None,
        "dataset_scores": oracle_scores,
    }
    focus_rows = [
        oracle_focus,
        {
            **best_fixed,
            "method_label": f"Best Fixed Subscore ({best_fixed['method_label']})",
        },
        {
            **fixed_fusion,
            "method_label": f"Fixed Fusion ({fixed_fusion['method_label']})",
        },
    ]

    fieldnames = [
        "method_key",
        "method_label",
        "mean_score",
        "std_score",
        "average_rank",
        "win_count",
        "top2_count",
        "mean_regret",
        "median_regret",
        "max_regret",
    ]
    write_csv(args.output_dir / "stability_analysis.csv", focus_rows, fieldnames)
    payload = {
        "datasets": datasets,
        "metric_key": args.stability_metric_key,
        "fixed_fusion_key": args.fixed_fusion_key,
        "best_fixed_subscore_key": best_fixed["method_key"],
        "all_methods": all_stats,
        "focus_rows": focus_rows,
    }
    write_json(args.output_dir / "stability_analysis.json", payload)

    md_rows = []
    for row in focus_rows:
        md_rows.append(
            [
                str(row["method_label"]),
                format_metric_value(row.get("mean_score")),
                format_metric_value(row.get("std_score")),
                format_metric_value(row.get("average_rank")),
                str(int(row["win_count"])) if row.get("win_count") is not None else "N/A",
                str(int(row["top2_count"])) if row.get("top2_count") is not None else "N/A",
                format_metric_value(row.get("mean_regret")),
                format_metric_value(row.get("median_regret")),
                format_metric_value(row.get("max_regret")),
            ]
        )
    write_markdown_table(
        args.output_dir / "stability_analysis.md",
        f"Stability Analysis ({args.stability_metric_key})",
        ["Method", "Mean Score", "Std", "Avg Rank", "Wins", "Top-2", "Mean Regret", "Median Regret", "Max Regret"],
        md_rows,
    )
    return payload


def load_mechanism_payload(exp_dir: Path | None) -> dict[str, Any] | None:
    if exp_dir is None:
        return None
    path = exp_dir / "tep_mechanism" / "mechanism_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_tep_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    for ablation in args.tep_ablations:
        payload, exp_dir, experiment_name = load_payload_and_dir(
            args.artifact_root,
            args.experiment_name_template,
            "TEP",
            ablation,
        )
        score_payload = resolve_nested(payload, args.tep_score_key) if payload is not None else None
        mechanism_payload = load_mechanism_payload(exp_dir)
        experiment_metrics: dict[str, Any] = {}
        a4_metrics: dict[str, Any] = {}
        if mechanism_payload is not None:
            experiments = mechanism_payload.get("experiments", {})
            if exp_dir is not None and exp_dir.name in experiments:
                experiment_metrics = experiments.get(exp_dir.name, {})
            elif experiments:
                experiment_metrics = next(iter(experiments.values()))
            a4_metrics = mechanism_payload.get("A4", {})

        row: dict[str, Any] = {
            "ablation_key": ablation,
            "ablation_label": ABLATION_LABELS.get(ablation, ablation),
            "experiment_name": experiment_name,
            "experiment_dir": str(exp_dir) if exp_dir is not None else "",
            "score_key": args.tep_score_key,
            "evaluation_protocol": resolve_nested(payload, "evaluation_protocol") if payload is not None else None,
        }
        for metric_key in TEP_MAIN_METRIC_KEYS:
            row[metric_key] = resolve_nested(score_payload, metric_key) if isinstance(score_payload, dict) else None
        for metric_key in TEP_MECHANISM_METRICS:
            row[metric_key] = experiment_metrics.get(metric_key)
        row["CG_mean"] = a4_metrics.get("CG_mean")
        row["%_multi_best"] = a4_metrics.get("%_multi_best")
        rows.append(row)

        if exp_dir is not None:
            figure_dir = exp_dir / "tep_mechanism"
            for figure_name in TEP_FIGURES:
                figure_path = figure_dir / figure_name
                if figure_path.exists():
                    figure_rows.append(
                        {
                            "ablation_key": ablation,
                            "ablation_label": ABLATION_LABELS.get(ablation, ablation),
                            "figure_name": figure_name,
                            "path": str(figure_path),
                        }
                    )

    write_csv(
        args.output_dir / "tep_sequence_results.csv",
        rows,
        [
            "ablation_key",
            "ablation_label",
            "experiment_name",
            "experiment_dir",
            "score_key",
            "evaluation_protocol",
            *TEP_MAIN_METRIC_KEYS,
            *TEP_MECHANISM_METRICS,
            "CG_mean",
            "%_multi_best",
        ],
    )
    write_json(
        args.output_dir / "tep_sequence_results.json",
        {
            "rows": rows,
            "figures": figure_rows,
        },
    )

    metric_rows = [
        [
            str(row["ablation_label"]),
            format_metric_value(row.get("roc_auc")),
            format_metric_value(row.get("pr_auc")),
            format_metric_value(row.get("best_f1")),
        ]
        for row in rows
    ]
    mechanism_rows = [
        [
            str(row["ablation_label"]),
            format_metric_value(row.get("SMC@K")),
            format_metric_value(row.get("SFR")),
            format_metric_value(row.get("Mode-FPR-Std")),
            format_metric_value(row.get("SMR@K")),
            format_metric_value(row.get("delta_mem_mode")),
            format_metric_value(row.get("EE95")),
            format_metric_value(row.get("Tail@0.99_error")),
            format_metric_value(row.get("Fault-Consistency-Std")),
            format_metric_value(row.get("Evidence-Dom-Consistency")),
            format_metric_value(row.get("Proto-Purity")),
            format_metric_value(row.get("Proto-Entropy")),
            format_metric_value(row.get("Cross-mode-margin")),
            format_metric_value(row.get("CG_mean")),
            format_metric_value(row.get("%_multi_best")),
        ]
        for row in rows
    ]
    figure_md_rows = [
        [
            str(item["ablation_label"]),
            str(item["figure_name"]),
            str(item["path"]),
        ]
        for item in figure_rows
    ]

    md_lines = [
        "# TEP Sequence-level Results and Mechanism Assets",
        "",
        "## Sequence-level Main Metrics",
        "",
        "| Ablation | AUROC | AUPRC | F1 |",
        "| --- | --- | --- | --- |",
    ]
    md_lines.extend("| " + " | ".join(row) + " |" for row in metric_rows)
    md_lines.extend(
        [
            "",
            "## Mechanism Metrics",
            "",
            "| Ablation | SMC@K | SFR | Mode-FPR-Std | SMR@K | delta_mem_mode | EE95 | Tail@0.99_error | Fault-Consistency-Std | Evidence-Dom-Consistency | Proto-Purity | Proto-Entropy | Cross-mode-margin | CG_mean | %_multi_best |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    md_lines.extend("| " + " | ".join(row) + " |" for row in mechanism_rows)
    md_lines.extend(
        [
            "",
            "## Figure Inventory",
            "",
            "| Ablation | Figure | Path |",
            "| --- | --- | --- |",
        ]
    )
    md_lines.extend("| " + " | ".join(row) + " |" for row in figure_md_rows)
    (args.output_dir / "tep_sequence_results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {"rows": rows, "figures": figure_rows}


def build_tep_dataset_statistics(args: argparse.Namespace) -> dict[str, Any]:
    config = CoReMADConfig(
        dataset="TEP",
        data_root=str(args.tep_data_root),
        n_channels=TEP_NUM_INPUT_CHANNELS,
    )
    bundle = _load_tep_raw_dataset_bundle(config)

    normal_pairs = [parse_tep_mode_fault(name) for name in TEP_NORMAL_FILES]
    fault_pairs = [parse_tep_mode_fault(name) for name in TEP_FAULT_FILES]
    mode_ids = sorted({mode_id for mode_id, _ in normal_pairs})
    normal_ids = sorted({fault_id for _, fault_id in normal_pairs})
    fault_ids = sorted({fault_id for _, fault_id in fault_pairs})

    mode_labels = [f"M{mode_id}" for mode_id in mode_ids]
    normal_labels = [f"IDV{fault_id}" for fault_id in normal_ids]
    fault_labels = [f"IDV{fault_id}" for fault_id in fault_ids]

    row = {
        "dataset": "TEP",
        "domain": "Industrial process",
        "channels": TEP_NUM_INPUT_CHANNELS,
        "mode_ids": mode_ids,
        "normal_ids": normal_ids,
        "fault_ids": fault_ids,
        "normal_train_steps": int(bundle.train.shape[0]),
        "normal_val_steps": int(bundle.val.shape[0]) if bundle.val is not None else 0,
        "fault_test_sequences": len(bundle.test_sequences or []),
        "total_fault_steps": int(sum(len(sequence) for sequence in (bundle.test_sequences or []))),
        "label_granularity": "Sequence-level",
        "validation_split": "tail",
        "source_description": "Tennessee Eastman selected multimode fault data",
        "normal_train_files": [Path(name).stem for name in TEP_NORMAL_FILES],
        "fault_test_files": [Path(name).stem for name in TEP_FAULT_FILES],
    }

    write_json(args.output_dir / "dataset_statistics_tep.json", row)
    write_markdown_table(
        args.output_dir / "dataset_statistics_tep.md",
        "TEP Dataset Statistics",
        [
            "Dataset",
            "Domain",
            "#Channels",
            "Mode IDs",
            "Normal IDs",
            "Fault IDs",
            "Normal Train",
            "Normal Val",
            "Fault Test Seq.",
            "Total Fault Steps",
            "Label Granularity",
        ],
        [
            [
                row["dataset"],
                row["domain"],
                f"{row['channels']}",
                ", ".join(mode_labels),
                ", ".join(normal_labels),
                ", ".join(fault_labels),
                f"{row['normal_train_steps']:,}",
                f"{row['normal_val_steps']:,}",
                f"{row['fault_test_sequences']:,}",
                f"{row['total_fault_steps']:,}",
                row["label_granularity"],
            ]
        ],
    )

    tex_mode_ids = tex_shortstack([
        ", ".join(mode_labels),
        f"(normal: {', '.join(normal_labels)})",
    ])
    tex_fault_ids = tex_shortstack(chunked_text(fault_labels, 4))
    tex_lines = [
        r"\textbf{(b) TEP sequence-level benchmark}",
        "",
        r"\vspace{2pt}",
        r"\begin{tabular}{llr p{2.2cm} p{3.8cm} rrrrll}",
        r"\toprule",
        r"Dataset & Domain & \#Channels & Mode IDs & Fault IDs & Normal Train & Normal Val & Fault Test Seq. & Total Fault Steps & Label Granularity & Source / Description \\",
        r"\midrule",
        (
            f"TEP & Industrial process & {row['channels']} & {tex_mode_ids} & {tex_fault_ids} & "
            f"{row['normal_train_steps']:,} & {row['normal_val_steps']:,} & {row['fault_test_sequences']:,} & "
            f"{row['total_fault_steps']:,} & Sequence-level & {row['source_description']} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2pt}",
        r"\parbox{0.98\textwidth}{\footnotesize",
        (
            "For TEP, normal training/validation files come from modes "
            f"{', '.join(mode_labels)} with normal identifier {', '.join(normal_labels)}. "
            "Fault evaluation uses sequence-level labels over fault files with IDs "
            f"{', '.join(fault_labels)}, and the validation split is constructed from normal "
            "training files using a tail split."
        ),
        r"}",
        "",
    ]
    (args.output_dir / "dataset_statistics_tep.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pointwise_main": build_pointwise_main_table(args),
        "a3_fusion_family": build_a3_fusion_table(args),
        "stability": build_stability_table(args),
        "tep": build_tep_summary(args),
        "tep_dataset_statistics": build_tep_dataset_statistics(args),
    }
    write_json(args.output_dir / "paper_tables_manifest.json", manifest)
    print(f"[PaperTables] wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
