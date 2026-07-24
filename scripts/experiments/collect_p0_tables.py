from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.score_outputs import flatten_score_metrics


MAIN_DATASETS = ("MSL", "PSM", "SMAP", "SMD", "SWAT")
BASELINES = ("PaAno", "MEMTO", "PUAD", "PGRF-Net")
E9_RATIOS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)
CATCH_DATASET_COUNT = 30
TSB_EXPECTED = {("M", "tuning"): 20, ("M", "eval"): 180, ("U", "tuning"): 48, ("U", "eval"): 350}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect formal P0 outputs as tables only.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            nested = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, nested))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        result[prefix] = value
    return result


def _metric_row(path: Path, *, require_core: bool = True) -> dict[str, Any]:
    return {
        **flatten_score_metrics(_load(path), require_core=require_core),
        "metric_file": str(path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ratio_tag(value: float) -> str:
    return str(value).replace(".", "p")


def _finite_mean(values: Iterable[Any]) -> float | None:
    numeric = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numeric.append(number)
    return sum(numeric) / len(numeric) if numeric else None


def collect(artifact_root: Path, output_dir: Path, *, strict: bool) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    from scripts.tsb_ad.collect_results import collect as collect_tsb

    official_table_names: list[str] = []
    for edition, split in TSB_EXPECTED:
        tsb_collection = collect_tsb(
            artifact_root / "tsb_ad",
            edition,
            split,
            seed=42,
        )
        if tsb_collection["failures"]:
            missing.append(
                f"TSB-AD {edition}/{split} collector failures: "
                f"{tsb_collection['failures']}"
            )
        source_dir = artifact_root / "tsb_ad" / edition / split
        for source_name, suffix in (
            ("results_official_average_wide.csv", "official_average"),
            ("results_dataset_macro_average_wide.csv", "dataset_macro_average"),
        ):
            target_name = f"table_p0_tsb_{edition.lower()}_{split}_{suffix}.csv"
            shutil.copyfile(source_dir / source_name, output_dir / target_name)
            official_table_names.append(target_name)

    def required(path: Path) -> Path | None:
        if path.is_file() and path.stat().st_size > 0:
            return path
        missing.append(str(path))
        return None

    main_rows = []
    for dataset in MAIN_DATASETS:
        experiment = artifact_root / f"scar_main_{dataset.lower()}_seed42"
        metrics = required(experiment / "test_metrics.json")
        if metrics:
            main_rows.append({"method": "SCAR", "dataset": dataset, **_metric_row(metrics)})

    baseline_rows = []
    efficiency_rows = []
    for dataset in MAIN_DATASETS:
        scar_dir = artifact_root / f"scar_main_{dataset.lower()}_seed42"
        scar_resource = required(scar_dir / "resource_metrics.json")
        scar_timing = required(
            artifact_root / f"scar_efficiency_{dataset.lower()}_seed42" / "timing.json"
        )
        if scar_resource and scar_timing:
            efficiency_rows.append(
                {
                    "method": "SCAR",
                    "dataset": dataset,
                    **{f"resource.{key}": value for key, value in _flatten(_load(scar_resource)).items()},
                    **{f"timing.{key}": value for key, value in _flatten(_load(scar_timing)).items()},
                }
            )
        for method in BASELINES:
            name = f"baseline_{method.lower().replace('-', '_')}_{dataset.lower()}_seed42"
            experiment = artifact_root / name
            metrics = required(experiment / "metrics.json")
            timing = required(experiment / "timing.json")
            resource = required(experiment / "resource_metrics.json")
            if metrics:
                baseline_rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        **_metric_row(metrics, require_core=False),
                    }
                )
            if timing and resource:
                efficiency_rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        **{
                            f"resource.{key}": value
                            for key, value in _flatten(_load(resource)).items()
                        },
                        **{
                            f"timing.{key}": value
                            for key, value in _flatten(_load(timing)).items()
                        },
                    }
                )

    catch_rows = []
    for metrics in sorted(artifact_root.glob("scar_catch_*_seed42/test_metrics.json")):
        dataset = metrics.parent.name.removeprefix("scar_catch_").removesuffix("_seed42")
        catch_rows.append({"dataset": dataset, **_metric_row(metrics)})
    if len(catch_rows) != CATCH_DATASET_COUNT:
        missing.append(f"CATCH result count: expected {CATCH_DATASET_COUNT}, found {len(catch_rows)}")

    e9_rows = []
    for dataset in MAIN_DATASETS:
        for ratio in E9_RATIOS:
            experiment = (
                artifact_root / f"scar_main_{dataset.lower()}_seed42"
                if ratio == 0.02
                else artifact_root / f"scar_e9_{dataset.lower()}_clean_{_ratio_tag(ratio)}"
            )
            metrics = required(experiment / "test_metrics.json")
            if metrics:
                e9_rows.append(
                    {"dataset": dataset, "clean_ratio": ratio, **_metric_row(metrics)}
                )

    e10_rows = []
    for protocol_path in sorted(
        artifact_root.glob("scar_e10_*_contam_*_clean_*_fold_*/contamination_protocol.json")
    ):
        payload = _load(protocol_path)
        heldout = payload.get("heldout_point_metrics", {})
        heldout_scores = payload.get("heldout_score_metrics", {})
        score_columns = (
            flatten_score_metrics(
                {
                    "selected_score_key": payload.get("selected_score_key", "cdf_mean"),
                    "selected": heldout,
                    **heldout_scores,
                }
            )
            if heldout_scores
            else {"roc_auc": heldout.get("roc_auc"), "pr_auc": heldout.get("pr_auc")}
        )
        e10_rows.append(
            {
                "dataset": protocol_path.parent.name.split("_")[2].upper(),
                "fold": payload.get("fold"),
                "contamination_ratio": payload.get("contamination_ratio_requested"),
                "clean_ratio": _load(protocol_path.parent / "config.json").get("clean_ratio"),
                **score_columns,
                "protocol_file": str(protocol_path),
            }
        )
    expected_e10 = len(MAIN_DATASETS) * 3 * (len((0.01, 0.03, 0.05)) * 2 + 3)
    if len(e10_rows) != expected_e10:
        missing.append(f"E10 nonzero count: expected {expected_e10}, found {len(e10_rows)}")
    zero_rows = []
    for heldout_path in sorted(
        artifact_root.glob("scar_e10_zero_*_*_fold_*/heldout_metrics.json")
    ):
        payload = _load(heldout_path)
        metrics = payload.get("metrics", {})
        all_scores = payload.get("score_metrics", {})
        score_columns = (
            flatten_score_metrics(
                {
                    "selected_score_key": payload.get("selected_score_key", "cdf_mean"),
                    "selected": metrics,
                    **all_scores,
                }
            )
            if all_scores
            else {"roc_auc": metrics.get("roc_auc"), "pr_auc": metrics.get("pr_auc")}
        )
        parts = heldout_path.parent.name.split("_")
        zero_rows.append(
            {
                "dataset": parts[3].upper(),
                "fold": payload.get("fold"),
                "contamination_ratio": 0.0,
                "purification": parts[4],
                **score_columns,
                "protocol_file": str(heldout_path),
            }
        )
    expected_zero = len(MAIN_DATASETS) * 3 * 2
    if len(zero_rows) != expected_zero:
        missing.append(f"E10 zero count: expected {expected_zero}, found {len(zero_rows)}")
    e10_rows.extend(zero_rows)

    tsb_rows = []
    tsb_summary_rows = []
    for (edition, split), expected_count in TSB_EXPECTED.items():
        metric_files = sorted(
            (artifact_root / "tsb_ad" / edition / split / "seed_42").glob(
                "*/test_metrics.json"
            )
        )
        if len(metric_files) != expected_count:
            missing.append(
                f"TSB-AD {edition}/{split}: expected {expected_count}, found {len(metric_files)}"
            )
        track_rows = []
        for metrics in metric_files:
            row = {
                "edition": edition,
                "split": split,
                "series": metrics.parent.name,
                **_metric_row(metrics),
            }
            track_rows.append(row)
            tsb_rows.append(row)
        metric_keys = sorted(
            {
                key
                for row in track_rows
                for key, value in row.items()
                if key not in {"edition", "split", "series", "metric_file", "selected_score_key"}
                and isinstance(value, (int, float))
            }
        )
        tsb_summary_rows.append(
            {
                "edition": edition,
                "split": split,
                "series_count": len(track_rows),
                **{
                    f"mean_{key}": _finite_mean(row.get(key) for row in track_rows)
                    for key in metric_keys
                },
            }
        )

    tables = {
        "table_p0_main5.csv": main_rows,
        "table_p0_baselines.csv": baseline_rows,
        "table_p0_efficiency.csv": efficiency_rows,
        "table_p0_catch.csv": catch_rows,
        "table_p0_e9.csv": e9_rows,
        "table_p0_e10.csv": e10_rows,
        "table_p0_tsb_series.csv": tsb_rows,
        "table_p0_tsb_summary.csv": tsb_summary_rows,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / name, rows)

    summary = {
        "schema_version": 1,
        "formal_output_mode": "tables_and_text_only",
        "missing": missing,
        "counts": {name: len(rows) for name, rows in tables.items()},
        "official_tsb_tables": official_table_names,
    }
    (output_dir / "p0_collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    text_lines = [
        "SCAR rebuttal P0 collection",
        "Output mode: tables and text only",
        *(f"{name}: {len(rows)} rows" for name, rows in tables.items()),
        f"Missing or invalid requirements: {len(missing)}",
    ]
    (output_dir / "p0_collection_summary.txt").write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )
    if strict and missing:
        raise RuntimeError(
            "P0 collection is incomplete. See p0_collection_summary.json; "
            f"missing_count={len(missing)}"
        )
    return summary


def main() -> None:
    args = parse_args()
    summary = collect(args.artifact_root, args.output_dir, strict=args.strict)
    print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
