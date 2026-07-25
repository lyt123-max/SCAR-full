from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.score_outputs import flatten_score_metrics


DATASETS = ("MSL", "PSM", "SMAP", "SMD", "SWAT")
METHODS = ("KNN", "LOF")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the five-dataset three-day rebuttal tables."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, name))
        return result
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {prefix: value}
    return {}


def metrics(path: Path) -> dict[str, Any]:
    return flatten_score_metrics(load_json(path), require_core=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing lightweight rebuttal artifact: {path}")
    return path


def collect(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    e10_rows = []
    for protocol_path in sorted(
        root.glob("scar_lite_e10_*_contam_*_clean_*_fold_*/contamination_protocol.json")
    ):
        payload = load_json(protocol_path)
        parts = protocol_path.parent.name.split("_")
        score_metrics = payload.get("heldout_score_metrics", {})
        e10_rows.append(
            {
                "dataset": parts[3].upper(),
                "fold": payload.get("fold"),
                "contamination_ratio": payload.get("contamination_ratio_requested"),
                "clean_ratio": load_json(protocol_path.parent / "config.json").get(
                    "clean_ratio"
                ),
                **flatten_score_metrics(
                    {
                        "selected_score_key": payload.get(
                            "selected_score_key", "cdf_mean"
                        ),
                        "selected": payload.get("heldout_point_metrics", {}),
                        **score_metrics,
                    },
                    require_core=False,
                ),
            }
        )
    if len(e10_rows) != 50:
        raise RuntimeError(f"Lightweight E10 requires 50 nonzero rows, found {len(e10_rows)}.")

    zero_rows = []
    for path in sorted(root.glob("scar_lite_e10_zero_*_*_fold_*/heldout_metrics.json")):
        payload = load_json(path)
        parts = path.parent.name.split("_")
        zero_rows.append(
            {
                "dataset": parts[4].upper(),
                "fold": payload.get("fold"),
                "contamination_ratio": 0.0,
                "purification": parts[5],
                **flatten_score_metrics(
                    {
                        "selected_score_key": payload.get(
                            "selected_score_key", "cdf_mean"
                        ),
                        "selected": payload.get("metrics", {}),
                        **payload.get("score_metrics", {}),
                    },
                    require_core=False,
                ),
            }
        )
    if len(zero_rows) != 20:
        raise RuntimeError(f"Lightweight E10 requires 20 zero rows, found {len(zero_rows)}.")
    e10_rows.extend(zero_rows)

    e29_rows = []
    for dataset in DATASETS:
        token = dataset.lower()
        sources = {
            (128, "full"): root / f"scar_main_{token}_seed42" / "test_metrics.json",
            (128, "global"): root
            / f"scar_lite_e29_{token}_l128_global"
            / "metrics.json",
            (512, "full"): root
            / f"scar_lite_e29_{token}_l512_seed42"
            / "test_metrics.json",
            (512, "global"): root
            / f"scar_lite_e29_{token}_l512_global"
            / "metrics.json",
        }
        for (seq_len, strategy), path in sources.items():
            e29_rows.append(
                {
                    "dataset": dataset,
                    "seq_len": seq_len,
                    "strategy": strategy,
                    **metrics(require(path)),
                }
            )

    e30_rows = []
    for dataset in DATASETS:
        token = dataset.lower()
        e30_rows.append(
            {
                "dataset": dataset,
                "patch": "8+32",
                "strategy": "full",
                **metrics(require(root / f"scar_main_{token}_seed42" / "test_metrics.json")),
            }
        )
        for patch in (8, 64):
            fit = root / f"scar_lite_e30_{token}_p{patch}_seed42"
            e30_rows.append(
                {
                    "dataset": dataset,
                    "patch": str(patch),
                    "strategy": "full",
                    **metrics(require(fit / "test_metrics.json")),
                }
            )
            e30_rows.append(
                {
                    "dataset": dataset,
                    "patch": str(patch),
                    "strategy": "global",
                    **metrics(
                        require(
                            root
                            / f"scar_lite_e30_{token}_p{patch}_global"
                            / "metrics.json"
                        )
                    ),
                }
            )

    e31_rows = []
    for dataset in DATASETS:
        token = dataset.lower()
        result_dir = root / f"scar_lite_e31_{token}_budget_global"
        e31_rows.append(
            {
                "dataset": dataset,
                "method": "SCAR",
                **metrics(require(root / f"scar_main_{token}_seed42" / "test_metrics.json")),
            }
        )
        e31_rows.append(
            {
                "dataset": dataset,
                "method": "budget-matched global",
                **metrics(require(result_dir / "metrics.json")),
                **{
                    f"budget.{key}": value
                    for key, value in flatten(
                        load_json(require(result_dir / "budget_match.json"))
                    ).items()
                },
            }
        )

    baseline_rows = []
    resource_rows = []
    scalability_rows = []
    for method in METHODS:
        for dataset in DATASETS:
            directory = root / f"baseline_{method.lower()}_{dataset.lower()}_seed42"
            baseline_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    **metrics(require(directory / "metrics.json")),
                }
            )
            resource_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    **{
                        f"timing.{key}": value
                        for key, value in flatten(
                            load_json(require(directory / "timing.json"))
                        ).items()
                    },
                    **{
                        f"memory.{key}": value
                        for key, value in flatten(
                            load_json(require(directory / "memory_metrics.json"))
                        ).items()
                    },
                    **{
                        f"resource.{key}": value
                        for key, value in flatten(
                            load_json(require(directory / "resource_metrics.json"))
                        ).items()
                    },
                }
            )
            scalability_rows.extend(
                {
                    "method": method,
                    "dataset": dataset,
                    **row,
                }
                for row in load_json(require(directory / "scalability.json"))["rows"]
            )

    purification = root / "scar_lite_e11_e12_purification_summary"
    table_sources = {
        "table_lite_e10.csv": e10_rows,
        "table_lite_e29_window.csv": e29_rows,
        "table_lite_e30_patch.csv": e30_rows,
        "table_lite_e31_budget.csv": e31_rows,
        "table_lite_knn_lof_performance.csv": baseline_rows,
        "table_lite_knn_lof_resources.csv": resource_rows,
        "table_lite_knn_lof_scalability.csv": scalability_rows,
    }
    for name, rows in table_sources.items():
        write_csv(output / name, rows)
    for source_name, target_name in (
        ("e11_rare_normal.csv", "table_lite_e11_rare_normal.csv"),
        ("e12_low_error_survival.csv", "table_lite_e12_low_error_survival.csv"),
    ):
        shutil.copyfile(require(purification / source_name), output / target_name)

    tep_case = root / "scar_lite_real_condition_case"
    shutil.copyfile(
        require(tep_case / "table_t10_real_condition_cases.csv"),
        output / "table_lite_real_condition_cases.csv",
    )
    summary = {
        "schema_version": 1,
        "scope": "five-dataset-three-day-lite",
        "output_mode": "tables_and_text_only",
        "counts": {name: len(rows) for name, rows in table_sources.items()},
        "copied_tables": [
            "table_lite_e11_rare_normal.csv",
            "table_lite_e12_low_error_survival.csv",
            "table_lite_real_condition_cases.csv",
        ],
    }
    (output / "lite_tables.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "lite_tables.txt").write_text(
        "\n".join(
            [
                "SCAR five-dataset three-day rebuttal package",
                "Output mode: tables and text only",
                *(f"{name}: {count}" for name, count in summary["counts"].items()),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = collect(args.artifact_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
