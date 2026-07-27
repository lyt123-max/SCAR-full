from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DATASETS = ("MSL", "PSM", "SMAP", "SMD", "SWAT")
E9_RATIOS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)
E10_RATIOS = (0.01, 0.03, 0.05, 0.10)
E12_ERROR_BANDS = ("all", "low_q25", "mid_q25_q75", "high_q75")


def token(value: float) -> str:
    return str(value).replace(".", "p")


def _analyze(experiment: Path, detail_dir: Path) -> dict:
    script = Path(__file__).resolve().parent / "analyze_purification_audit.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--experiment_dir",
            str(experiment),
            "--rare_normal_quantile",
            "0.90",
            "--low_error_quantile",
            "0.25",
            "--high_error_quantile",
            "0.75",
            "--output_dir",
            str(detail_dir),
        ],
        check=True,
    )
    return json.loads(
        (detail_dir / "purification_audit.json").read_text(encoding="utf-8")
    )


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_e12_rows(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("E12 contains no rows")
    grouped_bands: dict[tuple, set[str]] = defaultdict(set)
    key_fields = (
        "dataset",
        "fold",
        "contamination_ratio",
        "clean_ratio",
        "patch_size",
    )
    finite_fields = (
        "purification_retention",
        "purification_removal",
        "final_retention",
        "completion_mean",
        "completion_p95",
    )
    for row in rows:
        if row.get("group") != "injected_anomaly":
            raise ValueError(f"Unexpected E12 group: {row.get('group')}")
        if int(row.get("count", 0)) <= 0:
            raise ValueError(
                f"E12 zero denominator for {row.get('dataset')} "
                f"patch_size={row.get('patch_size')} "
                f"error_band={row.get('error_band')}"
            )
        for field in finite_fields:
            if not math.isfinite(float(row[field])):
                raise ValueError(
                    f"E12 non-finite {field} for {row.get('dataset')} "
                    f"patch_size={row.get('patch_size')} "
                    f"error_band={row.get('error_band')}"
                )
        grouped_bands[tuple(row.get(field) for field in key_fields)].add(
            str(row.get("error_band"))
        )
    expected = set(E12_ERROR_BANDS)
    for key, bands in grouped_bands.items():
        if bands != expected:
            raise ValueError(
                f"E12 error bands for {key} are {sorted(bands)}, "
                f"expected {sorted(expected)}"
            )


def aggregate_e12_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    key_fields = (
        "dataset",
        "contamination_ratio",
        "clean_ratio",
        "patch_size",
        "error_band",
        "error_quantile_low",
        "error_quantile_high",
    )
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        aggregate = grouped.setdefault(
            key,
            {
                **{field: row[field] for field in key_fields},
                "fold_count": 0,
                "count": 0,
                "purification_kept_count": 0,
                "final_kept_count": 0,
            },
        )
        aggregate["fold_count"] += 1
        aggregate["count"] += int(row["count"])
        aggregate["purification_kept_count"] += int(row["purification_kept_count"])
        aggregate["final_kept_count"] += int(row["final_kept_count"])

    summaries = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        aggregate = grouped[key]
        count = aggregate["count"]
        purification_retention = aggregate["purification_kept_count"] / count
        final_retention = aggregate["final_kept_count"] / count
        summaries.append(
            {
                **aggregate,
                "purification_retention": purification_retention,
                "purification_removal": 1.0 - purification_retention,
                "final_retention": final_retention,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect E11/E12 from E9/E10 audits.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--e9-ratios", type=float, nargs="+", default=list(E9_RATIOS))
    parser.add_argument("--e10-ratios", type=float, nargs="+", default=list(E10_RATIOS))
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--experiment-prefix", default="scar")
    parser.add_argument(
        "--lightweight-e10",
        action="store_true",
        help="Use default purification at 1/5% and no/default/strong at 10%.",
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e11_rows = []
    e12_rows = []
    for dataset in args.datasets:
        for ratio in args.e9_ratios:
            experiment = (
                root / f"scar_main_{dataset.lower()}_seed42"
                if ratio == 0.02
                else root
                / f"{args.experiment_prefix}_e9_{dataset.lower()}_clean_{token(ratio)}"
            )
            detail = args.output_dir / "details" / f"e9_{dataset}_{token(ratio)}"
            payload = _analyze(experiment, detail)
            for row in payload["groups"]:
                if row["group"] == "rare_normal":
                    e11_rows.append(
                        {
                            "dataset": dataset,
                            "clean_ratio": ratio,
                            "rarity_quantile": 0.90,
                            "rarity_threshold": payload["rarity_threshold"],
                            **row,
                        }
                    )
        for fold in range(args.n_folds):
            for contamination in args.e10_ratios:
                if args.lightweight_e10:
                    clean_values = (
                        (0.0, 0.02, 0.10) if contamination == 0.10 else (0.02,)
                    )
                else:
                    clean_values = (
                        (0.0, 0.02, 0.10) if contamination == 0.10 else (0.0, 0.02)
                    )
                for clean_ratio in clean_values:
                    name = (
                        f"{args.experiment_prefix}_e10_{dataset.lower()}_contam_{token(contamination)}"
                        f"_clean_{token(clean_ratio)}_fold_{fold}"
                    )
                    payload = _analyze(root / name, args.output_dir / "details" / name)
                    for row in payload["groups"]:
                        if row["group"] == "injected_anomaly":
                            e12_rows.append(
                                {
                                    "dataset": dataset,
                                    "fold": fold,
                                    "contamination_ratio": contamination,
                                    "clean_ratio": clean_ratio,
                                    **row,
                                }
                            )
    validate_e12_rows(e12_rows)
    e12_summary_rows = aggregate_e12_rows(e12_rows)
    _write(args.output_dir / "e11_rare_normal.csv", e11_rows)
    _write(args.output_dir / "e12_low_error_survival.csv", e12_rows)
    _write(
        args.output_dir / "e12_low_error_survival_summary.csv",
        e12_summary_rows,
    )
    (args.output_dir / "purification_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "E11": e11_rows,
                "E12": e12_rows,
                "E12_summary": e12_summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
