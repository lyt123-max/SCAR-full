from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DATASETS = ("MSL", "PSM", "SMAP", "SMD", "SWAT")
E9_RATIOS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)
E10_RATIOS = (0.01, 0.03, 0.05, 0.10)


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
            "--output_dir",
            str(detail_dir),
        ],
        check=True,
    )
    return json.loads(
        (detail_dir / "purification_audit.json").read_text(encoding="utf-8")
    )


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect E11/E12 from E9/E10 audits.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e11_rows = []
    e12_rows = []
    for dataset in DATASETS:
        for ratio in E9_RATIOS:
            experiment = (
                root / f"scar_main_{dataset.lower()}_seed42"
                if ratio == 0.02
                else root / f"scar_e9_{dataset.lower()}_clean_{token(ratio)}"
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
        for fold in range(3):
            for contamination in E10_RATIOS:
                clean_values = (0.0, 0.02, 0.10) if contamination == 0.10 else (0.0, 0.02)
                for clean_ratio in clean_values:
                    name = (
                        f"scar_e10_{dataset.lower()}_contam_{token(contamination)}"
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
    _write(args.output_dir / "e11_rare_normal.csv", e11_rows)
    _write(args.output_dir / "e12_low_error_survival.csv", e12_rows)
    (args.output_dir / "purification_summary.json").write_text(
        json.dumps(
            {"schema_version": 1, "E11": e11_rows, "E12": e12_rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
