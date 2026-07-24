from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GDFlex applicability to project datasets.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    source_dir = repo_root / "third_party" / "baselines" / "GDFlex"
    execute_path = source_dir / "src" / "Pub_GDFlex_execute.m"
    source = execute_path.read_text(encoding="utf-8", errors="replace")
    report = {
        "schema_version": 1,
        "baseline": "GDFlex",
        "implementation": str(source_dir),
        "entrypoint": str(execute_path),
        "third_party_modified": False,
        "input_contract": {
            "dataset_selector": "numeric ML/UCR dataset ID or all",
            "repository_data_format": "preprocessed MATLAB records in third_party/baselines/GDFlex/data",
            "multivariate_project_dataset_supported": False,
        },
        "evidence": {
            "uses_fixed_repository_data_directory": "TSAD2021_Dir = '../data'" in source,
            "loads_mat_records": "load(rawData_i)" in source,
            "supports_official_ablation_modes": all(
                mode in source
                for mode in ("baseline", "znormBias", "intra", "locMis", "interNoise")
            ),
        },
        "recommended_scope": (
            "Run GDFlex only on its documented univariate ML/UCR protocol or a separately "
            "validated univariate projection. Do not report it as a native multivariate "
            "baseline for MSL/SMAP/PSM/SWAT/SMD."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[muQn] wrote {args.output}")


if __name__ == "__main__":
    main()
