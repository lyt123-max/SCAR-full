from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "sensitivity" / "run_dataset_sensitivity.py"
PLOTTER = REPO_ROOT / "scripts" / "sensitivity" / "plot_dataset_sensitivity.py"
COLLECTOR = REPO_ROOT / "scripts" / "sensitivity" / "collect_results.py"
SUPPORTED_DATASETS = ("GECCO", "GENESIS", "MSL", "SMAP", "PSM", "SWAT", "SMD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensitivity sweeps for multiple datasets.")
    parser.add_argument("--datasets", type=str, nargs="*", default=list(SUPPORTED_DATASETS), choices=SUPPORTED_DATASETS)
    parser.add_argument("--preset", type=str, default="priority", choices=["priority", "full", "retrieval", "representation", "memory"])
    parser.add_argument("--artifact-root", type=str, default="./artifacts/sensitivity")
    parser.add_argument("--data-root", type=str, default="./dataset/anomaly_detect")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional shared date tag for all datasets in this batch.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-after-run", action="store_true", help="Draw per-dataset figures after each dataset finishes.")
    parser.add_argument(
        "--skip-collect-results",
        action="store_true",
        help="Skip the final combined CSV/JSON collection step.",
    )
    parser.add_argument(
        "--summary-output-prefix",
        type=str,
        default=None,
        help="Optional output prefix for the combined CSV/JSON summary. Defaults to <artifact-root>/sensitivity_summary.",
    )
    return parser.parse_args()


def default_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"[SensitivityAll] $ {printable}")
    if dry_run:
        return
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    run_tag = args.run_tag or default_run_tag()
    summary_output_prefix = (
        Path(args.summary_output_prefix)
        if args.summary_output_prefix
        else artifact_root / f"sensitivity_summary_{run_tag}"
    )
    manifest_rows: list[dict[str, str]] = []

    for dataset in args.datasets:
        dataset_root = artifact_root / f"{dataset.lower()}_sensitivity"
        study_name = f"{dataset.lower()}_sensitivity_{run_tag}"
        summary_json = dataset_root / "summary" / f"{study_name}_summary.json"

        command = [
            args.python_exe,
            str(RUNNER),
            "--dataset",
            dataset,
            "--preset",
            args.preset,
            "--artifact-root",
            str(dataset_root),
            "--data-root",
            args.data_root,
            "--device",
            args.device,
            "--study-name",
            study_name,
            "--run-tag",
            run_tag,
        ]
        if args.skip_existing:
            command.append("--skip-existing")
        if args.dry_run:
            command.append("--dry-run")
        run_command(command, args.dry_run)

        manifest_rows.append(
            {
                "dataset": dataset,
                "artifact_root": str(dataset_root),
                "summary_json": str(summary_json),
                "study_name": study_name,
            }
        )

        if args.plot_after_run:
            plot_command = [
                args.python_exe,
                str(PLOTTER),
                "--summary-json",
                str(summary_json),
            ]
            run_command(plot_command, args.dry_run)

    manifest_path = artifact_root / f"sensitivity_manifest_{run_tag}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"run_tag": run_tag, "datasets": manifest_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SensitivityAll] manifest={manifest_path}")

    if args.dry_run:
        print("[SensitivityAll] dry-run mode: skip combined summary collection.")
        return

    if args.skip_collect_results:
        print("[SensitivityAll] skip combined summary collection.")
        return

    collect_command = [
        args.python_exe,
        str(COLLECTOR),
        "--artifact-root",
        str(artifact_root),
        "--datasets",
        *args.datasets,
        "--study-name-template",
        "{dataset_lower}_sensitivity_" + run_tag,
        "--output-prefix",
        str(summary_output_prefix),
    ]
    run_command(collect_command, dry_run=False)


if __name__ == "__main__":
    main()
