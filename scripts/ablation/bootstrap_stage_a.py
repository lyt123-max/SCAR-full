from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy Stage-A artifacts into a new ablation experiment.")
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--source_artifact_root", type=Path, default=None)
    parser.add_argument("--source_experiment", type=str, required=True)
    parser.add_argument("--target_experiment", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_artifact_root or args.artifact_root
    source_dir = source_root / args.source_experiment
    target_dir = args.artifact_root / args.target_experiment
    target_dir.mkdir(parents=True, exist_ok=True)

    required_files = ["stage_a.pt"]
    optional_files = ["stage_a_last.pt", "stage_a_loss_curve.png", "config.json"]

    for filename in required_files:
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required Stage-A artifact: {source_path}")
        shutil.copy2(source_path, target_dir / filename)

    for filename in optional_files:
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, target_dir / filename)

    print(
        "[Ablation] Stage-A artifacts copied: "
        f"source={source_dir} target={target_dir}"
    )


if __name__ == "__main__":
    main()
