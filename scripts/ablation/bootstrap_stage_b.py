from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy Stage-B artifacts into a new ablation experiment.")
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--source_artifact_root", type=Path, default=None)
    parser.add_argument("--source_experiment", type=str, required=True)
    parser.add_argument("--target_experiment", type=str, required=True)
    return parser.parse_args()


def copy_if_exists(source_dir: Path, target_dir: Path, filename: str) -> None:
    source_path = source_dir / filename
    if source_path.exists():
        shutil.copy2(source_path, target_dir / filename)


def main() -> None:
    args = parse_args()
    source_root = args.source_artifact_root or args.artifact_root
    source_dir = source_root / args.source_experiment
    target_dir = args.artifact_root / args.target_experiment
    target_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "memory.pt",
        "memory_meta.json",
        "cdf_fusion.npz",
        "cdf_fusion.json",
        "zscore_fusion.json",
    ]
    optional_files = [
        "faiss_state.index",
    ]

    for filename in required_files:
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required Stage-B artifact: {source_path}")
        shutil.copy2(source_path, target_dir / filename)

    for filename in optional_files:
        copy_if_exists(source_dir, target_dir, filename)

    print(
        "[Ablation] Stage-B artifacts copied: "
        f"source={source_dir} target={target_dir}"
    )


if __name__ == "__main__":
    main()
