from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official multidimensional DAMP via MATLAB.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--matlab", type=str, default="matlab")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _matlab_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    adapter_dir = Path(__file__).resolve().parent
    official_dir = repo_root / "third_party" / "baselines" / "DAMP"
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    expression = (
        f"addpath('{_matlab_quote(adapter_dir)}'); "
        f"run_damp_multidim('{_matlab_quote(args.input_csv)}', "
        f"'{_matlab_quote(args.output_csv)}', '{_matlab_quote(official_dir)}');"
    )
    command = [args.matlab, "-batch", expression]
    manifest = {
        "baseline": "DAMP_Multidim",
        "implementation": str(official_dir),
        "adapter": str(adapter_dir / "run_damp_multidim.m"),
        "command": command,
        "third_party_modified": False,
        "adapter_behavior": "Generates a temporary output-returning copy of the official function.",
    }
    (args.output_csv.parent / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
