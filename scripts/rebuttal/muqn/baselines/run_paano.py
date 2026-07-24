from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FORMAL_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vendored official PaAno baseline.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--use_revin", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    official_dir = repo_root / "third_party" / "baselines" / "PaAno"
    command = [
        str(args.python),
        str(official_dir / "main.py"),
        "--data_dir",
        str(args.data_dir.resolve()),
        "--output_dir",
        str(args.output_dir.resolve()),
        "--patch_size",
        str(args.patch_size),
        "--num_iters",
        str(args.num_iters),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ]
    if args.use_revin:
        command.append("--use_revin")
    manifest = {
        "baseline": "PaAno",
        "implementation": str(official_dir),
        "command": command,
        "third_party_modified": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=args.output_dir.resolve(), check=True)


if __name__ == "__main__":
    main()
