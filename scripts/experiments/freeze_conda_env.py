from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def capture(command: list[str]) -> dict[str, object]:
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return {
            "command": command,
            "return_code": process.returncode,
            "output": process.stdout,
        }
    except OSError as exc:
        return {
            "command": command,
            "return_code": None,
            "output": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one remote rebuttal environment.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "conda_explicit": ["conda", "list", "--explicit"],
        "conda_export": ["conda", "env", "export", "--no-builds"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze", "--all"],
        "nvidia_smi": ["nvidia-smi", "-q"],
        "gpu_query": [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
    }
    payload = {name: capture(command) for name, command in commands.items()}
    for name, result in payload.items():
        (args.output_dir / f"{name}.txt").write_text(
            str(result["output"]), encoding="utf-8"
        )
    (args.output_dir / "environment_lock_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
