from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / "plot_dataset_sensitivity.py"


def main() -> None:
    command = [sys.executable, str(TARGET), *sys.argv[1:]]
    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
