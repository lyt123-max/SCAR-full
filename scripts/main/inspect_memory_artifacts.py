from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad.config import CoReMADConfig
from coremad.faiss_index import HAS_FAISS
from coremad.memory import MemoryBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an existing CoReM-AD memory bank without rerunning Stage B."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        help="Experiment directory containing config.json and memory.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.json. If omitted, uses <experiment-dir>/config.json",
    )
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    if args.experiment_dir:
        return Path(args.experiment_dir) / "config.json"
    raise ValueError("Provide either --experiment-dir or --config.")


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = CoReMADConfig.load(config_path)
    print(f"[Inspect] experiment_dir={config.experiment_dir}")
    print(f"[Inspect] memory_path={config.memory_path}")
    print(f"[Inspect] faiss_index_path={config.faiss_index_path}")

    if not config.memory_path.exists():
        raise FileNotFoundError(f"Memory artifact not found: {config.memory_path}")

    memory = MemoryBank.load(config.memory_path, config)
    print(f"[Inspect] num_windows={int(memory.state_bank.size(0))}")
    print(f"[Inspect] num_scales={len(memory.scales)}")
    print(f"[Inspect] num_prototypes={int(memory.prototype_centers.size(0))}")
    if memory.prototype_members:
        proto_sizes = [int(member.numel()) for member in memory.prototype_members]
        print(
            f"[Inspect] prototype_sizes: min={min(proto_sizes)} "
            f"median={float(sorted(proto_sizes)[len(proto_sizes)//2]):.1f} "
            f"max={max(proto_sizes)}"
        )

    for scale_idx, scale_memory in enumerate(memory.scales):
        patch_size = config.patch_sizes[scale_idx] if scale_idx < len(config.patch_sizes) else "unknown"
        print(
            f"[Inspect] scale patch_size={patch_size}: "
            f"{int(scale_memory.z.size(0))} vectors in scale memory"
        )

    if config.use_faiss and HAS_FAISS and memory.state_index.index is not None:
        print(f"[Inspect] state_index.backend={memory.state_index.describe()}")
        print(f"[Inspect] state_index.ntotal={int(memory.state_index.index.ntotal)}")
    else:
        print(
            "[Inspect] state index is not using a loaded Faiss index; "
            f"backend={memory.state_index.describe()}"
        )


if __name__ == "__main__":
    main()
