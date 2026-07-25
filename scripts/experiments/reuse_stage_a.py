from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_STAGE_B_OVERRIDES = {
    "memory_build_stride",
    "memory_batch_size",
    "test_batch_size",
    "test_stride",
    "max_train_windows",
    "max_test_windows",
    "top_M",
    "top_K",
    "knn_k",
    "clean_ratio",
    "use_faiss",
    "faiss_use_gpu",
    "faiss_exact_threshold",
    "faiss_ivf_nprobe",
    "coreset_keep_ratio",
    "coreset_max_patches_per_scale",
    "coreset_fps_threshold",
    "memory_seed",
    "memory_audit_mode",
    "resource_monitor_enabled",
    "resource_sample_interval",
    "evaluation_score_key",
    "export_visualizations",
    "use_two_level_retrieval",
    "use_context_key_retrieval",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_stage_a_artifacts(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    required_source = source_dir / "stage_a.pt"
    if not required_source.is_file():
        raise FileNotFoundError(f"Missing Stage-A artifact: {required_source}")

    for filename in ("stage_a.pt", "stage_a_last.pt"):
        source = source_dir / filename
        if not source.is_file():
            continue
        target = target_dir / filename
        if target.exists():
            if not target.is_file() or _sha256(target) != _sha256(source):
                raise RuntimeError(
                    f"Refusing to overwrite different Stage-A artifact: {target}"
                )
            continue
        shutil.copy2(source, target)


def parse_overrides(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in ALLOWED_STAGE_B_OVERRIDES:
            raise ValueError(f"Stage-B override is not allowed: {key}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse an immutable Stage-A checkpoint and rebuild Stage B/Test."
    )
    parser.add_argument("--base-experiment-dir", type=Path, required=True)
    parser.add_argument("--target-experiment-dir", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.base_experiment_dir.resolve()
    target_dir = args.target_experiment_dir.resolve()
    overrides = parse_overrides(args.override)
    manifest = {
        "schema_version": 1,
        "base_experiment_dir": str(base_dir),
        "target_experiment_dir": str(target_dir),
        "overrides": overrides,
        "stages": ["stage_b", "test"],
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from coremad import CoReMADConfig, CoReMADTrainer

    config_path = base_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing base config: {config_path}")
    copy_stage_a_artifacts(base_dir, target_dir)
    config = CoReMADConfig.load(config_path)
    config.artifact_root = str(target_dir.parent)
    config.experiment_name = target_dir.name
    config.resume = False
    for key, value in overrides.items():
        setattr(config, key, value)
    if args.device is not None:
        config.device = args.device
    config.__post_init__()
    config.save()
    (target_dir / "reuse_stage_a_protocol.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    trainer = CoReMADTrainer(config)
    trainer.run_stage_b()
    trainer.run_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
