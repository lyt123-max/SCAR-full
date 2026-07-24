from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from collect_results import collect
from common import (
    FORMAL_SEED,
    PROJECT_ROOT,
    default_data_root,
    normalize_edition,
    normalize_split,
    parse_file_name,
    read_manifest,
    resolve_data_dir,
    validate_local_files,
)


FORMAL_SCORE_FILES = (
    "test_scores_raw_max.npy",
    "test_scores_zscore_mean.npy",
    "test_scores_cdf_mean.npy",
    "test_scores_cdf_max.npy",
)
DIAGNOSTIC_SCORE_FILES = (
    "test_scores_knn_distance.npy",
    "test_scores_state_novelty.npy",
    "test_diagnostic_scores.npz",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_protocol(
    protocol: str,
    edition: str,
    split: str,
    seeds: list[int],
) -> None:
    if protocol != "formal-rebuttal":
        return
    normalized_split = normalize_split(split)
    if edition.upper() == "U" and normalized_split == "eval_full":
        raise ValueError("Formal rebuttal protocol forbids U/eval_full.")
    if normalized_split not in {"tuning", "eval"}:
        raise ValueError("Formal rebuttal protocol permits only tuning or eval splits.")
    if seeds != [FORMAL_SEED]:
        raise ValueError("Formal rebuttal protocol requires fixed seed 42.")


def _required_test_files(patch_sizes: list[int]) -> tuple[str, ...]:
    completion = tuple(f"test_scores_completion_scale{size}.npy" for size in patch_sizes)
    return ("test_metrics.json",) + FORMAL_SCORE_FILES + DIAGNOSTIC_SCORE_FILES + completion


def _test_artifact_error(experiment_dir: Path, patch_sizes: list[int]) -> str | None:
    required = _required_test_files(patch_sizes)
    missing = [name for name in required if not (experiment_dir / name).is_file()]
    if missing:
        return f"missing artifacts: {', '.join(missing)}"
    try:
        metrics = json.loads((experiment_dir / "test_metrics.json").read_text(encoding="utf-8"))
        total_length = int(metrics["dataset_metadata"]["total_length"])
        if total_length < 1:
            return "dataset_metadata.total_length must be positive"
        for name in required:
            if not name.endswith(".npy"):
                continue
            values = np.load(experiment_dir / name, mmap_mode="r", allow_pickle=False)
            if values.shape != (total_length,):
                return f"{name} has shape {values.shape}; expected ({total_length},)"
        with np.load(experiment_dir / "test_diagnostic_scores.npz", allow_pickle=False) as payload:
            if "labels" not in payload or payload["labels"].shape != (total_length,):
                return "test_diagnostic_scores.npz labels do not match the full series length"
    except Exception as exc:
        return f"invalid test artifacts: {exc}"
    return None


def _is_complete(experiment_dir: Path, stage: str, patch_sizes: list[int]) -> bool:
    if stage in {"full", "test"}:
        return _test_artifact_error(experiment_dir, patch_sizes) is None
    if stage == "stage_a":
        return (experiment_dir / "stage_a.pt").is_file()
    return (
        (experiment_dir / "memory.pt").is_file()
        and (experiment_dir / "cdf_fusion.npz").is_file()
        and (experiment_dir / "cdf_fusion.json").is_file()
        and (experiment_dir / "zscore_fusion.json").is_file()
    )


def _format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)


def _write_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def _build_command(
    args: argparse.Namespace,
    data_dir: Path,
    seed_root: Path,
    file_name: str,
    seed: int,
) -> list[str]:
    command = [
        str(args.python),
        str(PROJECT_ROOT / "run.py"),
        "--stage",
        args.stage,
        "--data_format",
        "tsb_ad",
        "--dataset",
        file_name,
        "--data_root",
        str(data_dir),
        "--artifact_root",
        str(seed_root),
        "--experiment_name",
        Path(file_name).stem,
        "--seed",
        str(seed),
        "--seq_len",
        str(args.seq_len),
        "--patch_sizes",
        *(str(size) for size in args.patch_sizes),
        "--batch_size",
        str(args.batch_size),
        "--stage_a_epochs",
        str(args.stage_a_epochs),
        "--num_workers",
        str(args.num_workers),
        "--evaluation_score_key",
        "cdf_mean",
        "--resume",
        str(int(args.resume or args.stage in {"stage_b", "test"})),
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.max_train_windows:
        command.extend(["--max_train_windows", str(args.max_train_windows)])
    if args.max_test_windows:
        command.extend(["--max_test_windows", str(args.max_test_windows)])
    return command


def run(args: argparse.Namespace) -> int:
    edition = normalize_edition(args.edition)
    split = normalize_split(args.split)
    validate_protocol(args.protocol, edition, split, args.seeds)
    names = read_manifest(edition, split)
    data_dir = resolve_data_dir(args.data_root or default_data_root(edition), edition)
    validate_local_files(data_dir, names)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        names = names[: args.limit]

    artifact_root = args.artifact_root.resolve()
    split_root = artifact_root / edition / split
    total = len(names) * len(args.seeds)
    completed = skipped = failed = 0
    print(
        f"[TSB-AD] edition={edition} split={split} files={len(names)} "
        f"seeds={args.seeds} stage={args.stage} total_runs={total}"
    )

    for seed in args.seeds:
        seed_root = split_root / f"seed_{seed}"
        for index, file_name in enumerate(names, start=1):
            metadata = parse_file_name(file_name)
            experiment_dir = seed_root / str(metadata["file_stem"])
            record_path = experiment_dir / "run_record.json"
            command = _build_command(args, data_dir, seed_root, file_name, seed)
            command_text = _format_command(command)
            prefix = f"[{index}/{len(names)}][seed={seed}]"

            if args.resume and _is_complete(experiment_dir, args.stage, args.patch_sizes):
                skipped += 1
                print(f"{prefix} skip complete: {file_name}")
                continue
            if args.dry_run:
                print(f"{prefix} {command_text}")
                continue

            experiment_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            record: dict[str, Any] = {
                "edition": edition,
                "split": split,
                "file": file_name,
                "source_dataset": metadata["source_dataset"],
                "seed": seed,
                "stage": args.stage,
                "status": "running",
                "started_at": _utc_now(),
                "command": command,
            }
            _write_record(record_path, record)
            print(f"{prefix} start: {file_name}")

            log_path = experiment_dir / f"{args.stage}.log"
            try:
                with log_path.open("w", encoding="utf-8", newline="") as log_handle:
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="")
                        log_handle.write(line)
                    return_code = process.wait()
            except Exception as exc:
                runtime = time.perf_counter() - started
                record.update(
                    {
                        "status": "failed",
                        "runtime_seconds": runtime,
                        "finished_at": _utc_now(),
                        "artifacts_complete": False,
                        "error": f"failed to launch or monitor subprocess: {exc}",
                        "log": str(log_path.resolve()),
                    }
                )
                _write_record(record_path, record)
                failed += 1
                print(f"{prefix} failed to launch or monitor subprocess: {exc}")
                if args.fail_fast:
                    if args.stage in {"full", "test"}:
                        collect(artifact_root, edition, split)
                    return 1
                continue

            runtime = time.perf_counter() - started
            complete = return_code == 0 and _is_complete(experiment_dir, args.stage, args.patch_sizes)
            record.update(
                {
                    "status": "completed" if complete else "failed",
                    "return_code": return_code,
                    "runtime_seconds": runtime,
                    "finished_at": _utc_now(),
                    "artifacts_complete": complete,
                    "log": str(log_path.resolve()),
                }
            )
            if return_code == 0 and not complete:
                record["error"] = _test_artifact_error(experiment_dir, args.patch_sizes)
            _write_record(record_path, record)

            if complete:
                completed += 1
                print(f"{prefix} completed in {runtime:.1f}s")
            else:
                failed += 1
                print(f"{prefix} failed (return_code={return_code}, complete={complete})")
                if args.fail_fast:
                    if args.stage in {"full", "test"}:
                        collect(artifact_root, edition, split)
                    return 1

    if not args.dry_run and args.stage in {"full", "test"}:
        summary = collect(artifact_root, edition, split)
        print(f"[TSB-AD] collection: {json.dumps(summary)}")
    print(f"[TSB-AD] completed={completed} skipped={skipped} failed={failed}")
    return int(failed > 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCAR independently on official TSB-AD series.")
    parser.add_argument("--edition", required=True, choices=["M", "U", "m", "u"])
    parser.add_argument(
        "--protocol",
        choices=["formal-rebuttal", "extended"],
        default="formal-rebuttal",
    )
    parser.add_argument("--split", default="eval", choices=["all", "tuning", "eval", "eval_full", "eval-full"])
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "artifacts" / "tsb_ad")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage", choices=["stage_a", "stage_b", "test", "full"], default="full")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stage-a-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    args = parser.parse_args()
    args.seeds = (
        args.seeds
        if args.seeds is not None
        else [args.seed if args.seed is not None else FORMAL_SEED]
    )
    return args


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
