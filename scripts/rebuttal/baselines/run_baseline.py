from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FORMAL_METHODS = ("PaAno", "PUAD", "PGRF-Net", "KNN", "LOF")
NATIVE_METHODS = ("KNN", "LOF")
ADAPTERS = {
    "PaAno": "run_paano_adapter.py",
    "MEMTO": "run_memto_adapter.py",
    "PUAD": "run_puad_adapter.py",
    "PGRF-Net": "run_pgrf_adapter.py",
    "CATCH": "run_catch_adapter.py",
    "KNN": "run_classical_adapter.py",
    "LOF": "run_classical_adapter.py",
}
UPSTREAM_DIRS = {
    "PaAno": "PaAno",
    "MEMTO": "MEMTO",
    "PUAD": "PUAD",
    "PGRF-Net": "PGRF-Net",
    "CATCH": "CATCH",
}
PINNED_COMMITS = {
    "PaAno": "d4c67116190efa4592dc6a8a157ced0def68b6af",
    "MEMTO": "5a3287103021c5c7e7cac9377c626cf18bdea50c",
    "PUAD": "41e8b4377e6baa83e56b8f3acdb60ff04ed6c892",
    "PGRF-Net": "5dc6f7522d20043eb31f6b2b13091c80ad394dcb",
    "CATCH": "3647c69be5eb56649b072596cf89098e689e20c3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one formal baseline task.")
    parser.add_argument("--method", choices=FORMAL_METHODS, required=True)
    parser.add_argument("--dataset", choices=("MSL", "PSM", "SMAP", "SMD", "SWAT"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scar-python", default=sys.executable)
    parser.add_argument("--baseline-python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_adapter_command(
    *,
    method: str,
    baseline_python: str,
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    seed: int,
    device: str,
    smoke: bool = False,
) -> list[str]:
    if method not in ADAPTERS:
        raise ValueError(f"Unsupported baseline method: {method}.")
    if int(seed) != 42:
        raise ValueError("Formal baseline runs require seed 42.")
    command = [
        str(baseline_python),
        str(Path(__file__).resolve().parent / ADAPTERS[method]),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if method in NATIVE_METHODS:
        command.extend(["--method", method])
    if smoke:
        smoke_args = {
            "PaAno": ["--num-iters", "1", "--batch-size", "128"],
            "MEMTO": ["--epochs", "1"],
            "PUAD": ["--epochs", "1"],
            "PGRF-Net": [
                "--epochs-stage1",
                "1",
                "--epochs-stage2",
                "1",
                "--patience-stage1",
                "1",
                "--patience-stage2",
                "1",
            ],
            "KNN": ["--smoke"],
            "LOF": ["--smoke"],
        }[method]
        command.extend(smoke_args)
    return command


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _source_hash(source_experiment: Path) -> str:
    config = source_experiment / "config.json"
    if not config.is_file():
        raise FileNotFoundError(
            f"Baseline source anchor is incomplete: missing {config}."
        )
    return hashlib.sha256(config.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Formal baseline runs require seed 42.")
    output_dir = args.output_dir.resolve()
    source = (
        args.source_experiment.resolve()
        if args.source_experiment is not None
        else output_dir.parent / f"scar_main_{args.dataset.lower()}_seed42"
    )
    commit = None
    if args.method not in NATIVE_METHODS:
        upstream = REPO_ROOT / "third_party" / "baselines" / UPSTREAM_DIRS[args.method]
        if not upstream.is_dir():
            raise FileNotFoundError(f"Missing pinned upstream repository: {upstream}.")
        commit = _git_commit(upstream)
        if commit != PINNED_COMMITS[args.method]:
            raise ValueError(
                f"{args.method} upstream commit mismatch: {commit} != "
                f"{PINNED_COMMITS[args.method]}."
            )
    source_hash = _source_hash(source)
    data_dir = output_dir / "prepared_data"
    prepare_command = [
        str(args.scar_python),
        str(
            REPO_ROOT
            / "scripts"
            / "rebuttal"
            / "muqn"
            / "baselines"
            / "prepare_baseline_data.py"
        ),
        "--experiment_dir",
        str(source),
        "--output_dir",
        str(data_dir),
    ]
    effective_device = "cpu" if args.method in NATIVE_METHODS else args.device
    adapter_command = build_adapter_command(
        method=args.method,
        baseline_python=str(args.baseline_python),
        data_dir=data_dir,
        output_dir=output_dir,
        dataset=args.dataset,
        seed=args.seed,
        device=effective_device,
        smoke=args.smoke,
    )
    monitor_command = [
        str(args.scar_python),
        str(REPO_ROOT / "scripts" / "efficiency" / "monitor_command.py"),
        "--output",
        str(output_dir / "resource_metrics.json"),
        "--method",
        args.method,
        "--dataset",
        args.dataset,
        "--seed",
        str(args.seed),
        "--stage",
        "train",
        "--device",
        effective_device,
        "--",
        *adapter_command,
    ]
    plan = {
        "schema_version": 1,
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
        "source_experiment": str(source),
        "source_config_sha256": source_hash,
        "upstream_commit": commit,
        "implementation_origin": (
            "project_native_sklearn"
            if args.method in NATIVE_METHODS
            else "pinned_upstream_adapter"
        ),
        "prepare_command": prepare_command,
        "adapter_command": adapter_command,
        "monitor_command": monitor_command,
        "smoke": bool(args.smoke),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "baseline_protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise FileExistsError(
                f"Refusing to reuse {output_dir} with a different baseline protocol."
            )
    else:
        protocol_path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    subprocess.run(prepare_command, cwd=REPO_ROOT, check=True)
    subprocess.run(monitor_command, cwd=output_dir, check=True)
    required = (
        "scores.npy",
        "labels.npy",
        "metrics.json",
        "run_manifest.json",
        "timing.json",
        "resource_metrics.json",
    )
    if args.method in NATIVE_METHODS:
        required = required + (
            "memory_metrics.json",
            "scalability.json",
            "scalability.csv",
            "model.pkl",
        )
    missing = [
        name
        for name in required
        if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Baseline adapter completed with missing outputs: {missing}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
