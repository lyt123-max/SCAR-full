from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PY = REPO_ROOT / "run.py"
SUPPORTED_DATASETS = ("GECCO", "GENESIS", "MSL", "SMAP", "PSM", "SWAT", "SMD")
SUMMARY_SCORE_KEYS = ("cdf_max", "cdf_mean", "cdf_softmax")
SUMMARY_METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "point_best_f1",
    "pa_best_f1",
    "aff_precision",
    "aff_recall",
    "aff_f1",
    "range_precision",
    "range_recall",
    "range_f1",
    "vus_roc",
    "vus_pr",
)

STAGE_B_COMPAT_FIELDS = (
    "dataset",
    "data_root",
    "seq_len",
    "memory_build_stride",
    "max_train_windows",
    "patch_sizes",
    "d_z",
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
    "seed",
)


@dataclass(frozen=True)
class SweepSpec:
    name: str
    cli_key: str
    values: tuple[Any, ...]
    default: Any
    group: str
    stage_mode: str
    description: str
    paper_panel: str
    value_labels: tuple[str, ...] | None = None


COMMON_DEFAULT_ARGS: dict[str, Any] = {
    "seq_len": 128,
    "batch_size": 128,
    "train_stride": 1,
    "test_stride": 1,
    "memory_build_stride": 1,
    "val_ratio": 0.15,
    "val_gap": 1,
    "val_min_train_windows": 50,
    "val_split_mode": "tail",
    "stage_a_epochs": 100,
    "early_stop_patience": 15,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "scheduler_eta_min_ratio": 0.01,
    "mask_ratio": 0.25,
    "n_mask_groups": 4,
    "lambda_pred": 0.5,
    "lambda_smooth": 0.01,
    "patch_sizes": [8, 32],
    "d_z": 128,
    "top_M": 50,
    "top_K": 20,
    "knn_k": 5,
    "clean_ratio": 0.02,
    "use_faiss": 1,
    "faiss_use_gpu": 1,
    "faiss_exact_threshold": 100000,
    "faiss_ivf_nprobe": 16,
    "coreset_keep_ratio": 1.0,
    "coreset_max_patches_per_scale": 200000,
    "coreset_fps_threshold": 100000,
    "num_workers": 4,
    "max_train_windows": 0,
    "max_test_windows": 0,
    "seed": 42,
    "resume": 1,
}


DATASET_OVERRIDES: dict[str, dict[str, Any]] = {
    "GECCO": {"dataset": "GECCO"},
    "GENESIS": {"dataset": "GENESIS"},
    # MSL sweeps are expected to reuse the historical 256-batch Stage-A baseline.
    "MSL": {"dataset": "MSL", "batch_size": 256},
    "SMAP": {"dataset": "SMAP"},
    "PSM": {"dataset": "PSM", "val_split_mode": "interleaved", "lr": 5e-4},
    "SWAT": {"dataset": "SWAT"},
    # SMD sweeps also reuse the historical 256-batch Stage-A baseline.
    "SMD": {"dataset": "SMD", "batch_size": 256},
}


SWEEP_SPECS: dict[str, SweepSpec] = {
    "top_M": SweepSpec(
        name="top_M",
        cli_key="top_M",
        values=(10, 25, 50, 100, 200),
        default=50,
        group="retrieval",
        stage_mode="reuse_stage_a",
        description="Number of coarse retrieval windows.",
        paper_panel="retrieval",
    ),
    "top_K": SweepSpec(
        name="top_K",
        cli_key="top_K",
        values=(5, 10, 20, 40, 80),
        default=20,
        group="retrieval",
        stage_mode="reuse_stage_a",
        description="Number of fine re-ranking candidates.",
        paper_panel="retrieval",
    ),
    "knn_k": SweepSpec(
        name="knn_k",
        cli_key="knn_k",
        values=(1, 3, 5, 10, 20),
        default=5,
        group="retrieval",
        stage_mode="reuse_stage_a",
        description="Number of nearest neighbors for final scoring.",
        paper_panel="retrieval",
    ),
    "d_z": SweepSpec(
        name="d_z",
        cli_key="d_z",
        values=(32, 64, 128, 256, 512),
        default=128,
        group="representation",
        stage_mode="full",
        description="Patch representation dimension.",
        paper_panel="d_z",
    ),
    "seq_len": SweepSpec(
        name="seq_len",
        cli_key="seq_len",
        values=(64, 128, 160, 192, 256),
        default=128,
        group="representation",
        stage_mode="full",
        description=(
            "Window length. 96 was intentionally replaced with 160 because the current "
            "model requires seq_len to be divisible by every patch size (8 and 32)."
        ),
        paper_panel="appendix",
    ),
    "mask_ratio": SweepSpec(
        name="mask_ratio",
        cli_key="mask_ratio",
        values=(0.10, 0.15, 0.25, 0.35, 0.50),
        default=0.25,
        group="representation",
        stage_mode="full",
        description="Masked-patch ratio in Stage A.",
        paper_panel="mask_ratio",
    ),
    "coreset_max_patches_per_scale": SweepSpec(
        name="coreset_max_patches_per_scale",
        cli_key="coreset_max_patches_per_scale",
        values=(50000, 100000, 200000, 500000, 0),
        default=200000,
        group="memory",
        stage_mode="reuse_stage_a",
        description="Per-scale memory cap. 0 means uncapped.",
        paper_panel="appendix",
        value_labels=("50k", "100k", "200k", "500k", "uncapped"),
    ),
}


PRESETS: dict[str, tuple[str, ...]] = {
    "priority": ("top_M", "top_K", "knn_k", "d_z", "mask_ratio"),
    "full": (
        "top_M",
        "top_K",
        "knn_k",
        "d_z",
        "seq_len",
        "mask_ratio",
        "coreset_max_patches_per_scale",
    ),
    "retrieval": ("top_M", "top_K", "knn_k"),
    "representation": ("d_z", "seq_len", "mask_ratio"),
    "memory": ("coreset_max_patches_per_scale",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-dataset hyperparameter sensitivity sweeps.")
    parser.add_argument("--dataset", type=str, required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--preset",
        type=str,
        default="priority",
        choices=sorted(PRESETS.keys()),
        help="Sweep preset. `priority` matches the recommended execution order.",
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs="*",
        default=None,
        help="Explicit parameter names to sweep. Overrides --preset when provided.",
    )
    parser.add_argument(
        "--param-values",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional subset of values for a single selected parameter. Accepts raw values "
            "(for example `0.1`) or value labels (for example `50k`, `uncapped`)."
        ),
    )
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="./dataset/anomaly_detect")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional shared date tag for experiment and summary names. Defaults to the current timestamp.",
    )
    parser.add_argument(
        "--base-exp-name",
        type=str,
        default=None,
        help="Experiment name for the reusable default Stage-A checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=None, help="Optional shared batch size override.")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Optional Stage-A train batch size override.")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Optional validation batch size override.")
    parser.add_argument("--memory-batch-size", type=int, default=None, help="Optional memory-building batch size override.")
    parser.add_argument("--test-batch-size", type=int, default=None, help="Optional test batch size override.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing test_metrics.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep writing the summary even if one or more sweep points fail.",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default=None,
        help="Optional output directory for summary CSV/JSON. Defaults to <artifact-root>/summary.",
    )
    return parser.parse_args()


def ensure_repo_paths() -> None:
    if not RUN_PY.exists():
        raise FileNotFoundError(f"run.py not found: {RUN_PY}")


def dataset_display_name(dataset: str) -> str:
    return "SWaT" if dataset.upper() == "SWAT" else dataset.upper()


def resolve_dataset_defaults(dataset: str) -> dict[str, Any]:
    merged = dict(COMMON_DEFAULT_ARGS)
    merged.update(DATASET_OVERRIDES[dataset])
    return merged


def apply_runtime_overrides(default_args: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(default_args)
    cli_to_config = {
        "batch_size": args.batch_size,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "memory_batch_size": args.memory_batch_size,
        "test_batch_size": args.test_batch_size,
    }
    for key, value in cli_to_config.items():
        if value is not None:
            merged[key] = int(value)
    # CoReMADConfig saves the legacy shared `batch_size` as an alias of
    # `train_batch_size`, so keep them aligned when the user only overrides
    # the train batch size. Otherwise baseline validation will falsely detect
    # a config mismatch and rebuild Stage-A every time.
    if args.batch_size is None and args.train_batch_size is not None:
        merged["batch_size"] = int(args.train_batch_size)
    return merged


def default_artifact_root(dataset: str) -> str:
    return f"./artifacts/sensitivity/{dataset.lower()}_sensitivity"


def default_study_name(dataset: str) -> str:
    return f"{dataset.lower()}_sensitivity"


def default_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_selected_params(args: argparse.Namespace) -> list[str]:
    if args.params:
        invalid = [name for name in args.params if name not in SWEEP_SPECS]
        if invalid:
            raise ValueError(f"Unknown sweep parameters: {invalid}")
        return list(dict.fromkeys(args.params))
    return list(PRESETS[args.preset])


def resolve_selected_points(
    args: argparse.Namespace,
    selected_params: list[str],
) -> dict[str, list[tuple[int, Any, str | None]]]:
    selected_points: dict[str, list[tuple[int, Any, str | None]]] = {}
    param_values = list(args.param_values or [])
    if param_values and len(selected_params) != 1:
        raise ValueError("--param-values requires exactly one selected parameter.")

    for param_name in selected_params:
        spec = SWEEP_SPECS[param_name]
        points = [
            (idx, value, spec.value_labels[idx] if spec.value_labels else None)
            for idx, value in enumerate(spec.values)
        ]
        if not param_values:
            selected_points[param_name] = points
            continue

        token_to_point: dict[str, tuple[int, Any, str | None]] = {}
        for idx, value, value_label in points:
            tokens = {
                str(value),
                format_value_for_cli(value),
                format_value_for_name(value),
                format_value_for_name(value, value_label),
            }
            if value_label:
                tokens.add(value_label)
            for token in tokens:
                token_to_point[token] = (idx, value, value_label)

        resolved: list[tuple[int, Any, str | None]] = []
        missing_tokens: list[str] = []
        seen_indices: set[int] = set()
        for token in param_values:
            point = token_to_point.get(token)
            if point is None:
                missing_tokens.append(token)
                continue
            idx = point[0]
            if idx in seen_indices:
                continue
            resolved.append(point)
            seen_indices.add(idx)
        if missing_tokens:
            allowed = [value_label or format_value_for_cli(value) for _, value, value_label in points]
            raise ValueError(
                f"Unknown values for `{param_name}`: {missing_tokens}. Allowed values: {allowed}"
            )
        if not resolved:
            raise ValueError(f"No sweep values selected for `{param_name}`.")
        selected_points[param_name] = resolved

    return selected_points


def format_value_for_name(value: Any, label: str | None = None) -> str:
    if label:
        token = label
    elif isinstance(value, float):
        token = f"{value:.4g}"
    else:
        token = str(value)
    return token.replace(".", "p").replace("-", "m").replace("/", "_")


def format_value_for_cli(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:g}"
    return str(value)


def experiment_name(study_name: str, spec: SweepSpec, value: Any, value_label: str | None) -> str:
    return f"{study_name}_{spec.name}_{format_value_for_name(value, value_label)}"


def has_pending_reuse_stage_a_points(
    artifact_root: Path,
    study_name: str,
    selected_points: dict[str, list[tuple[int, Any, str | None]]],
    skip_existing: bool,
) -> bool:
    for param_name, points in selected_points.items():
        spec = SWEEP_SPECS[param_name]
        if spec.stage_mode != "reuse_stage_a":
            continue
        for _, value, value_label in points:
            exp_dir = artifact_root / experiment_name(study_name, spec, value, value_label)
            if not should_skip(exp_dir, skip_existing):
                return True
    return False


def build_run_command(
    python_exe: str,
    artifact_root: Path,
    data_root: str,
    device: str,
    stage: str,
    exp_name: str,
    default_args: dict[str, Any],
    overrides: dict[str, Any],
) -> list[str]:
    merged = dict(default_args)
    merged.update(overrides)
    command = [
        python_exe,
        str(RUN_PY),
        "--stage",
        stage,
        "--artifact_root",
        str(artifact_root),
        "--experiment_name",
        exp_name,
        "--data_root",
        data_root,
        "--device",
        device,
    ]

    for key, value in merged.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, list):
            command.append(flag)
            command.extend(format_value_for_cli(item) for item in value)
        else:
            command.extend([flag, format_value_for_cli(value)])
    return command


def stage_log_path(exp_dir: Path, stage: str) -> Path:
    log_dir = exp_dir / "sensitivity_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{stage}.log"


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"[Sensitivity] $ {printable}")
    if dry_run:
        return

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {printable}\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
        handle.write(f"[exit_code] {return_code}\n")
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def reset_experiment_dir(exp_dir: Path, dry_run: bool, reason: str) -> None:
    if not exp_dir.exists():
        return
    print(f"[Sensitivity] reset experiment dir: {exp_dir} ({reason})")
    if dry_run:
        return
    shutil.rmtree(exp_dir)


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            _values_match(left, right) for left, right in zip(expected, actual)
        )
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
        except Exception:
            return False
    if isinstance(expected, (bool, int)) and isinstance(actual, (bool, int)):
        return int(expected) == int(actual)
    return expected == actual


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _file_signature(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def validate_stage_a_seed(base_dir: Path, default_args: dict[str, Any]) -> tuple[bool, str]:
    stage_a_path = base_dir / "stage_a.pt"
    stage_a_last_path = base_dir / "stage_a_last.pt"
    config_path = base_dir / "config.json"

    missing = [path.name for path in (stage_a_path, stage_a_last_path, config_path) if not path.exists()]
    if missing:
        return False, f"missing baseline artifacts: {', '.join(missing)}"

    try:
        payload = torch.load(stage_a_last_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"failed to read Stage-A checkpoint: {exc}"

    completed = bool(payload.get("completed", False))
    epoch = int(payload.get("epoch", 0))
    epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
    stage_a_epochs = int(default_args["stage_a_epochs"])
    early_stop_patience = int(default_args["early_stop_patience"])
    if not (completed or epoch >= stage_a_epochs or epochs_without_improvement >= early_stop_patience):
        return False, (
            "baseline Stage-A checkpoint is incomplete "
            f"(epoch={epoch}, completed={completed}, patience_count={epochs_without_improvement})"
        )

    try:
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"failed to read baseline config: {exc}"
    if not isinstance(saved_config, dict):
        return False, "baseline config.json is invalid"

    for key, expected_value in default_args.items():
        if key not in saved_config:
            return False, f"baseline config missing `{key}`"
        if not _values_match(expected_value, saved_config[key]):
            return False, (
                f"baseline config mismatch for `{key}`: "
                f"expected={expected_value!r} actual={saved_config[key]!r}"
            )
    return True, "baseline is complete and matches the expected defaults"


def validate_stage_b_artifacts(
    exp_dir: Path,
    expected_args: dict[str, Any],
    expected_stage_a_path: Path,
) -> tuple[bool, str]:
    memory_path = exp_dir / "memory.pt"
    memory_meta_path = exp_dir / "memory_meta.json"
    cdf_npz_path = exp_dir / "cdf_fusion.npz"
    cdf_json_path = exp_dir / "cdf_fusion.json"
    zscore_json_path = exp_dir / "zscore_fusion.json"

    missing = [
        path.name
        for path in (memory_path, memory_meta_path, cdf_npz_path, cdf_json_path, zscore_json_path)
        if not path.exists()
    ]
    if missing:
        return False, f"missing Stage-B artifacts: {', '.join(missing)}"

    try:
        meta = json.loads(memory_meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"failed to read Stage-B metadata: {exc}"
    if not isinstance(meta, dict):
        return False, "Stage-B metadata is invalid"

    stored_config = meta.get("config")
    if not isinstance(stored_config, dict):
        return False, "Stage-B metadata missing config snapshot"

    mismatches: list[str] = []
    for key in STAGE_B_COMPAT_FIELDS:
        if key not in expected_args:
            continue
        expected_value = _normalize_config_value(expected_args[key])
        actual_value = _normalize_config_value(stored_config.get(key, "<missing>"))
        if not _values_match(expected_value, actual_value):
            mismatches.append(f"{key}: expected={expected_value!r} actual={actual_value!r}")
    if mismatches:
        preview = "; ".join(mismatches[:5])
        if len(mismatches) > 5:
            preview += f"; ... +{len(mismatches) - 5} more"
        return False, f"Stage-B config mismatch: {preview}"

    expected_stage_a_sig = _file_signature(expected_stage_a_path)
    actual_stage_a_sig = meta.get("stage_a_artifact_signature")
    if actual_stage_a_sig != expected_stage_a_sig:
        return False, "Stage-B metadata points to a different Stage-A artifact"
    return True, "Stage-B artifacts are complete and match the expected configuration"


def resolve_point_stages(
    exp_dir: Path,
    spec: SweepSpec,
    run_args: dict[str, Any],
    base_dir: Path | None,
) -> tuple[tuple[str, ...], str]:
    if spec.stage_mode == "reuse_stage_a":
        if base_dir is None:
            raise RuntimeError("Missing Stage-A baseline for reuse-stage-a sweep.")
        copy_stage_a_artifacts(base_dir, exp_dir)
        stage_b_ready, stage_b_reason = validate_stage_b_artifacts(
            exp_dir,
            run_args,
            base_dir / "stage_a.pt",
        )
        if stage_b_ready:
            return ("test",), "reuse existing Stage-B artifacts and rerun test only"
        return ("stage_b", "test"), f"reuse Stage-A baseline and rebuild/resume Stage B ({stage_b_reason})"

    stage_a_ready, stage_a_reason = validate_stage_a_seed(exp_dir, run_args)
    stage_b_ready, stage_b_reason = validate_stage_b_artifacts(
        exp_dir,
        run_args,
        exp_dir / "stage_a.pt",
    )
    if stage_b_ready:
        return ("test",), "existing Stage-B artifacts are complete, rerun test only"
    if stage_a_ready:
        return ("stage_b", "test"), f"existing Stage-A artifacts are complete ({stage_a_reason})"
    return ("stage_a", "stage_b", "test"), f"Stage-A artifacts need build/resume ({stage_a_reason})"


def ensure_stage_a_seed(
    python_exe: str,
    artifact_root: Path,
    data_root: str,
    device: str,
    default_args: dict[str, Any],
    base_exp_name: str,
    dry_run: bool,
) -> Path:
    base_dir = artifact_root / base_exp_name
    is_reusable, reason = validate_stage_a_seed(base_dir, default_args)
    if is_reusable:
        print(f"[Sensitivity] reuse Stage-A baseline: {base_dir}")
        return base_dir

    if base_dir.exists():
        reset_experiment_dir(base_dir, dry_run, reason)

    print(f"[Sensitivity] building Stage-A baseline: {base_dir}")
    command = build_run_command(python_exe, artifact_root, data_root, device, "stage_a", base_exp_name, default_args, {})
    run_command(command, stage_log_path(base_dir, "stage_a"), dry_run)
    return base_dir


def copy_stage_a_artifacts(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("stage_a.pt", "stage_a_last.pt", "config.json"):
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, target_dir / filename)


def should_skip(exp_dir: Path, skip_existing: bool) -> bool:
    return skip_existing and (exp_dir / "test_metrics.json").exists()


def nested_get(metrics: dict[str, Any], *keys: str) -> float:
    current: Any = metrics
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return float("nan")
        current = current[key]
    try:
        return float(current)
    except Exception:
        return float("nan")


def extract_metric_summary(metric_root: dict[str, Any]) -> dict[str, float]:
    return {
        "roc_auc": nested_get({"root": metric_root}, "root", "roc_auc"),
        "pr_auc": nested_get({"root": metric_root}, "root", "pr_auc"),
        "point_best_f1": float(metric_root.get("point_best_f1", metric_root.get("best_f1", float("nan")))),
        "pa_best_f1": float(metric_root.get("pa_best_f1", float("nan"))),
        "aff_precision": float(metric_root.get("aff_precision", float("nan"))),
        "aff_recall": float(metric_root.get("aff_recall", float("nan"))),
        "aff_f1": float(metric_root.get("aff_f1", float("nan"))),
        "range_precision": float(metric_root.get("range_precision", float("nan"))),
        "range_recall": float(metric_root.get("range_recall", float("nan"))),
        "range_f1": float(metric_root.get("range_f1", float("nan"))),
        "vus_roc": float(metric_root.get("vus_roc", float("nan"))),
        "vus_pr": float(metric_root.get("vus_pr", float("nan"))),
    }


def read_metric_summary(exp_dir: Path) -> dict[str, Any]:
    metrics_path = exp_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing test metrics: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = metrics if isinstance(metrics, dict) else {}
    score_metrics: dict[str, dict[str, float]] = {}
    flattened: dict[str, Any] = {}
    for score_key in SUMMARY_SCORE_KEYS:
        metric_root = payload.get(score_key, {}) if isinstance(payload, dict) else {}
        if not isinstance(metric_root, dict):
            metric_root = {}
        summary = extract_metric_summary(metric_root)
        score_metrics[score_key] = summary
        for metric_key, metric_value in summary.items():
            flattened[f"{score_key}_{metric_key}"] = metric_value

    # Keep legacy top-level metrics mapped to cdf_max so existing plots
    # and downstream consumers remain compatible.
    flattened.update(score_metrics["cdf_max"])
    flattened["score_metrics"] = score_metrics
    return flattened


def write_summary(
    summary_dir: Path,
    dataset: str,
    study_name: str,
    selected_params: list[str],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / f"{study_name}_summary.json"
    csv_path = summary_dir / f"{study_name}_summary.csv"
    plan_path = summary_dir / f"{study_name}_plan.json"

    payload = {
        "dataset": dataset,
        "dataset_display": dataset_display_name(dataset),
        "study_name": study_name,
        "selected_params": selected_params,
        "specs": {
            name: {
                "group": SWEEP_SPECS[name].group,
                "stage_mode": SWEEP_SPECS[name].stage_mode,
                "values": list(SWEEP_SPECS[name].values),
                "default": SWEEP_SPECS[name].default,
                "description": SWEEP_SPECS[name].description,
                "paper_panel": SWEEP_SPECS[name].paper_panel,
                "value_labels": list(SWEEP_SPECS[name].value_labels) if SWEEP_SPECS[name].value_labels else None,
            }
            for name in selected_params
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "dataset",
        "param",
        "group",
        "stage_mode",
        "status",
        "error",
        "value",
        "value_label",
        "default",
        "experiment_name",
        "experiment_dir",
        "roc_auc",
        "pr_auc",
        "point_best_f1",
        "pa_best_f1",
        "aff_precision",
        "aff_recall",
        "aff_f1",
        "range_precision",
        "range_recall",
        "range_f1",
        "vus_roc",
        "vus_pr",
    ]
    for score_key in SUMMARY_SCORE_KEYS:
        fieldnames.extend(f"{score_key}_{metric_key}" for metric_key in SUMMARY_METRIC_KEYS)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    plan_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "note": (
                    "seq_len sweep uses [64, 128, 160, 192, 256] because seq_len must "
                    "be divisible by patch sizes 8 and 32."
                ),
                "recommended_main_figure_params": ["top_M", "top_K", "knn_k", "d_z", "mask_ratio"],
                "appendix_params": ["seq_len", "coreset_max_patches_per_scale"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return json_path, csv_path


def main() -> None:
    ensure_repo_paths()
    args = parse_args()
    dataset = args.dataset.upper()
    run_tag = args.run_tag or default_run_tag()
    selected_params = resolve_selected_params(args)
    selected_points = resolve_selected_points(args, selected_params)
    default_args = apply_runtime_overrides(resolve_dataset_defaults(dataset), args)
    artifact_root = Path(args.artifact_root or default_artifact_root(dataset))
    study_name = args.study_name or f"{default_study_name(dataset)}_{run_tag}"
    summary_dir = Path(args.summary_dir) if args.summary_dir else artifact_root / "summary"
    base_exp_name = args.base_exp_name or f"{study_name}_stage_a_default"

    print(f"[Sensitivity] repo_root={REPO_ROOT}")
    print(f"[Sensitivity] dataset={dataset_display_name(dataset)}")
    print(f"[Sensitivity] artifact_root={artifact_root}")
    print(f"[Sensitivity] run_tag={run_tag}")
    print(f"[Sensitivity] preset={args.preset} selected_params={selected_params}")
    if args.param_values:
        print(f"[Sensitivity] param_values={args.param_values}")

    needs_stage_a_reuse = has_pending_reuse_stage_a_points(
        artifact_root=artifact_root,
        study_name=study_name,
        selected_points=selected_points,
        skip_existing=args.skip_existing,
    )
    base_dir: Path | None = None
    if needs_stage_a_reuse:
        base_dir = ensure_stage_a_seed(
            args.python_exe,
            artifact_root,
            args.data_root,
            args.device,
            default_args,
            base_exp_name,
            args.dry_run,
        )

    rows: list[dict[str, Any]] = []
    for param_name in selected_params:
        spec = SWEEP_SPECS[param_name]
        for _, value, value_label in selected_points[param_name]:
            exp_name = experiment_name(study_name, spec, value, value_label)
            exp_dir = artifact_root / exp_name
            overrides = {spec.cli_key: value}

            print(
                f"[Sensitivity] running dataset={dataset_display_name(dataset)} "
                f"{param_name}={value_label or value} "
                f"(group={spec.group}, mode={spec.stage_mode}) -> {exp_name}"
            )

            error_message: str | None = None
            status = "ok"
            if should_skip(exp_dir, args.skip_existing):
                print(f"[Sensitivity] skip existing result: {exp_dir / 'test_metrics.json'}")
                status = "skipped_existing"
            else:
                try:
                    run_args = dict(default_args)
                    run_args.update(overrides)
                    run_args["data_root"] = args.data_root
                    stages, stage_reason = resolve_point_stages(exp_dir, spec, run_args, base_dir)
                    print(f"[Sensitivity] stage plan: {stages} ({stage_reason})")

                    for stage in stages:
                        command = build_run_command(
                            args.python_exe,
                            artifact_root,
                            args.data_root,
                            args.device,
                            stage,
                            exp_name,
                            default_args,
                            overrides,
                        )
                        run_command(command, stage_log_path(exp_dir, stage), args.dry_run)
                except Exception as exc:
                    error_message = str(exc)
                    status = "failed"
                    print(f"[Sensitivity] failed point: {exp_name}: {error_message}")
                    if not args.continue_on_error:
                        raise

            row = {
                "dataset": dataset,
                "param": param_name,
                "group": spec.group,
                "stage_mode": spec.stage_mode,
                "status": status,
                "error": error_message or "",
                "value": value,
                "value_label": value_label or str(value),
                "default": spec.default,
                "experiment_name": exp_name,
                "experiment_dir": str(exp_dir),
            }
            if not args.dry_run and (exp_dir / "test_metrics.json").exists():
                row.update(read_metric_summary(exp_dir))
            rows.append(row)

    json_path, csv_path = write_summary(summary_dir, dataset, study_name, selected_params, rows)
    print(f"[Sensitivity] summary_json={json_path}")
    print(f"[Sensitivity] summary_csv={csv_path}")
    print(
        "[Sensitivity] next step: "
        f"{args.python_exe} {REPO_ROOT / 'scripts' / 'sensitivity' / 'plot_dataset_sensitivity.py'} "
        f"--summary-json {json_path}"
    )


if __name__ == "__main__":
    main()
