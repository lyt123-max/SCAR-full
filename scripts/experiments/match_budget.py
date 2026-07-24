from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.budget import select_budget_candidate


D_Z = (64, 128, 256, 512)
KEEP_RATIOS = (1.0, 0.5, 0.25, 0.10, 0.05)
TOP_K = (5, 10, 20, 40, 80)


def closest_axis(
    target: float,
    candidates: list[dict[str, Any]],
    axis: str,
    *,
    smaller_key: str,
) -> dict[str, Any]:
    if target <= 0 or not candidates:
        raise ValueError("Axis matching requires a positive target and candidates.")
    return min(
        candidates,
        key=lambda row: (
            round(abs(float(row[axis]) - target) / target, 12),
            float(row[smaller_key]),
            float(row[axis]),
        ),
    )


def _config_and_model(experiment: Path, *, d_z: int | None = None, seq_len: int | None = None):
    from coremad import CoReMADConfig
    from coremad.model import CoReMADModel

    config = CoReMADConfig.load(experiment / "config.json")
    if d_z is not None:
        config.d_z = int(d_z)
    if seq_len is not None:
        config.seq_len = int(seq_len)
    model = CoReMADModel(config)
    return config, model


def _parameter_count(experiment: Path, *, d_z: int | None = None, seq_len: int | None = None) -> int:
    _, model = _config_and_model(experiment, d_z=d_z, seq_len=seq_len)
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _bank_bytes(experiment: Path) -> int:
    paths = (experiment / "memory.pt", experiment / "memory_meta.json")
    total = sum(path.stat().st_size for path in paths if path.is_file())
    if total <= 0:
        raise FileNotFoundError(f"No memory bank files found in {experiment}.")
    return total


def _train(args: argparse.Namespace) -> None:
    from coremad import CoReMADConfig, CoReMADTrainer

    target = args.target_experiment.resolve()
    target_parameters = _parameter_count(target)
    rows = [
        {
            "d_z": d_z,
            "parameters": _parameter_count(target, d_z=d_z, seq_len=512),
        }
        for d_z in D_Z
    ]
    selected = closest_axis(
        target_parameters, rows, "parameters", smaller_key="d_z"
    )
    output = args.output_dir.resolve()
    config = CoReMADConfig.load(target / "config.json")
    config.artifact_root = str(output.parent)
    config.experiment_name = output.name
    config.seq_len = 512
    config.d_z = int(selected["d_z"])
    config.seed = 42
    config.memory_seed = 42
    config.memory_audit_mode = "full"
    config.resource_monitor_enabled = True
    trainer = CoReMADTrainer(config)
    trainer.run_full()
    payload = {
        "schema_version": 1,
        "target_experiment": str(target),
        "target_parameters": target_parameters,
        "candidates": rows,
        "selected": selected,
        "selection_axis": "parameters",
        "tolerance": 0.15,
    }
    (output / "parameter_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_for_timing(experiment: Path, top_k: int):
    from coremad import CoReMADConfig, CoReMADTrainer

    config = CoReMADConfig.load(experiment / "config.json")
    config.artifact_root = str(experiment.parent)
    config.experiment_name = experiment.name
    config.top_K = int(top_k)
    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    memory = trainer.load_memory_bank()
    raw = trainer._load_raw_bundle()
    bundle = trainer.transform_bundle_with_normalizer(raw, normalizer)
    return config, trainer, model, memory, raw, bundle


def _time_top_k(experiment: Path, top_k: int) -> float:
    import torch

    config, trainer, model, memory, raw, bundle = _load_for_timing(
        experiment, top_k
    )

    def infer() -> None:
        trainer._aggregate_point_diagnostics(
            data=bundle.test,
            labels=raw.test_labels,
            model=model,
            memory=memory,
            batch_size=config.test_batch_size,
            stride=config.test_stride,
            max_windows=config.max_test_windows,
            print_stsd_stats=False,
            segment_ranges=bundle.test_segment_ranges,
        )

    infer()
    durations = []
    for _ in range(3):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        infer()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return float(np.mean(durations) * 1000.0)


def _finalize(args: argparse.Namespace) -> None:
    target = args.target_experiment.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = output.parent / f"scar_e31_{args.dataset.lower()}_global_l512_base"
    selection = json.loads(
        (base / "parameter_selection.json").read_text(encoding="utf-8")
    )
    selected_parameters = int(selection["selected"]["parameters"])
    target_parameters = int(selection["target_parameters"])
    target_bank = _bank_bytes(target)
    timing_path = output.parent / f"scar_efficiency_{args.dataset.lower()}_seed42" / "timing.json"
    target_timing = json.loads(timing_path.read_text(encoding="utf-8"))
    target_latency_ms = float(target_timing["mean_seconds"]) * 1000.0

    q_rows = []
    for ratio in KEEP_RATIOS:
        experiment = (
            base
            if ratio == 1.0
            else output.parent
            / f"scar_e31_{args.dataset.lower()}_global_l512_q{str(ratio).replace('.', 'p')}"
        )
        q_rows.append(
            {
                "keep_ratio": ratio,
                "bank_bytes": _bank_bytes(experiment),
                "experiment": str(experiment),
            }
        )
    selected_q = closest_axis(
        target_bank, q_rows, "bank_bytes", smaller_key="keep_ratio"
    )
    selected_experiment = Path(selected_q["experiment"])
    candidates = []
    for top_k in TOP_K:
        latency_ms = _time_top_k(selected_experiment, top_k)
        candidates.append(
            {
                "name": f"d{selection['selected']['d_z']}_q{selected_q['keep_ratio']}_k{top_k}",
                "d_z": int(selection["selected"]["d_z"]),
                "keep_ratio": float(selected_q["keep_ratio"]),
                "top_K": top_k,
                "parameters": selected_parameters,
                "bank_bytes": int(selected_q["bank_bytes"]),
                "latency_ms": latency_ms,
                "config_size": int(top_k),
            }
        )
    result = select_budget_candidate(
        {
            "parameters": target_parameters,
            "bank_bytes": target_bank,
            "latency_ms": target_latency_ms,
        },
        candidates,
        tolerance=0.15,
    )
    result["parameter_selection"] = selection
    result["bank_selection"] = {"candidates": q_rows, "selected": selected_q}
    result["top_k_timing_protocol"] = {"warmup_runs": 1, "timed_runs": 3}
    (output / "budget_match.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output / "budget_match.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "rebuttal"
                / "muqn"
                / "score_retrieval_strategies.py"
            ),
            "--experiment-dir",
            str(selected_experiment),
            "--strategy",
            "global",
            "--top-k",
            str(result["candidate"]["top_K"]),
            "--output-dir",
            str(output),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/finalize E31 budget matching.")
    parser.add_argument("action", choices=("train", "finalize"))
    parser.add_argument("--target-experiment", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "train":
        _train(args)
    else:
        _finalize(args)


if __name__ == "__main__":
    main()
