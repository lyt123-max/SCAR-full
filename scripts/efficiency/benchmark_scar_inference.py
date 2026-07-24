from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one saved SCAR checkpoint.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    import torch
    from coremad import CoReMADConfig, CoReMADTrainer

    args = parse_args()
    experiment = args.experiment_dir.resolve()
    config = CoReMADConfig.load(experiment / "config.json")
    config.artifact_root = str(experiment.parent)
    config.experiment_name = experiment.name
    if args.device is not None:
        config.device = args.device
    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    memory = trainer.load_memory_bank()
    cdf = trainer.load_cdf_fusion()
    zscore = trainer.load_zscore_fusion()
    raw = trainer._load_raw_bundle()
    bundle = trainer.transform_bundle_with_normalizer(raw, normalizer)

    def infer() -> np.ndarray:
        diagnostics, _ = trainer._aggregate_point_diagnostics(
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
        _, _, _, selected = trainer._build_metric_sources_from_point_diags(
            diagnostics, cdf, zscore
        )
        return np.asarray(selected)

    infer()
    durations = []
    reference = None
    for _ in range(3):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        scores = infer()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        if reference is None:
            reference = scores
        elif not np.allclose(reference, scores, equal_nan=True):
            raise ValueError("SCAR repeated inference produced different scores.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_experiment": str(experiment),
        "warmup_runs": 1,
        "timed_runs": 3,
        "durations_seconds": durations,
        "mean_seconds": float(np.mean(durations)),
        "std_seconds": float(np.std(durations)),
        "median_seconds": float(np.median(durations)),
        "test_points": int(len(raw.test_labels)),
        "seconds_per_point": float(np.mean(durations) / len(raw.test_labels)),
    }
    (args.output_dir / "timing.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
