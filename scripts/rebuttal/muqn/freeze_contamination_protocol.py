from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad import CoReMADConfig, CoReMADTrainer
from coremad.data import build_loader
from scripts.rebuttal.muqn.contamination import (
    build_nested_candidate_order,
    event_window_starts,
    filter_event_disjoint_window_starts,
    find_positive_events,
    split_events_into_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the event folds and nested E10 contamination prefixes."
    )
    parser.add_argument("--base-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contamination-ratios",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05, 0.10],
    )
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-window-stride", type=int, default=1)
    return parser.parse_args()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_frozen_protocol(
    *,
    base_dir: Path,
    ratios: list[float],
    n_folds: int,
    seed: int,
    event_window_stride: int,
) -> dict[str, object]:
    base_dir = base_dir.resolve()
    config_path = base_dir / "config.json"
    config = CoReMADConfig.load(config_path)
    config.artifact_root = str(base_dir.parent)
    config.experiment_name = base_dir.name
    trainer = CoReMADTrainer(config)
    _, normalizer, _ = trainer.load_stage_a_model()
    raw_bundle = trainer._load_raw_bundle()
    bundle = trainer.transform_bundle_with_normalizer(raw_bundle, normalizer)

    events = find_positive_events(raw_bundle.test_labels)
    if len(events) < n_folds:
        raise ValueError(
            f"Found only {len(events)} anomaly events, fewer than n_folds={n_folds}."
        )
    folds = split_events_into_folds(events, n_folds, seed)
    clean_loader = build_loader(
        data=bundle.train_full,
        labels=None,
        seq_len=config.seq_len,
        stride=config.memory_build_stride,
        batch_size=config.memory_batch_size,
        num_workers=0,
        shuffle=False,
        max_windows=config.max_train_windows,
        drop_last=False,
        segment_ranges=bundle.train_full_segment_ranges,
    )
    clean_window_count = len(clean_loader.dataset)

    frozen_folds: list[dict[str, object]] = []
    for fold_index, injection_events in enumerate(folds):
        candidates = event_window_starts(
            injection_events,
            data_length=len(bundle.test),
            seq_len=config.seq_len,
            stride=event_window_stride,
            segment_ranges=raw_bundle.test_segment_ranges,
        )
        candidates = filter_event_disjoint_window_starts(
            candidates,
            raw_bundle.test_labels,
            injection_events,
            config.seq_len,
        )
        targets = {
            ratio: int(round(clean_window_count * ratio / (1.0 - ratio)))
            for ratio in ratios
        }
        candidate_order = build_nested_candidate_order(
            candidates,
            required_count=max(targets.values(), default=0),
            seed=seed,
            allow_replacement=True,
        )
        nested = {
            ratio: np.sort(candidate_order[: targets[ratio]]) for ratio in ratios
        }
        frozen_folds.append(
            {
                "fold": fold_index,
                "injection_events": [list(event) for event in injection_events],
                "candidate_pool": candidates.tolist(),
                "candidate_order": candidate_order.tolist(),
                "sampling_with_replacement": bool(
                    len(candidate_order) > len(candidates)
                ),
                "starts_by_ratio": {
                    f"{ratio:.12g}": nested[float(ratio)].tolist() for ratio in ratios
                },
            }
        )

    config_bytes = config_path.read_bytes()
    return {
        "schema_version": 1,
        "dataset": config.dataset,
        "base_experiment_dir": str(base_dir),
        "base_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "test_labels_sha256": _sha256_array(np.asarray(raw_bundle.test_labels)),
        "seq_len": int(config.seq_len),
        "memory_build_stride": int(config.memory_build_stride),
        "event_window_stride": int(event_window_stride),
        "clean_window_count": int(clean_window_count),
        "test_length": int(len(raw_bundle.test_labels)),
        "seed": int(seed),
        "n_folds": int(n_folds),
        "contamination_ratios": [float(value) for value in ratios],
        "folds": frozen_folds,
    }


def main() -> None:
    args = parse_args()
    ratios = sorted({float(value) for value in args.contamination_ratios})
    protocol = build_frozen_protocol(
        base_dir=args.base_experiment_dir,
        ratios=ratios,
        n_folds=int(args.n_folds),
        seed=int(args.seed),
        event_window_stride=int(args.event_window_stride),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "contamination_fold_manifest.json"
    output_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
