from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rebuttal.muqn.contamination import (
    Event,
    event_evaluation_mask,
    find_positive_events,
    split_events_into_folds,
)


def _load_binary_point_metrics():
    path = REPO_ROOT / "coremad" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("scar_standalone_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.binary_point_metrics


def masked_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    excluded_events: Sequence[Event],
) -> tuple[dict, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError(f"Label/score length mismatch: {len(labels)} vs {len(scores)}.")
    mask = event_evaluation_mask(len(labels), excluded_events)
    metrics = _load_binary_point_metrics()(labels, scores, evaluation_mask=mask)
    metrics["n_evaluation_points"] = int(mask.sum())
    return metrics, mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply held-out contamination folds to an existing zero-contamination score."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-file", default="test_scores_selected.npy")
    parser.add_argument("--fold-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fold < 0 or args.fold >= args.n_folds:
        raise ValueError(f"Invalid fold {args.fold} for n_folds={args.n_folds}.")
    from coremad import CoReMADConfig, CoReMADTrainer

    experiment_dir = args.experiment_dir.resolve()
    config = CoReMADConfig.load(experiment_dir / "config.json")
    config.artifact_root = str(experiment_dir.parent)
    config.experiment_name = experiment_dir.name
    raw = CoReMADTrainer(config)._load_raw_bundle()
    raw_labels = np.asarray(raw.test_labels)
    labels = raw_labels.reshape(-1)
    if args.fold_manifest is None:
        events = find_positive_events(labels)
        folds = split_events_into_folds(events, args.n_folds, args.seed)
    else:
        protocol = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
        label_array = np.ascontiguousarray(raw_labels)
        digest = hashlib.sha256()
        digest.update(str(label_array.dtype).encode("ascii"))
        digest.update(str(label_array.shape).encode("ascii"))
        digest.update(label_array.tobytes())
        if (
            int(protocol.get("seed", -1)) != args.seed
            or int(protocol.get("n_folds", -1)) != args.n_folds
            or protocol.get("test_labels_sha256") != digest.hexdigest()
        ):
            raise ValueError("Frozen fold manifest does not match zero-contamination data.")
        folds = [
            [tuple(map(int, event)) for event in fold["injection_events"]]
            for fold in protocol["folds"]
        ]
    scores = np.load(experiment_dir / args.score_file)
    metrics, mask = masked_binary_metrics(labels, scores, folds[args.fold])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "evaluation_mask.npy", mask)
    payload = {
        "schema_version": 1,
        "source_experiment": str(experiment_dir),
        "score_file": args.score_file,
        "fold": int(args.fold),
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "excluded_events": [list(event) for event in folds[args.fold]],
        "frozen_fold_manifest": (
            str(args.fold_manifest.resolve()) if args.fold_manifest is not None else None
        ),
        "metrics": metrics,
    }
    (args.output_dir / "heldout_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
