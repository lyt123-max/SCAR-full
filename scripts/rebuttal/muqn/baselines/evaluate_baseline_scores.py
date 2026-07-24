from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from coremad.evaluation import binary_point_metrics
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    evaluation_path = REPO_ROOT / "coremad" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("coremad_standalone_evaluation", evaluation_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {evaluation_path}.") from exc
    evaluation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation_module)
    binary_point_metrics = evaluation_module.binary_point_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline scores with a shared point protocol.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--score_column", type=int, default=1)
    parser.add_argument("--label_column", type=int, default=0)
    parser.add_argument("--skip_rows", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def align_labels_and_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    train_end: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if train_end is None:
        if len(labels) != len(scores):
            raise ValueError(
                f"Label/score length mismatch and no train_end was provided: "
                f"{len(labels)} vs {len(scores)}."
            )
        return labels, scores
    boundary = int(train_end)
    if boundary < 0:
        raise ValueError("train_end must be non-negative.")
    if len(scores) == len(labels) - boundary:
        return labels[boundary:], scores
    if len(scores) == len(labels) and boundary <= len(labels):
        return labels[boundary:], scores[boundary:]
    raise ValueError(
        f"Label/score lengths cannot align at train_end={boundary}: "
        f"{len(labels)} vs {len(scores)}."
    )


def main() -> None:
    args = parse_args()
    train_end = args.train_end
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if train_end is None:
            train_end = int(manifest["train_end"])
    if args.scores.suffix.lower() == ".npy":
        scores = np.load(args.scores).reshape(-1)
        if args.labels is None:
            raise ValueError("--labels is required for NPY scores.")
        labels = np.load(args.labels).reshape(-1)
    else:
        table = np.loadtxt(args.scores, delimiter=",", skiprows=args.skip_rows)
        labels = table[:, args.label_column]
        scores = table[:, args.score_column]
    labels, scores = align_labels_and_scores(labels, scores, train_end=train_end)
    result = binary_point_metrics(labels, scores)
    payload = {
        "schema_version": 1,
        "score_file": str(args.scores.resolve()),
        "metrics": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[muQn] wrote {args.output}")


if __name__ == "__main__":
    main()
