from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


A3_VARIANTS = {
    "pit_fusion": "cdf_mean",
    "raw_max": "raw_max",
    "zscore_mean": "zscore_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize A3 ablation result directories from one full experiment."
    )
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--source_artifact_root", type=Path, default=None)
    parser.add_argument("--source_experiment", type=str, required=True)
    parser.add_argument("--target_pit", type=str, default=None)
    parser.add_argument("--target_raw_max", type=str, default=None)
    parser.add_argument("--target_zscore_mean", type=str, default=None)
    return parser.parse_args()


def infer_target_experiment(source_experiment: str, ablation_key: str) -> str:
    marker = "_ablation_full"
    if marker in source_experiment:
        return source_experiment.replace(marker, f"_ablation_{ablation_key}", 1)
    fallback_marker = "_full"
    if fallback_marker in source_experiment:
        return source_experiment.replace(fallback_marker, f"_{ablation_key}", 1)
    return f"{source_experiment}_{ablation_key}"


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_if_exists(source_path: Path, target_path: Path) -> None:
    if source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def materialize_variant(
    artifact_root: Path,
    source_artifact_root: Path,
    source_experiment: str,
    target_experiment: str,
    score_key: str,
) -> None:
    source_dir = source_artifact_root / source_experiment
    target_dir = artifact_root / target_experiment
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = source_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing source metrics: {metrics_path}")
    metrics = load_metrics(metrics_path)

    if score_key not in metrics:
        raise KeyError(f"Score key '{score_key}' is not present in {metrics_path}")

    materialized = json.loads(json.dumps(metrics))
    materialized["selected_score_key"] = score_key
    materialized["selected"] = materialized[score_key]
    materialized["materialized_from"] = {
        "source_experiment": source_experiment,
        "source_score_key": score_key,
    }
    if isinstance(materialized.get("weak_pointwise"), dict) and score_key in materialized["weak_pointwise"]:
        materialized["weak_pointwise"]["selected"] = materialized["weak_pointwise"][score_key]

    save_metrics(target_dir / "test_metrics.json", materialized)

    copy_if_exists(source_dir / "test_diagnostic_scores.npz", target_dir / "test_diagnostic_scores.npz")
    copy_if_exists(source_dir / "test_sequence_scores.npz", target_dir / "test_sequence_scores.npz")
    copy_if_exists(source_dir / "test_sequence_scores.csv", target_dir / "test_sequence_scores.csv")

    for source_path in source_dir.glob("test_scores*.npy"):
        copy_if_exists(source_path, target_dir / source_path.name)

    score_files = materialized.get("score_files", {})
    selected_score_filename = score_files.get(score_key)
    if isinstance(selected_score_filename, str):
        selected_source = source_dir / selected_score_filename
        copy_if_exists(selected_source, target_dir / "test_scores_selected.npy")
        copy_if_exists(selected_source, target_dir / "test_scores_final_selected.npy")
        copy_if_exists(selected_source, target_dir / "test_scores.npy")

    print(
        "[Ablation] A3 materialized from full: "
        f"source={source_dir} target={target_dir} score_key={score_key}"
    )


def main() -> None:
    args = parse_args()
    source_artifact_root = args.source_artifact_root or args.artifact_root
    target_names = {
        "pit_fusion": args.target_pit or infer_target_experiment(args.source_experiment, "pit_fusion"),
        "raw_max": args.target_raw_max or infer_target_experiment(args.source_experiment, "raw_max"),
        "zscore_mean": args.target_zscore_mean or infer_target_experiment(args.source_experiment, "zscore_mean"),
    }

    for ablation_key, score_key in A3_VARIANTS.items():
        materialize_variant(
            artifact_root=args.artifact_root,
            source_artifact_root=source_artifact_root,
            source_experiment=args.source_experiment,
            target_experiment=target_names[ablation_key],
            score_key=score_key,
        )


if __name__ == "__main__":
    main()
