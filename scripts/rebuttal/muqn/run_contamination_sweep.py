from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad import CoReMADConfig, CoReMADTrainer
from coremad.data import build_loader
from coremad.evaluation import binary_point_metrics
from scripts.experiments.score_outputs import resolve_score_files, score_metric_groups
from scripts.rebuttal.muqn.contamination import (
    event_evaluation_mask,
    event_window_starts,
    filter_event_disjoint_window_starts,
    find_positive_events,
    sample_nested_contamination_starts,
    split_events_into_folds,
)
from scripts.experiments.reuse_stage_a import copy_stage_a_artifacts


class ExplicitWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        starts: np.ndarray,
        seq_len: int,
        provenance_offset: int,
    ) -> None:
        self.data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.provenance_offset = int(provenance_offset)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        source_start = int(self.starts[index])
        return {
            "x": self.data[source_start : source_start + self.seq_len],
            "start": self.provenance_offset + source_start,
            "idx": int(index),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run event-disjoint Stage-B memory contamination sweeps."
    )
    parser.add_argument("--base_experiment_dir", type=Path, required=True)
    parser.add_argument("--fold_manifest", type=Path, default=None)
    parser.add_argument("--artifact_root", type=Path, default=None)
    parser.add_argument("--target_experiment_dir", type=Path, default=None)
    parser.add_argument(
        "--contamination_ratios",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05, 0.10],
    )
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event_window_stride", type=int, default=1)
    parser.add_argument("--clean_ratio", type=float, default=0.02)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _ratio_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_frozen_protocol(
    path: Path,
    *,
    base_dir: Path,
    config: CoReMADConfig,
    test_labels: np.ndarray,
    ratios: list[float],
    n_folds: int,
    seed: int,
) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "base_config_sha256": hashlib.sha256(
            (base_dir / "config.json").read_bytes()
        ).hexdigest(),
        "test_labels_sha256": _sha256_array(test_labels),
        "seq_len": int(config.seq_len),
        "seed": int(seed),
        "n_folds": int(n_folds),
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    available = {float(value) for value in protocol.get("contamination_ratios", [])}
    missing_ratios = sorted(set(ratios) - available)
    if mismatches or missing_ratios:
        raise ValueError(
            "Frozen contamination protocol does not match this run: "
            f"mismatches={mismatches}, missing_ratios={missing_ratios}"
        )
    clean_count = int(protocol.get("clean_window_count", -1))
    folds = protocol.get("folds", [])
    if not isinstance(folds, list) or len(folds) != n_folds:
        raise ValueError("Frozen contamination protocol has an invalid fold list.")
    for fold_index, fold in enumerate(folds):
        if int(fold.get("fold", -1)) != fold_index:
            raise ValueError("Frozen contamination protocol fold indices are invalid.")
        pool = np.asarray(fold.get("candidate_pool", []), dtype=np.int64)
        order = np.asarray(fold.get("candidate_order", []), dtype=np.int64)
        replacement = bool(fold.get("sampling_with_replacement", False))
        if not replacement and len(np.unique(order)) != len(order):
            raise ValueError(f"Frozen fold {fold_index} candidate order contains duplicates.")
        if len(pool) == 0 or not np.isin(order, pool).all():
            raise ValueError(f"Frozen fold {fold_index} candidate order is outside its pool.")
        for ratio in ratios:
            target = int(round(clean_count * ratio / (1.0 - ratio)))
            expected_starts = np.sort(order[:target])
            actual_starts = np.asarray(
                fold.get("starts_by_ratio", {}).get(f"{ratio:.12g}", []),
                dtype=np.int64,
            )
            if target > len(order) or not np.array_equal(actual_starts, expected_starts):
                raise ValueError(
                    f"Frozen fold {fold_index} ratio={ratio} is not the declared "
                    "nested candidate prefix."
                )
    return protocol


def main() -> None:
    args = parse_args()
    base_dir = args.base_experiment_dir.resolve()
    base_config = CoReMADConfig.load(base_dir / "config.json")
    base_config.artifact_root = str(base_dir.parent)
    base_config.experiment_name = base_dir.name
    if args.device is not None:
        base_config.device = args.device
    base_trainer = CoReMADTrainer(base_config)
    _, normalizer, _ = base_trainer.load_stage_a_model()
    raw_bundle = base_trainer._load_raw_bundle()
    bundle = base_trainer.transform_bundle_with_normalizer(raw_bundle, normalizer)

    ratios = [float(value) for value in args.contamination_ratios]
    frozen_protocol = (
        _load_frozen_protocol(
            args.fold_manifest.resolve(),
            base_dir=base_dir,
            config=base_config,
            test_labels=raw_bundle.test_labels,
            ratios=ratios,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        if args.fold_manifest is not None
        else None
    )
    if frozen_protocol is None:
        events = find_positive_events(raw_bundle.test_labels)
        if len(events) < args.n_folds:
            raise ValueError(
                f"Found only {len(events)} anomaly events, fewer than "
                f"n_folds={args.n_folds}."
            )
        event_folds = split_events_into_folds(events, args.n_folds, args.seed)
    else:
        event_folds = [
            [tuple(map(int, event)) for event in fold["injection_events"]]
            for fold in frozen_protocol["folds"]
        ]
    selected_folds = list(range(args.n_folds)) if args.folds is None else args.folds
    artifact_root = (
        args.artifact_root.resolve()
        if args.artifact_root is not None
        else base_dir.parent / "rebuttal_muqn_contamination"
    )
    if args.target_experiment_dir is not None and (
        len(selected_folds) != 1 or len(args.contamination_ratios) != 1
    ):
        raise ValueError(
            "--target_experiment_dir requires exactly one fold and one contamination ratio."
        )

    for fold_index in selected_folds:
        if fold_index < 0 or fold_index >= args.n_folds:
            raise ValueError(f"Invalid fold index {fold_index}.")
        injection_events = event_folds[fold_index]
        evaluation_mask = event_evaluation_mask(len(raw_bundle.test_labels), injection_events)
        clean_loader = build_loader(
            data=bundle.train_full,
            labels=None,
            seq_len=base_config.seq_len,
            stride=base_config.memory_build_stride,
            batch_size=base_config.memory_batch_size,
            num_workers=0,
            shuffle=False,
            max_windows=base_config.max_train_windows,
            drop_last=False,
            segment_ranges=bundle.train_full_segment_ranges,
        )
        clean_window_count = len(clean_loader.dataset)
        if frozen_protocol is None:
            candidate_starts = event_window_starts(
                injection_events,
                data_length=len(bundle.test),
                seq_len=base_config.seq_len,
                stride=args.event_window_stride,
                segment_ranges=raw_bundle.test_segment_ranges,
            )
            candidate_starts = filter_event_disjoint_window_starts(
                candidate_starts,
                raw_bundle.test_labels,
                injection_events,
                base_config.seq_len,
            )
            nested_starts = sample_nested_contamination_starts(
                candidate_starts,
                clean_window_count=clean_window_count,
                contamination_ratios=ratios,
                seed=args.seed,
                allow_replacement=True,
            )
        else:
            expected_clean = int(frozen_protocol["clean_window_count"])
            if clean_window_count != expected_clean:
                raise ValueError(
                    f"Frozen clean_window_count={expected_clean}, "
                    f"current={clean_window_count}."
                )
            frozen_fold = frozen_protocol["folds"][fold_index]
            nested_starts = {
                ratio: np.asarray(
                    frozen_fold["starts_by_ratio"][f"{ratio:.12g}"],
                    dtype=np.int64,
                )
                for ratio in ratios
            }
        for ratio in ratios:
            injected_starts = nested_starts[float(ratio)]
            experiment_name = (
                f"{base_dir.name}__muqn_contam_{_ratio_tag(ratio)}"
                f"__fold_{fold_index}__seed_{args.seed}"
            )
            target_dir = (
                args.target_experiment_dir.resolve()
                if args.target_experiment_dir is not None
                else artifact_root / experiment_name
            )
            manifest = {
                "schema_version": 1,
                "base_experiment_dir": str(base_dir),
                "experiment_dir": str(target_dir),
                "fold": int(fold_index),
                "n_folds": int(args.n_folds),
                "seed": int(args.seed),
                "contamination_ratio_requested": float(ratio),
                "clean_window_count": int(clean_window_count),
                "injected_provenance_offset": int(len(bundle.train_full)),
                "injected_window_count": int(len(injected_starts)),
                "sampling_with_replacement": (
                    bool(
                        frozen_protocol["folds"][fold_index].get(
                            "sampling_with_replacement", False
                        )
                    )
                    if frozen_protocol is not None
                    else bool(len(np.unique(injected_starts)) < len(injected_starts))
                ),
                "contamination_ratio_actual": float(
                    len(injected_starts) / (clean_window_count + len(injected_starts))
                ),
                "injection_events": [list(event) for event in injection_events],
                "injected_test_window_starts": injected_starts.tolist(),
                "evaluation_excludes_injection_events": True,
                "n_evaluation_points": int(evaluation_mask.sum()),
                "frozen_fold_manifest": (
                    str(args.fold_manifest.resolve())
                    if args.fold_manifest is not None
                    else None
                ),
            }
            if args.dry_run:
                print(json.dumps(manifest, ensure_ascii=False))
                continue

            copy_stage_a_artifacts(base_dir, target_dir)
            config = CoReMADConfig.load(base_dir / "config.json")
            config.artifact_root = str(target_dir.parent)
            config.experiment_name = target_dir.name
            config.clean_ratio = float(args.clean_ratio)
            config.memory_seed = int(args.seed)
            config.memory_audit_mode = "full"
            config.resume = False
            config.num_workers = 0
            if args.device is not None:
                config.device = args.device
            config.save()

            injected_dataset = ExplicitWindowDataset(
                bundle.test,
                injected_starts,
                config.seq_len,
                provenance_offset=len(bundle.train_full),
            )
            memory_dataset = ConcatDataset([clean_loader.dataset, injected_dataset])
            memory_loader = DataLoader(
                memory_dataset,
                batch_size=config.memory_batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
            )
            trainer = CoReMADTrainer(config)
            trainer.run_stage_b(memory_loader=memory_loader)
            trainer.run_test()

            test_metrics = json.loads(
                (target_dir / "test_metrics.json").read_text(encoding="utf-8")
            )
            score_names = [
                "selected",
                "raw_max",
                "zscore_mean",
                "cdf_mean",
                "cdf_max",
                *sorted(test_metrics.get("subscores", {})),
            ]
            score_files = resolve_score_files(target_dir, test_metrics, score_names)
            heldout_score_metrics = {}
            for score_name in dict.fromkeys(score_names):
                score_path = target_dir / score_files[score_name]
                heldout_score_metrics[score_name] = binary_point_metrics(
                    raw_bundle.test_labels,
                    np.load(score_path),
                    evaluation_mask=evaluation_mask,
                )
            score_metric_groups(
                {"scores": heldout_score_metrics},
                require_core=True,
                require_report_metrics=True,
                require_finite_report_metrics=True,
            )
            heldout_metrics = heldout_score_metrics["selected"]
            manifest["heldout_point_metrics"] = heldout_metrics
            manifest["heldout_score_metrics"] = heldout_score_metrics
            manifest["selected_score_key"] = test_metrics.get(
                "selected_score_key", "cdf_mean"
            )
            manifest["score_files"] = {
                name: str(target_dir / score_files[name])
                for name in heldout_score_metrics
            }
            manifest["score_file"] = manifest["score_files"]["selected"]
            (target_dir / "contamination_protocol.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"[muQn] fold={fold_index} contamination={ratio:.3f} "
                f"AUROC={heldout_metrics['roc_auc']:.6f} AP={heldout_metrics['pr_auc']:.6f}"
            )


if __name__ == "__main__":
    main()
