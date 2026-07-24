from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_STRATEGY_FLAGS = {
    "full": (True, True),
    "no_state": (False, True),
    "context_only": (False, True),
    "no_context": (True, False),
    "state_only": (True, False),
    "global": (False, False),
}


def strategy_flags(strategy: str) -> tuple[bool, bool]:
    try:
        return _STRATEGY_FLAGS[str(strategy)]
    except KeyError as exc:
        raise ValueError(f"Unknown retrieval strategy: {strategy!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a frozen SCAR checkpoint/memory under one retrieval strategy."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=sorted(_STRATEGY_FLAGS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from coremad import CoReMADConfig, CoReMADTrainer
    from coremad.data import build_loader
    from coremad.scorer import CDFPITFusion, ZScoreMeanFusion
    from scripts.rebuttal.muqn.export_retrieval_logs import _export_strategy

    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CoReMADConfig.load(experiment_dir / "config.json")
    config.artifact_root = str(experiment_dir.parent)
    config.experiment_name = experiment_dir.name
    if args.top_k is not None:
        if args.top_k <= 0:
            raise ValueError("--top-k must be positive.")
        config.top_K = int(args.top_k)
    if args.device is not None:
        config.device = args.device
    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    memory = trainer.load_memory_bank()
    use_state, use_context = strategy_flags(args.strategy)
    memory.config.use_two_level_retrieval = use_state
    memory.config.use_context_key_retrieval = use_context

    raw = trainer._load_raw_bundle()
    bundle = trainer.transform_bundle_with_normalizer(raw, normalizer)
    started = time.perf_counter()
    train_diags, train_count = trainer._aggregate_point_diagnostics(
        data=bundle.train_full,
        labels=None,
        model=model,
        memory=memory,
        batch_size=config.memory_batch_size,
        stride=config.memory_build_stride,
        max_windows=config.max_train_windows,
        print_stsd_stats=False,
        segment_ranges=bundle.train_full_segment_ranges,
    )
    cdf_names = trainer._cdf_fit_score_names()
    cdf = CDFPITFusion(cdf_names)
    cdf.fit(
        trainer._select_observed_train_diagnostics(train_diags, train_count, cdf_names)
    )
    zscore_names = trainer._fusion_score_names()
    zscore = ZScoreMeanFusion(zscore_names)
    zscore.fit(
        trainer._select_observed_train_diagnostics(
            train_diags, train_count, zscore_names
        )
    )
    cdf.save(output_dir / "strategy_cdf_fusion")
    zscore.save(output_dir / "strategy_zscore_fusion")

    test_diags, test_count = trainer._aggregate_point_diagnostics(
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
    test_diags, _, metric_sources, selected = trainer._build_metric_sources_from_point_diags(
        test_diags,
        cdf,
        zscore,
    )
    metrics = {
        name: trainer._compute_metrics(
            raw.test_labels,
            score,
            vus_window=raw.evaluation_vus_window,
        )
        for name, score in metric_sources.items()
    }
    np.save(output_dir / "scores.npy", selected)
    for name, values in metric_sources.items():
        np.save(output_dir / f"scores_{name}.npy", values)
    np.save(output_dir / "coverage_count.npy", test_count)
    np.savez_compressed(output_dir / "diagnostics.npz", **test_diags)
    payload = {
        "schema_version": 1,
        "strategy": args.strategy,
        "use_state_filtering": use_state,
        "use_context_retrieval": use_context,
        "selected_score_key": config.evaluation_score_key,
        "selected": metrics["selected"],
        "scores": metrics,
        "score_files": {
            name: f"scores_{name}.npy" for name in metric_sources
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_experiment": str(experiment_dir),
                "model_retrained": False,
                "memory_rebuilt": False,
                "fusion_recalibrated": True,
                "strategy": args.strategy,
                "top_K": int(config.top_K),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    retrieval_loader = build_loader(
        data=bundle.test,
        labels=bundle.test_labels,
        seq_len=config.seq_len,
        stride=config.test_stride,
        batch_size=config.test_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        max_windows=config.max_test_windows,
        drop_last=False,
        segment_ranges=bundle.test_segment_ranges,
    )
    _export_strategy(
        args.strategy,
        {
            "config": config,
            "trainer": trainer,
            "model": model,
            "memory": memory,
            "bundle": bundle,
        },
        retrieval_loader,
        output_dir / f"retrieval_{args.strategy}.npz",
        aggregate_top_k=20,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
