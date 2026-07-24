from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .manifest import RunSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_SEED = 42
MAIN_DATASETS = ("MSL", "PSM", "SMAP", "SMD", "SWAT")
PURIFICATION_RATIOS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)
CONTAMINATION_RATIOS = (0.01, 0.03, 0.05, 0.10)
CORESET_KEEP_RATIOS = (1.0, 0.5, 0.25, 0.10, 0.05)
WINDOW_LENGTHS = (64, 128, 256, 512)
SINGLE_PATCH_SIZES = (8, 16, 32, 64)
BUDGET_DIMS = (64, 128, 256, 512)
BUDGET_TOP_K = (5, 10, 20, 40, 80)


_COMMON_PROFILE: dict[str, Any] = {
    "seq_len": 128,
    "batch_size": 128,
    "train_batch_size": 128,
    "val_batch_size": 128,
    "memory_batch_size": 128,
    "test_batch_size": 128,
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
    "completion_dropout": 0.1,
    "lambda_pred": 0.5,
    "lambda_smooth": 0.01,
    "patch_sizes": [8, 32],
    "d_z": 128,
    "top_M": 50,
    "top_K": 20,
    "knn_k": 5,
    "clean_ratio": 0.02,
    "coreset_keep_ratio": 1.0,
    "coreset_max_patches_per_scale": 200000,
    "coreset_fps_threshold": 100000,
    "evaluation_score_key": "cdf_mean",
    "seed": FORMAL_SEED,
    "memory_seed": FORMAL_SEED,
    "memory_audit_mode": "full",
    "resource_monitor": 1,
}


def _profile(**overrides: Any) -> dict[str, Any]:
    result = dict(_COMMON_PROFILE)
    result.update(overrides)
    return result


FORMAL_DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "MSL": _profile(),
    "PSM": _profile(val_split_mode="interleaved", lr=5e-4),
    "SMAP": _profile(),
    "SMD": _profile(
        memory_build_stride=4,
        val_split_mode="interleaved",
        early_stop_patience=6,
        lr=1e-3,
    ),
    "SWAT": _profile(
        memory_build_stride=2,
        early_stop_patience=10,
        lr=1e-3,
        completion_dropout=0.2,
    ),
}


CATCH_EXTENSION_DATASETS = (
    *(f"ASD_dataset_{index}" for index in range(1, 13)),
    "CalIt2",
    "CICIDS",
    "Creditcard",
    "GECCO",
    "Genesis",
    "NYC",
    "synthetic_con0.0494",
    "synthetic_con0.072",
    "synthetic_glo0.048",
    "synthetic_glo0.0718",
    "synthetic_sea0.0482",
    "synthetic_sea0.0774",
    "synthetic_sha0.049",
    "synthetic_sha0.0742",
    "synthetic_sub_mix0.0574",
    "synthetic_sub_mix0.089",
    "synthetic_tre0.0482",
    "synthetic_tre0.0778",
)
SYNTHETIC_DATASETS = tuple(
    dataset for dataset in CATCH_EXTENSION_DATASETS if dataset.startswith("synthetic_")
)


CORE_REQUIRED_BASE = (
    "config.json",
    "stage_a.pt",
    "memory.pt",
    "faiss_state.index",
    "memory_meta.json",
    "cdf_fusion.npz",
    "cdf_fusion.json",
    "zscore_fusion.json",
    "test_diagnostic_scores.npz",
    "test_scores_selected.npy",
    "test_scores_raw_max.npy",
    "test_scores_zscore_mean.npy",
    "test_scores_cdf_mean.npy",
    "test_scores_cdf_max.npy",
    "test_scores_knn_distance.npy",
    "test_scores_state_novelty.npy",
    "test_metrics.json",
    "resource_metrics.json",
)
CORE_REQUIRED = CORE_REQUIRED_BASE + (
    "test_scores_completion_scale8.npy",
    "test_scores_completion_scale32.npy",
)
FULL_AUDIT_REQUIRED = CORE_REQUIRED + (
    "memory_audit/memory_audit_state.npz",
    "memory_audit/memory_audit_scale8.npz",
    "memory_audit/memory_audit_scale32.npz",
)

SEQUENCE_CORE_REQUIRED = (
    "config.json",
    "stage_a.pt",
    "memory.pt",
    "faiss_state.index",
    "memory_meta.json",
    "cdf_fusion.npz",
    "cdf_fusion.json",
    "zscore_fusion.json",
    "test_sequence_scores_selected.npy",
    "test_sequence_scores_raw_max.npy",
    "test_sequence_scores_zscore_mean.npy",
    "test_sequence_scores_cdf_mean.npy",
    "test_sequence_scores_cdf_max.npy",
    "test_sequence_scores_knn_distance.npy",
    "test_sequence_scores_state_novelty.npy",
    "test_sequence_scores_completion_scale8.npy",
    "test_sequence_scores_completion_scale32.npy",
    "test_sequence_scores.npz",
    "test_sequence_scores.csv",
    "test_metrics.json",
    "resource_metrics.json",
)
SEQUENCE_FULL_AUDIT_REQUIRED = SEQUENCE_CORE_REQUIRED + (
    "memory_audit/memory_audit_state.npz",
    "memory_audit/memory_audit_scale8.npz",
    "memory_audit/memory_audit_scale32.npz",
)


def full_audit_required(patch_sizes: Iterable[int]) -> tuple[str, ...]:
    patch_sizes = tuple(int(value) for value in patch_sizes)
    return CORE_REQUIRED_BASE + (
        *(f"test_scores_completion_scale{patch_size}.npy" for patch_size in patch_sizes),
        "memory_audit/memory_audit_state.npz",
        *(
            f"memory_audit/memory_audit_scale{patch_size}.npz"
            for patch_size in patch_sizes
        ),
    )


def validate_formal_request(
    *,
    seed: int,
    tsb_edition: str | None = None,
    tsb_split: str | None = None,
    allow_plots: bool = False,
) -> None:
    if int(seed) != FORMAL_SEED:
        raise ValueError("Formal rebuttal runs require fixed seed 42.")
    if (
        str(tsb_edition or "").upper() == "U"
        and str(tsb_split or "").lower().replace("-", "_") == "eval_full"
    ):
        raise ValueError("Formal rebuttal runs forbid TSB-AD U/eval_full.")
    if allow_plots:
        raise ValueError("Formal rebuttal runs forbid plot generation.")


def _cli_args(config: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        if value is None:
            continue
        args.append(f"--{key}")
        if isinstance(value, (list, tuple)):
            args.extend(str(item) for item in value)
        else:
            args.append(str(value))
    return args


def _scar_task(
    *,
    artifact_root: Path,
    python_exe: str,
    dataset: str,
    group: str,
    compute_kind: str,
    experiment_name: str,
    config: dict[str, Any],
    stage: str = "full",
    required: tuple[str, ...] | None = None,
    dependencies: tuple[str, ...] = (),
) -> RunSpec:
    artifact_dir = artifact_root / experiment_name
    if stage == "stage_b_test":
        base_experiment = config.get("base_experiment_dir")
        if not base_experiment:
            raise ValueError(f"{experiment_name} requires base_experiment_dir.")
        override_keys = (
            "memory_build_stride",
            "memory_batch_size",
            "test_batch_size",
            "test_stride",
            "top_M",
            "top_K",
            "knn_k",
            "clean_ratio",
            "coreset_keep_ratio",
            "coreset_max_patches_per_scale",
            "coreset_fps_threshold",
            "memory_seed",
            "memory_audit_mode",
            "evaluation_score_key",
        )
        overrides = [
            item
            for key in override_keys
            if key in config
            for item in ("--override", f"{key}={canonical_value(config[key])}")
        ]
        overrides.extend(
            ("--override", f"resource_monitor_enabled={canonical_value(bool(config.get('resource_monitor', 1)))}")
        )
        command = (
            python_exe,
            str(REPO_ROOT / "scripts" / "experiments" / "reuse_stage_a.py"),
            "--base-experiment-dir",
            str(base_experiment),
            "--target-experiment-dir",
            str(artifact_dir),
            *overrides,
        )
    else:
        command = (
            python_exe,
            str(REPO_ROOT / "run.py"),
            "--stage",
            stage,
            "--dataset",
            dataset,
            "--data_root",
            "./dataset/anomaly_detect",
            "--artifact_root",
            str(artifact_root),
            "--experiment_name",
            experiment_name,
            *_cli_args(config),
        )
    required_artifacts = (
        required
        if required is not None
        else full_audit_required(config.get("patch_sizes", (8, 32)))
    )
    return RunSpec(
        method="SCAR",
        dataset=dataset,
        seed=FORMAL_SEED,
        stage=stage,
        group=group,
        compute_kind=compute_kind,
        config=config,
        command=tuple(command),
        artifact_dir=artifact_dir,
        required_artifacts=required_artifacts,
        dependencies=dependencies,
        metadata={"environment": {"SCAR_METRIC_WORKERS": "8"}},
    )


def canonical_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_value(item) for item in value) + "]"
    return str(value)


def _read_tsb_manifest(edition: str, split: str) -> list[str]:
    suffix = {
        ("M", "tuning"): "TSB-AD-M-Tuning.csv",
        ("M", "eval"): "TSB-AD-M-Eva.csv",
        ("U", "tuning"): "TSB-AD-U-Tuning.csv",
        ("U", "eval"): "TSB-AD-U-Eva.csv",
    }[(edition, split)]
    path = REPO_ROOT / "scripts" / "tsb_ad" / "manifests" / suffix
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row["file_name"] for row in csv.DictReader(handle)]


def _analysis_task(
    *,
    artifact_root: Path,
    method: str,
    dataset: str,
    group: str,
    name: str,
    config: dict[str, Any],
    command: Iterable[str],
    required: tuple[str, ...],
    dependencies: tuple[str, ...] = (),
) -> RunSpec:
    return RunSpec(
        method=method,
        dataset=dataset,
        seed=FORMAL_SEED,
        stage="analysis",
        group=group,
        compute_kind="analysis",
        config=config,
        command=tuple(command),
        artifact_dir=artifact_root / name,
        required_artifacts=required,
        dependencies=dependencies,
    )


def _with_metric_workers(task: RunSpec) -> RunSpec:
    if task.method != "SCAR":
        return task
    metadata = dict(task.metadata)
    environment = dict(metadata.get("environment", {}))
    environment["SCAR_METRIC_WORKERS"] = "8"
    metadata["environment"] = environment
    return replace(task, metadata=metadata)


def build_p0_tasks(artifact_root: Path, *, python_exe: str) -> list[RunSpec]:
    artifact_root = Path(artifact_root)
    tasks: list[RunSpec] = []
    anchors: dict[str, RunSpec] = {}
    e9_specs: dict[tuple[str, float], RunSpec] = {}
    e10_protocol_specs: dict[str, RunSpec] = {}
    e10_specs: list[RunSpec] = []
    e38_specs: list[RunSpec] = []

    for dataset in MAIN_DATASETS:
        config = dict(FORMAL_DATASET_PROFILES[dataset])
        anchor = _scar_task(
            artifact_root=artifact_root,
            python_exe=python_exe,
            dataset=dataset,
            group="anchors",
            compute_kind="full_model_fit",
            experiment_name=f"scar_main_{dataset.lower()}_seed42",
            config=config,
        )
        anchors[dataset] = anchor
        tasks.append(anchor)

    for dataset in CATCH_EXTENSION_DATASETS:
        config = dict(_COMMON_PROFILE)
        tasks.append(
            _scar_task(
                artifact_root=artifact_root,
                python_exe=python_exe,
                dataset=dataset,
                group="catch",
                compute_kind="full_model_fit",
                experiment_name=f"scar_catch_{dataset.lower()}_seed42",
                config=config,
            )
        )

    tep_name = "scar_tep_full_seed42"
    tep_config = {
        **_COMMON_PROFILE,
        "tep_protocol": "full",
        "max_test_sequences": 0,
        "allow_plots": False,
    }
    tasks.append(
        RunSpec(
            method="SCAR",
            dataset="TEP",
            seed=FORMAL_SEED,
            stage="full",
            group="tep",
            compute_kind="full_model_fit",
            config=tep_config,
            command=(
                "bash",
                str(REPO_ROOT / "scripts" / "main" / "train_tep_full.sh"),
                "full",
                tep_name,
                tep_name,
            ),
            artifact_dir=artifact_root / tep_name,
            required_artifacts=SEQUENCE_FULL_AUDIT_REQUIRED
            + ("tep_rebuttal_tables/table_t2_tep_availability.csv",),
            metadata={
                "environment": {
                    "FORMAL_REBUTTAL": "1",
                    "TEP_ALLOW_PLOTS": "0",
                    "MAX_TEST_SEQUENCES": "0",
                    "SCAR_METRIC_WORKERS": "8",
                }
            },
        )
    )

    for edition, split in (("M", "tuning"), ("M", "eval"), ("U", "tuning"), ("U", "eval")):
        validate_formal_request(seed=FORMAL_SEED, tsb_edition=edition, tsb_split=split)
        for file_name in _read_tsb_manifest(edition, split):
            stem = Path(file_name).stem
            config = {
                "edition": edition,
                "split": split,
                "file_name": file_name,
                "seq_len": 128,
                "patch_sizes": [8, 32],
                "stage_a_epochs": 100,
                "evaluation_score_key": "cdf_mean",
                "seed": FORMAL_SEED,
            }
            run_root = artifact_root / "tsb_ad" / edition / split / "seed_42"
            tasks.append(
                RunSpec(
                    method="SCAR",
                    dataset=file_name,
                    seed=FORMAL_SEED,
                    stage="full",
                    group="tsb",
                    compute_kind="full_model_fit",
                    config=config,
                    command=(
                        python_exe,
                        str(REPO_ROOT / "run.py"),
                        "--stage",
                        "full",
                        "--data_format",
                        "tsb_ad",
                        "--dataset",
                        file_name,
                        "--data_root",
                        f"./dataset/TSB-AD-{edition}/TSB-AD-{edition}",
                        "--artifact_root",
                        str(run_root),
                        "--experiment_name",
                        stem,
                        "--seed",
                        "42",
                        "--seq_len",
                        "128",
                        "--patch_sizes",
                        "8",
                        "32",
                        "--stage_a_epochs",
                        "100",
                        "--evaluation_score_key",
                        "cdf_mean",
                    ),
                    artifact_dir=run_root / stem,
                    required_artifacts=CORE_REQUIRED,
                )
            )

    # CATCH is compared using its official reported benchmark results.  Do not
    # spend rebuttal compute reproducing it under a second protocol.
    for method in ("PaAno", "MEMTO", "PUAD", "PGRF-Net"):
        for dataset in MAIN_DATASETS:
            name = f"baseline_{method.lower().replace('-', '_')}_{dataset.lower()}_seed42"
            tasks.append(
                RunSpec(
                    method=method,
                    dataset=dataset,
                    seed=FORMAL_SEED,
                    stage="full",
                    group="baselines",
                    compute_kind="full_model_fit",
                    config={"seed": FORMAL_SEED, "dataset": dataset},
                    command=(
                        python_exe,
                        str(REPO_ROOT / "scripts" / "rebuttal" / "baselines" / "run_baseline.py"),
                        "--method",
                        method,
                        "--dataset",
                        dataset,
                        "--output-dir",
                        str(artifact_root / name),
                        "--source-experiment",
                        str(artifact_root / f"scar_main_{dataset.lower()}_seed42"),
                        "--scar-python",
                        python_exe,
                        "--seed",
                        "42",
                    ),
                    artifact_dir=artifact_root / name,
                    required_artifacts=(
                        "scores.npy",
                        "labels.npy",
                        "metrics.json",
                        "run_manifest.json",
                        "baseline_protocol.json",
                        "timing.json",
                        "resource_metrics.json",
                    ),
                    dependencies=(anchors[dataset].run_id,),
                )
            )

    for dataset in MAIN_DATASETS:
        base = artifact_root / f"scar_main_{dataset.lower()}_seed42"
        for ratio in PURIFICATION_RATIOS:
            if ratio == 0.02:
                continue
            name = f"scar_e9_{dataset.lower()}_clean_{str(ratio).replace('.', 'p')}"
            config = {
                **FORMAL_DATASET_PROFILES[dataset],
                "clean_ratio": ratio,
                "base_experiment_dir": str(base),
            }
            task = _scar_task(
                artifact_root=artifact_root,
                python_exe=python_exe,
                dataset=dataset,
                group="e9",
                compute_kind="stage_b_test",
                experiment_name=name,
                config=config,
                stage="stage_b_test",
                dependencies=(anchors[dataset].run_id,),
            )
            e9_specs[(dataset, ratio)] = task
            tasks.append(task)

        protocol_name = f"scar_e10_{dataset.lower()}_frozen_protocol"
        protocol_dir = artifact_root / protocol_name
        e10_protocol = _analysis_task(
            artifact_root=artifact_root,
            method="SCAR",
            dataset=dataset,
            group="e10_protocol",
            name=protocol_name,
            config={
                "dataset": dataset,
                "base_experiment_dir": str(base),
                "contamination_ratios": list(CONTAMINATION_RATIOS),
                "n_folds": 3,
                "seed": FORMAL_SEED,
            },
            command=(
                python_exe,
                str(
                    REPO_ROOT
                    / "scripts"
                    / "rebuttal"
                    / "muqn"
                    / "freeze_contamination_protocol.py"
                ),
                "--base-experiment-dir",
                str(base),
                "--output-dir",
                str(protocol_dir),
                "--contamination-ratios",
                *(str(value) for value in CONTAMINATION_RATIOS),
                "--n-folds",
                "3",
                "--seed",
                "42",
            ),
            required=("contamination_fold_manifest.json",),
            dependencies=(anchors[dataset].run_id,),
        )
        e10_protocol_specs[dataset] = e10_protocol
        tasks.append(e10_protocol)

        for fold in range(3):
            for contamination in CONTAMINATION_RATIOS:
                clean_values = [0.0, 0.02]
                if contamination == 0.10:
                    clean_values.append(0.10)
                for clean_ratio in clean_values:
                    name = (
                        f"scar_e10_{dataset.lower()}_contam_"
                        f"{str(contamination).replace('.', 'p')}_clean_"
                        f"{str(clean_ratio).replace('.', 'p')}_fold_{fold}"
                    )
                    config = {
                        "dataset": dataset,
                        "contamination_ratio": contamination,
                        "clean_ratio": clean_ratio,
                        "fold": fold,
                        "seed": FORMAL_SEED,
                    }
                    task = RunSpec(
                            method="SCAR",
                            dataset=dataset,
                            seed=FORMAL_SEED,
                            stage="stage_b_test",
                            group="e10",
                            compute_kind="stage_b_test",
                            config=config,
                            command=(
                                python_exe,
                                str(
                                    REPO_ROOT
                                    / "scripts"
                                    / "rebuttal"
                                    / "muqn"
                                    / "run_contamination_sweep.py"
                                ),
                                "--base_experiment_dir",
                                str(base),
                                "--fold_manifest",
                                str(protocol_dir / "contamination_fold_manifest.json"),
                                "--artifact_root",
                                str(artifact_root),
                                "--target_experiment_dir",
                                str(artifact_root / name),
                                "--contamination_ratios",
                                str(contamination),
                                "--folds",
                                str(fold),
                                "--clean_ratio",
                                str(clean_ratio),
                                "--seed",
                                "42",
                            ),
                            artifact_dir=artifact_root / name,
                            required_artifacts=FULL_AUDIT_REQUIRED
                            + ("contamination_protocol.json",),
                            dependencies=(
                                anchors[dataset].run_id,
                                e10_protocol_specs[dataset].run_id,
                            ),
                        )
                    e10_specs.append(task)
                    tasks.append(task)

        for keep_ratio in CORESET_KEEP_RATIOS:
            if keep_ratio == 1.0:
                continue
            name = f"scar_e38_{dataset.lower()}_keep_{str(keep_ratio).replace('.', 'p')}"
            config = {
                **FORMAL_DATASET_PROFILES[dataset],
                "coreset_keep_ratio": keep_ratio,
                "base_experiment_dir": str(base),
            }
            task = _scar_task(
                    artifact_root=artifact_root,
                    python_exe=python_exe,
                    dataset=dataset,
                    group="e38",
                    compute_kind="stage_b_test",
                    experiment_name=name,
                    config=config,
                    stage="stage_b_test",
                    dependencies=(anchors[dataset].run_id,),
                )
            e38_specs.append(task)
            tasks.append(task)

    for dataset in MAIN_DATASETS:
        anchor = artifact_root / f"scar_main_{dataset.lower()}_seed42"
        mechanism_ids: list[str] = []
        for strategy in ("full", "no_state", "no_context"):
            name = f"scar_e1_e5_{dataset.lower()}_{strategy}"
            strategy_task = _strategy_task(
                artifact_root=artifact_root,
                python_exe=python_exe,
                dataset=dataset,
                group="mechanism",
                source_experiment=anchor,
                strategy=strategy,
                name=name,
                extra_config={"recalibrate_fusion": True},
                dependencies=(anchors[dataset].run_id,),
            )
            tasks.append(strategy_task)
            mechanism_ids.append(strategy_task.run_id)

        timing_name = f"scar_efficiency_{dataset.lower()}_seed42"
        tasks.append(
            _analysis_task(
                artifact_root=artifact_root,
                method="SCAR",
                dataset=dataset,
                group="efficiency",
                name=timing_name,
                config={
                    "source_experiment": str(anchor),
                    "warmup_runs": 1,
                    "timed_runs": 3,
                },
                command=(
                    python_exe,
                    str(
                        REPO_ROOT
                        / "scripts"
                        / "efficiency"
                        / "benchmark_scar_inference.py"
                    ),
                    "--experiment-dir",
                    str(anchor),
                    "--output-dir",
                    str(artifact_root / timing_name),
                ),
                required=("timing.json",),
                dependencies=(anchors[dataset].run_id,),
            )
        )

        evidence_name = f"scar_e1_e5_{dataset.lower()}_evidence"
        strategy_root = f"scar_e1_e5_{dataset.lower()}"
        tasks.append(
            _analysis_task(
                artifact_root=artifact_root,
                method="SCAR",
                dataset=dataset,
                group="mechanism",
                name=evidence_name,
                config={
                    "strategies": ["full", "no_state", "no_context"],
                    "block_size": 128,
                    "n_bootstrap": 2000,
                },
                command=(
                    python_exe,
                    str(
                        REPO_ROOT
                        / "scripts"
                        / "rebuttal"
                        / "muqn"
                        / "compute_retrieval_evidence.py"
                    ),
                    "--log_dir",
                    str(artifact_root / evidence_name),
                    "--log-files",
                    str(artifact_root / f"{strategy_root}_full" / "retrieval_full.npz"),
                    str(
                        artifact_root
                        / f"{strategy_root}_no_state"
                        / "retrieval_no_state.npz"
                    ),
                    str(
                        artifact_root
                        / f"{strategy_root}_no_context"
                        / "retrieval_no_context.npz"
                    ),
                    "--output_dir",
                    str(artifact_root / evidence_name),
                    "--seed",
                    "42",
                ),
                required=("retrieval_evidence.json", "retrieval_evidence.csv"),
                dependencies=tuple(mechanism_ids),
            )
        )

        clean_zero = artifact_root / f"scar_e9_{dataset.lower()}_clean_0p0"
        for fold in range(3):
            for purification, source in (("no", clean_zero), ("default", anchor)):
                name = (
                    f"scar_e10_zero_{dataset.lower()}_{purification}_fold_{fold}"
                )
                tasks.append(
                    _analysis_task(
                        artifact_root=artifact_root,
                        method="SCAR",
                        dataset=dataset,
                        group="e10_zero",
                        name=name,
                        config={
                            "source_experiment": str(source),
                            "fold": fold,
                            "purification": purification,
                            "seed": FORMAL_SEED,
                        },
                        command=(
                            python_exe,
                            str(
                                REPO_ROOT
                                / "scripts"
                                / "rebuttal"
                                / "muqn"
                                / "reevaluate_zero_contamination.py"
                            ),
                            "--experiment-dir",
                            str(source),
                            "--fold",
                            str(fold),
                            "--fold-manifest",
                            str(
                                artifact_root
                                / f"scar_e10_{dataset.lower()}_frozen_protocol"
                                / "contamination_fold_manifest.json"
                            ),
                            "--seed",
                            "42",
                            "--output-dir",
                            str(artifact_root / name),
                        ),
                        required=("heldout_metrics.json", "evaluation_mask.npy"),
                        dependencies=(
                            (
                                e9_specs[(dataset, 0.0)].run_id
                                if purification == "no"
                                else anchors[dataset].run_id
                            ),
                            e10_protocol_specs[dataset].run_id,
                        ),
                    )
                )

    tasks.append(
        _analysis_task(
            artifact_root=artifact_root,
            method="SCAR",
            dataset="ALL",
            group="e11_e12",
            name="scar_e11_e12_purification_summary",
            config={"rare_normal_quantile": 0.90, "source_groups": ["e9", "e10"]},
            command=(
                python_exe,
                str(
                    REPO_ROOT
                    / "scripts"
                    / "rebuttal"
                    / "muqn"
                    / "collect_purification_sweep.py"
                ),
                "--artifact-root",
                str(artifact_root),
                "--output-dir",
                str(artifact_root / "scar_e11_e12_purification_summary"),
            ),
            required=(
                "e11_rare_normal.csv",
                "e12_low_error_survival.csv",
                "purification_summary.json",
            ),
            dependencies=tuple(
                [
                    *(task.run_id for task in e9_specs.values()),
                    *(task.run_id for task in e10_specs),
                    *(task.run_id for task in anchors.values()),
                ]
            ),
        )
    )
    tasks.append(
        _analysis_task(
            artifact_root=artifact_root,
            method="SCAR",
            dataset="ALL",
            group="e38_e39",
            name="scar_e38_e39_coreset_summary",
            config={"oracle_keep_ratio": 1.0, "keep_ratios": list(CORESET_KEEP_RATIOS)},
            command=(
                python_exe,
                str(
                    REPO_ROOT
                    / "scripts"
                    / "rebuttal"
                    / "muqn"
                    / "collect_coreset_sweep.py"
                ),
                "--artifact-root",
                str(artifact_root),
                "--output-dir",
                str(artifact_root / "scar_e38_e39_coreset_summary"),
            ),
            required=("e38_coreset.csv", "e39_resource_tradeoff.json"),
            dependencies=tuple(
                [
                    *(task.run_id for task in e38_specs),
                    *(task.run_id for task in anchors.values()),
                    *(
                        task.run_id
                        for task in tasks
                        if task.group == "mechanism"
                        and task.config.get("strategy") == "full"
                    ),
                ]
            ),
        )
    )
    tasks.append(
        _analysis_task(
            artifact_root=artifact_root,
            method="SCAR",
            dataset="CATCH-EXTENSION",
            group="catch",
            name="scar_catch_dataset_audit",
            config={"asd": 12, "real": 6, "synthetic": 12},
            command=(
                python_exe,
                str(REPO_ROOT / "scripts" / "experiments" / "audit_catch_datasets.py"),
                "--data-root",
                str(REPO_ROOT / "dataset" / "anomaly_detect"),
                "--output-dir",
                str(artifact_root / "scar_catch_dataset_audit"),
            ),
            required=("catch_dataset_audit.json",),
        )
    )

    table_dependencies = tuple(task.run_id for task in tasks)
    tasks.append(
        _analysis_task(
            artifact_root=artifact_root,
            method="SCAR",
            dataset="ALL",
            group="p0_tables",
            name="rebuttal_tables_p0",
            config={
                "output_mode": "tables_and_text_only",
                "strict": True,
                "seed": FORMAL_SEED,
            },
            command=(
                python_exe,
                str(REPO_ROOT / "scripts" / "experiments" / "collect_p0_tables.py"),
                "--artifact-root",
                str(artifact_root),
                "--output-dir",
                str(artifact_root / "rebuttal_tables_p0"),
                "--strict",
            ),
            required=(
                "table_p0_main5.csv",
                "table_p0_retrieval_strategies.csv",
                "table_p0_tep_scores.csv",
                "table_p0_baselines.csv",
                "table_p0_efficiency.csv",
                "table_p0_catch.csv",
                "table_p0_e9.csv",
                "table_p0_e10.csv",
                "table_p0_tsb_series.csv",
                "table_p0_tsb_summary.csv",
                "table_p0_tsb_m_tuning_official_average.csv",
                "table_p0_tsb_m_tuning_dataset_macro_average.csv",
                "table_p0_tsb_m_eval_official_average.csv",
                "table_p0_tsb_m_eval_dataset_macro_average.csv",
                "table_p0_tsb_u_tuning_official_average.csv",
                "table_p0_tsb_u_tuning_dataset_macro_average.csv",
                "table_p0_tsb_u_eval_official_average.csv",
                "table_p0_tsb_u_eval_dataset_macro_average.csv",
                "p0_collection_summary.json",
                "p0_collection_summary.txt",
            ),
            dependencies=table_dependencies,
        )
    )

    return [_with_metric_workers(task) for task in tasks]


def _strategy_task(
    *,
    artifact_root: Path,
    python_exe: str,
    dataset: str,
    group: str,
    source_experiment: Path,
    strategy: str,
    name: str,
    extra_config: dict[str, Any] | None = None,
    dependencies: tuple[str, ...] = (),
) -> RunSpec:
    patch_sizes = tuple((extra_config or {}).get("patch_sizes", (8, 32)))
    config = {
        "source_experiment": str(source_experiment),
        "strategy": strategy,
        "seed": FORMAL_SEED,
        **(extra_config or {}),
    }
    return _analysis_task(
        artifact_root=artifact_root,
        method="SCAR",
        dataset=dataset,
        group=group,
        name=name,
        config=config,
        command=(
            python_exe,
            str(
                REPO_ROOT
                / "scripts"
                / "rebuttal"
                / "muqn"
                / "score_retrieval_strategies.py"
            ),
            "--experiment-dir",
            str(source_experiment),
            "--strategy",
            strategy,
            "--output-dir",
            str(artifact_root / name),
        ),
        required=(
            "scores.npy",
            "scores_raw_max.npy",
            "scores_zscore_mean.npy",
            "scores_cdf_mean.npy",
            "scores_cdf_max.npy",
            "scores_knn_distance.npy",
            "scores_state_novelty.npy",
            *(f"scores_completion_scale{size}.npy" for size in patch_sizes),
            "diagnostics.npz",
            "metrics.json",
            "run_manifest.json",
            f"retrieval_{strategy}.npz",
        ),
        dependencies=dependencies,
    )


def build_p1_tasks(artifact_root: Path, *, python_exe: str) -> list[RunSpec]:
    artifact_root = Path(artifact_root)
    tasks: list[RunSpec] = []

    for dataset in MAIN_DATASETS:
        for seq_len in WINDOW_LENGTHS:
            if seq_len == 128:
                source = artifact_root / f"scar_main_{dataset.lower()}_seed42"
                source_dependencies: tuple[str, ...] = ()
            else:
                name = f"scar_e29_{dataset.lower()}_l{seq_len}_seed42"
                config = {**FORMAL_DATASET_PROFILES[dataset], "seq_len": seq_len}
                fit = _scar_task(
                    artifact_root=artifact_root,
                    python_exe=python_exe,
                    dataset=dataset,
                    group="e29",
                    compute_kind="full_model_fit",
                    experiment_name=name,
                    config=config,
                )
                tasks.append(fit)
                source = fit.artifact_dir
                source_dependencies = (fit.run_id,)
            for strategy in ("global", "context_only", "full"):
                name = f"scar_e29_{dataset.lower()}_l{seq_len}_{strategy}"
                tasks.append(
                    _strategy_task(
                        artifact_root=artifact_root,
                        python_exe=python_exe,
                        dataset=dataset,
                        group="e29",
                        source_experiment=source,
                        strategy=strategy,
                        name=name,
                        extra_config={"seq_len": seq_len},
                        dependencies=source_dependencies,
                    )
                )

    e30_datasets = (*MAIN_DATASETS, *SYNTHETIC_DATASETS)
    for dataset in e30_datasets:
        base_profile = (
            FORMAL_DATASET_PROFILES[dataset]
            if dataset in FORMAL_DATASET_PROFILES
            else _COMMON_PROFILE
        )
        for patch_size in SINGLE_PATCH_SIZES:
            name = f"scar_e30_{dataset.lower()}_p{patch_size}_seed42"
            config = {**base_profile, "patch_sizes": [patch_size]}
            fit = _scar_task(
                artifact_root=artifact_root,
                python_exe=python_exe,
                dataset=dataset,
                group="e30",
                compute_kind="full_model_fit",
                experiment_name=name,
                config=config,
            )
            tasks.append(fit)
            for strategy in ("global", "full"):
                strategy_name = f"{name}_{strategy}"
                tasks.append(
                    _strategy_task(
                        artifact_root=artifact_root,
                        python_exe=python_exe,
                        dataset=dataset,
                        group="e30",
                        source_experiment=fit.artifact_dir,
                        strategy=strategy,
                        name=strategy_name,
                        extra_config={"patch_sizes": [patch_size]},
                        dependencies=(fit.run_id,),
                    )
                )
        anchor = (
            artifact_root / f"scar_main_{dataset.lower()}_seed42"
            if dataset in MAIN_DATASETS
            else artifact_root / f"scar_catch_{dataset.lower()}_seed42"
        )
        tasks.append(
            _strategy_task(
                artifact_root=artifact_root,
                python_exe=python_exe,
                dataset=dataset,
                group="e30",
                source_experiment=anchor,
                strategy="full",
                name=f"scar_e30_{dataset.lower()}_multiscale_full",
                extra_config={"patch_sizes": [8, 32], "synthetic_type_summary": True},
            )
        )

    for dataset in MAIN_DATASETS:
        target = artifact_root / f"scar_main_{dataset.lower()}_seed42"
        selection_name = f"scar_e31_{dataset.lower()}_global_l512_base"
        base_task = RunSpec(
                method="SCAR",
                dataset=dataset,
                seed=FORMAL_SEED,
                stage="full",
                group="e31",
                compute_kind="full_model_fit",
                config={
                    "target_experiment": str(target),
                    "seq_len": 512,
                    "d_z_candidates": list(BUDGET_DIMS),
                    "tolerance": 0.15,
                },
                command=(
                    python_exe,
                    str(REPO_ROOT / "scripts" / "experiments" / "match_budget.py"),
                    "train",
                    "--target-experiment",
                    str(target),
                    "--dataset",
                    dataset,
                    "--output-dir",
                    str(artifact_root / selection_name),
                ),
                artifact_dir=artifact_root / selection_name,
                required_artifacts=FULL_AUDIT_REQUIRED + ("parameter_selection.json",),
            )
        tasks.append(base_task)
        q_dependencies: list[str] = []
        for keep_ratio in CORESET_KEEP_RATIOS:
            if keep_ratio == 1.0:
                continue
            name = f"scar_e31_{dataset.lower()}_global_l512_q{str(keep_ratio).replace('.', 'p')}"
            config = {
                **FORMAL_DATASET_PROFILES[dataset],
                "coreset_keep_ratio": keep_ratio,
                "base_experiment_dir": str(artifact_root / selection_name),
            }
            q_task = _scar_task(
                    artifact_root=artifact_root,
                    python_exe=python_exe,
                    dataset=dataset,
                    group="e31",
                    compute_kind="stage_b_test",
                    experiment_name=name,
                    config=config,
                    stage="stage_b_test",
                    dependencies=(base_task.run_id,),
                )
            tasks.append(q_task)
            q_dependencies.append(q_task.run_id)
        timing_name = f"scar_e31_{dataset.lower()}_timing_selection"
        tasks.append(
            _analysis_task(
                artifact_root=artifact_root,
                method="SCAR",
                dataset=dataset,
                group="e31",
                name=timing_name,
                config={
                    "target_experiment": str(target),
                    "top_K_candidates": list(BUDGET_TOP_K),
                    "tolerance": 0.15,
                },
                command=(
                    python_exe,
                    str(REPO_ROOT / "scripts" / "experiments" / "match_budget.py"),
                    "finalize",
                    "--target-experiment",
                    str(target),
                    "--dataset",
                    dataset,
                    "--output-dir",
                    str(artifact_root / timing_name),
                ),
                required=(
                    "budget_match.json",
                    "budget_match.csv",
                    "scores.npy",
                    "scores_raw_max.npy",
                    "scores_zscore_mean.npy",
                    "scores_cdf_mean.npy",
                    "scores_cdf_max.npy",
                    "scores_knn_distance.npy",
                    "scores_state_novelty.npy",
                    "scores_completion_scale8.npy",
                    "scores_completion_scale32.npy",
                    "diagnostics.npz",
                    "metrics.json",
                    "run_manifest.json",
                ),
                dependencies=(base_task.run_id, *q_dependencies),
            )
        )

    tasks.append(
        _analysis_task(
            artifact_root=artifact_root,
            method="SCAR",
            dataset="ALL",
            group="p1_collect",
            name="scar_p1_tables",
            config={
                "experiments": ["E29", "E30", "E31"],
                "output_mode": "table_only",
            },
            command=(
                python_exe,
                str(REPO_ROOT / "scripts" / "experiments" / "collect_p1_tables.py"),
                "--artifact-root",
                str(artifact_root),
                "--output-dir",
                str(artifact_root / "scar_p1_tables"),
            ),
            required=(
                "e29_window_strategy.csv",
                "e30_patch_strategy.csv",
                "e30_synthetic_macro.csv",
                "e31_budget_match.csv",
                "p1_tables.json",
            ),
            dependencies=tuple(task.run_id for task in tasks),
        )
    )

    return [_with_metric_workers(task) for task in tasks]
