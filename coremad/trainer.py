from __future__ import annotations

import hashlib
import json
import random
import shutil
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    HAS_SKLEARN_METRICS = True
except Exception:
    average_precision_score = None
    precision_recall_curve = None
    roc_auc_score = None
    HAS_SKLEARN_METRICS = False

try:
    from TSB_AD.evaluation.metrics import get_metrics as vus_get_metrics

    HAS_VUS_METRICS = True
except Exception:
    try:
        from vus.metrics import get_metrics as vus_get_metrics

        HAS_VUS_METRICS = True
    except Exception:
        vus_get_metrics = None
        HAS_VUS_METRICS = False

from .config import CoReMADConfig
from .data import (
    DataBundle,
    RawDatasetBundle,
    TimeSeriesNormalizer,
    build_data_bundle,
    build_loader,
    load_raw_dataset_bundle,
    resolve_tep_files,
    transform_raw_bundle,
)
from .faiss_index import HAS_FAISS
from .memory import MemoryBank
from .model import CoReMADModel
from .resource_monitor import ResourceMonitor
from .scorer import CDFPITFusion, ZScoreMeanFusion, fuse_raw_max
from .visualization import plot_score_distribution, plot_score_timeline, plot_training_curves


BASE_TEST_DIAGNOSTIC_SPECS = [
    ("knn_distance", "kNN Distance", "Distance"),
    ("soft_support_score", "Soft Candidate Support", "Score"),
    ("state_novelty", "State Novelty ($a_w$)", "Novelty"),
    ("raw_max_score", "Raw Max Fusion", "Score"),
    ("zscore_mean_score", "Z-Score Mean Fusion", "Score"),
    ("cdf_max_score", "CDF-PIT Max Fusion", "Score"),
    ("cdf_mean_score", "CDF-PIT Mean Fusion", "Score"),
    ("cdf_mean_soft_support_score", "CDF-PIT Mean Fusion (Soft Support Swap)", "Score"),
    ("cdf_softmax_score", "CDF-PIT Softmax Fusion", "Score"),
    ("final_fused_score", "Final Fused Score", "Score"),
]

STAGE_A_COMPAT_FIELDS = (
    "dataset",
    "data_root",
    "tep_protocol",
    "seq_len",
    "train_stride",
    "val_ratio",
    "val_gap",
    "val_min_train_windows",
    "val_split_mode",
    "batch_size",
    "max_train_windows",
    "stsd_hidden",
    "stsd_lowpass_center",
    "d_state",
    "state_hidden",
    "patch_sizes",
    "d_trunk",
    "d_z",
    "context_k",
    "use_stsd_decomposition",
    "use_channel_modulation",
    "mask_ratio",
    "n_mask_groups",
    "completion_n_heads",
    "completion_n_layers",
    "completion_dropout",
    "use_completion_head",
    "stage_a_epochs",
    "early_stop_patience",
    "lr",
    "weight_decay",
    "grad_clip_norm",
    "scheduler_eta_min_ratio",
    "lambda_pred",
    "lambda_smooth",
    "seed",
)

STAGE_B_COMPAT_FIELDS = (
    "dataset",
    "data_root",
    "tep_protocol",
    "seq_len",
    "memory_build_stride",
    "max_train_windows",
    "patch_sizes",
    "d_z",
    "top_M",
    "top_K",
    "knn_k",
    "clean_ratio",
    "state_prototype_count",
    "prototype_top_p",
    "prototype_candidate_cap",
    "soft_candidate_tau",
    "support_score_tau",
    "support_score_eps",
    "include_soft_support_in_fusion",
    "use_prototype_support",
    "use_two_level_retrieval",
    "use_context_key_retrieval",
    "use_completion_head",
    "use_completion_self_cleaning",
    "use_completion_score_fusion",
    "use_faiss",
    "faiss_use_gpu",
    "faiss_exact_threshold",
    "faiss_ivf_nprobe",
    "coreset_keep_ratio",
    "coreset_max_patches_per_scale",
    "coreset_fps_threshold",
    "seed",
    "memory_seed",
    "memory_audit_mode",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CoReMADTrainer:
    def __init__(self, config: CoReMADConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.config.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._resource_invocation_id = uuid.uuid4().hex
        self._active_resource_monitor: Optional[ResourceMonitor] = None
        self._last_diagnostic_workload: dict[str, int] = {}
        set_seed(config.seed)

    def _run_monitored_stage(self, stage: str, operation: Callable[[], Any]) -> Any:
        if not self.config.resource_monitor_enabled:
            return operation()
        monitor = ResourceMonitor(
            output_path=self.config.resource_metrics_path,
            method="SCAR",
            dataset=self.config.dataset,
            seed=self.config.seed,
            stage=stage,
            device=str(self.device),
            invocation_id=self._resource_invocation_id,
            sample_interval=self.config.resource_sample_interval,
        )
        previous_monitor = self._active_resource_monitor
        try:
            with monitor:
                self._active_resource_monitor = monitor
                result = operation()
                if isinstance(result, bool) and not result:
                    monitor.mark_skipped("stage artifacts are already complete")
                self._record_resource_artifacts(stage, monitor)
                return result
        finally:
            self._active_resource_monitor = previous_monitor

    def _resource_span(self, name: str):
        if self._active_resource_monitor is None:
            return nullcontext(None)
        return self._active_resource_monitor.span(name)

    def _record_model_resource_stats(self, model: CoReMADModel) -> None:
        if self._active_resource_monitor is None:
            return
        total = sum(int(parameter.numel()) for parameter in model.parameters())
        trainable = sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self._active_resource_monitor.set_model_stats(
            total_parameters=total,
            trainable_parameters=trainable,
        )

    def _record_resource_artifacts(
        self,
        stage: str,
        monitor: ResourceMonitor,
    ) -> None:
        artifact_paths: dict[str, Path] = {"config_bytes": self.config.config_path}
        if stage == "stage_a":
            artifact_paths.update(
                {
                    "best_checkpoint_bytes": self.config.stage_a_path,
                    "last_checkpoint_bytes": self.config.stage_a_last_path,
                }
            )
        elif stage == "stage_b":
            artifact_paths.update(
                {
                    "memory_bytes": self.config.memory_path,
                    "faiss_index_bytes": self.config.faiss_index_path,
                    "memory_meta_bytes": self.config.memory_meta_path,
                    "cdf_npz_bytes": self.config.cdf_fusion_path.with_suffix(".npz"),
                    "cdf_json_bytes": self.config.cdf_fusion_path.with_suffix(".json"),
                    "zscore_npz_bytes": self.config.zscore_fusion_path.with_suffix(".npz"),
                    "zscore_json_bytes": self.config.zscore_fusion_path.with_suffix(".json"),
                }
            )
        elif stage == "test":
            artifact_paths.update(
                {
                    "metrics_bytes": self.config.experiment_dir / "test_metrics.json",
                    "scores_bytes": (
                        self.config.experiment_dir
                        / f"test_scores_{self.config.evaluation_score_key}.npy"
                    ),
                }
            )
        monitor.set_artifact_bytes(**artifact_paths)

    def _completion_score_names(self) -> list[str]:
        return self.config.completion_score_names()

    def _raw_diagnostic_keys(self) -> list[str]:
        keys = [*self._completion_score_names(), "knn_distance", "state_novelty"]
        if self.config.use_prototype_support:
            keys.insert(len(self._completion_score_names()) + 1, "soft_support_score")
        return keys

    def _fusion_score_names(self) -> list[str]:
        return self.config.fusion_score_names()

    def _legacy_cdf_mean_score_names(self) -> list[str]:
        return [*self._completion_score_names(), "knn_distance", "state_novelty"]

    def _soft_support_cdf_mean_score_names(self) -> list[str]:
        if not self.config.use_prototype_support:
            return self._legacy_cdf_mean_score_names()
        return [*self._completion_score_names(), "soft_support_score", "state_novelty"]

    def _cdf_fit_score_names(self) -> list[str]:
        return self._raw_diagnostic_keys()

    def _test_diagnostic_specs(self) -> list[tuple[str, str, str]]:
        completion_specs = [
            (name, f"Completion Score (Scale {patch_size})", "Completion Score")
            for name, patch_size in zip(self._completion_score_names(), self.config.patch_sizes)
        ]
        base_specs = list(BASE_TEST_DIAGNOSTIC_SPECS)
        if not self.config.use_prototype_support:
            base_specs = [
                spec
                for spec in base_specs
                if spec[0] not in {"soft_support_score", "cdf_mean_soft_support_score"}
            ]
        return [*completion_specs, *base_specs]

    def _stage_a_monitor_key(self) -> str:
        if self.config.use_completion_head:
            return "mask_loss"
        if self.config.lambda_pred > 0.0:
            return "pred_loss"
        return "loss"

    def _load_raw_bundle(self) -> RawDatasetBundle:
        return load_raw_dataset_bundle(self.config.dataset, self.config.data_root, config=self.config)

    def prepare_stage_a_data(self) -> DataBundle:
        bundle = build_data_bundle(self.config)
        self.config.n_channels = int(bundle.train_full.shape[1])
        self.config.save()
        return bundle

    def transform_bundle_with_normalizer(
        self,
        raw_bundle: RawDatasetBundle,
        normalizer: TimeSeriesNormalizer,
    ) -> DataBundle:
        bundle = transform_raw_bundle(raw_bundle, normalizer, self.config)
        self.config.n_channels = int(bundle.train_full.shape[1])
        self.config.save()
        return bundle

    def save_stage_a_checkpoint(
        self,
        payload: dict,
        path: Path,
    ) -> None:
        torch.save(payload, path)

    def _build_stage_a_payload(
        self,
        model: CoReMADModel,
        normalizer: TimeSeriesNormalizer,
        optimizer: AdamW,
        scheduler: CosineAnnealingLR,
        epoch: int,
        best_loss: float,
        history: list[dict],
        epochs_without_improvement: int,
        completed: bool,
        monitor_key: str,
    ) -> dict:
        return {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "normalizer": normalizer.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "history": history,
            "epochs_without_improvement": int(epochs_without_improvement),
            "completed": bool(completed),
            "monitor_key": monitor_key,
            "config": self.config.to_dict(),
        }

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _move_optimizer_state(optimizer: AdamW, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _load_model_state(self, model: CoReMADModel, state_dict: dict) -> None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing: set[str] = set()
        allowed_unexpected = {"patch_encoder.bypass_alpha"}
        actual_missing = set(missing) - allowed_missing
        actual_unexpected = set(unexpected) - allowed_unexpected
        if actual_missing or actual_unexpected:
            raise RuntimeError(
                "Stage-A checkpoint is incompatible with the current model config. "
                f"missing={sorted(actual_missing)} unexpected={sorted(actual_unexpected)}"
            )

    @staticmethod
    def _print_stage_banner(title: str) -> None:
        print("=" * 60)
        print(title)
        print("=" * 60)

    @staticmethod
    def _require_file(path: Path, description: str, hint: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}. {hint}")

    @staticmethod
    def _normalize_config_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    def _config_snapshot(self, field_names: tuple[str, ...]) -> dict[str, Any]:
        return {
            name: self._normalize_config_value(getattr(self.config, name))
            for name in field_names
        }

    def _config_mismatches(
        self,
        stored_config: Optional[dict[str, Any]],
        field_names: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        current = self._config_snapshot(field_names)
        if not isinstance(stored_config, dict):
            return {
                name: {"current": value, "stored": "<missing>"}
                for name, value in current.items()
            }

        mismatches: dict[str, dict[str, Any]] = {}
        for name, current_value in current.items():
            stored_raw = stored_config.get(name, "<missing>")
            if name == "tep_protocol" and stored_raw == "<missing>":
                # Checkpoints predating the protocol field were created from the
                # historical selected subset.
                stored_raw = "selected"
            stored_value = self._normalize_config_value(stored_raw)
            if stored_value != current_value:
                mismatches[name] = {"current": current_value, "stored": stored_value}
        return mismatches

    @staticmethod
    def _format_mismatches(mismatches: dict[str, dict[str, Any]], max_items: int = 5) -> str:
        items = list(mismatches.items())
        preview = []
        for name, payload in items[:max_items]:
            preview.append(
                f"{name}: current={payload['current']} stored={payload['stored']}"
            )
        if len(items) > max_items:
            preview.append(f"... +{len(items) - max_items} more")
        return "; ".join(preview)

    @staticmethod
    def _file_signature(path: Path) -> Optional[dict[str, int]]:
        if not path.exists():
            return None
        stat = path.stat()
        return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

    def _tep_data_file_signatures(self) -> Optional[dict[str, dict[str, Any]]]:
        if str(self.config.dataset).upper() != "TEP":
            return None
        paths, _, _ = resolve_tep_files(
            self.config.data_root,
            protocol=self.config.tep_protocol,
        )
        return {
            name: {
                "path": str(path.resolve()),
                **(self._file_signature(path) or {}),
            }
            for name, path in sorted(paths.items())
        }

    def _stage_a_payload_compatible(
        self,
        payload: Optional[dict[str, Any]],
        reason: str,
        verbose: bool = True,
    ) -> bool:
        mismatches = self._config_mismatches(
            None if payload is None else payload.get("config"),
            STAGE_A_COMPAT_FIELDS,
        )
        if not mismatches:
            return True
        if verbose:
            print(
                f"[ResumeGuard] ignore existing Stage-A artifacts for {reason}: "
                f"{self._format_mismatches(mismatches)}"
            )
        return False

    def _load_stage_b_meta(self) -> Optional[dict[str, Any]]:
        if not self.config.memory_meta_path.exists():
            return None
        return json.loads(self.config.memory_meta_path.read_text(encoding="utf-8"))

    def _stage_b_meta_compatible(
        self,
        meta: Optional[dict[str, Any]],
        reason: str,
        verbose: bool = True,
    ) -> bool:
        if not isinstance(meta, dict):
            if verbose:
                print(f"[ResumeGuard] ignore existing Stage-B artifacts for {reason}: missing metadata.")
            return False

        mismatches = self._config_mismatches(
            meta.get("config"),
            STAGE_B_COMPAT_FIELDS,
        )
        stored_stage_a_sig = meta.get("stage_a_artifact_signature")
        current_stage_a_sig = self._file_signature(self.config.stage_a_path)
        if stored_stage_a_sig != current_stage_a_sig:
            mismatches["stage_a_artifact_signature"] = {
                "current": current_stage_a_sig,
                "stored": stored_stage_a_sig,
            }
        if not mismatches:
            return True
        if verbose:
            print(
                f"[ResumeGuard] ignore existing Stage-B artifacts for {reason}: "
                f"{self._format_mismatches(mismatches)}"
            )
        return False

    def _compute_window_subset_stats(
        self,
        data: np.ndarray,
        start_indices: Optional[np.ndarray],
    ) -> tuple[float, float]:
        if start_indices is None:
            return float(data.mean()), float(data.std())

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0
        for start in np.asarray(start_indices, dtype=np.int64):
            window = data[start : start + self.config.seq_len]
            total_sum += float(window.sum(dtype=np.float64))
            total_sq_sum += float(np.square(window, dtype=np.float64).sum())
            total_count += int(window.size)
        if total_count == 0:
            return 0.0, 0.0
        mean = total_sum / total_count
        var = max(total_sq_sum / total_count - mean * mean, 0.0)
        return float(mean), float(np.sqrt(var))

    def _is_stage_a_complete(self) -> bool:
        if not self.config.stage_a_path.exists():
            return False
        stage_a_payload = self._load_checkpoint(self.config.stage_a_path)
        if not self._stage_a_payload_compatible(stage_a_payload, "skip Stage A"):
            return False
        if not self.config.stage_a_last_path.exists():
            return True
        payload = self._load_checkpoint(self.config.stage_a_last_path)
        if not self._stage_a_payload_compatible(payload, "resume Stage A"):
            return False
        if bool(payload.get("completed", False)):
            return True
        epoch = int(payload.get("epoch", 0))
        if epoch >= self.config.stage_a_epochs:
            return True
        if int(payload.get("epochs_without_improvement", 0)) >= self.config.early_stop_patience:
            return True
        return False

    def _is_stage_b_complete(self) -> bool:
        if not self.config.memory_path.exists():
            return False
        if not self.config.cdf_fusion_path.with_suffix(".npz").exists():
            return False
        if not self.config.cdf_fusion_path.with_suffix(".json").exists():
            return False
        if not self.config.zscore_fusion_path.with_suffix(".json").exists():
            return False
        meta = self._load_stage_b_meta()
        if not self._stage_b_meta_compatible(meta, "skip Stage B"):
            return False
        if self.config.use_faiss and HAS_FAISS:
            return self.config.faiss_index_path.exists()
        return True

    def _require_stage_a_artifacts(self, consumer_stage: str) -> None:
        self._require_file(
            self.config.stage_a_path,
            "Stage A checkpoint",
            f"Run Stage A before {consumer_stage}.",
        )

    def _require_memory_artifacts(self, consumer_stage: str) -> None:
        self._require_file(
            self.config.memory_path,
            "memory bank artifact",
            f"Run Stage B before {consumer_stage}.",
        )
        meta = self._load_stage_b_meta()
        if not self._stage_b_meta_compatible(meta, f"use Stage B artifacts for {consumer_stage}"):
            raise RuntimeError(
                f"Stage B artifacts are incompatible with the current configuration for {consumer_stage}. "
                "Rebuild Stage B with the requested dataset protocol."
            )
        if self.config.use_faiss and HAS_FAISS:
            self._require_file(
                self.config.faiss_index_path,
                "Faiss state index artifact",
                f"Stage B may be incomplete. Re-run Stage B before {consumer_stage}.",
            )

    def run_stage_a(self) -> bool:
        return bool(self._run_monitored_stage("stage_a", self._run_stage_a_impl))

    def _run_stage_a_impl(self) -> bool:
        self._print_stage_banner("Stage A: Self-supervised representation learning")
        bundle = self.prepare_stage_a_data()
        train_mean, train_std = self._compute_window_subset_stats(
            bundle.train_stage_a,
            bundle.train_stage_a_start_indices,
        )
        print(f"[Diag] train mean={train_mean:.4f}, std={train_std:.4f}")
        if bundle.val_stage_a is not None:
            val_mean, val_std = self._compute_window_subset_stats(
                bundle.val_stage_a,
                bundle.val_stage_a_start_indices,
            )
            print(f"[Diag] val   mean={val_mean:.4f}, std={val_std:.4f}")
        model = CoReMADModel(self.config).to(self.device)
        self._record_model_resource_stats(model)
        optimizer = AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.config.stage_a_epochs),
            eta_min=self.config.lr * self.config.scheduler_eta_min_ratio,
        )

        start_epoch = 1
        best_val = float("inf")
        epochs_without_improvement = 0
        history: list[dict] = []
        trained = False
        monitor_key = self._stage_a_monitor_key()

        if self.config.resume and self._is_stage_a_complete():
            print("[Stage A] completed checkpoint detected, skip training.")
            return False

        if self.config.resume and self.config.stage_a_last_path.exists():
            payload = self._load_checkpoint(self.config.stage_a_last_path)
            if self._stage_a_payload_compatible(payload, "resume Stage A"):
                self._load_model_state(model, payload["model_state"])
                optimizer.load_state_dict(payload["optimizer_state"])
                scheduler.load_state_dict(payload["scheduler_state"])
                self._move_optimizer_state(optimizer, self.device)
                start_epoch = int(payload["epoch"]) + 1
                best_val = float(payload.get("best_loss", float("inf")))
                history = list(payload.get("history", []))
                epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
                monitor_key = str(payload.get("monitor_key", monitor_key))
                print(
                    f"[Stage A] resume from epoch={start_epoch:03d} "
                    f"(best_{monitor_key}={best_val:.6f}, patience_count={epochs_without_improvement})"
                )
                if start_epoch > self.config.stage_a_epochs:
                    print("[Stage A] already finished, skip training.")
                    return False

        train_loader = build_loader(
            bundle.train_stage_a,
            None,
            seq_len=self.config.seq_len,
            stride=self.config.train_stride,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            max_windows=self.config.max_train_windows,
            drop_last=True,
            start_indices=bundle.train_stage_a_start_indices,
            segment_ranges=bundle.train_stage_a_segment_ranges,
        )
        val_loader = None
        if bundle.val_stage_a is not None:
            val_loader = build_loader(
                bundle.val_stage_a,
                None,
                seq_len=self.config.seq_len,
                stride=self.config.train_stride,
                batch_size=self.config.val_batch_size,
                num_workers=self.config.num_workers,
                shuffle=False,
                drop_last=False,
                start_indices=bundle.val_stage_a_start_indices,
                segment_ranges=bundle.val_stage_a_segment_ranges,
            )

        best_payload: Optional[dict] = None
        if self.config.stage_a_path.exists():
            existing_best_payload = self._load_checkpoint(self.config.stage_a_path)
            if self._stage_a_payload_compatible(existing_best_payload, "reuse best Stage A checkpoint"):
                best_payload = existing_best_payload
                best_val = min(best_val, float(best_payload.get("best_loss", best_val)))
        for epoch in range(start_epoch, self.config.stage_a_epochs + 1):
            trained = True
            model.train()
            train_metrics = self._run_stage_a_epoch(model, train_loader, optimizer)
            val_metrics = self._evaluate_stage_a(model, val_loader) if val_loader is not None else train_metrics
            scheduler.step()
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            current_metric = val_metrics[monitor_key]

            if current_metric < best_val:
                best_val = current_metric
                epochs_without_improvement = 0
                best_payload = self._build_stage_a_payload(
                    model=model,
                    normalizer=bundle.normalizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_loss=best_val,
                    history=history,
                    epochs_without_improvement=epochs_without_improvement,
                    completed=False,
                    monitor_key=monitor_key,
                )
                self.save_stage_a_checkpoint(best_payload, self.config.stage_a_path)
            else:
                epochs_without_improvement += 1
            last_payload = self._build_stage_a_payload(
                model=model,
                normalizer=bundle.normalizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_loss=best_val,
                history=history,
                epochs_without_improvement=epochs_without_improvement,
                completed=False,
                monitor_key=monitor_key,
            )
            self.save_stage_a_checkpoint(last_payload, self.config.stage_a_last_path)
            print(
                f"[Stage A] epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_{monitor_key}={val_metrics[monitor_key]:.6f}"
            )
            if val_loader is not None and epochs_without_improvement >= self.config.early_stop_patience:
                print(
                    f"[Stage A] early stop triggered at epoch={epoch:03d} "
                    f"(patience={self.config.early_stop_patience}, best_{monitor_key}={best_val:.6f})"
                )
                break

        if best_payload is None:
            final_payload = self._build_stage_a_payload(
                model=model,
                normalizer=bundle.normalizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=max(0, start_epoch - 1),
                best_loss=best_val,
                history=history,
                epochs_without_improvement=epochs_without_improvement,
                completed=True,
                monitor_key=monitor_key,
            )
            self.save_stage_a_checkpoint(final_payload, self.config.stage_a_path)
        else:
            self.save_stage_a_checkpoint(best_payload, self.config.stage_a_path)
        final_last_payload = self._build_stage_a_payload(
            model=model,
            normalizer=bundle.normalizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=history[-1]["epoch"] if history else max(0, start_epoch - 1),
            best_loss=best_val,
            history=history,
            epochs_without_improvement=epochs_without_improvement,
            completed=True,
            monitor_key=monitor_key,
        )
        self.save_stage_a_checkpoint(final_last_payload, self.config.stage_a_last_path)
        plot_training_curves(history, self.config.experiment_dir / "stage_a_loss_curve.png")
        return trained

    def _run_stage_a_epoch(self, model: CoReMADModel, loader: DataLoader, optimizer: AdamW) -> dict[str, float]:
        totals = {"loss": 0.0, "mask_loss": 0.0, "pred_loss": 0.0, "smooth_loss": 0.0}
        count = 0
        batch_count = 0
        with self._resource_span("train_loop") as resource_span:
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True)
                losses = model.compute_pretraining_losses(x)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.grad_clip_norm)
                optimizer.step()
                batch_size = x.size(0)
                count += batch_size
                batch_count += 1
                for key in totals:
                    totals[key] += float(losses[key].item()) * batch_size
            workload = {
                "points": count * self.config.seq_len,
                "windows": count,
                "batches": batch_count,
                "epochs": 1,
            }
            if resource_span is not None:
                resource_span.set_workload(**workload)
            if self._active_resource_monitor is not None:
                self._active_resource_monitor.add_workload(**workload)
        return {key: value / max(1, count) for key, value in totals.items()}

    def _evaluate_stage_a(self, model: CoReMADModel, loader: Optional[DataLoader]) -> dict[str, float]:
        if loader is None:
            return {"loss": 0.0, "mask_loss": 0.0, "pred_loss": 0.0, "smooth_loss": 0.0}
        model.eval()
        totals = {"loss": 0.0, "mask_loss": 0.0, "pred_loss": 0.0, "smooth_loss": 0.0}
        count = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True)
                losses = model.compute_pretraining_losses(x)
                batch_size = x.size(0)
                count += batch_size
                for key in totals:
                    totals[key] += float(losses[key].item()) * batch_size
        return {key: value / max(1, count) for key, value in totals.items()}

    def run_stage_b(self, memory_loader: Optional[DataLoader] = None) -> bool:
        return bool(
            self._run_monitored_stage(
                "stage_b",
                lambda: self._run_stage_b_impl(memory_loader=memory_loader),
            )
        )

    def _run_stage_b_impl(self, memory_loader: Optional[DataLoader] = None) -> bool:
        self._print_stage_banner("Stage B: Building memory bank + fitting fusion calibrators")
        if self._is_stage_b_complete():
            print("[Stage B] completed artifacts detected, skip rebuilding memory bank.")
            return False
        self._require_stage_a_artifacts("Stage B")
        model, normalizer, _ = self.load_stage_a_model()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self._record_model_resource_stats(model)
        assert not model.training, "Model must be in eval mode for Stage B."
        raw_bundle = self._load_raw_bundle()
        bundle = self.transform_bundle_with_normalizer(raw_bundle, normalizer)
        loader = memory_loader
        if loader is None:
            loader = build_loader(
                bundle.train_full,
                None,
                seq_len=self.config.seq_len,
                stride=self.config.memory_build_stride,
                batch_size=self.config.memory_batch_size,
                num_workers=self.config.num_workers,
                shuffle=False,
                max_windows=self.config.max_train_windows,
                drop_last=False,
                segment_ranges=bundle.train_full_segment_ranges,
            )
            loader_source = "normalized training split"
        else:
            loader_source = "caller-provided memory dataset"
        n_windows = len(loader.dataset)
        n_batches = len(loader)
        stage_b_workload = {
            "points": len(bundle.train_full),
            "windows": n_windows,
            "batches": n_batches,
        }
        if self._active_resource_monitor is not None:
            self._active_resource_monitor.set_workload(**stage_b_workload)
        print(
            f"[Stage B] loader ready: source={loader_source}, windows={n_windows}, batches={n_batches}, "
            f"batch_size={self.config.memory_batch_size}, stride={self.config.memory_build_stride}, "
            f"num_workers={self.config.num_workers}, device={self.device}"
        )
        for patch_size in self.config.patch_sizes:
            n_patches = self.config.seq_len // patch_size
            print(
                f"[Stage B] scale patch_size={patch_size}: "
                f"raw_patch_count~={n_windows * n_patches} ({n_patches} patches/window)"
            )
        with self._resource_span("memory_build") as resource_span:
            memory = MemoryBank.build(model, loader, self.config, self.device)
            memory.sanity_check()
            memory.save(self.config.memory_path)
            if resource_span is not None:
                resource_span.set_workload(**stage_b_workload)
        with self._resource_span("fusion_fit") as resource_span:
            train_diags, train_scores_count = self._aggregate_point_diagnostics(
                data=bundle.train_full,
                labels=None,
                model=model,
                memory=memory,
                batch_size=self.config.memory_batch_size,
                stride=self.config.memory_build_stride,
                max_windows=self.config.max_train_windows,
                print_stsd_stats=False,
                segment_ranges=bundle.train_full_segment_ranges,
            )
            cdf_fusion = self._fit_and_save_cdf_fusion_from_diagnostics(train_diags, train_scores_count)
            zscore_fusion = self._fit_and_save_zscore_fusion_from_diagnostics(train_diags, train_scores_count)
            if resource_span is not None:
                resource_span.set_workload(**self._last_diagnostic_workload)
        self.config.memory_meta_path.write_text(
            json.dumps(
                {
                    "num_windows": int(memory.state_bank.size(0)),
                    "num_scales": len(memory.scales),
                    "scale_sizes": [int(scale.z.size(0)) for scale in memory.scales],
                    "memory_build_stats": memory.build_stats,
                    "memory_seed": int(self.config.effective_memory_seed),
                    "memory_audit_mode": self.config.memory_audit_mode,
                    "coreset_cap_active": any(
                        int(stats.get("final_count", 0))
                        < int(stats.get("clean_count", 0))
                        for stats in memory.build_stats
                    ),
                    "num_prototypes": int(memory.prototype_centers.size(0)),
                    "prototype_sizes": [int(member.numel()) for member in memory.prototype_members],
                    "cdf_fusion_path": str(self.config.cdf_fusion_path),
                    "cdf_fusion_scores": list(cdf_fusion.cdfs.keys()),
                    "zscore_fusion_path": str(self.config.zscore_fusion_path),
                    "zscore_fusion_scores": list(zscore_fusion.stats.keys()),
                    "config": self._config_snapshot(STAGE_B_COMPAT_FIELDS),
                    "config_fields": list(STAGE_B_COMPAT_FIELDS),
                    "stage_a_artifact_signature": self._file_signature(self.config.stage_a_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"[Stage B] saved memory bank: path={self.config.memory_path}, "
            f"cdf_fusion_path={self.config.cdf_fusion_path}, "
            f"zscore_fusion_path={self.config.zscore_fusion_path}, "
            f"faiss_index_path={self.config.faiss_index_path}, meta_path={self.config.memory_meta_path}"
        )
        print(
            f"[Stage B] summary: num_windows={int(memory.state_bank.size(0))}, "
            f"num_prototypes={int(memory.prototype_centers.size(0))}, "
            f"scale_sizes={[int(scale.z.size(0)) for scale in memory.scales]}"
        )
        for scale_idx, scale_memory in enumerate(memory.scales):
            patch_size = self.config.patch_sizes[scale_idx]
            print(
                f"[Stage B] scale patch_size={patch_size}: "
                f"{int(scale_memory.z.size(0))} vectors in scale memory"
            )
        if self.config.use_faiss and HAS_FAISS and memory.state_index.index is not None:
            print(f"[Stage B] state_index.ntotal={int(memory.state_index.index.ntotal)}")
        return True

    def _prepare_test_fusions(
        self,
        raw_bundle: RawDatasetBundle,
        normalizer: TimeSeriesNormalizer,
        model: CoReMADModel,
        memory: MemoryBank,
    ) -> tuple[CDFPITFusion, ZScoreMeanFusion]:
        cdf_fusion = self.load_cdf_fusion(optional=False)
        required_cdf_names = set(self._cdf_fit_score_names())
        if not required_cdf_names.issubset(set(cdf_fusion.cdfs.keys())):
            print(
                "[Test] missing CDF fusion sub-scores for the soft-support swap metric; "
                "rebuilding CDF fusion from normalized training data."
            )
            train_norm = normalizer.transform(raw_bundle.train)
            cdf_fusion = self._fit_and_save_cdf_fusion(
                train_norm,
                model,
                memory,
                segment_ranges=raw_bundle.train_segment_ranges,
            )
        zscore_fusion = self.load_zscore_fusion(optional=True)
        if zscore_fusion is None:
            print(
                "[Test] missing z-score fusion artifact; "
                "building a compatibility fallback from normalized training data."
            )
            train_norm = normalizer.transform(raw_bundle.train)
            zscore_fusion = self._fit_and_save_zscore_fusion(
                train_norm,
                model,
                memory,
                segment_ranges=raw_bundle.train_segment_ranges,
            )
        return cdf_fusion, zscore_fusion

    def _build_metric_sources_from_point_diags(
        self,
        final_diags: dict[str, np.ndarray],
        cdf_fusion: CDFPITFusion,
        zscore_fusion: ZScoreMeanFusion,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        enriched_diags = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in final_diags.items()
        }
        raw_max_scores = np.asarray(fuse_raw_max(enriched_diags, self._fusion_score_names()), dtype=np.float64)
        zscore_mean_scores = np.asarray(zscore_fusion.fuse(enriched_diags), dtype=np.float64)
        cdf_max_scores = np.asarray(
            cdf_fusion.fuse(enriched_diags, mode="max", score_names=self._fusion_score_names()),
            dtype=np.float64,
        )
        cdf_mean_scores = np.asarray(
            cdf_fusion.fuse(enriched_diags, mode="mean", score_names=self._fusion_score_names()),
            dtype=np.float64,
        )
        if self.config.use_prototype_support:
            cdf_mean_soft_support_scores = np.asarray(
                cdf_fusion.fuse(enriched_diags, mode="mean", score_names=self._soft_support_cdf_mean_score_names()),
                dtype=np.float64,
            )
        else:
            cdf_mean_soft_support_scores = cdf_mean_scores.copy()
        cdf_softmax_scores = np.asarray(
            cdf_fusion.fuse(enriched_diags, mode="softmax", score_names=self._fusion_score_names()),
            dtype=np.float64,
        )

        enriched_diags["raw_max_score"] = raw_max_scores
        enriched_diags["zscore_mean_score"] = zscore_mean_scores
        enriched_diags["cdf_max_score"] = cdf_max_scores
        enriched_diags["cdf_mean_score"] = cdf_mean_scores
        enriched_diags["cdf_mean_soft_support_score"] = cdf_mean_soft_support_scores
        enriched_diags["cdf_softmax_score"] = cdf_softmax_scores
        selected_scores = {
            "raw_max": raw_max_scores,
            "zscore_mean": zscore_mean_scores,
            "cdf_max": cdf_max_scores,
            "cdf_mean": cdf_mean_scores,
            "cdf_mean_soft_support": cdf_mean_soft_support_scores,
            "cdf_softmax": cdf_softmax_scores,
        }[self.config.evaluation_score_key]
        enriched_diags["final_fused_score"] = selected_scores

        raw_metric_sources = {
            name: np.asarray(enriched_diags[name], dtype=np.float64)
            for name in self._raw_diagnostic_keys()
            if name in enriched_diags
        }
        metric_sources = {
            **raw_metric_sources,
            "raw_max": raw_max_scores,
            "zscore_mean": zscore_mean_scores,
            "cdf_max": cdf_max_scores,
            "cdf_mean": cdf_mean_scores,
            "cdf_mean_soft_support": cdf_mean_soft_support_scores,
            "cdf_softmax": cdf_softmax_scores,
            "selected": selected_scores,
        }
        return enriched_diags, raw_metric_sources, metric_sources, selected_scores

    def _aggregate_sequence_scalar(
        self,
        scores: np.ndarray,
        observed_mask: Optional[np.ndarray] = None,
    ) -> tuple[float, bool, float]:
        seq_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if observed_mask is None:
            observed = np.ones_like(seq_scores, dtype=bool)
        else:
            observed = np.asarray(observed_mask, dtype=bool).reshape(-1)
            if observed.shape != seq_scores.shape:
                raise ValueError("observed_mask must have the same shape as scores.")
        coverage_ratio = float(observed.mean()) if observed.size else 0.0
        if not np.any(observed):
            return float("nan"), False, coverage_ratio
        visible_scores = seq_scores[observed]
        if self.config.sequence_score_aggregation == "max":
            value = float(np.max(visible_scores))
        elif self.config.sequence_score_aggregation == "top5_mean":
            k = max(1, int(np.ceil(visible_scores.size * 0.05)))
            topk = np.partition(visible_scores, -k)[-k:]
            value = float(np.mean(topk))
        else:
            value = float(np.percentile(visible_scores, 95))
        return value, True, coverage_ratio

    def _run_independent_sequence_test(
        self,
        raw_bundle: RawDatasetBundle,
        normalizer: TimeSeriesNormalizer,
        model: CoReMADModel,
        memory: MemoryBank,
        cdf_fusion: CDFPITFusion,
        zscore_fusion: ZScoreMeanFusion,
    ) -> dict[str, float]:
        if raw_bundle.test_sequences is None or raw_bundle.test_sequence_labels is None:
            raise RuntimeError("Independent sequence test requires test_sequences and test_sequence_labels.")

        sequence_names = raw_bundle.test_sequence_names or [
            f"sequence_{idx}" for idx in range(len(raw_bundle.test_sequence_labels))
        ]
        labels = np.asarray(raw_bundle.test_sequence_labels, dtype=np.int32)
        test_sequences = raw_bundle.test_sequences
        if self.config.max_test_sequences:
            limit = min(int(self.config.max_test_sequences), len(sequence_names))
            sequence_names = sequence_names[:limit]
            labels = labels[:limit]
            test_sequences = test_sequences[:limit]
        print(
            f"[Test][SequenceOnly] enabled: n_sequences={len(sequence_names)} "
            f"aggregation={self.config.sequence_score_aggregation}"
        )
        if np.unique(labels).size < 2:
            print(
                "[Test][SequenceOnly] sequence labels contain a single class only; "
                "binary summary metrics such as ROC-AUC/PR-AUC/F1 will be reported as NaN."
            )

        sequence_score_store: dict[str, list[float]] = {}
        covered_mask = np.zeros(len(sequence_names), dtype=bool)
        coverage_ratio = np.zeros(len(sequence_names), dtype=np.float64)
        use_sequence_shards = (
            str(self.config.dataset).upper() == "TEP"
            and str(self.config.tep_protocol).lower() == "full"
        )
        shard_dir = self.config.experiment_dir / "test_sequence_shards"
        signature_payload = {
            "schema_version": 1,
            "dataset": str(self.config.dataset),
            "data_root": str(Path(self.config.data_root).resolve()),
            "tep_protocol": str(self.config.tep_protocol),
            "seq_len": int(self.config.seq_len),
            "test_stride": int(self.config.test_stride),
            "max_test_windows": int(self.config.max_test_windows),
            "max_test_sequences": int(self.config.max_test_sequences),
            "sequence_score_aggregation": str(self.config.sequence_score_aggregation),
            "evaluation_score_key": str(self.config.evaluation_score_key),
            "stage_a": self._file_signature(self.config.stage_a_path),
            "memory": self._file_signature(self.config.memory_path),
            "cdf_npz": self._file_signature(self.config.cdf_fusion_path.with_suffix(".npz")),
            "cdf_json": self._file_signature(self.config.cdf_fusion_path.with_suffix(".json")),
            "zscore_json": self._file_signature(self.config.zscore_fusion_path.with_suffix(".json")),
            "tep_data_files": self._tep_data_file_signatures(),
        }
        shard_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        if use_sequence_shards:
            shard_dir.mkdir(parents=True, exist_ok=True)

        for idx, (seq_name, seq_label, sequence) in enumerate(
            zip(sequence_names, labels.tolist(), test_sequences)
        ):
            shard_path = shard_dir / f"{Path(str(seq_name)).stem}.npz"
            cached_scores: Optional[dict[str, float]] = None
            if use_sequence_shards and self.config.resume and shard_path.exists():
                try:
                    cached = np.load(shard_path, allow_pickle=False)
                    cached_signature = str(np.asarray(cached["signature"]).item())
                    cached_name = str(np.asarray(cached["sequence_name"]).item())
                    cached_label = int(np.asarray(cached["label"]).item())
                    if (
                        cached_signature == shard_signature
                        and cached_name == str(seq_name)
                        and cached_label == int(seq_label)
                    ):
                        cached_scores = {
                            key[len("score__") :]: float(np.asarray(cached[key]).item())
                            for key in cached.files
                            if key.startswith("score__")
                        }
                        required_score_names = {
                            *self._raw_diagnostic_keys(),
                            "raw_max",
                            "zscore_mean",
                            "cdf_max",
                            "cdf_mean",
                            "cdf_mean_soft_support",
                            "cdf_softmax",
                            "selected",
                        }
                        if not required_score_names.issubset(cached_scores):
                            cached_scores = None
                        covered_mask[idx] = bool(int(np.asarray(cached["covered"]).item()))
                        coverage_ratio[idx] = float(np.asarray(cached["coverage_ratio"]).item())
                except Exception as exc:  # noqa: BLE001
                    print(f"[Test][SequenceOnly] ignore invalid shard={shard_path}: {exc}")
            if cached_scores:
                print(f"[Test][SequenceOnly] resume shard sequence={seq_name}")
                for name, scalar in cached_scores.items():
                    sequence_score_store.setdefault(name, []).append(float(scalar))
                continue

            seq_array = np.asarray(sequence, dtype=np.float32)
            seq_norm = normalizer.transform(seq_array)
            print(
                f"[Test][SequenceOnly] sequence={seq_name} label={int(seq_label)} "
                f"length={len(seq_array)}"
            )
            with self._resource_span("scoring") as resource_span:
                seq_diags, seq_scores_count = self._aggregate_point_diagnostics(
                    data=seq_norm,
                    labels=None,
                    model=model,
                    memory=memory,
                    batch_size=self.config.test_batch_size,
                    stride=self.config.test_stride,
                    max_windows=self.config.max_test_windows,
                    print_stsd_stats=(idx == 0),
                    segment_ranges=None,
                )
                _, _, metric_sources, _ = self._build_metric_sources_from_point_diags(
                    seq_diags,
                    cdf_fusion,
                    zscore_fusion,
                )
                if resource_span is not None:
                    resource_span.set_workload(**self._last_diagnostic_workload)
                if self._active_resource_monitor is not None:
                    self._active_resource_monitor.add_workload(**self._last_diagnostic_workload)
            observed_mask = np.asarray(seq_scores_count, dtype=np.float64) > 0
            covered_here = bool(np.any(observed_mask))
            coverage_here = float(observed_mask.mean()) if observed_mask.size else 0.0
            covered_mask[idx] = covered_here
            coverage_ratio[idx] = coverage_here
            sequence_result: dict[str, float] = {}
            for name, point_scores in metric_sources.items():
                scalar, _, _ = self._aggregate_sequence_scalar(point_scores, observed_mask=observed_mask)
                sequence_score_store.setdefault(name, []).append(float(scalar))
                sequence_result[name] = float(scalar)
            if use_sequence_shards:
                temp_path = shard_path.with_suffix(".npz.part")
                with temp_path.open("wb") as handle:
                    np.savez(
                        handle,
                        signature=np.asarray(shard_signature),
                        sequence_name=np.asarray(str(seq_name)),
                        label=np.asarray(int(seq_label), dtype=np.int32),
                        covered=np.asarray(int(covered_here), dtype=np.int8),
                        coverage_ratio=np.asarray(coverage_here, dtype=np.float64),
                        **{
                            f"score__{name}": np.asarray(value, dtype=np.float64)
                            for name, value in sequence_result.items()
                        },
                    )
                temp_path.replace(shard_path)

        sequence_scores = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in sequence_score_store.items()
        }
        n_covered = int(covered_mask.sum())
        if n_covered < len(covered_mask):
            print(
                f"[Test][SequenceOnly] coverage warning: only {n_covered}/{len(covered_mask)} "
                "sequences received at least one scored point."
            )

        sequence_metrics: dict[str, dict[str, float]] = {}
        with self._resource_span("evaluation") as resource_span:
            for name, scores in sequence_scores.items():
                seq_metrics = self._compute_sequence_metrics(labels, scores)
                sequence_metrics[name] = seq_metrics
                print(
                    f"[Test][SequenceOnly] {name}: roc_auc={seq_metrics['roc_auc']:.6f} "
                    f"pr_auc={seq_metrics['pr_auc']:.6f} "
                    f"f1={seq_metrics['best_f1']:.6f}"
                )
            if resource_span is not None:
                resource_span.set_workload(points=len(labels))

        coverage_summary = {
            "n_total_sequences": int(len(labels)),
            "n_covered_sequences": int(covered_mask.sum()),
            "mean_covered_fraction": float(coverage_ratio.mean()) if coverage_ratio.size else float("nan"),
            "min_covered_fraction": float(coverage_ratio.min()) if coverage_ratio.size else float("nan"),
        }
        raw_subscore_metrics = {
            name: sequence_metrics[name]
            for name in self._raw_diagnostic_keys()
            if name in sequence_metrics
        }
        metrics = {
            "evaluation_protocol": "sequence_level",
            "sequence_evaluation_mode": "independent_sequences",
            "classification_metrics_available": bool(np.unique(labels).size >= 2),
            "classification_metrics_unavailable_reason": (
                None
                if np.unique(labels).size >= 2
                else "single_class_fault_only_mechanism_protocol"
            ),
            "sequence_score_aggregation": self.config.sequence_score_aggregation,
            "sequence_coverage": coverage_summary,
            "selected_score_key": self.config.evaluation_score_key,
            "selected": sequence_metrics["selected"],
            "raw_max": sequence_metrics["raw_max"],
            "zscore_mean": sequence_metrics["zscore_mean"],
            "cdf_max": sequence_metrics["cdf_max"],
            "cdf_mean": sequence_metrics["cdf_mean"],
            "cdf_mean_soft_support": sequence_metrics["cdf_mean_soft_support"],
            "cdf_softmax": sequence_metrics["cdf_softmax"],
            "subscores": raw_subscore_metrics,
        }
        if raw_bundle.dataset_metadata is not None:
            metrics["dataset_metadata"] = raw_bundle.dataset_metadata
        if use_sequence_shards:
            metrics["sequence_shards"] = {
                "directory": str(shard_dir),
                "signature": shard_signature,
                "count": int(len(sequence_names)),
            }

        score_file_map = {
            name: f"test_sequence_scores_{name}.npy"
            for name in sequence_scores
        }
        score_file_map["sequence_table_npz"] = "test_sequence_scores.npz"
        score_file_map["sequence_table_csv"] = "test_sequence_scores.csv"
        metrics["score_files"] = score_file_map

        for name, values in sequence_scores.items():
            np.save(self.config.experiment_dir / f"test_sequence_scores_{name}.npy", values)
            print(f"[Test] saved sequence score {name}: {self.config.experiment_dir / f'test_sequence_scores_{name}.npy'}")

        np.savez(
            self.config.experiment_dir / "test_sequence_scores.npz",
            labels=labels,
            names=np.asarray(sequence_names, dtype=object),
            covered=covered_mask,
            coverage_ratio=coverage_ratio,
            **sequence_scores,
        )
        csv_lines = ["sequence_name,label,covered,coverage_ratio," + ",".join(sequence_scores.keys())]
        for idx, seq_name in enumerate(sequence_names):
            row = [
                seq_name,
                str(int(labels[idx])),
                str(int(covered_mask[idx])),
                f"{float(coverage_ratio[idx]):.10f}",
            ]
            row.extend(f"{float(sequence_scores[name][idx]):.10f}" for name in sequence_scores)
            csv_lines.append(",".join(row))
        (self.config.experiment_dir / "test_sequence_scores.csv").write_text(
            "\n".join(csv_lines) + "\n",
            encoding="utf-8",
        )
        print(f"[Test] saved sequence scores: {self.config.experiment_dir / 'test_sequence_scores.npz'}")
        print(f"[Test] saved sequence scores csv: {self.config.experiment_dir / 'test_sequence_scores.csv'}")

        with self._resource_span("result_export"):
            (self.config.experiment_dir / "test_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        selected_metrics = sequence_metrics["selected"]
        print(
            f"[Test] final={self.config.evaluation_score_key} protocol=sequence_level "
            f"mode=independent_sequences aggregation={self.config.sequence_score_aggregation} "
            f"roc_auc={selected_metrics['roc_auc']:.6f} "
            f"pr_auc={selected_metrics['pr_auc']:.6f} "
            f"precision={selected_metrics['precision_at_best_f1']:.6f} "
            f"recall={selected_metrics['recall_at_best_f1']:.6f} "
            f"f1={selected_metrics['best_f1']:.6f}"
        )
        return selected_metrics

    def run_test(self) -> dict[str, float]:
        return self._run_monitored_stage("test", self._run_test_impl)

    def _run_test_impl(self) -> dict[str, float]:
        self._print_stage_banner("Testing")
        if self.config.data_format == "tsb_ad" and not HAS_VUS_METRICS:
            raise RuntimeError(
                "TSB-AD evaluation requires the official VUS metrics package. "
                "Install the dependencies listed in scripts/tsb_ad/requirements.txt."
            )
        self._require_stage_a_artifacts("testing")
        self._require_memory_artifacts("testing")
        model, normalizer, _ = self.load_stage_a_model()
        self._record_model_resource_stats(model)
        memory = self.load_memory_bank()
        raw_bundle = self._load_raw_bundle()
        cdf_fusion, zscore_fusion = self._prepare_test_fusions(raw_bundle, normalizer, model, memory)
        if raw_bundle.test_sequences is not None and raw_bundle.test_sequence_labels is not None:
            return self._run_independent_sequence_test(
                raw_bundle=raw_bundle,
                normalizer=normalizer,
                model=model,
                memory=memory,
                cdf_fusion=cdf_fusion,
                zscore_fusion=zscore_fusion,
            )
        test_norm = normalizer.transform(raw_bundle.test)
        with self._resource_span("scoring") as resource_span:
            final_diags, scores_count = self._aggregate_point_diagnostics(
                data=test_norm,
                labels=raw_bundle.test_labels,
                model=model,
                memory=memory,
                batch_size=self.config.test_batch_size,
                stride=self.config.test_stride,
                max_windows=self.config.max_test_windows,
                print_stsd_stats=True,
                segment_ranges=raw_bundle.test_segment_ranges,
            )
            final_diags, raw_metric_sources, metric_sources, selected_scores = self._build_metric_sources_from_point_diags(
                final_diags,
                cdf_fusion,
                zscore_fusion,
            )
            if resource_span is not None:
                resource_span.set_workload(**self._last_diagnostic_workload)
            if self._active_resource_monitor is not None:
                self._active_resource_monitor.add_workload(**self._last_diagnostic_workload)
        self._print_knn_distribution_stats(raw_bundle.test_labels, final_diags["knn_distance"])
        raw_max_scores = metric_sources["raw_max"]
        zscore_mean_scores = metric_sources["zscore_mean"]
        cdf_max_scores = metric_sources["cdf_max"]
        cdf_mean_scores = metric_sources["cdf_mean"]
        cdf_mean_soft_support_scores = metric_sources["cdf_mean_soft_support"]
        cdf_softmax_scores = metric_sources["cdf_softmax"]
        all_metrics: dict[str, dict[str, float]] = {}
        with self._resource_span("evaluation") as resource_span:
            for name, scores in metric_sources.items():
                mode_metrics = self._compute_metrics(
                    raw_bundle.test_labels,
                    scores,
                    vus_window=raw_bundle.evaluation_vus_window,
                )
                all_metrics[name] = mode_metrics
                print(
                    f"[Test] {name}: roc_auc={mode_metrics['roc_auc']:.6f} "
                    f"pr_auc={mode_metrics['pr_auc']:.6f} "
                    f"point_f1={mode_metrics['best_f1']:.6f} "
                    f"pa_f1={mode_metrics['pa_best_f1']:.6f}"
                )
            if resource_span is not None:
                resource_span.set_workload(points=len(raw_bundle.test_labels))

        sequence_metrics: dict[str, dict[str, float]] = {}
        sequence_scores: dict[str, np.ndarray] = {}
        sequence_covered_mask: Optional[np.ndarray] = None
        sequence_coverage_ratio: Optional[np.ndarray] = None
        if raw_bundle.test_sequence_ranges is not None and raw_bundle.test_sequence_labels is not None:
            print(
                f"[Test][Sequence] enabled: n_sequences={len(raw_bundle.test_sequence_labels)} "
                f"aggregation={self.config.sequence_score_aggregation}"
            )
            point_observed_mask = scores_count > 0
            for name, scores in metric_sources.items():
                seq_scores, covered_mask, coverage_ratio = self._aggregate_sequence_scores(
                    scores=scores,
                    sequence_ranges=raw_bundle.test_sequence_ranges,
                    aggregation=self.config.sequence_score_aggregation,
                    observed_mask=point_observed_mask,
                )
                if sequence_covered_mask is None:
                    sequence_covered_mask = covered_mask
                    sequence_coverage_ratio = coverage_ratio
                    n_covered = int(covered_mask.sum())
                    if n_covered < len(covered_mask):
                        print(
                            f"[Test][Sequence] coverage warning: only {n_covered}/{len(covered_mask)} "
                            "sequences received at least one scored point. "
                            "Metrics below exclude uncovered sequences."
                        )
                sequence_scores[name] = seq_scores
                seq_metrics = self._compute_sequence_metrics(raw_bundle.test_sequence_labels, seq_scores)
                sequence_metrics[name] = seq_metrics
                print(
                    f"[Test][Sequence] {name}: roc_auc={seq_metrics['roc_auc']:.6f} "
                    f"pr_auc={seq_metrics['pr_auc']:.6f} "
                    f"f1={seq_metrics['best_f1']:.6f}"
                )

        if sequence_metrics:
            coverage_summary = {
                "n_total_sequences": int(len(raw_bundle.test_sequence_labels)),
                "n_covered_sequences": int(sequence_covered_mask.sum()) if sequence_covered_mask is not None else 0,
                "mean_covered_fraction": (
                    float(sequence_coverage_ratio.mean()) if sequence_coverage_ratio is not None and sequence_coverage_ratio.size else float("nan")
                ),
                "min_covered_fraction": (
                    float(sequence_coverage_ratio.min()) if sequence_coverage_ratio is not None and sequence_coverage_ratio.size else float("nan")
                ),
            }
            metrics = {
                "evaluation_protocol": "sequence_level",
                "sequence_score_aggregation": self.config.sequence_score_aggregation,
                "sequence_coverage": coverage_summary,
                "selected_score_key": self.config.evaluation_score_key,
                "selected": sequence_metrics["selected"],
                "raw_max": sequence_metrics["raw_max"],
                "zscore_mean": sequence_metrics["zscore_mean"],
                "cdf_max": sequence_metrics["cdf_max"],
                "cdf_mean": sequence_metrics["cdf_mean"],
                "cdf_mean_soft_support": sequence_metrics["cdf_mean_soft_support"],
                "cdf_softmax": sequence_metrics["cdf_softmax"],
                "subscores": {name: sequence_metrics[name] for name in raw_metric_sources},
                "weak_pointwise": {
                    "selected": all_metrics["selected"],
                    "raw_max": all_metrics["raw_max"],
                    "zscore_mean": all_metrics["zscore_mean"],
                    "cdf_max": all_metrics["cdf_max"],
                    "cdf_mean": all_metrics["cdf_mean"],
                    "cdf_mean_soft_support": all_metrics["cdf_mean_soft_support"],
                    "cdf_softmax": all_metrics["cdf_softmax"],
                    "subscores": {name: all_metrics[name] for name in raw_metric_sources},
                },
            }
        else:
            metrics = {
                "evaluation_protocol": "point_level",
                "point_coverage_ratio": float(np.mean(scores_count > 0)) if scores_count.size else 0.0,
                "selected_score_key": self.config.evaluation_score_key,
                "selected": all_metrics["selected"],
                "raw_max": all_metrics["raw_max"],
                "zscore_mean": all_metrics["zscore_mean"],
                "cdf_max": all_metrics["cdf_max"],
                "cdf_mean": all_metrics["cdf_mean"],
                "cdf_mean_soft_support": all_metrics["cdf_mean_soft_support"],
                "cdf_softmax": all_metrics["cdf_softmax"],
                "subscores": {name: all_metrics[name] for name in raw_metric_sources},
            }
        if raw_bundle.dataset_metadata is not None:
            metrics["dataset_metadata"] = raw_bundle.dataset_metadata
        score_file_map = {
            "final_fused_score": "test_scores_final_selected.npy",
            "final_fused_score_legacy_alias": "test_scores.npy",
            "final_cdf_max_legacy_alias": "test_scores_final_cdf_max.npy",
            "selected": "test_scores_selected.npy",
            "raw_max": "test_scores_raw_max.npy",
            "zscore_mean": "test_scores_zscore_mean.npy",
            "cdf_max": "test_scores_cdf_max.npy",
            "cdf_mean": "test_scores_cdf_mean.npy",
            "cdf_mean_soft_support": "test_scores_cdf_mean_soft_support.npy",
            "cdf_softmax": "test_scores_cdf_softmax.npy",
            "knn_distance": "test_scores_knn_distance.npy",
            "state_novelty": "test_scores_state_novelty.npy",
        }
        for name in self._completion_score_names():
            score_file_map[name] = f"test_scores_{name}.npy"
        metrics["score_files"] = score_file_map

        np.save(self.config.experiment_dir / "test_scores_final_selected.npy", selected_scores)
        np.save(self.config.experiment_dir / "test_scores.npy", selected_scores)
        np.save(self.config.experiment_dir / "test_scores_final_cdf_max.npy", cdf_max_scores)
        np.save(self.config.experiment_dir / "test_scores_selected.npy", selected_scores)
        np.save(self.config.experiment_dir / "test_scores_raw_max.npy", raw_max_scores)
        np.save(self.config.experiment_dir / "test_scores_zscore_mean.npy", zscore_mean_scores)
        np.save(self.config.experiment_dir / "test_scores_cdf_max.npy", cdf_max_scores)
        np.save(self.config.experiment_dir / "test_scores_cdf_mean.npy", cdf_mean_scores)
        np.save(self.config.experiment_dir / "test_scores_cdf_mean_soft_support.npy", cdf_mean_soft_support_scores)
        np.save(self.config.experiment_dir / "test_scores_cdf_softmax.npy", cdf_softmax_scores)
        np.save(self.config.experiment_dir / "test_scores_knn_distance.npy", metric_sources["knn_distance"])
        np.save(self.config.experiment_dir / "test_scores_state_novelty.npy", metric_sources["state_novelty"])
        for name in self._completion_score_names():
            if name in metric_sources:
                np.save(self.config.experiment_dir / f"test_scores_{name}.npy", metric_sources[name])
        np.savez(
            self.config.experiment_dir / "test_diagnostic_scores.npz",
            labels=np.asarray(raw_bundle.test_labels, dtype=np.int32),
            **final_diags,
        )
        if sequence_scores and raw_bundle.test_sequence_labels is not None:
            np.savez(
                self.config.experiment_dir / "test_sequence_scores.npz",
                labels=np.asarray(raw_bundle.test_sequence_labels, dtype=np.int32),
                names=np.asarray(raw_bundle.test_sequence_names or [], dtype=object),
                covered=np.asarray(sequence_covered_mask, dtype=bool) if sequence_covered_mask is not None else np.zeros(0, dtype=bool),
                coverage_ratio=(
                    np.asarray(sequence_coverage_ratio, dtype=np.float64)
                    if sequence_coverage_ratio is not None
                    else np.zeros(0, dtype=np.float64)
                ),
                **{name: np.asarray(values, dtype=np.float64) for name, values in sequence_scores.items()},
            )
            csv_lines = ["sequence_name,label,covered,coverage_ratio," + ",".join(sequence_scores.keys())]
            sequence_names = raw_bundle.test_sequence_names or [f"sequence_{idx}" for idx in range(len(raw_bundle.test_sequence_labels))]
            for idx, seq_name in enumerate(sequence_names):
                covered = (
                    bool(sequence_covered_mask[idx])
                    if sequence_covered_mask is not None and idx < len(sequence_covered_mask)
                    else False
                )
                coverage_ratio = (
                    float(sequence_coverage_ratio[idx])
                    if sequence_coverage_ratio is not None and idx < len(sequence_coverage_ratio)
                    else float("nan")
                )
                row = [
                    seq_name,
                    str(int(raw_bundle.test_sequence_labels[idx])),
                    str(int(covered)),
                    f"{coverage_ratio:.10f}",
                ]
                row.extend(f"{float(sequence_scores[name][idx]):.10f}" for name in sequence_scores)
                csv_lines.append(",".join(row))
            (self.config.experiment_dir / "test_sequence_scores.csv").write_text(
                "\n".join(csv_lines) + "\n",
                encoding="utf-8",
            )
            print(f"[Test] saved sequence scores: {self.config.experiment_dir / 'test_sequence_scores.npz'}")
            print(f"[Test] saved sequence scores csv: {self.config.experiment_dir / 'test_sequence_scores.csv'}")
        print(f"[Test] saved diagnostic scores: {self.config.experiment_dir / 'test_diagnostic_scores.npz'}")
        for score_name, filename in score_file_map.items():
            print(f"[Test] saved {score_name}: {self.config.experiment_dir / filename}")
        with self._resource_span("result_export"):
            (self.config.experiment_dir / "test_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._export_test_visualizations(selected_scores, raw_bundle.test_labels, final_diags)
        if sequence_metrics:
            selected_metrics = sequence_metrics["selected"]
            print(
                f"[Test] final={self.config.evaluation_score_key} protocol=sequence_level "
                f"aggregation={self.config.sequence_score_aggregation} "
                f"roc_auc={selected_metrics['roc_auc']:.6f} "
                f"pr_auc={selected_metrics['pr_auc']:.6f} "
                f"precision={selected_metrics['precision_at_best_f1']:.6f} "
                f"recall={selected_metrics['recall_at_best_f1']:.6f} "
                f"f1={selected_metrics['best_f1']:.6f}"
            )
            weak_selected_metrics = all_metrics["selected"]
            print(
                f"[Test][WeakPoint] final={self.config.evaluation_score_key} "
                f"roc_auc={weak_selected_metrics['roc_auc']:.6f} "
                f"pr_auc={weak_selected_metrics['pr_auc']:.6f} "
                f"point_f1={weak_selected_metrics['point_best_f1']:.6f} "
                f"pa_f1={weak_selected_metrics['pa_best_f1']:.6f}"
            )
        else:
            selected_metrics = all_metrics["selected"]
            print(
                f"[Test] final={self.config.evaluation_score_key} "
                f"roc_auc={selected_metrics['roc_auc']:.6f} "
                f"pr_auc={selected_metrics['pr_auc']:.6f} "
                f"point_precision={selected_metrics['point_best_precision']:.6f} "
                f"point_recall={selected_metrics['point_best_recall']:.6f} "
                f"point_f1={selected_metrics['point_best_f1']:.6f} "
                f"pa_precision={selected_metrics['pa_precision_at_best_f1']:.6f} "
                f"pa_recall={selected_metrics['pa_recall_at_best_f1']:.6f} "
                f"pa_f1={selected_metrics['pa_best_f1']:.6f}"
            )
            print(
                f"[Test] aff_p={selected_metrics['aff_precision']:.6f} "
                f"aff_r={selected_metrics['aff_recall']:.6f} "
                f"aff_f1={selected_metrics['aff_f1']:.6f} "
                f"range_p={selected_metrics['range_precision']:.6f} "
                f"range_r={selected_metrics['range_recall']:.6f} "
                f"range_f1={selected_metrics['range_f1']:.6f} "
                f"r_auc_roc={selected_metrics['r_auc_roc']:.6f} "
                f"r_auc_pr={selected_metrics['r_auc_pr']:.6f} "
                f"vus_roc={selected_metrics['vus_roc']:.6f} "
                f"vus_pr={selected_metrics['vus_pr']:.6f} "
                f"vus_window={selected_metrics['vus_window']:.0f}"
            )
        return metrics

    @staticmethod
    def _print_knn_distribution_stats(labels: np.ndarray, knn_scores: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=np.int32)
        knn_scores = np.asarray(knn_scores, dtype=np.float64)
        normal_scores = knn_scores[labels == 0]
        anomaly_scores = knn_scores[labels == 1]
        if normal_scores.size == 0 or anomaly_scores.size == 0:
            print("[TestDiag] skip kNN distribution stats: missing normal or anomaly points.")
            return

        normal_median = float(np.median(normal_scores))
        normal_p95 = float(np.percentile(normal_scores, 95))
        anomaly_median = float(np.median(anomaly_scores))
        anomaly_p5 = float(np.percentile(anomaly_scores, 5))
        overlap = normal_p95 > anomaly_p5

        print(f"[TestDiag] kNN normal:  median={normal_median:.4f}, P95={normal_p95:.4f}")
        print(f"[TestDiag] kNN anomaly: median={anomaly_median:.4f}, P5={anomaly_p5:.4f}")
        print(
            f"[TestDiag] kNN overlap: normal_P95 {'>' if overlap else '<='} anomaly_P5 "
            f"({normal_p95:.4f} vs {anomaly_p5:.4f})"
        )

    def _export_test_visualizations(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        diagnostics: dict[str, np.ndarray],
    ) -> None:
        for name, title, xlabel in self._test_diagnostic_specs():
            if name not in diagnostics:
                print(f"[Viz] skip {name}: diagnostic score not available.")
                continue
            filename = f"{name}_distribution.png"
            plot_score_distribution(
                diagnostics[name],
                labels,
                self.config.experiment_dir / filename,
                title=f"{title}: Normal vs Anomaly",
                xlabel=xlabel,
                plot_name=name,
            )
        explicit_timeline_path = (
            self.config.experiment_dir
            / f"score_timeline_{self.config.evaluation_score_key}_full.png"
        )
        if plot_score_timeline(
            scores,
            labels,
            explicit_timeline_path,
            title="Detection Result (Full Timeline)",
        ):
            self._write_legacy_alias(
                explicit_timeline_path,
                self.config.experiment_dir / "score_timeline_full.png",
            )

        segments = self._find_anomaly_segments(labels)
        if not segments:
            print("[Viz] no anomaly segments found in labels, skip local timeline plots.")
            return

        margin = 500
        for seg_idx, (start, end) in enumerate(segments[:3], start=1):
            left = max(0, start - margin)
            right = min(len(labels), end + margin)
            explicit_segment_path = (
                self.config.experiment_dir
                / f"score_timeline_{self.config.evaluation_score_key}_segment_{seg_idx}.png"
            )
            if plot_score_timeline(
                scores,
                labels,
                explicit_segment_path,
                time_range=(left, right),
                title=f"Detection Result (Segment {seg_idx}: {left}-{right})",
            ):
                self._write_legacy_alias(
                    explicit_segment_path,
                    self.config.experiment_dir / f"score_timeline_segment_{seg_idx}.png",
                )

    @staticmethod
    def _write_legacy_alias(source_path: Path, alias_path: Path) -> None:
        if source_path == alias_path:
            return
        shutil.copyfile(source_path, alias_path)
        print(f"[Viz] saved legacy alias: {alias_path}")

    def run_full(self) -> dict[str, float]:
        if self.config.resume and self._is_stage_a_complete():
            print("[Full] stage_a checkpoint detected, enabling resume/skip behavior.")
        self.run_stage_a()
        self.run_stage_b()
        return self.run_test()

    def _restore_config_from_stage_a_payload(self, payload: dict) -> None:
        payload_config = payload.get("config")
        if not isinstance(payload_config, dict):
            return

        current = self.config
        valid_keys = set(CoReMADConfig.__dataclass_fields__.keys())
        filtered_payload_config = {
            key: value for key, value in payload_config.items() if key in valid_keys
        }
        restored = CoReMADConfig(**filtered_payload_config)
        # Keep runtime / Stage-B / Stage-test overrides from the current invocation
        # so we can safely reuse a Stage-A checkpoint while sweeping retrieval or
        # memory hyperparameters in a new experiment directory.
        runtime_override_fields = (
            "artifact_root",
            "experiment_name",
            "data_root",
            "data_format",
            "tep_protocol",
            "device",
            "resume",
            "batch_size",
            "train_batch_size",
            "val_batch_size",
            "memory_batch_size",
            "test_batch_size",
            "train_stride",
            "test_stride",
            "memory_build_stride",
            "val_ratio",
            "val_gap",
            "val_min_train_windows",
            "val_split_mode",
            "stage_a_epochs",
            "early_stop_patience",
            "lr",
            "weight_decay",
            "grad_clip_norm",
            "scheduler_eta_min_ratio",
            "mask_ratio",
            "n_mask_groups",
            "lambda_pred",
            "lambda_smooth",
            "patch_sizes",
            "d_z",
            "top_M",
            "top_K",
            "knn_k",
            "clean_ratio",
            "state_prototype_count",
            "prototype_top_p",
            "prototype_candidate_cap",
            "soft_candidate_tau",
            "support_score_tau",
            "support_score_eps",
            "include_soft_support_in_fusion",
            "use_stsd_decomposition",
            "use_channel_modulation",
            "use_two_level_retrieval",
            "use_context_key_retrieval",
            "use_completion_head",
            "use_completion_self_cleaning",
            "use_completion_score_fusion",
            "evaluation_score_key",
            "use_faiss",
            "faiss_use_gpu",
            "faiss_exact_threshold",
            "faiss_ivf_nprobe",
            "coreset_keep_ratio",
            "coreset_max_patches_per_scale",
            "coreset_fps_threshold",
            "num_workers",
            "max_train_windows",
            "max_test_windows",
            "max_test_sequences",
            "seed",
            "memory_seed",
            "memory_audit_mode",
            "resource_monitor_enabled",
            "resource_sample_interval",
        )
        for field_name in runtime_override_fields:
            setattr(restored, field_name, getattr(current, field_name))
        self.config = restored

    def load_stage_a_model(self) -> tuple[CoReMADModel, TimeSeriesNormalizer, dict]:
        payload = torch.load(self.config.stage_a_path, map_location="cpu", weights_only=False)
        if not self._stage_a_payload_compatible(payload, "load Stage A checkpoint"):
            raise RuntimeError(
                "Stage A checkpoint is incompatible with the current configuration. "
                "Train or select a checkpoint created with the requested dataset protocol."
            )
        self._restore_config_from_stage_a_payload(payload)
        model = CoReMADModel(self.config)
        self._load_model_state(model, payload["model_state"])
        model.to(self.device)
        model.eval()
        normalizer = TimeSeriesNormalizer.from_state_dict(payload["normalizer"])
        return model, normalizer, payload

    def load_memory_bank(self) -> MemoryBank:
        return MemoryBank.load(self.config.memory_path, self.config)

    def load_cdf_fusion(self, optional: bool = False) -> Optional[CDFPITFusion]:
        npz_path = self.config.cdf_fusion_path.with_suffix(".npz")
        json_path = self.config.cdf_fusion_path.with_suffix(".json")
        if not npz_path.exists() or not json_path.exists():
            if optional:
                return None
            raise FileNotFoundError(f"Missing CDF fusion artifact: {self.config.cdf_fusion_path}")
        fusion = CDFPITFusion(self._cdf_fit_score_names())
        fusion.load(self.config.cdf_fusion_path)
        return fusion

    def load_zscore_fusion(self, optional: bool = False) -> Optional[ZScoreMeanFusion]:
        json_path = self.config.zscore_fusion_path.with_suffix(".json")
        if not json_path.exists():
            if optional:
                return None
            raise FileNotFoundError(f"Missing z-score fusion artifact: {self.config.zscore_fusion_path}")
        fusion = ZScoreMeanFusion(self._fusion_score_names())
        fusion.load(self.config.zscore_fusion_path)
        return fusion

    def _aggregate_point_diagnostics(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        model: CoReMADModel,
        memory: MemoryBank,
        batch_size: int,
        stride: int,
        max_windows: int,
        print_stsd_stats: bool = False,
        segment_ranges: Optional[np.ndarray] = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        loader = build_loader(
            data,
            labels,
            seq_len=self.config.seq_len,
            stride=stride,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            max_windows=max_windows,
            drop_last=False,
            segment_ranges=segment_ranges,
        )
        total_windows = len(loader.dataset) if hasattr(loader, "dataset") else None
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        self._last_diagnostic_workload = {
            "points": 0,
            "covered_points": 0,
            "input_points": int(len(data)),
            "window_points": int(total_windows or 0) * self.config.seq_len,
            "windows": int(total_windows or 0),
            "batches": int(total_batches or 0),
        }
        print(
            f"[DiagAgg] start: windows={total_windows}, batches={total_batches}, "
            f"batch_size={batch_size}, stride={stride}, "
            f"seq_len={self.config.seq_len}, num_workers={self.config.num_workers}"
        )
        total_length = len(data) if labels is None else len(labels)
        raw_keys = self._raw_diagnostic_keys()
        diag_sum = {name: np.zeros(total_length, dtype=np.float64) for name in raw_keys}
        scores_count = np.zeros(total_length, dtype=np.float64)
        printed_stats = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                x = batch["x"].to(self.device, non_blocking=True)
                if print_stsd_stats and not printed_stats:
                    encoded = model.encode(x)
                    slow = encoded["slow"]
                    residual = encoded["residual"]
                    slow_mean_abs = slow.abs().mean().item()
                    residual_mean_abs = residual.abs().mean().item()
                    total_mean_abs = slow_mean_abs + residual_mean_abs
                    slow_ratio = slow_mean_abs / total_mean_abs if total_mean_abs > 0.0 else 0.0
                    print(f"[TestDiag] slow   std={slow.std().item():.4f}  mean_abs={slow_mean_abs:.4f}")
                    print(f"[TestDiag] resid  std={residual.std().item():.4f}  mean_abs={residual_mean_abs:.4f}")
                    print(f"[TestDiag] ratio  slow/total = {slow_ratio:.2%}")
                    printed_stats = True

                with self._resource_span("inference") as resource_span:
                    diag_tensors = self.extract_point_feature_dict(x, model, memory)
                    if resource_span is not None:
                        batch_windows = int(x.size(0))
                        resource_span.set_workload(
                            points=batch_windows * self.config.seq_len,
                            window_points=batch_windows * self.config.seq_len,
                            windows=batch_windows,
                            batches=1,
                        )
                batch_diags = {
                    name: diag_tensors[name].detach().cpu().numpy()
                    for name in raw_keys
                }
                starts = batch["start"].cpu().numpy().astype(np.int64, copy=False)
                point_indices = starts[:, None] + np.arange(self.config.seq_len, dtype=np.int64)[None, :]
                for name in raw_keys:
                    np.add.at(diag_sum[name], point_indices, batch_diags[name])
                np.add.at(scores_count, point_indices, 1.0)
                if total_batches is not None and (
                    batch_idx == 1
                    or batch_idx == total_batches
                    or batch_idx % max(1, total_batches // 5) == 0
                ):
                    processed_windows = min(batch_idx * int(getattr(loader, "batch_size", x.size(0))), total_windows or batch_idx * x.size(0))
                    print(
                        f"[DiagAgg] progress: batch={batch_idx}/{total_batches}, "
                        f"processed_windows={processed_windows}"
                    )

        covered_points = int(np.count_nonzero(scores_count > 0))
        self._last_diagnostic_workload["points"] = covered_points
        self._last_diagnostic_workload["covered_points"] = covered_points
        final_diags = {
            name: values / np.maximum(scores_count, 1.0)
            for name, values in diag_sum.items()
        }
        return final_diags, scores_count

    def _select_observed_train_diagnostics(
        self,
        train_diags: dict[str, np.ndarray],
        scores_count: np.ndarray,
        score_names: list[str],
    ) -> dict[str, np.ndarray]:
        observed_mask = np.asarray(scores_count, dtype=np.float64) > 0
        if not np.any(observed_mask):
            raise RuntimeError("Stage B diagnostics produced no covered training points for fusion fitting.")
        return {
            name: np.asarray(train_diags[name], dtype=np.float64)[observed_mask]
            for name in score_names
        }

    def _fit_and_save_cdf_fusion_from_diagnostics(
        self,
        train_diags: dict[str, np.ndarray],
        scores_count: np.ndarray,
    ) -> CDFPITFusion:
        score_names = self._cdf_fit_score_names()
        fusion = CDFPITFusion(score_names)
        observed_diags = self._select_observed_train_diagnostics(train_diags, scores_count, score_names)
        fusion.fit(observed_diags)
        fusion.save(self.config.cdf_fusion_path)
        first_key = next(iter(observed_diags))
        print(f"[Stage B] CDF fusion fitted on {int(observed_diags[first_key].size)} covered training points.")
        return fusion

    def _fit_and_save_zscore_fusion_from_diagnostics(
        self,
        train_diags: dict[str, np.ndarray],
        scores_count: np.ndarray,
    ) -> ZScoreMeanFusion:
        score_names = self._fusion_score_names()
        fusion = ZScoreMeanFusion(score_names)
        observed_diags = self._select_observed_train_diagnostics(train_diags, scores_count, score_names)
        fusion.fit(observed_diags)
        fusion.save(self.config.zscore_fusion_path)
        first_key = next(iter(observed_diags))
        print(f"[Stage B] z-score fusion fitted on {int(observed_diags[first_key].size)} covered training points.")
        return fusion

    def _fit_and_save_cdf_fusion(
        self,
        train_data: np.ndarray,
        model: CoReMADModel,
        memory: MemoryBank,
        segment_ranges: Optional[np.ndarray] = None,
    ) -> CDFPITFusion:
        train_diags, scores_count = self._aggregate_point_diagnostics(
            data=train_data,
            labels=None,
            model=model,
            memory=memory,
            batch_size=self.config.memory_batch_size,
            stride=self.config.memory_build_stride,
            max_windows=self.config.max_train_windows,
            print_stsd_stats=False,
            segment_ranges=segment_ranges,
        )
        return self._fit_and_save_cdf_fusion_from_diagnostics(train_diags, scores_count)

    def _fit_and_save_zscore_fusion(
        self,
        train_data: np.ndarray,
        model: CoReMADModel,
        memory: MemoryBank,
        segment_ranges: Optional[np.ndarray] = None,
    ) -> ZScoreMeanFusion:
        train_diags, scores_count = self._aggregate_point_diagnostics(
            data=train_data,
            labels=None,
            model=model,
            memory=memory,
            batch_size=self.config.memory_batch_size,
            stride=self.config.memory_build_stride,
            max_windows=self.config.max_train_windows,
            print_stsd_stats=False,
            segment_ranges=segment_ranges,
        )
        return self._fit_and_save_zscore_fusion_from_diagnostics(train_diags, scores_count)

    def extract_point_feature_dict(
        self,
        x: torch.Tensor,
        model: CoReMADModel,
        memory: MemoryBank,
        z_override: Optional[list[torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        if self._completion_score_names():
            encoded, comp_scores = model.deterministic_completion_scores(x)
        else:
            encoded = model.encode(x)
            comp_scores = []
        z_list = z_override if z_override is not None else encoded["z"]
        memory_out = memory.query_preencoded(encoded["state_vec"], z_list, encoded["c"])
        mem_point = self._fuse_patch_scores(memory_out["mem_scores"])
        novelty = memory_out["state_novelty"].to(x.device).unsqueeze(-1).expand(-1, self.config.seq_len)
        features = {
            "knn_distance": mem_point,
            "state_novelty": novelty,
        }
        if self.config.use_prototype_support:
            features["soft_support_score"] = self._fuse_patch_scores(memory_out["soft_mem_scores"])
        for score_name, patch_size, comp_score in zip(self._completion_score_names(), self.config.patch_sizes, comp_scores):
            features[score_name] = self._expand_patch_score(comp_score, patch_size)
        return features

    def _fuse_patch_scores(self, patch_scores: list[torch.Tensor]) -> torch.Tensor:
        point_scores = None
        weights = self.config.scale_weights()
        for score, patch_size, weight in zip(patch_scores, self.config.patch_sizes, weights):
            expanded = self._expand_patch_score(score, patch_size)
            point_scores = expanded * weight if point_scores is None else point_scores + expanded * weight
        return point_scores

    def _expand_patch_score(self, score: torch.Tensor, patch_size: int) -> torch.Tensor:
        expanded = score.repeat_interleave(patch_size, dim=1)
        if expanded.size(1) < self.config.seq_len:
            pad = expanded[:, -1:].expand(-1, self.config.seq_len - expanded.size(1))
            expanded = torch.cat([expanded, pad], dim=1)
        return expanded[:, : self.config.seq_len]

    @staticmethod
    def _binary_clf_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(scores, kind="mergesort")[::-1]
        y_true = labels[order].astype(np.float64)
        y_score = scores[order]
        distinct = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1.0 + threshold_idxs - tps
        thresholds = y_score[threshold_idxs]
        return fps, tps, thresholds

    @classmethod
    def _precision_recall_curve(cls, labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fps, tps, thresholds = cls._binary_clf_curve(labels, scores)
        precision = tps / np.maximum(tps + fps, 1e-12)
        recall = tps / np.maximum(tps[-1], 1e-12)
        precision = np.r_[1.0, precision]
        recall = np.r_[0.0, recall]
        return precision, recall, thresholds

    @staticmethod
    def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.size == 0 or neg.size == 0:
            return float("nan")
        combined = np.concatenate([pos, neg])
        ranks = np.argsort(np.argsort(combined, kind="mergesort"), kind="mergesort") + 1
        pos_ranks = ranks[: pos.size]
        auc = (pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
        return float(auc)

    @staticmethod
    def _average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
        return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

    @staticmethod
    def _normalize_scores_01(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            return np.zeros_like(scores, dtype=np.float64)
        lo = float(finite.min())
        hi = float(finite.max())
        if hi - lo < 1e-12:
            return np.zeros_like(scores, dtype=np.float64)
        norm = (scores - lo) / (hi - lo)
        return np.clip(norm, 0.0, 1.0)

    @staticmethod
    def _find_anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        start = None
        for idx, value in enumerate(labels.astype(np.int32)):
            if value == 1 and start is None:
                start = idx
            elif value == 0 and start is not None:
                segments.append((start, idx))
                start = None
        if start is not None:
            segments.append((start, len(labels)))
        return segments

    @staticmethod
    def _find_positive_segments(values: np.ndarray) -> list[tuple[int, int]]:
        binary = np.asarray(values, dtype=np.int32).reshape(-1)
        if binary.size == 0:
            return []
        padded = np.pad(binary, (1, 1), mode="constant")
        transitions = np.diff(padded)
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        return [(int(start), int(end)) for start, end in zip(starts.tolist(), ends.tolist())]

    @staticmethod
    def _count_segment_matches(
        true_segments: list[tuple[int, int]],
        pred_segments: list[tuple[int, int]],
    ) -> tuple[int, int]:
        if not true_segments or not pred_segments:
            return 0, 0

        true_hits = np.zeros(len(true_segments), dtype=bool)
        pred_hits = np.zeros(len(pred_segments), dtype=bool)
        true_idx = 0
        pred_idx = 0

        while true_idx < len(true_segments) and pred_idx < len(pred_segments):
            true_start, true_end = true_segments[true_idx]
            pred_start, pred_end = pred_segments[pred_idx]
            if true_end <= pred_start:
                true_idx += 1
                continue
            if pred_end <= true_start:
                pred_idx += 1
                continue

            true_hits[true_idx] = True
            pred_hits[pred_idx] = True
            if true_end <= pred_end:
                true_idx += 1
            else:
                pred_idx += 1

        return int(true_hits.sum()), int(pred_hits.sum())

    @classmethod
    def _compute_event_best_metrics(
        cls,
        labels: np.ndarray,
        scores: np.ndarray,
        max_thresholds: int = 256,
    ) -> dict[str, float]:
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if labels.size == 0:
            return {
                "range_precision": float("nan"),
                "range_recall": float("nan"),
                "range_f1": float("nan"),
                "range_best_threshold": float("nan"),
                "event_precision": float("nan"),
                "event_recall": float("nan"),
                "event_f1": float("nan"),
                "event_best_threshold": float("nan"),
            }

        true_segments = cls._find_anomaly_segments(labels)
        if not true_segments:
            return {
                "range_precision": float("nan"),
                "range_recall": float("nan"),
                "range_f1": float("nan"),
                "range_best_threshold": float("nan"),
                "event_precision": float("nan"),
                "event_recall": float("nan"),
                "event_f1": float("nan"),
                "event_best_threshold": float("nan"),
            }

        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            return {
                "range_precision": 0.0,
                "range_recall": 0.0,
                "range_f1": 0.0,
                "range_best_threshold": float("nan"),
                "event_precision": 0.0,
                "event_recall": 0.0,
                "event_f1": 0.0,
                "event_best_threshold": float("nan"),
            }

        unique_scores = np.unique(finite_scores)
        if unique_scores.size <= max_thresholds:
            candidate_thresholds = unique_scores
        else:
            quantiles = np.linspace(0.0, 1.0, num=max_thresholds)
            candidate_thresholds = np.unique(np.quantile(finite_scores, quantiles))

        best = {
            "range_precision": 0.0,
            "range_recall": 0.0,
            "range_f1": 0.0,
            "range_best_threshold": float(candidate_thresholds[0]),
        }
        total_true = max(1, len(true_segments))

        for threshold in candidate_thresholds.tolist():
            pred_segments = cls._find_positive_segments(scores >= float(threshold))
            if not pred_segments:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                matched_true, matched_pred = cls._count_segment_matches(true_segments, pred_segments)
                precision = matched_pred / max(1, len(pred_segments))
                recall = matched_true / total_true
                f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            if f1 > best["range_f1"]:
                best = {
                    "range_precision": float(precision),
                    "range_recall": float(recall),
                    "range_f1": float(f1),
                    "range_best_threshold": float(threshold),
                }

        return {
            **best,
            "event_precision": best["range_precision"],
            "event_recall": best["range_recall"],
            "event_f1": best["range_f1"],
            "event_best_threshold": best["range_best_threshold"],
        }

    @classmethod
    def _estimate_vus_window(cls, labels: np.ndarray) -> int:
        # Use the median anomaly segment length as a stable dataset-specific sliding window.
        seg_lengths = [end - start for start, end in cls._find_anomaly_segments(labels)]
        if not seg_lengths:
            return 1
        return max(1, int(np.median(np.asarray(seg_lengths, dtype=np.float64))))

    @classmethod
    def _compute_vus_metrics(
        cls,
        labels: np.ndarray,
        scores: np.ndarray,
        vus_window: Optional[int] = None,
    ) -> dict[str, float]:
        if not HAS_VUS_METRICS:
            return {
                "aff_precision": float("nan"),
                "aff_recall": float("nan"),
                "aff_f1": float("nan"),
                "r_auc_roc": float("nan"),
                "r_auc_pr": float("nan"),
                "vus_roc": float("nan"),
                "vus_pr": float("nan"),
                "vus_window": float("nan"),
            }

        labels = np.asarray(labels, dtype=np.int32)
        scores = cls._normalize_scores_01(scores)
        vus_window = (
            max(1, int(vus_window))
            if vus_window is not None
            else cls._estimate_vus_window(labels)
        )

        try:
            results = vus_get_metrics(scores, labels, metric="all", slidingWindow=vus_window)
            aff_precision = float(results.get("Affiliation_Precision", float("nan")))
            aff_recall = float(results.get("Affiliation_Recall", float("nan")))
            aff_f1 = 2.0 * aff_precision * aff_recall / max(aff_precision + aff_recall, 1e-12)
            return {
                "aff_precision": aff_precision,
                "aff_recall": aff_recall,
                "aff_f1": float(aff_f1),
                "r_auc_roc": float(results.get("R_AUC_ROC", float("nan"))),
                "r_auc_pr": float(results.get("R_AUC_PR", float("nan"))),
                "vus_roc": float(results.get("VUS_ROC", float("nan"))),
                "vus_pr": float(results.get("VUS_PR", float("nan"))),
                "vus_window": float(vus_window),
            }
        except Exception as exc:
            print(f"[Metrics] VUS metrics computation failed: {exc}")
            return {
                "aff_precision": float("nan"),
                "aff_recall": float("nan"),
                "aff_f1": float("nan"),
                "r_auc_roc": float("nan"),
                "r_auc_pr": float("nan"),
                "vus_roc": float("nan"),
                "vus_pr": float("nan"),
                "vus_window": float("nan"),
            }

    @classmethod
    def _compute_pa_best_metrics(cls, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
        labels = labels.astype(np.float32)
        segments = cls._find_anomaly_segments(labels)
        if not segments:
            return {
                "pa_best_f1": float("nan"),
                "pa_best_threshold": 0.5,
                "pa_precision_at_best_f1": float("nan"),
                "pa_recall_at_best_f1": float("nan"),
                "pa_positive_ratio": float("nan"),
                "pa_best_precision": float("nan"),
                "pa_best_recall": float("nan"),
            }

        total_anomaly_points = float(labels.sum())
        normal_scores = scores[labels == 0].astype(np.float64)
        normal_sorted = np.sort(normal_scores)

        seg_max = np.asarray([float(scores[start:end].max()) for start, end in segments], dtype=np.float64)
        seg_len = np.asarray([float(end - start) for start, end in segments], dtype=np.float64)

        candidate_thresholds = np.unique(scores.astype(np.float64))
        best = {
            "pa_best_f1": -1.0,
            "pa_best_threshold": 0.5,
            "pa_precision_at_best_f1": 0.0,
            "pa_recall_at_best_f1": 0.0,
            "pa_positive_ratio": 0.0,
        }

        total_points = float(len(labels))
        for threshold in candidate_thresholds:
            normal_idx = np.searchsorted(normal_sorted, threshold, side="left")
            fp = float(normal_sorted.size - normal_idx)

            # Point-adjust: if any point in a segment is predicted positive, count the whole segment as TP.
            tp = float(seg_len[seg_max >= threshold].sum())

            precision = tp / max(tp + fp, 1e-12)
            recall = tp / max(total_anomaly_points, 1e-12)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            positive_ratio = (tp + fp) / max(total_points, 1e-12)

            if f1 > best["pa_best_f1"]:
                best = {
                    "pa_best_f1": float(f1),
                    "pa_best_threshold": float(threshold),
                    "pa_precision_at_best_f1": float(precision),
                    "pa_recall_at_best_f1": float(recall),
                    "pa_positive_ratio": float(positive_ratio),
                    "pa_best_precision": float(precision),
                    "pa_best_recall": float(recall),
                }
        return best

    @classmethod
    def _compute_metrics(
        cls,
        labels: np.ndarray,
        scores: np.ndarray,
        vus_window: Optional[int] = None,
    ) -> dict[str, float]:
        labels = labels.astype(np.float32)
        if HAS_SKLEARN_METRICS:
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            roc_auc = float(roc_auc_score(labels, scores))
            pr_auc = float(average_precision_score(labels, scores))
        else:
            precision, recall, thresholds = cls._precision_recall_curve(labels, scores)
            roc_auc = cls._roc_auc(labels, scores)
            pr_auc = cls._average_precision(precision, recall)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
        best_idx = int(np.nanargmax(f1))
        if thresholds.size == 0:
            best_threshold = 0.5
        elif best_idx < thresholds.size:
            best_threshold = float(thresholds[best_idx])
        else:
            best_threshold = float(thresholds[-1])
        pred = (scores >= best_threshold).astype(np.float32)
        point_metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_f1": float(np.nanmax(f1)),
            "best_threshold": best_threshold,
            "precision_at_best_f1": float(precision[best_idx]),
            "recall_at_best_f1": float(recall[best_idx]),
            "positive_ratio": float(labels.mean()),
            "predicted_positive_ratio": float(pred.mean()),
            "point_best_f1": float(np.nanmax(f1)),
            "point_best_threshold": best_threshold,
            "point_precision_at_best_f1": float(precision[best_idx]),
            "point_recall_at_best_f1": float(recall[best_idx]),
            "point_positive_ratio": float(labels.mean()),
            "point_predicted_positive_ratio": float(pred.mean()),
            "point_best_precision": float(precision[best_idx]),
            "point_best_recall": float(recall[best_idx]),
        }
        point_metrics.update(cls._compute_pa_best_metrics(labels, scores))
        point_metrics.update(cls._compute_event_best_metrics(labels, scores))
        point_metrics.update(cls._compute_vus_metrics(labels, scores, vus_window=vus_window))
        return point_metrics

    @staticmethod
    def _aggregate_sequence_scores(
        scores: np.ndarray,
        sequence_ranges: np.ndarray,
        aggregation: str,
        observed_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=np.float64)
        ranges = np.asarray(sequence_ranges, dtype=np.int64)
        if observed_mask is None:
            observed = np.ones_like(scores, dtype=bool)
        else:
            observed = np.asarray(observed_mask, dtype=bool)
            if observed.shape != scores.shape:
                raise ValueError("observed_mask must have the same shape as scores.")
        out = np.full(len(ranges), np.nan, dtype=np.float64)
        covered = np.zeros(len(ranges), dtype=bool)
        coverage_ratio = np.zeros(len(ranges), dtype=np.float64)
        for idx, (start, end) in enumerate(ranges):
            start_idx = int(start)
            end_idx = int(end)
            seq_mask = observed[start_idx:end_idx]
            if seq_mask.size == 0:
                continue
            coverage_ratio[idx] = float(seq_mask.mean())
            if not np.any(seq_mask):
                continue
            covered[idx] = True
            seq_scores = scores[start_idx:end_idx][seq_mask]
            if seq_scores.size == 0:
                continue
            if aggregation == "max":
                out[idx] = float(np.max(seq_scores))
            elif aggregation == "top5_mean":
                k = max(1, int(np.ceil(seq_scores.size * 0.05)))
                topk = np.partition(seq_scores, -k)[-k:]
                out[idx] = float(np.mean(topk))
            else:
                out[idx] = float(np.percentile(seq_scores, 95))
        return out, covered, coverage_ratio

    @classmethod
    def _compute_sequence_metrics(cls, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
        labels = np.asarray(labels, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float64)
        valid = np.isfinite(scores)
        labels = labels[valid]
        scores = scores[valid]
        label_positive_ratio = float(labels.mean()) if labels.size else float("nan")
        if labels.size == 0:
            return {
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "best_f1": float("nan"),
                "best_threshold": float("nan"),
                "precision_at_best_f1": float("nan"),
                "recall_at_best_f1": float("nan"),
                "positive_ratio": label_positive_ratio,
                "predicted_positive_ratio": float("nan"),
                "n_sequences": 0,
                "n_positive_sequences": 0,
                "n_negative_sequences": 0,
            }
        if np.unique(labels).size < 2:
            return {
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "best_f1": float("nan"),
                "best_threshold": float("nan"),
                "precision_at_best_f1": float("nan"),
                "recall_at_best_f1": float("nan"),
                "positive_ratio": label_positive_ratio,
                "predicted_positive_ratio": float("nan"),
                "n_sequences": int(labels.size),
                "n_positive_sequences": int(labels.sum()),
                "n_negative_sequences": int(labels.size - labels.sum()),
            }
        if HAS_SKLEARN_METRICS:
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            roc_auc = float(roc_auc_score(labels, scores))
            pr_auc = float(average_precision_score(labels, scores))
        else:
            precision, recall, thresholds = cls._precision_recall_curve(labels, scores)
            roc_auc = cls._roc_auc(labels, scores)
            pr_auc = cls._average_precision(precision, recall)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
        best_idx = int(np.nanargmax(f1))
        if thresholds.size == 0:
            best_threshold = 0.5
        elif best_idx < thresholds.size:
            best_threshold = float(thresholds[best_idx])
        else:
            best_threshold = float(thresholds[-1])
        pred = (scores >= best_threshold).astype(np.float32)
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_f1": float(np.nanmax(f1)),
            "best_threshold": best_threshold,
            "precision_at_best_f1": float(precision[best_idx]),
            "recall_at_best_f1": float(recall[best_idx]),
            "positive_ratio": label_positive_ratio,
            "predicted_positive_ratio": float(pred.mean()),
            "n_sequences": int(labels.size),
            "n_positive_sequences": int(labels.sum()),
            "n_negative_sequences": int(labels.size - labels.sum()),
        }
