from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any

import torch


@dataclass
class CoReMADConfig:
    dataset: str = "MSL"
    data_root: str = "./dataset/anomaly_detect"
    artifact_root: str = "./artifacts"
    experiment_name: str = "coremad_default"

    seq_len: int = 128
    train_stride: int = 1
    test_stride: int = 1
    memory_build_stride: int = 1
    val_ratio: float = 0.15
    val_gap: bool = True
    val_min_train_windows: int = 50
    val_split_mode: str = "tail"
    batch_size: int = 128
    train_batch_size: int | None = None
    val_batch_size: int | None = None
    memory_batch_size: int | None = None
    test_batch_size: int | None = None
    num_workers: int = 8
    max_train_windows: int = 0
    max_test_windows: int = 0
    sequence_score_aggregation: str = "p95"

    n_channels: int = 55
    stsd_hidden: int = 64
    stsd_lowpass_center: float = 0.25

    d_state: int = 128
    state_hidden: int = 128
    patch_sizes: list[int] = field(default_factory=lambda: [8, 32])
    d_trunk: int = 128
    d_z: int = 128
    context_k: int = 2
    use_stsd_decomposition: bool = True
    use_channel_modulation: bool = True

    mask_ratio: float = 0.25
    n_mask_groups: int = 4
    completion_n_heads: int = 4
    completion_n_layers: int = 2
    completion_dropout: float = 0.1
    use_completion_head: bool = True
    use_completion_self_cleaning: bool = True
    use_completion_score_fusion: bool = True
    evaluation_score_key: str = "cdf_mean"

    stage_a_epochs: int = 100
    early_stop_patience: int = 15
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    scheduler_eta_min_ratio: float = 0.01
    lambda_pred: float = 0.5
    lambda_smooth: float = 0.01

    top_M: int = 50
    top_K: int = 20
    knn_k: int = 5
    clean_ratio: float = 0.02
    state_novelty_k: int = 10
    state_prototype_count: int = 0
    prototype_top_p: int = 3
    prototype_candidate_cap: int = 512
    soft_candidate_tau: float = 1.0
    support_score_tau: float = 1.0
    support_score_eps: float = 1e-8
    include_soft_support_in_fusion: bool = False
    use_prototype_support: bool = True
    use_two_level_retrieval: bool = True
    use_context_key_retrieval: bool = True
    use_faiss: bool = True
    faiss_use_gpu: bool = True
    faiss_exact_threshold: int = 100000
    faiss_ivf_nprobe: int = 16
    coreset_keep_ratio: float = 1.0
    coreset_max_patches_per_scale: int = 200000
    coreset_fps_threshold: int = 100000
    coreset_candidate_pool: int = 16384
    coverage_threshold_quantile: float = 0.9

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    resume: bool = False

    def __post_init__(self) -> None:
        self.dataset = str(self.dataset).strip()
        self.batch_size = int(self.batch_size)
        self.train_batch_size = self.batch_size if self.train_batch_size is None else int(self.train_batch_size)
        self.val_batch_size = self.batch_size if self.val_batch_size is None else int(self.val_batch_size)
        self.memory_batch_size = self.batch_size if self.memory_batch_size is None else int(self.memory_batch_size)
        self.test_batch_size = self.batch_size if self.test_batch_size is None else int(self.test_batch_size)
        # Keep the legacy shared field as an alias to the Stage-A train batch size.
        self.batch_size = int(self.train_batch_size)
        self.patch_sizes = [int(size) for size in self.patch_sizes]
        for name in ("train_batch_size", "val_batch_size", "memory_batch_size", "test_batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        for patch_size in self.patch_sizes:
            if self.seq_len % patch_size != 0:
                raise ValueError(
                    f"seq_len={self.seq_len} must be divisible by patch size {patch_size}."
                )
        if self.val_split_mode not in {"tail", "interleaved"}:
            raise ValueError("val_split_mode must be either 'tail' or 'interleaved'.")
        if self.sequence_score_aggregation not in {"p95", "top5_mean", "max"}:
            raise ValueError("sequence_score_aggregation must be one of {'p95', 'top5_mean', 'max'}.")
        if self.n_mask_groups <= 0:
            raise ValueError("n_mask_groups must be positive.")
        if self.state_prototype_count < 0:
            raise ValueError("state_prototype_count must be non-negative.")
        if self.prototype_top_p <= 0:
            raise ValueError("prototype_top_p must be positive.")
        if self.prototype_candidate_cap < 0:
            raise ValueError("prototype_candidate_cap must be non-negative.")
        if self.soft_candidate_tau <= 0.0:
            raise ValueError("soft_candidate_tau must be positive.")
        if self.support_score_tau <= 0.0:
            raise ValueError("support_score_tau must be positive.")
        if self.support_score_eps <= 0.0:
            raise ValueError("support_score_eps must be positive.")
        if self.evaluation_score_key not in {
            "cdf_max",
            "cdf_mean",
            "cdf_mean_soft_support",
            "cdf_softmax",
            "raw_max",
            "zscore_mean",
        }:
            raise ValueError(
                "evaluation_score_key must be one of "
                "{'cdf_max', 'cdf_mean', 'cdf_mean_soft_support', 'cdf_softmax', 'raw_max', 'zscore_mean'}."
            )
        if not self.use_completion_head:
            # Completion-dependent downstream paths cannot run without the head itself.
            self.use_completion_self_cleaning = False
            self.use_completion_score_fusion = False

    @property
    def experiment_dir(self) -> Path:
        return Path(self.artifact_root) / self.experiment_name

    @property
    def stage_a_path(self) -> Path:
        return self.experiment_dir / "stage_a.pt"

    @property
    def memory_path(self) -> Path:
        return self.experiment_dir / "memory.pt"

    @property
    def faiss_index_path(self) -> Path:
        return self.experiment_dir / "faiss_state.index"

    @property
    def cdf_fusion_path(self) -> Path:
        return self.experiment_dir / "cdf_fusion"

    @property
    def zscore_fusion_path(self) -> Path:
        return self.experiment_dir / "zscore_fusion"

    @property
    def stage_a_last_path(self) -> Path:
        return self.experiment_dir / "stage_a_last.pt"

    @property
    def config_path(self) -> Path:
        return self.experiment_dir / "config.json"

    @property
    def memory_meta_path(self) -> Path:
        return self.experiment_dir / "memory_meta.json"

    def scale_weights(self) -> list[float]:
        weights = [1.0 / float(size) for size in self.patch_sizes]
        total = sum(weights)
        return [weight / total for weight in weights]

    def completion_score_names(self) -> list[str]:
        if not self.use_completion_head or not self.use_completion_score_fusion:
            return []
        return [f"completion_scale{size}" for size in self.patch_sizes]

    def retrieval_score_names(self) -> list[str]:
        names = ["knn_distance", "state_novelty"]
        if self.include_soft_support_in_fusion and self.use_prototype_support:
            names.insert(1, "soft_support_score")
        return names

    def fusion_score_names(self) -> list[str]:
        return [*self.retrieval_score_names(), *self.completion_score_names()]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CoReMADConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid_keys = {item.name for item in dataclass_fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)
