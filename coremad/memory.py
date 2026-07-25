from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

try:
    from sklearn.cluster import MiniBatchKMeans

    HAS_SKLEARN_CLUSTER = True
except Exception:
    MiniBatchKMeans = None
    HAS_SKLEARN_CLUSTER = False

from .config import CoReMADConfig
from .faiss_index import ContextSearcher, StateIndex
from .model import CoReMADModel


@dataclass
class ScaleMemory:
    z: torch.Tensor
    c: torch.Tensor
    window_ids: torch.Tensor
    raw_starts: Optional[torch.Tensor] = None
    coverage_threshold: float = float("inf")

    def __post_init__(self) -> None:
        if self.raw_starts is None:
            self.raw_starts = torch.full(
                (self.window_ids.numel(),),
                -1,
                dtype=torch.long,
            )
        self.raw_starts = self.raw_starts.long().cpu().contiguous()
        if self.raw_starts.numel() != self.window_ids.numel():
            raise ValueError(
                "ScaleMemory raw_starts must align with window_ids: "
                f"raw_starts={self.raw_starts.numel()} window_ids={self.window_ids.numel()}"
            )
        self._lookup = self._build_lookup()

    def _build_lookup(self) -> dict[int, tuple[int, int]]:
        lookup: dict[int, tuple[int, int]] = {}
        if self.window_ids.numel() == 0:
            return lookup
        current_id = int(self.window_ids[0].item())
        start = 0
        count = 1
        for idx in range(1, self.window_ids.numel()):
            window_id = int(self.window_ids[idx].item())
            if window_id == current_id:
                count += 1
                continue
            lookup[current_id] = (start, count)
            current_id = window_id
            start = idx
            count = 1
        lookup[current_id] = (start, count)
        return lookup

    def candidate_indices(self, window_ids: np.ndarray | torch.Tensor) -> Optional[torch.Tensor]:
        if isinstance(window_ids, torch.Tensor):
            window_id_list = window_ids.tolist()
        else:
            window_id_list = np.asarray(window_ids).tolist()
        chunks = []
        for window_id in window_id_list:
            span = self._lookup.get(int(window_id))
            if span is None:
                continue
            start, count = span
            chunks.append(torch.arange(start, start + count, dtype=torch.long))
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)


@dataclass
class _ScaleDiskCache:
    patch_size: int
    raw_count: int
    z_mm: Optional[np.memmap]
    c_mm: Optional[np.memmap]
    window_ids_mm: Optional[np.memmap]
    raw_starts_mm: Optional[np.memmap]
    comp_mm: Optional[np.memmap]
    cursor: int = 0

    def append(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        window_ids: torch.Tensor,
        raw_starts: torch.Tensor,
        comp: Optional[torch.Tensor],
    ) -> None:
        if (
            self.z_mm is None
            or self.c_mm is None
            or self.window_ids_mm is None
            or self.raw_starts_mm is None
        ):
            raise RuntimeError("Scale disk cache has already been closed.")
        rows = int(z.size(0))
        start = self.cursor
        end = start + rows
        if end > self.raw_count:
            raise RuntimeError(
                f"Scale disk cache overflow for patch_size={self.patch_size}: "
                f"cursor={self.cursor}, rows={rows}, raw_count={self.raw_count}"
            )
        self.z_mm[start:end] = z.numpy()
        self.c_mm[start:end] = c.numpy()
        self.window_ids_mm[start:end] = window_ids.numpy()
        self.raw_starts_mm[start:end] = raw_starts.numpy()
        if self.comp_mm is not None:
            if comp is None:
                raise RuntimeError("Missing completion scores for a completion-enabled scale cache.")
            self.comp_mm[start:end] = comp.numpy()
        self.cursor = end

    def finalize(self) -> None:
        if self.cursor != self.raw_count:
            raise RuntimeError(
                f"Scale disk cache size mismatch for patch_size={self.patch_size}: "
                f"cursor={self.cursor}, raw_count={self.raw_count}"
            )
        if self.z_mm is not None:
            self.z_mm.flush()
        if self.c_mm is not None:
            self.c_mm.flush()
        if self.window_ids_mm is not None:
            self.window_ids_mm.flush()
        if self.raw_starts_mm is not None:
            self.raw_starts_mm.flush()
        if self.comp_mm is not None:
            self.comp_mm.flush()

    def close(self) -> None:
        for name in ("z_mm", "c_mm", "window_ids_mm", "raw_starts_mm", "comp_mm"):
            arr = getattr(self, name)
            if arr is None:
                continue
            arr.flush()
            mmap_obj = getattr(arr, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            setattr(self, name, None)


class MemoryBank:
    def __init__(
        self,
        config: CoReMADConfig,
        state_bank: torch.Tensor,
        scales: list[ScaleMemory],
        state_window_starts: Optional[torch.Tensor] = None,
        prototype_centers: Optional[torch.Tensor] = None,
        prototype_labels: Optional[torch.Tensor] = None,
        prototype_members: Optional[list[torch.Tensor]] = None,
        build_stats: Optional[list[dict[str, Any]]] = None,
        build_index: bool = True,
    ):
        self.config = config
        self.state_bank = state_bank.float().cpu().contiguous()
        self.scales = scales
        if state_window_starts is None:
            state_window_starts = torch.arange(self.state_bank.size(0), dtype=torch.long)
        self.state_window_starts = state_window_starts.long().cpu().contiguous()
        self.build_stats = list(build_stats or [])
        if self.config.use_prototype_support:
            if prototype_centers is None or prototype_labels is None or prototype_members is None:
                prototype_centers, prototype_labels, prototype_members = self._build_state_prototypes(
                    self.state_bank,
                    config,
                )
            self.prototype_centers = prototype_centers.float().cpu().contiguous()
            self.prototype_labels = prototype_labels.long().cpu().contiguous()
            self.prototype_members = [member.long().cpu().contiguous() for member in prototype_members]
        else:
            self.prototype_centers = torch.zeros(0, config.d_state, dtype=torch.float32)
            self.prototype_labels = torch.zeros(0, dtype=torch.long)
            self.prototype_members = []
        self.state_index = StateIndex(
            d_state=config.d_state,
            use_faiss=config.use_faiss,
            use_gpu=config.faiss_use_gpu,
            exact_threshold=config.faiss_exact_threshold,
            ivf_nprobe=config.faiss_ivf_nprobe,
        )
        self.context_searcher = ContextSearcher()
        self._device_tensor_cache: dict[str, dict[str, torch.Tensor]] = {}
        if build_index:
            self.state_index.build(self.state_bank.numpy())

    def _device_cache_for(self, device: torch.device) -> dict[str, torch.Tensor]:
        return self._device_tensor_cache.setdefault(str(device), {})

    def _get_cached_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        cache = self._device_cache_for(device)
        cached = cache.get(name)
        if cached is None:
            cached = tensor.to(device, non_blocking=True)
            cache[name] = cached
        return cached

    @staticmethod
    def _auto_prototype_count(num_windows: int) -> int:
        if num_windows <= 1:
            return 1
        return max(2, min(64, int(np.sqrt(num_windows))))

    @staticmethod
    def _torch_kmeans(
        state_bank: torch.Tensor,
        n_clusters: int,
        seed: int,
        max_iters: int = 25,
        chunk_size: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = state_bank.float().cpu().contiguous()
        num_windows = int(data.size(0))
        if n_clusters >= num_windows:
            centers = data.clone()
            labels = torch.arange(num_windows, dtype=torch.long)
            return centers, labels

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(num_windows, generator=generator)
        centers = data[perm[:n_clusters]].clone()
        labels = torch.zeros(num_windows, dtype=torch.long)

        for iter_idx in range(max_iters):
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(n_clusters, dtype=torch.long)
            cursor = 0
            for start in range(0, num_windows, chunk_size):
                batch = data[start : start + chunk_size]
                dists = torch.cdist(batch, centers, p=2.0).pow(2)
                batch_labels = dists.argmin(dim=1)
                labels[start : start + batch.size(0)] = batch_labels
                new_centers.index_add_(0, batch_labels, batch)
                counts += torch.bincount(batch_labels, minlength=n_clusters)
                cursor += batch.size(0)
            empty = counts == 0
            if empty.any():
                refill_idx = torch.randperm(num_windows, generator=generator)[: int(empty.sum())]
                new_centers[empty] = data[refill_idx[: int(empty.sum())]]
                counts[empty] = 1
            centers_next = new_centers / counts.clamp(min=1).unsqueeze(1).to(new_centers.dtype)
            if torch.allclose(centers, centers_next, atol=1e-4, rtol=1e-4):
                centers = centers_next
                break
            centers = centers_next

        final_labels = torch.zeros(num_windows, dtype=torch.long)
        for start in range(0, num_windows, chunk_size):
            batch = data[start : start + chunk_size]
            dists = torch.cdist(batch, centers, p=2.0).pow(2)
            final_labels[start : start + batch.size(0)] = dists.argmin(dim=1)
        return centers, final_labels

    @classmethod
    def _build_state_prototypes(
        cls,
        state_bank: torch.Tensor,
        config: CoReMADConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        num_windows = int(state_bank.size(0))
        if num_windows == 0:
            return (
                torch.zeros(0, config.d_state, dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
                [],
            )

        n_clusters = int(config.state_prototype_count)
        if n_clusters <= 0:
            n_clusters = cls._auto_prototype_count(num_windows)
        n_clusters = max(1, min(num_windows, n_clusters))

        if n_clusters == 1:
            labels = torch.zeros(num_windows, dtype=torch.long)
            members = [torch.arange(num_windows, dtype=torch.long)]
            centers = state_bank.mean(dim=0, keepdim=True).float().cpu()
            print(f"[Memory] State prototypes: trivial single prototype covering {num_windows} windows")
            return centers, labels, members

        if HAS_SKLEARN_CLUSTER:
            state_np = state_bank.cpu().numpy().astype(np.float32, copy=False)
            batch_size = min(max(256, n_clusters * 16), max(256, num_windows))
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=config.effective_memory_seed,
                batch_size=batch_size,
                n_init=10,
                reassignment_ratio=0.0,
            )
            labels_np = kmeans.fit_predict(state_np)
            centers = torch.from_numpy(kmeans.cluster_centers_).float().cpu()
            labels = torch.from_numpy(labels_np).long().cpu()
            backend = "sklearn-minibatch-kmeans"
        else:
            centers, labels = cls._torch_kmeans(
                state_bank=state_bank,
                n_clusters=n_clusters,
                seed=config.effective_memory_seed,
            )
            labels_np = labels.numpy()
            backend = "torch-kmeans"
        members = [
            torch.from_numpy(np.where(labels_np == cluster_idx)[0].astype(np.int64))
            for cluster_idx in range(n_clusters)
        ]
        counts = np.asarray([int(member.numel()) for member in members], dtype=np.int64)
        print(
            f"[Memory] State prototypes built: backend={backend}, n_clusters={n_clusters}, "
            f"size_min={int(counts.min())}, size_median={float(np.median(counts)):.1f}, "
            f"size_max={int(counts.max())}"
        )
        return centers, labels, members

    @staticmethod
    def _create_scale_disk_cache(
        temp_dir: Path,
        scale_idx: int,
        patch_size: int,
        raw_count: int,
        d_z: int,
        with_completion_scores: bool,
    ) -> _ScaleDiskCache:
        if raw_count <= 0:
            raise ValueError(f"raw_count must be positive for patch_size={patch_size}, got {raw_count}")
        prefix = f"scale_{scale_idx}_p{patch_size}"
        z_mm = np.memmap(temp_dir / f"{prefix}_z.dat", mode="w+", dtype=np.float32, shape=(raw_count, d_z))
        c_mm = np.memmap(temp_dir / f"{prefix}_c.dat", mode="w+", dtype=np.float32, shape=(raw_count, d_z))
        window_ids_mm = np.memmap(temp_dir / f"{prefix}_window_ids.dat", mode="w+", dtype=np.int64, shape=(raw_count,))
        raw_starts_mm = np.memmap(temp_dir / f"{prefix}_raw_starts.dat", mode="w+", dtype=np.int64, shape=(raw_count,))
        comp_mm = None
        if with_completion_scores:
            comp_mm = np.memmap(temp_dir / f"{prefix}_comp.dat", mode="w+", dtype=np.float32, shape=(raw_count,))
        return _ScaleDiskCache(
            patch_size=patch_size,
            raw_count=raw_count,
            z_mm=z_mm,
            c_mm=c_mm,
            window_ids_mm=window_ids_mm,
            raw_starts_mm=raw_starts_mm,
            comp_mm=comp_mm,
        )

    @staticmethod
    def _coreset_keep(total: int, config: CoReMADConfig) -> int:
        if total <= 0:
            return 0
        keep = int(total * config.coreset_keep_ratio)
        if config.coreset_keep_ratio <= 0.0:
            keep = total
        if config.coreset_max_patches_per_scale > 0:
            keep = min(keep if keep > 0 else total, config.coreset_max_patches_per_scale)
        return max(1, min(total, keep if keep > 0 else total))

    @staticmethod
    def _finalize_stratified_sample(
        selected: np.ndarray,
        total: int,
        keep: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if len(selected) > keep:
            selected = rng.choice(selected, keep, replace=False)
        elif len(selected) < keep:
            remaining = np.setdiff1d(np.arange(total, dtype=np.int64), selected, assume_unique=False)
            extra = rng.choice(remaining, keep - len(selected), replace=False)
            selected = np.concatenate([selected, extra], axis=0)
        return selected.astype(np.int64, copy=False)

    @staticmethod
    def _stratified_random_sampling_sorted_np(
        ids: np.ndarray,
        keep: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        starts = np.flatnonzero(np.concatenate(([True], ids[1:] != ids[:-1])))
        counts = np.diff(np.concatenate((starts, np.asarray([ids.size], dtype=np.int64))))
        total = int(ids.size)
        selected_chunks = []
        for start, count in zip(starts.tolist(), counts.tolist()):
            quota = max(1, int(round(keep * count / total)))
            quota = min(quota, count)
            chosen = rng.choice(count, quota, replace=False).astype(np.int64, copy=False) + int(start)
            selected_chunks.append(chosen)
        if not selected_chunks:
            return np.zeros(0, dtype=np.int64)
        selected = np.concatenate(selected_chunks, axis=0)
        return MemoryBank._finalize_stratified_sample(selected, total, keep, rng)

    @staticmethod
    def _stratified_random_sampling_unsorted_np(
        ids: np.ndarray,
        keep: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        unique_windows, counts = np.unique(ids, return_counts=True)
        total = len(ids)
        selected = []
        for window_id, count in zip(unique_windows, counts):
            w_indices = np.where(ids == window_id)[0]
            quota = max(1, int(round(keep * len(w_indices) / total)))
            quota = min(quota, len(w_indices))
            chosen = rng.choice(w_indices, quota, replace=False)
            selected.extend(chosen.tolist())
        selected_np = np.asarray(selected, dtype=np.int64)
        return MemoryBank._finalize_stratified_sample(selected_np, total, keep, rng)

    @staticmethod
    def _stratified_random_sampling_np(
        window_ids: np.ndarray | torch.Tensor,
        keep: int,
        seed: int,
    ) -> np.ndarray:
        if isinstance(window_ids, torch.Tensor):
            ids = window_ids.cpu().numpy()
        else:
            ids = np.asarray(window_ids)
        ids = ids.astype(np.int64, copy=False)
        if ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        rng = np.random.RandomState(seed)
        if ids.size == 1 or np.all(ids[1:] >= ids[:-1]):
            return MemoryBank._stratified_random_sampling_sorted_np(ids, keep, rng)
        return MemoryBank._stratified_random_sampling_unsorted_np(ids, keep, rng)

    @classmethod
    def _materialize_scale_from_disk(
        cls,
        scale_cache: _ScaleDiskCache,
        config: CoReMADConfig,
    ) -> tuple[ScaleMemory, int, int, int, float, str]:
        if (
            scale_cache.z_mm is None
            or scale_cache.c_mm is None
            or scale_cache.window_ids_mm is None
            or scale_cache.raw_starts_mm is None
        ):
            raise RuntimeError("Scale disk cache is not available for materialization.")
        raw_count = int(scale_cache.raw_count)
        clean_mask: Optional[np.ndarray] = None
        clean_threshold = float("inf")
        if scale_cache.comp_mm is not None:
            comp_tensor = torch.from_numpy(np.array(scale_cache.comp_mm, copy=True))
            clean_threshold = cls._self_clean_threshold(comp_tensor, config.clean_ratio)
            clean_mask = np.asarray(scale_cache.comp_mm <= clean_threshold, dtype=bool)

        clean_count = int(clean_mask.sum()) if clean_mask is not None else raw_count
        keep = cls._coreset_keep(clean_count, config)
        coreset_mode = cls._coreset_mode(clean_count, config)

        cleaned_raw_indices: Optional[np.ndarray]
        if clean_mask is None:
            cleaned_raw_indices = None
        else:
            cleaned_raw_indices = np.flatnonzero(clean_mask).astype(np.int64, copy=False)

        if keep >= clean_count:
            raw_keep_indices = (
                cleaned_raw_indices
                if cleaned_raw_indices is not None
                else np.arange(raw_count, dtype=np.int64)
            )
        elif coreset_mode == "stratified_random":
            clean_window_ids = (
                np.array(scale_cache.window_ids_mm[cleaned_raw_indices], copy=False)
                if cleaned_raw_indices is not None
                else np.asarray(scale_cache.window_ids_mm)
            )
            selected_clean_indices = cls._stratified_random_sampling_np(
                clean_window_ids,
                keep,
                config.effective_memory_seed,
            )
            raw_keep_indices = (
                cleaned_raw_indices[selected_clean_indices]
                if cleaned_raw_indices is not None
                else selected_clean_indices
            )
        else:
            base_indices = (
                cleaned_raw_indices
                if cleaned_raw_indices is not None
                else np.arange(raw_count, dtype=np.int64)
            )
            base_c = torch.from_numpy(np.array(scale_cache.c_mm[base_indices], copy=True)).float()
            base_window_ids = torch.from_numpy(
                np.array(scale_cache.window_ids_mm[base_indices], copy=True)
            ).long()
            selected_clean_indices = cls._select_indices(
                base_c,
                base_window_ids,
                keep,
                config,
            )
            raw_keep_indices = base_indices[selected_clean_indices.cpu().numpy()]

        cls._write_scale_audit(
            scale_cache=scale_cache,
            clean_mask=clean_mask,
            raw_keep_indices=np.asarray(raw_keep_indices, dtype=np.int64),
            clean_threshold=clean_threshold,
            config=config,
        )

        z = torch.from_numpy(np.array(scale_cache.z_mm[raw_keep_indices], copy=True)).float()
        c = torch.from_numpy(np.array(scale_cache.c_mm[raw_keep_indices], copy=True)).float()
        window_ids = torch.from_numpy(np.array(scale_cache.window_ids_mm[raw_keep_indices], copy=True)).long()
        raw_starts = torch.from_numpy(np.array(scale_cache.raw_starts_mm[raw_keep_indices], copy=True)).long()
        order = torch.argsort(window_ids)
        z = z[order].float().contiguous()
        c = c[order].float().contiguous()
        window_ids = window_ids[order].long().contiguous()
        raw_starts = raw_starts[order].long().contiguous()
        scale_memory = ScaleMemory(
            z=z,
            c=c,
            window_ids=window_ids,
            raw_starts=raw_starts,
        )
        return (
            scale_memory,
            raw_count,
            clean_count,
            int(scale_memory.z.size(0)),
            clean_threshold,
            coreset_mode,
        )

    @staticmethod
    def _write_scale_audit(
        scale_cache: _ScaleDiskCache,
        clean_mask: Optional[np.ndarray],
        raw_keep_indices: np.ndarray,
        clean_threshold: float,
        config: CoReMADConfig,
    ) -> None:
        if config.memory_audit_mode != "full":
            return
        if scale_cache.window_ids_mm is None or scale_cache.raw_starts_mm is None:
            raise RuntimeError("Cannot write memory audit after scale cache closure.")
        raw_count = int(scale_cache.raw_count)
        kept_after_purification = (
            np.ones(raw_count, dtype=bool)
            if clean_mask is None
            else np.asarray(clean_mask, dtype=bool)
        )
        kept_after_coreset = np.zeros(raw_count, dtype=bool)
        kept_after_coreset[np.asarray(raw_keep_indices, dtype=np.int64)] = True
        completion_score = (
            np.full(raw_count, np.nan, dtype=np.float32)
            if scale_cache.comp_mm is None
            else np.asarray(scale_cache.comp_mm, dtype=np.float32)
        )
        config.memory_audit_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.memory_audit_dir / f"memory_audit_scale{scale_cache.patch_size}.npz",
            patch_size=np.asarray(scale_cache.patch_size, dtype=np.int64),
            raw_start=np.asarray(scale_cache.raw_starts_mm, dtype=np.int64),
            window_id=np.asarray(scale_cache.window_ids_mm, dtype=np.int64),
            completion_score=completion_score,
            kept_after_purification=kept_after_purification,
            kept_after_coreset=kept_after_coreset,
            clean_threshold=np.asarray(clean_threshold, dtype=np.float64),
        )

    @staticmethod
    def _write_array_audit(
        patch_size: int,
        raw_starts: torch.Tensor,
        window_ids: torch.Tensor,
        completion_scores: torch.Tensor,
        clean_mask: torch.Tensor,
        raw_keep_indices: torch.Tensor,
        clean_threshold: float,
        config: CoReMADConfig,
    ) -> None:
        if config.memory_audit_mode != "full":
            return
        kept_after_coreset = torch.zeros(raw_starts.numel(), dtype=torch.bool)
        kept_after_coreset[raw_keep_indices.long()] = True
        config.memory_audit_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.memory_audit_dir / f"memory_audit_scale{int(patch_size)}.npz",
            patch_size=np.asarray(int(patch_size), dtype=np.int64),
            raw_start=raw_starts.cpu().numpy().astype(np.int64, copy=False),
            window_id=window_ids.cpu().numpy().astype(np.int64, copy=False),
            completion_score=completion_scores.cpu().numpy().astype(np.float32, copy=False),
            kept_after_purification=clean_mask.cpu().numpy().astype(bool, copy=False),
            kept_after_coreset=kept_after_coreset.cpu().numpy().astype(bool, copy=False),
            clean_threshold=np.asarray(clean_threshold, dtype=np.float64),
        )

    def _write_state_audit(self) -> None:
        if self.config.memory_audit_mode != "full":
            return
        n_windows = int(self.state_bank.size(0))
        if self.prototype_centers.numel() > 0 and self.prototype_labels.numel() == n_windows:
            prototype_ids = self.prototype_labels.long()
            assigned_centers = self.prototype_centers[prototype_ids]
            prototype_distances = torch.linalg.vector_norm(
                self.state_bank - assigned_centers,
                dim=1,
            )
        else:
            prototype_ids = torch.full((n_windows,), -1, dtype=torch.long)
            center = self.state_bank.mean(dim=0, keepdim=True)
            prototype_distances = torch.linalg.vector_norm(
                self.state_bank - center,
                dim=1,
            )
        self.config.memory_audit_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.config.memory_audit_dir / "memory_audit_state.npz",
            window_id=np.arange(n_windows, dtype=np.int64),
            raw_start=self.state_window_starts.cpu().numpy().astype(np.int64, copy=False),
            prototype_id=prototype_ids.cpu().numpy().astype(np.int64, copy=False),
            prototype_distance=prototype_distances.cpu().numpy().astype(
                np.float32, copy=False
            ),
        )

    @classmethod
    def _build_in_memory(
        cls,
        model: CoReMADModel,
        loader,
        config: CoReMADConfig,
        device: torch.device,
    ) -> "MemoryBank":
        model.eval()
        total_windows = len(loader.dataset) if hasattr(loader, "dataset") else None
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        print(
            f"[Memory] Build start: device={device}, windows={total_windows}, "
            f"batches={total_batches}, batch_size={getattr(loader, 'batch_size', 'n/a')}, "
            f"use_faiss={config.use_faiss}, clean_ratio={config.clean_ratio:.4f}, "
            f"coreset_keep_ratio={config.coreset_keep_ratio:.4f}"
        )
        state_chunks = []
        state_start_chunks = []
        z_chunks = [[] for _ in config.patch_sizes]
        c_chunks = [[] for _ in config.patch_sizes]
        wid_chunks = [[] for _ in config.patch_sizes]
        raw_start_chunks = [[] for _ in config.patch_sizes]
        need_completion_scores = config.use_completion_head and config.use_completion_self_cleaning
        comp_chunks = [[] for _ in config.patch_sizes] if need_completion_scores else None
        window_offset = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                x = batch["x"].to(device, non_blocking=True)
                if need_completion_scores:
                    encoded, comp_scores = model.deterministic_completion_scores(x)
                else:
                    encoded = model.encode(x)
                    comp_scores = []
                batch_size = x.size(0)
                state_chunks.append(encoded["state_vec"].detach().cpu())
                batch_starts = batch["start"].detach().cpu().long()
                state_start_chunks.append(batch_starts)
                for scale_idx in range(len(config.patch_sizes)):
                    z = encoded["z"][scale_idx].detach().cpu().reshape(-1, config.d_z)
                    c = encoded["c"][scale_idx].detach().cpu().reshape(-1, config.d_z)
                    n_patches = encoded["z"][scale_idx].size(1)
                    window_ids = torch.arange(window_offset, window_offset + batch_size).repeat_interleave(n_patches)
                    patch_size = int(config.patch_sizes[scale_idx])
                    raw_starts = batch_starts.repeat_interleave(n_patches) + (
                        torch.arange(n_patches, dtype=torch.long).repeat(batch_size) * patch_size
                    )
                    z_chunks[scale_idx].append(z)
                    c_chunks[scale_idx].append(c)
                    wid_chunks[scale_idx].append(window_ids)
                    raw_start_chunks[scale_idx].append(raw_starts)
                    if comp_chunks is not None:
                        comp_chunks[scale_idx].append(comp_scores[scale_idx].detach().cpu().reshape(-1))
                window_offset += batch_size
                if total_batches is not None and (
                    batch_idx == 1
                    or batch_idx == total_batches
                    or batch_idx % max(1, total_batches // 5) == 0
                ):
                    print(
                        f"[Memory] Encoding progress: batch={batch_idx}/{total_batches}, "
                        f"processed_windows={window_offset}"
                    )

        state_bank = torch.cat(state_chunks, dim=0)
        state_window_starts = torch.cat(state_start_chunks, dim=0) if state_start_chunks else torch.zeros(0, dtype=torch.long)
        print(f"[Memory] Encoded state bank: windows={state_bank.size(0)}, d_state={state_bank.size(1)}")
        if config.use_prototype_support:
            prototype_centers, prototype_labels, prototype_members = cls._build_state_prototypes(state_bank, config)
        else:
            prototype_centers = torch.zeros(0, config.d_state, dtype=torch.float32)
            prototype_labels = torch.zeros(0, dtype=torch.long)
            prototype_members = []
            print("[Memory] State prototypes skipped: use_prototype_support=False")
        scales = []
        build_stats: list[dict[str, Any]] = []
        for scale_idx in range(len(config.patch_sizes)):
            z = torch.cat(z_chunks[scale_idx], dim=0)
            c = torch.cat(c_chunks[scale_idx], dim=0)
            window_ids = torch.cat(wid_chunks[scale_idx], dim=0)
            raw_starts = torch.cat(raw_start_chunks[scale_idx], dim=0)
            raw_count = z.size(0)
            patch_size = config.patch_sizes[scale_idx]
            if comp_chunks is not None:
                comp = torch.cat(comp_chunks[scale_idx], dim=0)
                clean_threshold = cls._self_clean_threshold(comp, config.clean_ratio)
                clean_mask = comp <= clean_threshold
            else:
                clean_threshold = float("inf")
                comp = torch.full((raw_count,), float("nan"), dtype=torch.float32)
                clean_mask = torch.ones(raw_count, dtype=torch.bool)
            clean_indices = torch.flatnonzero(clean_mask)
            clean_count = int(clean_indices.numel())
            coreset_mode = cls._coreset_mode(clean_count, config)
            keep = cls._coreset_keep(clean_count, config)
            selected_clean_indices = cls._select_indices(
                c[clean_indices],
                window_ids[clean_indices],
                keep,
                config,
            )
            raw_keep_indices = clean_indices[selected_clean_indices]
            z = z[raw_keep_indices]
            c = c[raw_keep_indices]
            window_ids = window_ids[raw_keep_indices]
            selected_raw_starts = raw_starts[raw_keep_indices]
            coreset_count = int(z.size(0))
            cls._write_array_audit(
                patch_size=patch_size,
                raw_starts=raw_starts,
                window_ids=torch.cat(wid_chunks[scale_idx], dim=0),
                completion_scores=comp,
                clean_mask=clean_mask,
                raw_keep_indices=raw_keep_indices,
                clean_threshold=clean_threshold,
                config=config,
            )
            order = torch.argsort(window_ids)
            z = z[order].float().contiguous()
            c = c[order].float().contiguous()
            window_ids = window_ids[order].long().contiguous()
            selected_raw_starts = selected_raw_starts[order].long().contiguous()
            print(
                f"[Memory] Scale patch_size={patch_size}: raw_patches={raw_count}, "
                f"after_clean={clean_count}, after_coreset={coreset_count}, "
                f"clean_threshold={clean_threshold:.6f}, coreset_mode={coreset_mode}"
            )
            build_stats.append(
                {
                    "patch_size": int(patch_size),
                    "raw_count": int(raw_count),
                    "clean_threshold": float(clean_threshold),
                    "clean_count": int(clean_count),
                    "coreset_mode": str(coreset_mode),
                    "final_count": int(coreset_count),
                }
            )
            scales.append(
                ScaleMemory(
                    z=z,
                    c=c,
                    window_ids=window_ids,
                    raw_starts=selected_raw_starts,
                )
            )
        memory = cls(
            config=config,
            state_bank=state_bank,
            scales=scales,
            state_window_starts=state_window_starts,
            prototype_centers=prototype_centers,
            prototype_labels=prototype_labels,
            prototype_members=prototype_members,
            build_stats=build_stats,
        )
        memory._write_state_audit()
        print(
            f"[Memory] State index ready: backend={memory.state_index.describe()}, "
            f"n_vectors={memory.state_bank.size(0)}, faiss_use_gpu={config.faiss_use_gpu}"
        )
        return memory

    @classmethod
    def build(
        cls,
        model: CoReMADModel,
        loader,
        config: CoReMADConfig,
        device: torch.device,
    ) -> "MemoryBank":
        model.eval()
        total_windows = len(loader.dataset) if hasattr(loader, "dataset") else None
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        print(
            f"[Memory] Build start: device={device}, windows={total_windows}, "
            f"batches={total_batches}, batch_size={getattr(loader, 'batch_size', 'n/a')}, "
            f"use_faiss={config.use_faiss}, clean_ratio={config.clean_ratio:.4f}, "
            f"coreset_keep_ratio={config.coreset_keep_ratio:.4f}"
        )
        if total_windows is None:
            print("[Memory] Build strategy: fallback_in_memory (unknown dataset size)")
            return cls._build_in_memory(model, loader, config, device)

        config.experiment_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="memory_build_", dir=str(config.experiment_dir)))
        print(f"[Memory] Build strategy: exact_chunked_memmap (temp_dir={temp_dir})")
        state_chunks = []
        state_start_chunks = []
        need_completion_scores = config.use_completion_head and config.use_completion_self_cleaning
        window_offset = 0
        scale_caches = [
            cls._create_scale_disk_cache(
                temp_dir=temp_dir,
                scale_idx=scale_idx,
                patch_size=patch_size,
                raw_count=int(total_windows) * (config.seq_len // patch_size),
                d_z=config.d_z,
                with_completion_scores=need_completion_scores,
            )
            for scale_idx, patch_size in enumerate(config.patch_sizes)
        ]

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader, start=1):
                    x = batch["x"].to(device, non_blocking=True)
                    if need_completion_scores:
                        encoded, comp_scores = model.deterministic_completion_scores(x)
                    else:
                        encoded = model.encode(x)
                        comp_scores = []
                    batch_size = x.size(0)
                    state_chunks.append(encoded["state_vec"].detach().cpu())
                    batch_starts = batch["start"].detach().cpu().long()
                    state_start_chunks.append(batch_starts)
                    for scale_idx in range(len(config.patch_sizes)):
                        z = encoded["z"][scale_idx].detach().cpu().reshape(-1, config.d_z).contiguous()
                        c = encoded["c"][scale_idx].detach().cpu().reshape(-1, config.d_z).contiguous()
                        n_patches = encoded["z"][scale_idx].size(1)
                        window_ids = (
                            torch.arange(window_offset, window_offset + batch_size)
                            .repeat_interleave(n_patches)
                            .long()
                        )
                        patch_size = int(config.patch_sizes[scale_idx])
                        raw_starts = batch_starts.repeat_interleave(n_patches) + (
                            torch.arange(n_patches, dtype=torch.long).repeat(batch_size) * patch_size
                        )
                        comp = None
                        if need_completion_scores:
                            comp = comp_scores[scale_idx].detach().cpu().reshape(-1).contiguous()
                        scale_caches[scale_idx].append(
                            z=z,
                            c=c,
                            window_ids=window_ids,
                            raw_starts=raw_starts,
                            comp=comp,
                        )
                    window_offset += batch_size
                    if total_batches is not None and (
                        batch_idx == 1
                        or batch_idx == total_batches
                        or batch_idx % max(1, total_batches // 5) == 0
                    ):
                        print(
                            f"[Memory] Encoding progress: batch={batch_idx}/{total_batches}, "
                            f"processed_windows={window_offset}"
                        )

            for scale_cache in scale_caches:
                scale_cache.finalize()

            state_bank = torch.cat(state_chunks, dim=0)
            state_window_starts = (
                torch.cat(state_start_chunks, dim=0)
                if state_start_chunks
                else torch.zeros(0, dtype=torch.long)
            )
            print(f"[Memory] Encoded state bank: windows={state_bank.size(0)}, d_state={state_bank.size(1)}")
            if config.use_prototype_support:
                prototype_centers, prototype_labels, prototype_members = cls._build_state_prototypes(state_bank, config)
            else:
                prototype_centers = torch.zeros(0, config.d_state, dtype=torch.float32)
                prototype_labels = torch.zeros(0, dtype=torch.long)
                prototype_members = []
                print("[Memory] State prototypes skipped: use_prototype_support=False")

            scales = []
            build_stats: list[dict[str, Any]] = []
            for scale_cache in scale_caches:
                (
                    scale_memory,
                    raw_count,
                    clean_count,
                    coreset_count,
                    clean_threshold,
                    coreset_mode,
                ) = cls._materialize_scale_from_disk(scale_cache, config)
                print(
                    f"[Memory] Scale patch_size={scale_cache.patch_size}: raw_patches={raw_count}, "
                    f"after_clean={clean_count}, after_coreset={coreset_count}, "
                    f"clean_threshold={clean_threshold:.6f}, coreset_mode={coreset_mode}"
                )
                build_stats.append(
                    {
                        "patch_size": int(scale_cache.patch_size),
                        "raw_count": int(raw_count),
                        "clean_threshold": float(clean_threshold),
                        "clean_count": int(clean_count),
                        "coreset_mode": str(coreset_mode),
                        "final_count": int(coreset_count),
                    }
                )
                scales.append(scale_memory)

            memory = cls(
                config=config,
                state_bank=state_bank,
                scales=scales,
                state_window_starts=state_window_starts,
                prototype_centers=prototype_centers,
                prototype_labels=prototype_labels,
                prototype_members=prototype_members,
                build_stats=build_stats,
            )
            memory._write_state_audit()
            print(
                f"[Memory] State index ready: backend={memory.state_index.describe()}, "
                f"n_vectors={memory.state_bank.size(0)}, faiss_use_gpu={config.faiss_use_gpu}"
            )
            return memory
        finally:
            for scale_cache in scale_caches:
                scale_cache.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def sanity_check(self, num_samples: int = 10) -> None:
        if self.state_bank.size(0) == 0:
            return
        sample_size = min(num_samples, self.state_bank.size(0))
        sample_idx = np.random.choice(self.state_bank.size(0), sample_size, replace=False)
        sample_queries = self.state_bank[sample_idx].numpy().astype(np.float32)
        dists, indices = self.state_index.search(sample_queries, min(5, self.state_bank.size(0)))
        top1_dist = dists[:, 0]
        top1_idx = indices[:, 0]
        exact_index_hits = int(np.sum(top1_idx == sample_idx))
        near_zero_mask = top1_dist <= 1e-4
        equivalent_mask = near_zero_mask.copy()

        for row_idx, is_equivalent in enumerate(equivalent_mask):
            if is_equivalent:
                continue
            candidate_ids = torch.from_numpy(indices[row_idx]).long()
            candidate_vecs = self.state_bank[candidate_ids].numpy().astype(np.float32)
            candidate_dists = np.sum((candidate_vecs - sample_queries[row_idx : row_idx + 1]) ** 2, axis=1)
            equivalent_mask[row_idx] = bool(np.min(candidate_dists) <= 1e-4)

        if not np.all(equivalent_mask):
            bad_rows = np.where(~equivalent_mask)[0]
            preview = bad_rows[: min(3, len(bad_rows))]
            debug_rows = []
            for row_idx in preview:
                debug_rows.append(
                    {
                        "query_id": int(sample_idx[row_idx]),
                        "top1_id": int(top1_idx[row_idx]),
                        "top1_dist": float(top1_dist[row_idx]),
                        "top5_ids": indices[row_idx].tolist(),
                        "top5_dists": [float(val) for val in dists[row_idx].tolist()],
                    }
                )
            raise RuntimeError(
                "[Memory] State index sanity check failed: "
                f"{len(bad_rows)}/{sample_size} queries did not retrieve equivalent vectors. "
                f"examples={debug_rows}"
            )
        print(
            f"[Memory] Sanity check passed: {sample_size}/{sample_size} queries "
            f"retrieved equivalent vectors "
            f"(exact_index_hits={exact_index_hits}/{sample_size}, max_top1_dist={top1_dist.max():.2e})"
        )

    @staticmethod
    def _self_clean(
        z: torch.Tensor,
        c: torch.Tensor,
        window_ids: torch.Tensor,
        completion_scores: torch.Tensor,
        clean_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if clean_ratio <= 0.0 or z.numel() == 0:
            return z, c, window_ids
        threshold = torch.quantile(completion_scores, max(0.0, min(1.0, 1.0 - clean_ratio)))
        keep = completion_scores <= threshold
        return z[keep], c[keep], window_ids[keep]

    @staticmethod
    def _self_clean_threshold(completion_scores: torch.Tensor, clean_ratio: float) -> float:
        if clean_ratio <= 0.0 or completion_scores.numel() == 0:
            return float("inf")
        threshold = torch.quantile(completion_scores, max(0.0, min(1.0, 1.0 - clean_ratio)))
        return float(threshold.item())

    @classmethod
    def _coreset(
        cls,
        z: torch.Tensor,
        c: torch.Tensor,
        window_ids: torch.Tensor,
        config: CoReMADConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = z.size(0)
        keep = cls._coreset_keep(total, config)
        if keep >= total:
            return z, c, window_ids
        indices = cls._select_indices(c, window_ids, keep, config)
        return z[indices], c[indices], window_ids[indices]

    @staticmethod
    def _coreset_mode(total: int, config: CoReMADConfig) -> str:
        keep = MemoryBank._coreset_keep(total, config)
        if keep >= total:
            return "identity"
        if total > config.coreset_fps_threshold:
            return "stratified_random"
        return "approx_fps"

    @staticmethod
    def _select_indices(
        features: torch.Tensor,
        window_ids: torch.Tensor,
        keep: int,
        config: CoReMADConfig,
    ) -> torch.Tensor:
        total = features.size(0)
        if keep >= total:
            return torch.arange(total, dtype=torch.long)
        if total > config.coreset_fps_threshold:
            return MemoryBank._stratified_random_sampling(
                window_ids,
                keep,
                config.effective_memory_seed,
            )
        return MemoryBank._approx_farthest_point_sampling(features, keep, config)

    @staticmethod
    def _approx_farthest_point_sampling(
        features: torch.Tensor,
        keep: int,
        config: CoReMADConfig,
    ) -> torch.Tensor:
        total = features.size(0)
        pool_size = min(total, max(keep * 4, config.coreset_candidate_pool))
        pool_indices = torch.randperm(total)[:pool_size]
        pool = features[pool_indices]
        seed = torch.randint(0, pool_size, (1,)).item()
        selected = [seed]
        min_dist = torch.cdist(pool[seed : seed + 1], pool).squeeze(0)
        while len(selected) < keep:
            next_idx = int(torch.argmax(min_dist).item())
            selected.append(next_idx)
            min_dist = torch.minimum(min_dist, torch.cdist(pool[next_idx : next_idx + 1], pool).squeeze(0))
        return pool_indices[torch.tensor(selected, dtype=torch.long)]

    @staticmethod
    def _stratified_random_sampling(window_ids: torch.Tensor, keep: int, seed: int) -> torch.Tensor:
        selected = MemoryBank._stratified_random_sampling_np(window_ids, keep, seed)
        return torch.from_numpy(selected.astype(np.int64, copy=False))

    @staticmethod
    def _compute_memory_score(
        query_z: torch.Tensor,
        neighbor_z: torch.Tensor,
        neighbor_valid: torch.Tensor,
        knn_k: int,
    ) -> torch.Tensor:
        z_dists = torch.cdist(query_z.unsqueeze(-2), neighbor_z, p=2.0).squeeze(-2).pow(2)
        z_dists = z_dists.masked_fill(~neighbor_valid, float("inf"))
        n_valid = neighbor_valid.sum(dim=-1)
        sorted_dists, _ = torch.sort(z_dists, dim=-1)
        actual_k = torch.clamp(n_valid, min=1, max=knn_k)
        mem_scores = torch.zeros_like(actual_k, device=query_z.device, dtype=query_z.dtype)

        for k_val in range(1, knn_k + 1):
            mask = actual_k >= k_val
            if mask.any():
                mem_scores[mask] += sorted_dists[..., k_val - 1][mask]
        mem_scores = mem_scores / actual_k.float().clamp(min=1.0)

        no_neighbor = n_valid == 0
        if no_neighbor.any():
            finite_mask = torch.isfinite(sorted_dists)
            if finite_mask.any():
                high_value = sorted_dists[finite_mask].max() * 2.0
            else:
                high_value = torch.tensor(1e6, device=query_z.device, dtype=query_z.dtype)
            mem_scores[no_neighbor] = high_value
        return mem_scores

    @staticmethod
    def _compute_soft_candidate_weights(
        coarse_distances: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        finite_mask = torch.isfinite(coarse_distances)
        safe_distances = torch.where(finite_mask, coarse_distances, torch.zeros_like(coarse_distances))
        denom = finite_mask.sum(dim=1, keepdim=True).clamp(min=1)
        distance_scale = safe_distances.sum(dim=1, keepdim=True) / denom
        distance_scale = distance_scale.clamp(min=1e-6)
        normalized = coarse_distances / distance_scale
        logits = -normalized / max(float(tau), 1e-6)
        logits = logits.masked_fill(~finite_mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        no_finite = ~finite_mask.any(dim=1)
        if no_finite.any():
            weights[no_finite] = 1.0 / float(coarse_distances.size(1))
        return weights

    def _prototype_soft_gate(
        self,
        state_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.prototype_centers.numel() == 0:
            empty_idx = torch.zeros(state_vec.size(0), 0, dtype=torch.long, device=state_vec.device)
            empty_weight = torch.zeros(state_vec.size(0), 0, dtype=state_vec.dtype, device=state_vec.device)
            empty_dist = torch.zeros(state_vec.size(0), 0, dtype=state_vec.dtype, device=state_vec.device)
            return empty_idx, empty_weight, empty_dist

        centers = self._get_cached_tensor("prototype_centers", self.prototype_centers, state_vec.device)
        proto_dists = torch.cdist(state_vec, centers, p=2.0).pow(2)
        top_p = min(int(self.config.prototype_top_p), int(proto_dists.size(1)))
        top_proto_dists, top_proto_ids = torch.topk(proto_dists, top_p, dim=1, largest=False)
        top_proto_weights = self._compute_soft_candidate_weights(
            top_proto_dists,
            tau=self.config.soft_candidate_tau,
        )
        return top_proto_ids, top_proto_weights, top_proto_dists

    def _collect_prototype_candidates(
        self,
        query_state: torch.Tensor,
        prototype_ids: torch.Tensor,
        prototype_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prototype_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=query_state.dtype, device=query_state.device)

        candidate_windows: list[torch.Tensor] = []
        candidate_weights: list[torch.Tensor] = []
        for proto_id, proto_weight in zip(prototype_ids.tolist(), prototype_weights.tolist()):
            members = self.prototype_members[int(proto_id)]
            if members.numel() == 0:
                continue
            candidate_windows.append(members)
            candidate_weights.append(
                torch.full(
                    (members.numel(),),
                    float(proto_weight),
                    dtype=query_state.dtype,
                    device=query_state.device,
                )
            )
        if not candidate_windows:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=query_state.dtype, device=query_state.device)

        window_ids = torch.cat(candidate_windows, dim=0).long().cpu()
        window_weights = torch.cat(candidate_weights, dim=0)
        cap = int(self.config.prototype_candidate_cap)
        if cap > 0 and window_ids.numel() > cap:
            state_bank = self._get_cached_tensor("state_bank", self.state_bank, query_state.device)
            member_states = state_bank[window_ids.to(query_state.device)]
            state_dists = torch.cdist(query_state.unsqueeze(0), member_states, p=2.0).squeeze(0).pow(2)
            distance_scale = state_dists.mean().clamp(min=1e-6)
            rank_scores = state_dists / distance_scale - torch.log(window_weights.clamp(min=1e-12))
            keep_idx = torch.topk(rank_scores, k=cap, largest=False).indices
            window_ids = window_ids[keep_idx.cpu()]
            window_weights = window_weights[keep_idx]
        return window_ids, window_weights

    @staticmethod
    def _context_topk_neighbors(
        retrieval_query: torch.Tensor,
        candidate_key: torch.Tensor,
        candidate_z: torch.Tensor,
        candidate_window_ids: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_k = min(int(top_k), int(candidate_key.shape[0]))
        if actual_k > 0:
            dists = torch.cdist(retrieval_query, candidate_key, p=2.0)
            topk_dists, topk_idx = torch.topk(dists, actual_k, dim=1, largest=False)
            neighbor_z = candidate_z[topk_idx]
            window_ids = candidate_window_ids[topk_idx]
            neighbor_valid = torch.ones(
                retrieval_query.shape[0],
                actual_k,
                dtype=torch.bool,
                device=retrieval_query.device,
            )
            if actual_k < top_k:
                pad_n = top_k - actual_k
                d_z = int(candidate_z.shape[1])
                neighbor_z = torch.cat(
                    [
                        neighbor_z,
                        torch.zeros(
                            retrieval_query.shape[0],
                            pad_n,
                            d_z,
                            device=retrieval_query.device,
                            dtype=candidate_z.dtype,
                        ),
                    ],
                    dim=1,
                )
                topk_dists = torch.cat(
                    [
                        topk_dists,
                        torch.full(
                            (retrieval_query.shape[0], pad_n),
                            float("inf"),
                            device=retrieval_query.device,
                            dtype=retrieval_query.dtype,
                        ),
                    ],
                    dim=1,
                )
                window_ids = torch.cat(
                    [
                        window_ids,
                        torch.full(
                            (retrieval_query.shape[0], pad_n),
                            -1,
                            device=retrieval_query.device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
                )
                neighbor_valid = torch.cat(
                    [
                        neighbor_valid,
                        torch.zeros(
                            retrieval_query.shape[0],
                            pad_n,
                            device=retrieval_query.device,
                            dtype=torch.bool,
                        ),
                    ],
                    dim=1,
                )
                topk_idx = torch.cat(
                    [
                        topk_idx,
                        torch.zeros(
                            retrieval_query.shape[0],
                            pad_n,
                            device=retrieval_query.device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
                )
        else:
            d_z = int(candidate_z.shape[1])
            neighbor_z = torch.zeros(
                retrieval_query.shape[0],
                top_k,
                d_z,
                device=retrieval_query.device,
                dtype=candidate_z.dtype,
            )
            topk_dists = torch.full(
                (retrieval_query.shape[0], top_k),
                float("inf"),
                device=retrieval_query.device,
                dtype=retrieval_query.dtype,
            )
            window_ids = torch.full(
                (retrieval_query.shape[0], top_k),
                -1,
                device=retrieval_query.device,
                dtype=torch.long,
            )
            neighbor_valid = torch.zeros(
                (retrieval_query.shape[0], top_k),
                device=retrieval_query.device,
                dtype=torch.bool,
            )
            topk_idx = torch.zeros(
                retrieval_query.shape[0],
                top_k,
                device=retrieval_query.device,
                dtype=torch.long,
            )
        return neighbor_z, topk_dists, neighbor_valid, window_ids, topk_idx

    @staticmethod
    def _pack_scale_candidates(
        scale_memory: ScaleMemory,
        candidate_idx_list: list[Optional[torch.Tensor]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(candidate_idx_list)
        d_z = int(scale_memory.z.shape[1])
        lengths = [0 if idx is None else int(idx.numel()) for idx in candidate_idx_list]
        max_len = max(lengths, default=0)
        if max_len == 0:
            zeros = torch.zeros(batch, 0, d_z, device=device, dtype=scale_memory.z.dtype)
            window_ids = torch.zeros(batch, 0, device=device, dtype=torch.long)
            raw_starts = torch.zeros(batch, 0, device=device, dtype=torch.long)
            valid = torch.zeros(batch, 0, device=device, dtype=torch.bool)
            return zeros, zeros, window_ids, raw_starts, valid

        packed_z = torch.zeros(batch, max_len, d_z, dtype=scale_memory.z.dtype)
        packed_c = torch.zeros(batch, max_len, d_z, dtype=scale_memory.c.dtype)
        packed_window_ids = torch.full((batch, max_len), -1, dtype=torch.long)
        packed_raw_starts = torch.full((batch, max_len), -1, dtype=torch.long)
        valid = torch.zeros(batch, max_len, dtype=torch.bool)
        for batch_idx, candidate_idx in enumerate(candidate_idx_list):
            if candidate_idx is None or candidate_idx.numel() == 0:
                continue
            candidate_idx = candidate_idx.long()
            count = int(candidate_idx.numel())
            packed_z[batch_idx, :count] = scale_memory.z[candidate_idx]
            packed_c[batch_idx, :count] = scale_memory.c[candidate_idx]
            packed_window_ids[batch_idx, :count] = scale_memory.window_ids[candidate_idx]
            packed_raw_starts[batch_idx, :count] = scale_memory.raw_starts[candidate_idx]
            valid[batch_idx, :count] = True
        return (
            packed_z.to(device, non_blocking=True),
            packed_c.to(device, non_blocking=True),
            packed_window_ids.to(device, non_blocking=True),
            packed_raw_starts.to(device, non_blocking=True),
            valid.to(device, non_blocking=True),
        )

    @staticmethod
    def _pack_candidate_weights(
        scale_memory: ScaleMemory,
        candidate_idx_list: list[Optional[torch.Tensor]],
        candidate_window_ids_list: list[torch.Tensor],
        candidate_window_weights_list: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        batch = len(candidate_idx_list)
        lengths = [0 if idx is None else int(idx.numel()) for idx in candidate_idx_list]
        max_len = max(lengths, default=0)
        dtype = candidate_window_weights_list[0].dtype if candidate_window_weights_list else torch.float32
        if max_len == 0:
            return torch.zeros(batch, 0, device=device, dtype=dtype)

        packed_weights = torch.zeros(batch, max_len, dtype=dtype)
        for batch_idx, candidate_idx in enumerate(candidate_idx_list):
            if candidate_idx is None or candidate_idx.numel() == 0:
                continue
            count = int(candidate_idx.numel())
            candidate_patch_window_ids = scale_memory.window_ids[candidate_idx.long()]
            packed_weights[batch_idx, :count] = MemoryBank._expand_candidate_weights(
                candidate_patch_window_ids,
                candidate_window_ids_list[batch_idx],
                candidate_window_weights_list[batch_idx].cpu(),
            ).cpu()
        return packed_weights.to(device, non_blocking=True)

    @staticmethod
    def _batched_context_topk_neighbors(
        retrieval_query: torch.Tensor,
        candidate_key: torch.Tensor,
        candidate_z: torch.Tensor,
        candidate_window_ids: torch.Tensor,
        candidate_valid: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, d_z = candidate_z.shape
        max_candidates = int(candidate_key.shape[1])
        if max_candidates == 0:
            neighbor_z = torch.zeros(
                retrieval_query.shape[0],
                retrieval_query.shape[1],
                top_k,
                d_z,
                device=retrieval_query.device,
                dtype=candidate_z.dtype,
            )
            topk_dists = torch.full(
                (retrieval_query.shape[0], retrieval_query.shape[1], top_k),
                float("inf"),
                device=retrieval_query.device,
                dtype=retrieval_query.dtype,
            )
            window_ids = torch.full(
                (retrieval_query.shape[0], retrieval_query.shape[1], top_k),
                -1,
                device=retrieval_query.device,
                dtype=torch.long,
            )
            neighbor_valid_out = torch.zeros(
                (retrieval_query.shape[0], retrieval_query.shape[1], top_k),
                device=retrieval_query.device,
                dtype=torch.bool,
            )
            topk_idx = torch.zeros(
                (retrieval_query.shape[0], retrieval_query.shape[1], top_k),
                device=retrieval_query.device,
                dtype=torch.long,
            )
            return neighbor_z, topk_dists, neighbor_valid_out, window_ids, topk_idx

        actual_k = min(int(top_k), max_candidates)
        # Chunk the candidate axis to avoid materializing a full [B, Q, N] distance
        # tensor for global-retrieval ablations on large memories.
        candidate_chunk_size = 4096
        topk_dists = torch.full(
            (batch, retrieval_query.shape[1], actual_k),
            float("inf"),
            device=retrieval_query.device,
            dtype=retrieval_query.dtype,
        )
        topk_idx = torch.zeros(
            batch,
            retrieval_query.shape[1],
            actual_k,
            device=retrieval_query.device,
            dtype=torch.long,
        )

        for start in range(0, max_candidates, candidate_chunk_size):
            end = min(start + candidate_chunk_size, max_candidates)
            chunk_key = candidate_key[:, start:end, :]
            chunk_valid = candidate_valid[:, start:end]
            chunk_dists = torch.cdist(retrieval_query, chunk_key, p=2.0)
            chunk_dists = chunk_dists.masked_fill(~chunk_valid[:, None, :], float("inf"))

            chunk_k = min(actual_k, end - start)
            chunk_topk_dists, chunk_topk_idx = torch.topk(chunk_dists, chunk_k, dim=2, largest=False)
            chunk_topk_idx = chunk_topk_idx + start

            if chunk_k < actual_k:
                pad_n = actual_k - chunk_k
                chunk_topk_dists = torch.cat(
                    [
                        chunk_topk_dists,
                        torch.full(
                            (batch, retrieval_query.shape[1], pad_n),
                            float("inf"),
                            device=retrieval_query.device,
                            dtype=retrieval_query.dtype,
                        ),
                    ],
                    dim=2,
                )
                chunk_topk_idx = torch.cat(
                    [
                        chunk_topk_idx,
                        torch.zeros(
                            batch,
                            retrieval_query.shape[1],
                            pad_n,
                            device=retrieval_query.device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=2,
                )

            merged_dists = torch.cat([topk_dists, chunk_topk_dists], dim=2)
            merged_idx = torch.cat([topk_idx, chunk_topk_idx], dim=2)
            merged_order = torch.topk(merged_dists, actual_k, dim=2, largest=False).indices
            topk_dists = torch.gather(merged_dists, 2, merged_order)
            topk_idx = torch.gather(merged_idx, 2, merged_order)

        expand_z = candidate_z[:, None, :, :].expand(-1, retrieval_query.shape[1], -1, -1)
        neighbor_z = torch.gather(
            expand_z,
            2,
            topk_idx.unsqueeze(-1).expand(-1, -1, -1, d_z),
        )
        expand_window_ids = candidate_window_ids[:, None, :].expand(-1, retrieval_query.shape[1], -1)
        window_ids = torch.gather(expand_window_ids, 2, topk_idx)
        expand_valid = candidate_valid[:, None, :].expand(-1, retrieval_query.shape[1], -1)
        neighbor_valid_out = torch.gather(expand_valid, 2, topk_idx)
        window_ids = window_ids.masked_fill(~neighbor_valid_out, -1)

        if actual_k < top_k:
            pad_n = top_k - actual_k
            neighbor_z = torch.cat(
                [
                    neighbor_z,
                    torch.zeros(
                        batch,
                        retrieval_query.shape[1],
                        pad_n,
                        d_z,
                        device=retrieval_query.device,
                        dtype=candidate_z.dtype,
                    ),
                ],
                dim=2,
            )
            topk_dists = torch.cat(
                [
                    topk_dists,
                    torch.full(
                        (batch, retrieval_query.shape[1], pad_n),
                        float("inf"),
                        device=retrieval_query.device,
                        dtype=retrieval_query.dtype,
                    ),
                ],
                dim=2,
            )
            window_ids = torch.cat(
                [
                    window_ids,
                    torch.full(
                        (batch, retrieval_query.shape[1], pad_n),
                        -1,
                        device=retrieval_query.device,
                        dtype=torch.long,
                    ),
                ],
                dim=2,
            )
            neighbor_valid_out = torch.cat(
                [
                    neighbor_valid_out,
                    torch.zeros(
                        batch,
                        retrieval_query.shape[1],
                        pad_n,
                        device=retrieval_query.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=2,
            )
            topk_idx = torch.cat(
                [
                    topk_idx,
                    torch.zeros(
                        batch,
                        retrieval_query.shape[1],
                        pad_n,
                        device=retrieval_query.device,
                        dtype=torch.long,
                    ),
                ],
                dim=2,
            )
        return neighbor_z, topk_dists, neighbor_valid_out, window_ids, topk_idx

    @staticmethod
    def _batched_prototype_support_score(
        retrieval_query: torch.Tensor,
        query_z: torch.Tensor,
        candidate_z: torch.Tensor,
        candidate_c: torch.Tensor,
        candidate_window_ids: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_weights: torch.Tensor,
        *,
        use_context_key_retrieval: bool,
        top_k: int,
        tau: float,
        eps: float,
    ) -> torch.Tensor:
        candidate_key = candidate_c if use_context_key_retrieval else candidate_z
        neighbor_z, topk_distances, neighbor_valid, _, topk_idx = (
            MemoryBank._batched_context_topk_neighbors(
                retrieval_query,
                candidate_key,
                candidate_z,
                candidate_window_ids,
                candidate_valid,
                top_k,
            )
        )
        neighbor_valid = neighbor_valid & torch.isfinite(topk_distances)
        expanded_weights = candidate_weights[:, None, :].expand(
            -1,
            retrieval_query.shape[1],
            -1,
        )
        neighbor_weights = torch.gather(expanded_weights, 2, topk_idx)
        return MemoryBank._compute_support_score(
            query_z,
            neighbor_z,
            neighbor_valid,
            neighbor_weights,
            tau=tau,
            eps=eps,
        )

    @staticmethod
    def _expand_candidate_weights(
        expanded_window_ids: torch.Tensor,
        candidate_window_ids: torch.Tensor,
        candidate_window_weights: torch.Tensor,
    ) -> torch.Tensor:
        if expanded_window_ids.numel() == 0 or candidate_window_ids.numel() == 0:
            return torch.zeros(
                expanded_window_ids.shape[0],
                device=expanded_window_ids.device,
                dtype=candidate_window_weights.dtype,
            )
        matches = expanded_window_ids.unsqueeze(-1) == candidate_window_ids.view(1, -1)
        return (
            matches.to(dtype=candidate_window_weights.dtype)
            * candidate_window_weights.view(1, -1)
        ).sum(dim=-1)

    @staticmethod
    def _compute_support_score(
        query_z: torch.Tensor,
        neighbor_z: torch.Tensor,
        neighbor_valid: torch.Tensor,
        neighbor_weights: torch.Tensor,
        tau: float,
        eps: float,
    ) -> torch.Tensor:
        z_dists = torch.cdist(query_z.unsqueeze(-2), neighbor_z, p=2.0).squeeze(-2).pow(2)
        z_dists = z_dists.masked_fill(~neighbor_valid, float("inf"))
        support = neighbor_weights * torch.exp(-z_dists / max(float(tau), 1e-6))
        support = support.masked_fill(~neighbor_valid, 0.0)
        total_support = support.sum(dim=-1)
        support_scores = -torch.log(total_support + float(eps))

        no_neighbor = neighbor_valid.sum(dim=-1) == 0
        if no_neighbor.any():
            finite_mask = torch.isfinite(support_scores)
            if finite_mask.any():
                high_value = support_scores[finite_mask].max() * 2.0
            else:
                high_value = torch.tensor(1e6, device=query_z.device, dtype=query_z.dtype)
            support_scores[no_neighbor] = high_value
        return support_scores

    def _state_search(self, state_vec: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        query = state_vec.detach().cpu().numpy().astype(np.float32)
        search_k = max(self.config.top_M, self.config.state_novelty_k)
        return self.state_index.search(query, search_k)

    def query_preencoded(
        self,
        state_vec: torch.Tensor,
        z_list: list[torch.Tensor],
        c_list: list[torch.Tensor],
        return_details: bool = False,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        state_distances, coarse_indices = self._state_search(state_vec)
        novelty_k = min(self.config.state_novelty_k, state_distances.shape[1])
        novelty = torch.from_numpy(state_distances[:, :novelty_k].mean(axis=1)).float()
        coarse_windows = coarse_indices[:, : min(self.config.top_M, coarse_indices.shape[1])]
        coarse_distances = state_distances[:, : min(self.config.top_M, state_distances.shape[1])]
        prototype_ids, prototype_weights, prototype_distances = self._prototype_soft_gate(state_vec)

        mem_scores: list[torch.Tensor] = []
        soft_mem_scores: list[torch.Tensor] = []
        neighbor_window_ids: list[torch.Tensor] = []
        neighbor_raw_starts: list[torch.Tensor] = []
        neighbor_context_distances: list[torch.Tensor] = []
        neighbor_valid_masks: list[torch.Tensor] = []
        for scale_idx, scale_memory in enumerate(self.scales):
            z_query = z_list[scale_idx].detach()
            c_query = c_list[scale_idx].detach()
            batch, n_patches, _ = z_query.shape
            mem = torch.full((batch, n_patches), float("inf"), device=z_query.device, dtype=z_query.dtype)
            soft_mem = torch.full((batch, n_patches), float("inf"), device=z_query.device, dtype=z_query.dtype)
            if return_details:
                scale_neighbor_ids = torch.full(
                    (batch, n_patches, self.config.top_K),
                    -1,
                    device=z_query.device,
                    dtype=torch.long,
                )
                scale_neighbor_dists = torch.full(
                    (batch, n_patches, self.config.top_K),
                    float("inf"),
                    device=z_query.device,
                    dtype=z_query.dtype,
                )
                scale_neighbor_valid = torch.zeros(
                    (batch, n_patches, self.config.top_K),
                    device=z_query.device,
                    dtype=torch.bool,
                )
                scale_neighbor_raw_starts = torch.full(
                    (batch, n_patches, self.config.top_K),
                    -1,
                    device=z_query.device,
                    dtype=torch.long,
                )

            retrieval_query = c_query if self.config.use_context_key_retrieval else z_query
            if self.config.use_two_level_retrieval:
                coarse_candidate_idx = [
                    scale_memory.candidate_indices(coarse_windows[batch_idx])
                    for batch_idx in range(batch)
                ]
                (
                    candidate_z,
                    candidate_c,
                    candidate_window_ids,
                    candidate_raw_starts,
                    candidate_valid,
                ) = self._pack_scale_candidates(scale_memory, coarse_candidate_idx, z_query.device)
            else:
                full_z = self._get_cached_tensor(f"scale_{scale_idx}_z", scale_memory.z, z_query.device)
                full_c = self._get_cached_tensor(f"scale_{scale_idx}_c", scale_memory.c, c_query.device)
                full_window_ids = self._get_cached_tensor(
                    f"scale_{scale_idx}_window_ids",
                    scale_memory.window_ids,
                    z_query.device,
                )
                full_raw_starts = self._get_cached_tensor(
                    f"scale_{scale_idx}_raw_starts",
                    scale_memory.raw_starts,
                    z_query.device,
                )
                candidate_z = full_z.unsqueeze(0).expand(batch, -1, -1)
                candidate_c = full_c.unsqueeze(0).expand(batch, -1, -1)
                candidate_window_ids = full_window_ids.unsqueeze(0).expand(batch, -1)
                candidate_raw_starts = full_raw_starts.unsqueeze(0).expand(batch, -1)
                candidate_valid = torch.ones(
                    batch,
                    candidate_z.shape[1],
                    device=z_query.device,
                    dtype=torch.bool,
                )
            candidate_key = candidate_c if self.config.use_context_key_retrieval else candidate_z
            neighbor_z, ctx_dists, neighbor_valid, window_ids, topk_idx = self._batched_context_topk_neighbors(
                retrieval_query,
                candidate_key,
                candidate_z,
                candidate_window_ids,
                candidate_valid,
                self.config.top_K,
            )
            if candidate_raw_starts.shape[1] > 0:
                expanded_raw_starts = candidate_raw_starts[:, None, :].expand(
                    -1, n_patches, -1
                )
                selected_raw_starts = torch.gather(
                    expanded_raw_starts,
                    2,
                    topk_idx,
                ).masked_fill(~neighbor_valid, -1)
            else:
                selected_raw_starts = torch.full(
                    (batch, n_patches, self.config.top_K),
                    -1,
                    device=z_query.device,
                    dtype=torch.long,
                )
            mem = self._compute_memory_score(
                z_query,
                neighbor_z,
                neighbor_valid,
                self.config.knn_k,
            )
            # Fall back to the standard kNN memory score when prototype-based
            # support cannot be formed, so soft_support_score always stays finite.
            soft_mem = mem.clone()
            if return_details:
                scale_neighbor_ids = window_ids
                scale_neighbor_raw_starts = selected_raw_starts
                scale_neighbor_dists = ctx_dists
                scale_neighbor_valid = neighbor_valid

            proto_window_ids_list: list[torch.Tensor] = []
            proto_window_weights_list: list[torch.Tensor] = []
            proto_candidate_idx_list: list[Optional[torch.Tensor]] = []
            for batch_idx in range(batch):
                proto_window_ids, proto_window_weights = self._collect_prototype_candidates(
                    state_vec[batch_idx].detach(),
                    prototype_ids[batch_idx],
                    prototype_weights[batch_idx],
                )
                proto_window_ids_list.append(proto_window_ids)
                proto_window_weights_list.append(proto_window_weights)
                proto_candidate_idx_list.append(scale_memory.candidate_indices(proto_window_ids))

            (
                proto_candidate_z,
                proto_candidate_c,
                proto_candidate_window_ids,
                _,
                proto_candidate_valid,
            ) = self._pack_scale_candidates(
                scale_memory,
                proto_candidate_idx_list,
                z_query.device,
            )
            proto_candidate_weights = self._pack_candidate_weights(
                scale_memory,
                proto_candidate_idx_list,
                proto_window_ids_list,
                proto_window_weights_list,
                z_query.device,
            )
            active_proto_rows = proto_candidate_valid.any(dim=1)
            if active_proto_rows.any():
                soft_mem[active_proto_rows] = self._batched_prototype_support_score(
                    retrieval_query[active_proto_rows],
                    z_query[active_proto_rows],
                    proto_candidate_z[active_proto_rows],
                    proto_candidate_c[active_proto_rows],
                    proto_candidate_window_ids[active_proto_rows],
                    proto_candidate_valid[active_proto_rows],
                    proto_candidate_weights[active_proto_rows],
                    use_context_key_retrieval=self.config.use_context_key_retrieval,
                    top_k=self.config.top_K,
                    tau=self.config.support_score_tau,
                    eps=self.config.support_score_eps,
                )

            mem_scores.append(mem)
            soft_mem_scores.append(soft_mem)
            if return_details:
                neighbor_window_ids.append(scale_neighbor_ids)
                neighbor_raw_starts.append(scale_neighbor_raw_starts)
                neighbor_context_distances.append(scale_neighbor_dists)
                neighbor_valid_masks.append(scale_neighbor_valid)

        coarse_window_ids_tensor = torch.from_numpy(coarse_windows).long()
        coarse_window_starts = self.state_window_starts[
            coarse_window_ids_tensor.clamp(min=0)
        ].masked_fill(coarse_window_ids_tensor < 0, -1)
        out = {
            "mem_scores": mem_scores,
            "soft_mem_scores": soft_mem_scores,
            "state_novelty": novelty.to(state_vec.device),
            "coarse_windows": coarse_windows,
            "coarse_window_starts": coarse_window_starts.to(state_vec.device),
            "coarse_distances": torch.from_numpy(coarse_distances).to(state_vec.device),
            "prototype_ids": prototype_ids,
            "prototype_weights": prototype_weights,
            "prototype_distances": prototype_distances,
        }
        if return_details:
            out["neighbor_window_ids"] = neighbor_window_ids
            out["neighbor_raw_starts"] = neighbor_raw_starts
            out["neighbor_context_distances"] = neighbor_context_distances
            out["neighbor_valid_masks"] = neighbor_valid_masks
        return out

    def nearest_local_memory(
        self,
        state_vec: torch.Tensor,
        z_patch: torch.Tensor,
        c_patch: torch.Tensor,
        scale_idx: int,
    ) -> Optional[torch.Tensor]:
        state_distances, coarse_indices = self._state_search(state_vec.unsqueeze(0))
        _ = state_distances
        coarse_windows = coarse_indices[0, : min(self.config.top_M, coarse_indices.shape[1])]
        scale_memory = self.scales[scale_idx]
        if self.config.use_two_level_retrieval:
            candidate_idx = scale_memory.candidate_indices(coarse_windows)
        else:
            candidate_idx = None
        if candidate_idx is None:
            candidate_z = scale_memory.z
            candidate_c = scale_memory.c
        else:
            candidate_z = scale_memory.z[candidate_idx]
            candidate_c = scale_memory.c[candidate_idx]
        if candidate_c.size(0) == 0:
            return None

        query_key = (
            c_patch.detach().cpu().float().unsqueeze(0)
            if self.config.use_context_key_retrieval
            else z_patch.detach().cpu().float().unsqueeze(0)
        )
        query_z = z_patch.detach().cpu().float().unsqueeze(0)
        candidate_key = candidate_c if self.config.use_context_key_retrieval else candidate_z
        neighbor_z, _, neighbor_valid = self.context_searcher.search(
            query_key, candidate_key, candidate_z, self.config.top_K
        )
        z_dist = torch.cdist(query_z.unsqueeze(1), neighbor_z, p=2.0).squeeze(0).squeeze(0).pow(2)
        z_dist = z_dist.masked_fill(~neighbor_valid.squeeze(0), float("inf"))
        if not torch.isfinite(z_dist).any():
            return None
        return neighbor_z[0, int(torch.argmin(z_dist).item())].clone()

    def state_dict(self) -> dict:
        return {
            "state_bank": self.state_bank,
            "state_window_starts": self.state_window_starts,
            "prototypes": {
                "centers": self.prototype_centers,
                "labels": self.prototype_labels,
                "members": self.prototype_members,
            },
            "build_stats": self.build_stats,
            "scales": [
                {
                    "z": scale.z,
                    "c": scale.c,
                    "window_ids": scale.window_ids,
                    "raw_starts": scale.raw_starts,
                    "coverage_threshold": scale.coverage_threshold,
                }
                for scale in self.scales
            ],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict()
        tmp_path = path.with_name(f".{path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            try:
                torch.save(payload, tmp_path)
            except RuntimeError as exc:
                message = str(exc)
                if tmp_path.exists():
                    tmp_path.unlink()
                if "PytorchStreamWriter" not in message and "unexpected pos" not in message:
                    raise
                print(
                    f"[MemoryBank] zip serialization failed for {path}; "
                    "retrying with legacy torch.save serialization."
                )
                torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        self.state_index.save(self.config.faiss_index_path)

    @classmethod
    def load(cls, path: str | Path, config: CoReMADConfig) -> "MemoryBank":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        prototype_payload = payload.get("prototypes", {})
        scales = [
            ScaleMemory(
                z=scale_payload["z"],
                c=scale_payload["c"],
                window_ids=scale_payload["window_ids"],
                raw_starts=scale_payload.get("raw_starts"),
                coverage_threshold=float(scale_payload["coverage_threshold"]),
            )
            for scale_payload in payload["scales"]
        ]
        state_window_starts = payload.get("state_window_starts")
        if state_window_starts is None:
            state_window_starts = torch.arange(payload["state_bank"].size(0), dtype=torch.long)
        memory = cls(
            config=config,
            state_bank=payload["state_bank"],
            scales=scales,
            state_window_starts=state_window_starts,
            prototype_centers=prototype_payload.get("centers"),
            prototype_labels=prototype_payload.get("labels"),
            prototype_members=prototype_payload.get("members"),
            build_stats=payload.get("build_stats"),
            build_index=False,
        )
        loaded = memory.state_index.load(config.faiss_index_path, memory.state_bank.numpy())
        if not loaded:
            memory.state_index.build(memory.state_bank.numpy())
        return memory
