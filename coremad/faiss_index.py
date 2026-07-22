from __future__ import annotations

from typing import Tuple
from pathlib import Path

import numpy as np
import torch

try:
    import faiss

    HAS_FAISS = True
except Exception as exc:
    faiss = None
    HAS_FAISS = False
    print(f"[Faiss] unavailable, falling back to torch retrieval: {exc}")


class StateIndex:
    """
    Level-1 retrieval over window-level state vectors.
    Prefer Faiss when available, otherwise fall back to exact torch search.
    """

    def __init__(
        self,
        d_state: int,
        use_faiss: bool = True,
        use_gpu: bool = False,
        exact_threshold: int = 100000,
        ivf_nprobe: int = 16,
    ) -> None:
        self.d_state = int(d_state)
        self.exact_threshold = int(exact_threshold)
        self.ivf_nprobe = int(ivf_nprobe)
        self._use_faiss = bool(use_faiss) and HAS_FAISS
        self._want_gpu = bool(use_gpu)
        self._gpu_resources = None
        self.index = None
        self.cpu_index = None
        self.vectors: np.ndarray | None = None
        self.n_vectors = 0
        self.backend = "torch-exact"
        self._index_family = "flat"

    def build(self, state_vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(state_vectors.astype(np.float32))
        self.vectors = vectors
        self.n_vectors = len(vectors)
        self.index = None
        self.cpu_index = None
        self.backend = "torch-exact"
        self._index_family = "flat"

        if not self._use_faiss or self.n_vectors == 0:
            if not self._use_faiss:
                self.backend = "torch-exact"
            return

        cpu_index, backend = self._build_cpu_index(vectors)
        self.cpu_index = cpu_index
        self.index = cpu_index
        self.backend = backend

        if self._want_gpu:
            if faiss.get_num_gpus() > 0:
                try:
                    self._gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
                    self.backend = backend.replace("-cpu", "-gpu")
                except Exception as exc:
                    self.index = cpu_index
                    self.backend = backend
                    print(f"[Faiss] GPU index build failed, fallback to CPU index: {exc}")
            else:
                print("[Faiss] GPU index requested but no GPU detected, fallback to CPU index.")

    def search(self, query: np.ndarray, top_m: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise RuntimeError("StateIndex must be built before search.")

        q = np.ascontiguousarray(query.astype(np.float32))
        top_m = min(int(top_m), self.n_vectors)
        if top_m <= 0:
            return (
                np.empty((len(q), 0), dtype=np.float32),
                np.empty((len(q), 0), dtype=np.int64),
            )

        if self.index is not None:
            try:
                return self.index.search(q, top_m)
            except Exception as exc:
                if self.backend.endswith("-gpu") and self.cpu_index is not None:
                    print(f"[Faiss] GPU search failed, fallback to CPU index: {exc}")
                    self.index = self.cpu_index
                    self.backend = self.backend.replace("-gpu", "-cpu")
                    return self.index.search(q, top_m)
                print(f"[Faiss] index search failed, fallback to torch exact search: {exc}")
                self.index = None
                self.cpu_index = None
                self.backend = "torch-exact"
        return self._torch_search(q, top_m)

    def _torch_search(self, query: np.ndarray, top_m: int) -> Tuple[np.ndarray, np.ndarray]:
        q = torch.from_numpy(query)
        v = torch.from_numpy(self.vectors)
        dists_sq = torch.cdist(q, v, p=2.0).pow(2)
        topk = torch.topk(dists_sq, top_m, dim=1, largest=False)
        return topk.values.cpu().numpy(), topk.indices.cpu().numpy()

    def save(self, path: str | Path) -> bool:
        if not self._use_faiss or self.index is None:
            return False
        cpu_index = self.cpu_index
        if cpu_index is None:
            cpu_index = self.index
            if HAS_FAISS and hasattr(faiss, "index_gpu_to_cpu"):
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                except Exception:
                    cpu_index = self.index
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(cpu_index, str(path))
        return True

    def load(self, path: str | Path, state_vectors: np.ndarray) -> bool:
        vectors = np.ascontiguousarray(state_vectors.astype(np.float32))
        self.vectors = vectors
        self.n_vectors = len(vectors)
        self.index = None
        self.cpu_index = None
        self.backend = "torch-exact"
        self._index_family = "flat"
        path = Path(path)
        if not self._use_faiss or not path.exists() or self.n_vectors == 0:
            return False

        index = faiss.read_index(str(path))
        self.cpu_index = index
        self.index = index
        self.backend = "faiss-loaded-cpu"
        if self._want_gpu:
            if faiss.get_num_gpus() > 0:
                try:
                    self._gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
                    self.backend = "faiss-loaded-gpu"
                except Exception as exc:
                    self.index = index
                    self.backend = "faiss-loaded-cpu"
                    print(f"[Faiss] GPU index load failed, fallback to CPU index: {exc}")
            else:
                print("[Faiss] GPU index requested during load but no GPU detected, fallback to CPU index.")
        return True

    def describe(self) -> str:
        return self.backend

    def _build_cpu_index(self, vectors: np.ndarray):
        if self.n_vectors < self.exact_threshold:
            index = faiss.IndexFlatL2(self.d_state)
            backend = "faiss-flat-cpu"
            self._index_family = "flat"
        else:
            n_clusters = min(max(1, int(np.sqrt(self.n_vectors))), 1024)
            quantizer = faiss.IndexFlatL2(self.d_state)
            index = faiss.IndexIVFFlat(quantizer, self.d_state, n_clusters, faiss.METRIC_L2)
            index.train(vectors)
            index.nprobe = self.ivf_nprobe
            backend = "faiss-ivf-cpu"
            self._index_family = "ivf"
        index.add(vectors)
        return index, backend


class ContextSearcher:
    """
    Level-2 retrieval stays exact because the candidate set after state screening is small.
    """

    @staticmethod
    def search(
        query_key: torch.Tensor,
        candidate_key: torch.Tensor,
        candidate_z: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_k = min(int(top_k), int(candidate_key.shape[0]))
        if actual_k <= 0:
            d_z = int(candidate_z.shape[1])
            return (
                torch.zeros(query_key.shape[0], top_k, d_z, device=query_key.device, dtype=query_key.dtype),
                torch.full((query_key.shape[0], top_k), float("inf"), device=query_key.device, dtype=query_key.dtype),
                torch.zeros(query_key.shape[0], top_k, dtype=torch.bool, device=query_key.device),
            )

        dists = torch.cdist(query_key, candidate_key, p=2.0)
        topk_dists, topk_idx = torch.topk(dists, actual_k, dim=1, largest=False)
        neighbor_z = candidate_z[topk_idx]
        valid = torch.ones(query_key.shape[0], actual_k, dtype=torch.bool, device=query_key.device)

        if actual_k < top_k:
            pad_n = top_k - actual_k
            d_z = int(candidate_z.shape[1])
            neighbor_z = torch.cat(
                [
                    neighbor_z,
                    torch.zeros(query_key.shape[0], pad_n, d_z, device=query_key.device, dtype=query_key.dtype),
                ],
                dim=1,
            )
            topk_dists = torch.cat(
                [
                    topk_dists,
                    torch.full((query_key.shape[0], pad_n), float("inf"), device=query_key.device, dtype=query_key.dtype),
                ],
                dim=1,
            )
            valid = torch.cat(
                [
                    valid,
                    torch.zeros(query_key.shape[0], pad_n, dtype=torch.bool, device=query_key.device),
                ],
                dim=1,
            )

        return neighbor_z, topk_dists, valid
