from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad import CoReMADConfig, CoReMADTrainer
from coremad.data import build_loader


def load_experiment_context(
    experiment_dir: str | Path,
    device: Optional[str] = None,
) -> dict[str, Any]:
    exp_dir = Path(experiment_dir)
    config = CoReMADConfig.load(exp_dir / "config.json")
    config.artifact_root = str(exp_dir.parent)
    config.experiment_name = exp_dir.name
    if device is not None:
        config.device = device
    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    memory = trainer.load_memory_bank()
    raw_bundle = trainer._load_raw_bundle()
    cdf_fusion = trainer.load_cdf_fusion(optional=False)
    zscore_fusion = trainer.load_zscore_fusion(optional=False)
    return {
        "experiment_dir": exp_dir,
        "config": config,
        "trainer": trainer,
        "model": model,
        "normalizer": normalizer,
        "memory": memory,
        "raw_bundle": raw_bundle,
        "cdf_fusion": cdf_fusion,
        "zscore_fusion": zscore_fusion,
    }


def parse_tep_name(name: str) -> dict[str, Any]:
    base = str(name).strip()
    is_val = base.endswith("_val")
    stem = base[:-4] if is_val else base
    if len(stem) < 4 or stem[0] != "m" or "d" not in stem:
        raise ValueError(f"Unsupported TEP sequence/file name: {name}")
    mode_part, fault_part = stem.split("d", maxsplit=1)
    mode_id = int(mode_part[1:])
    fault_id = int(fault_part)
    is_normal = int(fault_id == 0)
    return {
        "name": base,
        "stem": stem,
        "mode_id": mode_id,
        "fault_id": fault_id,
        "is_normal": is_normal,
        "is_val": int(is_val),
        "file_id": stem,
    }


def metadata_for_starts(
    starts: np.ndarray,
    ranges: np.ndarray,
    names: list[str],
) -> dict[str, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    ranges = np.asarray(ranges, dtype=np.int64)
    out: dict[str, np.ndarray] = {
        "mode_id": np.full(starts.shape, -1, dtype=np.int32),
        "fault_id": np.full(starts.shape, -1, dtype=np.int32),
        "is_normal": np.zeros(starts.shape, dtype=np.int32),
        "is_val": np.zeros(starts.shape, dtype=np.int32),
        "segment_index": np.full(starts.shape, -1, dtype=np.int32),
        "name": np.empty(starts.shape, dtype=object),
        "file_id": np.empty(starts.shape, dtype=object),
    }
    for seg_idx, ((start, end), name) in enumerate(zip(ranges.tolist(), names)):
        meta = parse_tep_name(name)
        mask = (starts >= int(start)) & (starts < int(end))
        out["mode_id"][mask] = int(meta["mode_id"])
        out["fault_id"][mask] = int(meta["fault_id"])
        out["is_normal"][mask] = int(meta["is_normal"])
        out["is_val"][mask] = int(meta["is_val"])
        out["segment_index"][mask] = int(seg_idx)
        out["name"][mask] = meta["name"]
        out["file_id"][mask] = meta["file_id"]
    return out


def get_loader_starts(
    data: np.ndarray,
    config: CoReMADConfig,
    stride: int,
    max_windows: int,
    segment_ranges: np.ndarray,
) -> np.ndarray:
    loader = build_loader(
        data=data,
        labels=None,
        seq_len=config.seq_len,
        stride=stride,
        batch_size=config.memory_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        max_windows=max_windows,
        drop_last=False,
        segment_ranges=segment_ranges,
    )
    starts = loader.dataset.start_indices
    if starts is None:
        return np.arange(len(loader.dataset), dtype=np.int64)
    return np.asarray(starts, dtype=np.int64).reshape(-1)


def aggregate_point_scores(values: np.ndarray, mode: str) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(finite))
    if mode == "max":
        return float(np.max(finite))
    if mode == "top5_mean":
        k = max(1, int(np.ceil(finite.size * 0.05)))
        topk = np.partition(finite, -k)[-k:]
        return float(np.mean(topk))
    return float(np.percentile(finite, 95))


def aggregate_neighbor_lists(
    neighbor_ids: list[np.ndarray],
    neighbor_dists: list[np.ndarray],
    neighbor_valid: list[np.ndarray],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    best: dict[int, float] = {}
    for ids_arr, dist_arr, valid_arr in zip(neighbor_ids, neighbor_dists, neighbor_valid):
        ids_flat = np.asarray(ids_arr, dtype=np.int64).reshape(-1)
        dist_flat = np.asarray(dist_arr, dtype=np.float64).reshape(-1)
        valid_flat = np.asarray(valid_arr, dtype=bool).reshape(-1)
        for window_id, dist, is_valid in zip(ids_flat.tolist(), dist_flat.tolist(), valid_flat.tolist()):
            if not is_valid or window_id < 0 or not np.isfinite(dist):
                continue
            prev = best.get(int(window_id))
            if prev is None or dist < prev:
                best[int(window_id)] = float(dist)
    ordered = sorted(best.items(), key=lambda item: item[1])[:top_k]
    out_ids = np.full(top_k, -1, dtype=np.int64)
    out_dists = np.full(top_k, np.inf, dtype=np.float64)
    for idx, (window_id, dist) in enumerate(ordered):
        out_ids[idx] = int(window_id)
        out_dists[idx] = float(dist)
    return out_ids, out_dists


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    vals = np.asarray(values, dtype=np.float64)
    if ref.size == 0:
        return np.full(vals.shape, np.nan, dtype=np.float64)
    positions = np.searchsorted(ref, vals, side="right")
    return positions.astype(np.float64) / float(ref.size)


def load_window_logs(log_dir: str | Path, prefix: str = "window_logs") -> dict[str, np.ndarray]:
    payload = np.load(Path(log_dir) / f"{prefix}.npz", allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def load_sequence_scores_with_meta(
    log_dir: str | Path,
    filename: str = "sequence_scores_with_meta.json",
) -> list[dict[str, Any]]:
    path = Path(log_dir) / filename
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_state_meta(log_dir: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(log_dir) / "train_state_meta.npz", allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def np_to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
