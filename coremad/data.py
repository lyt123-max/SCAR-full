from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import scipy.io as sio

    HAS_SCIPY_IO = True
except Exception:
    sio = None
    HAS_SCIPY_IO = False

try:
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    StandardScaler = None
    HAS_SKLEARN = False

from .config import CoReMADConfig


def _flatten_labels(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1).max(axis=1)
    return arr.astype(np.float32)


def _validate_loaded_arrays(train: np.ndarray, test: np.ndarray, labels: np.ndarray, dataset: str) -> None:
    assert train.ndim == 2, f"{dataset} train shape {train.shape}, expected (T, C)"
    assert test.ndim == 2, f"{dataset} test shape {test.shape}, expected (T, C)"
    assert train.shape[1] == test.shape[1], (
        f"{dataset} channel mismatch: train={train.shape[1]}, test={test.shape[1]}"
    )
    assert len(labels) == len(test), (
        f"{dataset} label length {len(labels)} != test length {len(test)}"
    )
    uniq = set(np.unique(labels).tolist())
    assert uniq.issubset({0, 1, 0.0, 1.0, False, True}), (
        f"{dataset} labels contain unexpected values: {sorted(uniq)}"
    )
    assert not np.isnan(train).any(), f"{dataset} train contains NaN after loading"
    assert not np.isnan(test).any(), f"{dataset} test contains NaN after loading"
    assert not np.isinf(train).any(), f"{dataset} train contains Inf after loading"
    assert not np.isinf(test).any(), f"{dataset} test contains Inf after loading"


class TimeSeriesNormalizer:
    def __init__(self) -> None:
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        if self.scaler is not None:
            self.scaler.fit(data)
            self.mean_ = np.asarray(self.scaler.mean_, dtype=np.float64)
            self.scale_ = np.asarray(self.scaler.scale_, dtype=np.float64)
            self.scale_ = np.clip(self.scale_, a_min=1e-8, a_max=None)
            self.var_ = self.scale_ ** 2
            self.scaler.mean_ = self.mean_
            self.scaler.scale_ = self.scale_
            self.scaler.var_ = self.var_
            self.scaler.n_features_in_ = self.mean_.shape[0]
            self.scaler.n_samples_seen_ = int(data.shape[0])
        else:
            self.mean_ = np.mean(data, axis=0, dtype=np.float64)
            self.scale_ = np.std(data, axis=0, dtype=np.float64)
            self.scale_ = np.clip(self.scale_, a_min=1e-8, a_max=None)
            self.var_ = self.scale_ ** 2
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform.")
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array for transform, got shape {data.shape}")
        if data.shape[0] == 0:
            return np.empty((0, data.shape[1]), dtype=np.float32)
        if self.scaler is not None:
            out = self.scaler.transform(data)
        else:
            out = (data - self.mean_) / self.scale_
        out = out.astype(np.float32)
        if np.isnan(out).any() or np.isinf(out).any():
            raise RuntimeError("Normalizer transform produced NaN/Inf.")
        return out

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        if self.is_fitted:
            raise RuntimeError(
                "Normalizer already fitted. Use transform() instead of fit_transform()."
            )
        self.fit(data)
        return self.transform(data)

    def state_dict(self) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted normalizer.")
        return {
            "mean": self.mean_.astype(np.float32),
            "scale": self.scale_.astype(np.float32),
            "var": self.var_.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]) -> "TimeSeriesNormalizer":
        normalizer = cls()
        normalizer.mean_ = np.asarray(state["mean"], dtype=np.float64)
        normalizer.scale_ = np.asarray(state["scale"], dtype=np.float64)
        normalizer.var_ = np.asarray(state.get("var", normalizer.scale_ ** 2), dtype=np.float64)
        if normalizer.scaler is not None:
            normalizer.scaler.mean_ = normalizer.mean_
            normalizer.scaler.scale_ = normalizer.scale_
            normalizer.scaler.var_ = normalizer.var_
            normalizer.scaler.n_features_in_ = normalizer.mean_.shape[0]
            normalizer.scaler.n_samples_seen_ = 1
        normalizer.is_fitted = True
        return normalizer


@dataclass
class DataBundle:
    train_full: np.ndarray
    train_stage_a: np.ndarray
    val_stage_a: Optional[np.ndarray]
    test: np.ndarray
    test_labels: np.ndarray
    normalizer: TimeSeriesNormalizer
    train_stage_a_start_indices: Optional[np.ndarray] = None
    val_stage_a_start_indices: Optional[np.ndarray] = None
    train_full_segment_ranges: Optional[np.ndarray] = None
    train_stage_a_segment_ranges: Optional[np.ndarray] = None
    val_stage_a_segment_ranges: Optional[np.ndarray] = None
    test_segment_ranges: Optional[np.ndarray] = None


@dataclass
class RawDatasetBundle:
    train: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    val: Optional[np.ndarray] = None
    train_segment_ranges: Optional[np.ndarray] = None
    val_segment_ranges: Optional[np.ndarray] = None
    test_segment_ranges: Optional[np.ndarray] = None
    train_segment_names: Optional[list[str]] = None
    val_segment_names: Optional[list[str]] = None
    test_segment_names: Optional[list[str]] = None
    test_sequence_ranges: Optional[np.ndarray] = None
    test_sequence_labels: Optional[np.ndarray] = None
    test_sequence_names: Optional[list[str]] = None
    test_sequences: Optional[list[np.ndarray]] = None
    evaluation_vus_window: Optional[int] = None
    dataset_metadata: Optional[dict[str, Any]] = None


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        seq_len: int,
        stride: int = 1,
        max_windows: int = 0,
        start_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.data_tensor = torch.from_numpy(self.data)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.labels_tensor = None if self.labels is None else torch.from_numpy(self.labels)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        if start_indices is not None:
            starts = np.asarray(start_indices, dtype=np.int64)
            if max_windows > 0:
                starts = starts[: int(max_windows)]
            self.start_indices = starts
            self.n_windows = len(starts)
        else:
            total = max(0, (len(self.data) - self.seq_len) // self.stride + 1)
            self.start_indices = None
            self.n_windows = total if max_windows <= 0 else min(total, int(max_windows))

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        start = int(self.start_indices[idx]) if self.start_indices is not None else idx * self.stride
        end = start + self.seq_len
        item: dict[str, torch.Tensor | int] = {
            "x": self.data_tensor[start:end],
            "start": start,
            "idx": idx,
        }
        if self.labels_tensor is not None:
            item["labels"] = self.labels_tensor[start:end]
        return item


def _split_tail(
    data: np.ndarray,
    ratio: float,
    seq_len: int,
    gap: bool,
    min_train_windows: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    total_steps = len(data)
    if ratio <= 0.0 or total_steps < (2 * seq_len):
        print(f"[Split] Skipped (val_ratio={ratio}, T={total_steps})")
        return data, None

    gap_size = seq_len if gap else 0
    n_val = max(seq_len, int(total_steps * ratio))
    n_train = total_steps - n_val - gap_size
    min_train = seq_len * max(1, int(min_train_windows))

    if n_train < min_train:
        gap_size = 0
        n_train = total_steps - n_val
        if n_train < min_train:
            n_val = total_steps - min_train
            n_train = min_train
            if n_val < seq_len:
                print(f"[Split] Dataset too short (T={total_steps}), no val split")
                return data, None

    train_split = data[:n_train]
    val_split = data[n_train + gap_size :]
    print(
        f"[Split] T={total_steps} -> Train={len(train_split)}, "
        f"Gap={gap_size}, Val={len(val_split)} ({len(val_split) / total_steps:.1%})"
    )
    return train_split, val_split


def _normalize_segment_ranges(segment_ranges: Optional[np.ndarray | list[tuple[int, int]]]) -> Optional[np.ndarray]:
    if segment_ranges is None:
        return None
    arr = np.asarray(segment_ranges, dtype=np.int64)
    if arr.size == 0:
        return None
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"segment_ranges must have shape (N, 2), got {arr.shape}")
    return arr


def _compute_start_indices_from_segment_ranges(
    seq_len: int,
    stride: int,
    segment_ranges: Optional[np.ndarray | list[tuple[int, int]]],
    max_windows: int = 0,
) -> Optional[np.ndarray]:
    ranges = _normalize_segment_ranges(segment_ranges)
    if ranges is None:
        return None
    starts_list: list[np.ndarray] = []
    for start, end in ranges:
        start_i = int(start)
        end_i = int(end)
        last = end_i - int(seq_len)
        if last < start_i:
            continue
        starts = np.arange(start_i, last + 1, int(stride), dtype=np.int64)
        if starts.size > 0:
            starts_list.append(starts)
    if not starts_list:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(starts_list, axis=0)
    if max_windows > 0:
        out = out[: int(max_windows)]
    return out


def _concat_segments(segments: list[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    non_empty_segments = [np.asarray(segment, dtype=np.float32) for segment in segments if len(segment) > 0]
    if not non_empty_segments:
        raise ValueError("Expected at least one non-empty segment.")
    concatenated = np.concatenate(non_empty_segments, axis=0).astype(np.float32)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for segment in non_empty_segments:
        end = cursor + int(len(segment))
        ranges.append((cursor, end))
        cursor = end
    return concatenated, np.asarray(ranges, dtype=np.int64)


def _empty_feature_array(n_features: int) -> np.ndarray:
    return np.empty((0, int(n_features)), dtype=np.float32)


def _split_interleaved_windows(
    data: np.ndarray,
    ratio: float,
    seq_len: int,
    stride: int,
    min_train_windows: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    total_steps = len(data)
    n_windows = max(0, (total_steps - seq_len) // stride + 1)
    if ratio <= 0.0 or n_windows < 2:
        print(f"[Split] Skipped interleaved split (val_ratio={ratio}, n_windows={n_windows})")
        return None, None

    period = max(2, int(round(1.0 / max(ratio, 1e-12))))
    chunk_len = seq_len * period
    train_chunks = 0
    val_chunks = 0
    train_starts_list: list[np.ndarray] = []
    val_starts_list: list[np.ndarray] = []

    chunk_id = 0
    chunk_start = 0
    last_start = total_steps - seq_len
    while chunk_start <= last_start:
        chunk_end = min(total_steps, chunk_start + chunk_len)
        starts = np.arange(chunk_start, chunk_end - seq_len + 1, stride, dtype=np.int64)
        if starts.size > 0:
            if (chunk_id % period) == (period - 1):
                val_starts_list.append(starts)
                val_chunks += 1
            else:
                train_starts_list.append(starts)
                train_chunks += 1
        chunk_id += 1
        chunk_start += chunk_len

    train_starts = (
        np.concatenate(train_starts_list, axis=0).astype(np.int64)
        if train_starts_list
        else np.empty(0, dtype=np.int64)
    )
    val_starts = (
        np.concatenate(val_starts_list, axis=0).astype(np.int64)
        if val_starts_list
        else np.empty(0, dtype=np.int64)
    )

    if len(train_starts) < max(1, int(min_train_windows)) or len(val_starts) == 0:
        print(
            f"[Split] Interleaved split degenerate (train_windows={len(train_starts)}, "
            f"val_windows={len(val_starts)}), fallback to no val split"
        )
        return None, None

    print(
        f"[Split] interleaved chunks={train_chunks + val_chunks} "
        f"(chunk_len={chunk_len}, period={period}) -> "
        f"Train={len(train_starts)}, Val={len(val_starts)} "
        f"({len(val_starts) / max(1, len(train_starts) + len(val_starts)):.1%})"
    )
    return train_starts, val_starts


def _dataset_key(name: str) -> str:
    return str(name).strip().casefold()


def _resolve_detect_root(data_root: str | Path) -> Path:
    base = Path(data_root)
    candidates = [
        base,
        base / "anomaly_detect",
        base / "dataset" / "anomaly_detect",
    ]
    for candidate in candidates:
        if (candidate / "DETECT_META.csv").exists() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find CATCH-style anomaly dataset root under {data_root!r}. "
        "Expected a directory containing DETECT_META.csv and a data/ subdirectory."
    )


def _read_detect_meta(meta_path: Path) -> dict[str, dict[str, str]]:
    meta_index: dict[str, dict[str, str]] = {}
    with meta_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = str(row.get("file_name", "")).strip()
            if not file_name:
                continue
            meta_index[_dataset_key(file_name)] = {
                key: (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
            }
    return meta_index


def _resolve_detect_dataset_file(dataset: str, detect_root: Path) -> tuple[Path, int]:
    meta_index = _read_detect_meta(detect_root / "DETECT_META.csv")
    dataset_name = str(dataset).strip()
    candidates: list[str] = []
    if dataset_name.endswith(".csv"):
        candidates.append(dataset_name)
    else:
        candidates.extend(
            [
                f"{dataset_name}.csv",
                f"{dataset_name.upper()}.csv",
                f"{dataset_name.lower()}.csv",
            ]
        )

    matched_row: Optional[dict[str, str]] = None
    matched_file_name: Optional[str] = None
    for candidate in candidates:
        matched_row = meta_index.get(_dataset_key(candidate))
        if matched_row is not None:
            matched_file_name = str(matched_row["file_name"]).strip()
            break

    if matched_file_name is None:
        data_dir = detect_root / "data"
        for path in data_dir.glob("*.csv"):
            if _dataset_key(path.stem) == _dataset_key(dataset_name) or _dataset_key(path.name) == _dataset_key(
                dataset_name
            ):
                matched_file_name = path.name
                matched_row = meta_index.get(_dataset_key(path.name))
                break

    if matched_file_name is None:
        raise FileNotFoundError(
            f"Cannot find dataset {dataset!r} under {detect_root / 'data'}. "
            "Pass the CSV stem or file name listed in DETECT_META.csv."
        )

    if matched_row is None or not str(matched_row.get("train_lens", "")).strip():
        raise KeyError(
            f"Dataset {matched_file_name!r} is missing train_lens in {detect_root / 'DETECT_META.csv'}. "
            "Please add the train split length before training."
        )

    csv_path = detect_root / "data" / matched_file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV declared in metadata does not exist: {csv_path}")

    train_lens = int(float(str(matched_row["train_lens"])))
    return csv_path, train_lens


def _cache_path_for_dataset(detect_root: Path, csv_path: Path) -> Path:
    cache_dir = detect_root / ".coremad_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{csv_path.stem}.npz"


def _build_point_mask_from_start_indices(
    total_steps: int,
    seq_len: int,
    start_indices: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if start_indices is None:
        return None
    starts = np.asarray(start_indices, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        return None
    ends = np.clip(starts + int(seq_len), 0, int(total_steps))
    diff = np.zeros(int(total_steps) + 1, dtype=np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    mask = np.cumsum(diff[:-1]) > 0
    return mask if mask.any() else None


def _infer_block_length_from_long_csv(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        _ = next(reader, None)
        first_token: Optional[str] = None
        count = 0
        for row in reader:
            if not row:
                continue
            token = str(row[0]).strip()
            if first_token is None:
                first_token = token
                count = 1
                continue
            if token == first_token:
                break
            count += 1
    if count <= 0:
        raise ValueError(f"Failed to infer sequence length from {csv_path}")
    return count


def _csv_has_label_block(csv_path: Path) -> bool:
    with csv_path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        step = min(size, 65536)
        handle.seek(size - step)
        tail = handle.read(step).decode("utf-8", errors="ignore")
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty CSV tail while inspecting {csv_path}")
    return lines[-1].rsplit(",", 1)[-1].strip().casefold() == "label"


def _try_load_cached_detect_dataset(
    cache_path: Path,
    csv_path: Path,
    train_lens: int,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as payload:
        cached_mtime = int(payload["source_mtime_ns"])
        cached_train_lens = int(payload["train_lens"])
        if cached_mtime != int(csv_path.stat().st_mtime_ns) or cached_train_lens != int(train_lens):
            return None

        train = np.asarray(payload["train"], dtype=np.float32)
        test = np.asarray(payload["test"], dtype=np.float32)
        labels = np.asarray(payload["test_labels"], dtype=np.float32)

    _validate_loaded_arrays(train, test, labels, dataset_name)
    print(f"[Data] Loaded cache: {cache_path}")
    return train, test, labels


def _build_detect_dataset_cache(
    csv_path: Path,
    cache_path: Path,
    train_lens: int,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_steps = _infer_block_length_from_long_csv(csv_path)
    has_label_block = _csv_has_label_block(csv_path)
    values = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=1,
        dtype=np.float32,
    )
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"No numeric data loaded from {csv_path}")
    if values.size % total_steps != 0:
        raise ValueError(
            f"{csv_path} rows={values.size} cannot be reshaped into blocks of length {total_steps}."
        )

    n_blocks = values.size // total_steps
    if has_label_block and n_blocks < 2:
        raise ValueError(f"{csv_path} has a label block but no feature blocks.")

    wide = values.reshape(n_blocks, total_steps).T
    if has_label_block:
        full_data = wide[:, :-1]
        full_labels = _flatten_labels(wide[:, -1:])
    else:
        full_data = wide
        full_labels = np.zeros(total_steps, dtype=np.float32)

    if train_lens <= 0 or train_lens >= len(full_data):
        raise ValueError(
            f"Invalid train_lens={train_lens} for {dataset_name}: total_length={len(full_data)}."
        )

    train = np.nan_to_num(full_data[:train_lens], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test = np.nan_to_num(full_data[train_lens:], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    labels = _flatten_labels(full_labels[train_lens:])
    _validate_loaded_arrays(train, test, labels, dataset_name)

    np.savez(
        cache_path,
        train=train,
        test=test,
        test_labels=labels,
        train_lens=np.asarray(train_lens, dtype=np.int64),
        source_mtime_ns=np.asarray(csv_path.stat().st_mtime_ns, dtype=np.int64),
    )
    print(
        f"[Data] Built cache for {dataset_name}: "
        f"csv={csv_path.name}, train={train.shape}, test={test.shape}, cache={cache_path}"
    )
    return train, test, labels


TSB_AD_FILE_PATTERN = re.compile(
    r"^(?P<index>\d{3})_(?P<source>.+?)_id_(?P<series_id>\d+)_"
    r"(?P<domain>.+?)_tr_(?P<train_length>\d+)_1st_(?P<first_anomaly>\d+)\.csv$",
    flags=re.IGNORECASE,
)
TSB_AD_CACHE_VERSION = 1


def parse_tsb_ad_filename(file_name: str) -> dict[str, Any]:
    name = Path(str(file_name)).name
    if not name.casefold().endswith(".csv"):
        name += ".csv"
    match = TSB_AD_FILE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Invalid TSB-AD file name {file_name!r}. Expected "
            "'NNN_<dataset>_id_<id>_<domain>_tr_<length>_1st_<index>.csv'."
        )
    fields = match.groupdict()
    return {
        "file_name": name,
        "file_index": int(fields["index"]),
        "source_dataset": fields["source"],
        "series_id": int(fields["series_id"]),
        "domain": fields["domain"],
        "train_length": int(fields["train_length"]),
        "first_anomaly_index": int(fields["first_anomaly"]),
    }


def _resolve_tsb_ad_root(data_root: str | Path) -> Path:
    base = Path(data_root)
    candidates = [
        base,
        base / base.name,
        base / "TSB-AD-M",
        base / "TSB-AD-U",
        base / "dataset" / "TSB-AD-M",
        base / "dataset" / "TSB-AD-U",
    ]
    matches: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(TSB_AD_FILE_PATTERN.fullmatch(path.name) for path in candidate.glob("*.csv")):
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Cannot find TSB-AD CSV files under {data_root!r}. Expected a TSB-AD-M "
            "or TSB-AD-U directory containing the official numbered CSV files."
        )
    raise ValueError(
        f"Multiple TSB-AD editions found under {data_root!r}: {matches}. "
        "Pass the specific TSB-AD-M or TSB-AD-U directory."
    )


def _resolve_tsb_ad_file(dataset: str, tsb_root: Path) -> Path:
    requested = Path(str(dataset).strip()).name
    requested_name = requested if requested.casefold().endswith(".csv") else f"{requested}.csv"
    direct = tsb_root / requested_name
    if direct.is_file():
        parse_tsb_ad_filename(direct.name)
        return direct

    key = requested_name.casefold()
    matches = [path for path in tsb_root.glob("*.csv") if path.name.casefold() == key]
    if len(matches) == 1:
        parse_tsb_ad_filename(matches[0].name)
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous TSB-AD dataset {dataset!r} under {tsb_root}.")
    raise FileNotFoundError(
        f"Cannot find TSB-AD series {dataset!r} under {tsb_root}. "
        "Pass an exact official CSV file name or file stem."
    )


def _tsb_ad_vus_window(signal: np.ndarray) -> int:
    """Match TSB-AD's rank-1 ACF window rule without consulting labels."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)[:20000]
    if values.size < 4:
        return 1
    values = values - float(values.mean())
    energy = float(np.dot(values, values))
    if not np.isfinite(energy) or energy <= 1e-12:
        return 125

    max_lag = min(400, values.size - 1)
    fft_size = 1 << (2 * values.size - 1).bit_length()
    spectrum = np.fft.rfft(values, n=fft_size)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: max_lag + 1]
    acf = np.real(acf) / max(float(acf[0]), 1e-12)
    base = 3
    candidates = acf[base:]
    if candidates.size < 3:
        return 125
    local_maxima = np.flatnonzero(
        (candidates[1:-1] > candidates[:-2]) & (candidates[1:-1] > candidates[2:])
    ) + 1
    if local_maxima.size == 0:
        return 125
    best = int(local_maxima[np.argmax(candidates[local_maxima])]) + base
    return 125 if best > 300 else max(1, best)


def _tsb_ad_cache_path(tsb_root: Path, csv_path: Path) -> Path:
    cache_dir = tsb_root / ".coremad_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{csv_path.stem}_tsb_ad.npz"


def _read_tsb_ad_csv(csv_path: Path, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, int]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header or len(header) < 2 or str(header[-1]).strip().casefold() != "label":
        raise ValueError(f"TSB-AD CSV must end with a Label column: {csv_path}")

    wide = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    if wide.ndim != 2 or wide.shape[1] != len(header):
        raise ValueError(
            f"Unexpected TSB-AD array shape {wide.shape} for {csv_path}; "
            f"header declares {len(header)} columns."
        )
    finite_rows = np.isfinite(wide).all(axis=1)
    dropped_rows = int((~finite_rows).sum())
    if dropped_rows:
        wide = wide[finite_rows]
    if wide.shape[0] == 0:
        raise ValueError(f"No finite rows remain in TSB-AD CSV: {csv_path}")

    data = np.asarray(wide[:, :-1], dtype=np.float32)
    raw_labels = np.asarray(wide[:, -1], dtype=np.float64)
    labels = np.rint(raw_labels).astype(np.float32)
    if not np.allclose(raw_labels, labels, atol=1e-6) or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError(f"TSB-AD labels must be binary in {csv_path}.")

    train_length = int(metadata["train_length"])
    if train_length <= 0 or train_length >= len(data):
        raise ValueError(
            f"Invalid TSB-AD train length {train_length} for {csv_path.name} "
            f"with {len(data)} finite rows."
        )
    positive_indices = np.flatnonzero(labels > 0)
    first_anomaly = int(metadata["first_anomaly_index"])
    if positive_indices.size == 0 or int(positive_indices[0]) != first_anomaly:
        observed = None if positive_indices.size == 0 else int(positive_indices[0])
        raise ValueError(
            f"TSB-AD first anomaly mismatch in {csv_path.name}: "
            f"filename={first_anomaly}, labels={observed}."
        )
    vus_window = _tsb_ad_vus_window(data[:, 0])
    return data, labels, vus_window, dropped_rows


def _load_tsb_ad_raw_dataset_bundle(dataset: str, data_root: str | Path) -> RawDatasetBundle:
    tsb_root = _resolve_tsb_ad_root(data_root)
    csv_path = _resolve_tsb_ad_file(dataset, tsb_root)
    metadata = parse_tsb_ad_filename(csv_path.name)
    cache_path = _tsb_ad_cache_path(tsb_root, csv_path)
    source_mtime_ns = int(csv_path.stat().st_mtime_ns)

    cache_valid = False
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                cache_valid = (
                    int(payload["cache_version"]) == TSB_AD_CACHE_VERSION
                    and int(payload["source_mtime_ns"]) == source_mtime_ns
                    and int(payload["train_length"]) == int(metadata["train_length"])
                )
                if cache_valid:
                    full_data = np.asarray(payload["data"], dtype=np.float32)
                    full_labels = np.asarray(payload["labels"], dtype=np.float32)
                    vus_window = int(payload["vus_window"])
                    dropped_rows = int(payload["dropped_rows"])
        except (KeyError, OSError, ValueError):
            cache_valid = False
    if not cache_valid:
        full_data, full_labels, vus_window, dropped_rows = _read_tsb_ad_csv(csv_path, metadata)
        np.savez(
            cache_path,
            data=full_data,
            labels=full_labels,
            cache_version=np.asarray(TSB_AD_CACHE_VERSION, dtype=np.int64),
            source_mtime_ns=np.asarray(source_mtime_ns, dtype=np.int64),
            train_length=np.asarray(metadata["train_length"], dtype=np.int64),
            vus_window=np.asarray(vus_window, dtype=np.int64),
            dropped_rows=np.asarray(dropped_rows, dtype=np.int64),
        )
        print(f"[TSB-AD] Built cache: {cache_path}")
    else:
        print(f"[TSB-AD] Loaded cache: {cache_path}")

    train_length = int(metadata["train_length"])
    train = np.asarray(full_data[:train_length], dtype=np.float32)
    test = np.asarray(full_data, dtype=np.float32)
    labels = np.asarray(full_labels, dtype=np.float32)
    _validate_loaded_arrays(train, test, labels, csv_path.stem)
    training_prefix_anomaly_count = int(np.count_nonzero(labels[:train_length]))
    training_prefix_anomaly_ratio = float(training_prefix_anomaly_count / train_length)

    edition = "M" if test.shape[1] > 1 else "U"
    metadata.update(
        {
            "edition": edition,
            "csv_path": str(csv_path.resolve()),
            "total_length": int(len(test)),
            "n_channels": int(test.shape[1]),
            "dropped_nonfinite_rows": int(dropped_rows),
            "evaluation_scope": "full_series",
            "normalization_scope": "training_prefix",
            "training_prefix_protocol": "official_tr_N_boundary",
            "training_prefix_anomaly_count": training_prefix_anomaly_count,
            "training_prefix_anomaly_ratio": training_prefix_anomaly_ratio,
            "training_prefix_contains_anomalies": bool(training_prefix_anomaly_count),
            "vus_window": int(vus_window),
        }
    )
    if training_prefix_anomaly_count:
        print(
            f"[TSB-AD] warning: official tr_N prefix in {csv_path.name} contains "
            f"{training_prefix_anomaly_count}/{train_length} anomaly labels; "
            "retaining the complete prefix to match the official semisupervised split."
        )
    print(
        f"[TSB-AD-{edition}] {csv_path.name}: train={len(train)} full_eval={len(test)} "
        f"channels={test.shape[1]} vus_window={vus_window}"
    )
    return RawDatasetBundle(
        train=train,
        test=test,
        test_labels=labels,
        evaluation_vus_window=vus_window,
        dataset_metadata=metadata,
    )


TEP_SELECTED_MODES = (1, 3, 4)
TEP_SELECTED_FILE_FAULT_IDS = (1, 4, 6, 7, 10, 13, 14, 27)
TEP_FULL_MODES = tuple(range(1, 7))
TEP_NUM_INPUT_CHANNELS = 53
TEP_NUM_DISTURBANCES = 28


def _tep_file_name(mode_id: int, file_fault_id: int) -> str:
    return f"m{int(mode_id)}d{int(file_fault_id):02d}.mat"


def tep_protocol_files(protocol: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    protocol = str(protocol).strip().lower()
    if protocol == "selected":
        modes = TEP_SELECTED_MODES
        file_fault_ids = TEP_SELECTED_FILE_FAULT_IDS
    elif protocol == "full":
        modes = TEP_FULL_MODES
        file_fault_ids = tuple(range(1, TEP_NUM_DISTURBANCES + 1))
    else:
        raise ValueError(f"Unsupported TEP protocol: {protocol!r}.")
    normal_files = tuple(_tep_file_name(mode_id, 0) for mode_id in modes)
    fault_files = tuple(
        _tep_file_name(mode_id, file_fault_id)
        for mode_id in modes
        for file_fault_id in file_fault_ids
    )
    return normal_files, fault_files


TEP_NORMAL_FILES, TEP_FAULT_FILES = tep_protocol_files("selected")


def tep_file_fault_to_idv(file_fault_id: int) -> int:
    """Map the upstream dXX file suffix to the official physical IDV number."""
    file_fault_id = int(file_fault_id)
    if file_fault_id == 0:
        return 0
    if not 1 <= file_fault_id <= TEP_NUM_DISTURBANCES:
        raise ValueError(f"TEP file fault id must be in [0, {TEP_NUM_DISTURBANCES}], got {file_fault_id}.")
    return TEP_NUM_DISTURBANCES + 1 - file_fault_id


def tep_idv_to_file_fault(idv: int) -> int:
    """Map an official physical IDV number to the upstream dXX file suffix."""
    idv = int(idv)
    if idv == 0:
        return 0
    if not 1 <= idv <= TEP_NUM_DISTURBANCES:
        raise ValueError(f"TEP IDV must be in [0, {TEP_NUM_DISTURBANCES}], got {idv}.")
    return TEP_NUM_DISTURBANCES + 1 - idv


def _tep_mode_from_file_name(file_name: str) -> int:
    stem = Path(file_name).stem
    if not stem.startswith("m") or "d" not in stem:
        raise ValueError(f"Unsupported TEP file name: {file_name!r}.")
    return int(stem[1 : stem.index("d")])


def _tep_file_fault_from_file_name(file_name: str) -> int:
    stem = Path(file_name).stem
    if "d" not in stem:
        raise ValueError(f"Unsupported TEP file name: {file_name!r}.")
    return int(stem.split("d", maxsplit=1)[1])


def resolve_tep_files(
    data_root: str | Path,
    protocol: str = "selected",
) -> tuple[dict[str, Path], tuple[str, ...], tuple[str, ...]]:
    base = Path(data_root)
    roots = [
        base,
        base / "TEP_Selected_Data",
        base / "TEP-DATA" / "TEP_Selected_Data",
    ]
    normal_files, fault_files = tep_protocol_files(protocol)
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for file_name in (*normal_files, *fault_files):
        mode_id = _tep_mode_from_file_name(file_name)
        candidates = [
            candidate
            for root in roots
            for candidate in (root / file_name, root / f"M{mode_id}" / file_name)
        ]
        match = next((candidate for candidate in candidates if candidate.is_file()), None)
        if match is None:
            missing.append(file_name)
        else:
            resolved[file_name] = match
    if not missing:
        return resolved, normal_files, fault_files
    preview = ", ".join(missing[:8])
    if len(missing) > 8:
        preview += f", ... (+{len(missing) - 8} more)"
    raise FileNotFoundError(
        f"Cannot resolve TEP protocol={protocol!r} under {data_root!r}; "
        f"missing {len(missing)} files: {preview}. "
        "Both flat files and M1...M6 subdirectories are supported."
    )


def _load_tep_mat(file_path: Path) -> np.ndarray:
    if not HAS_SCIPY_IO or sio is None:
        raise RuntimeError("scipy is required to load TEP .mat files.")
    payload = sio.loadmat(file_path)
    key = file_path.stem
    if key not in payload:
        raise KeyError(f"Expected key {key!r} in {file_path}, found {sorted(payload.keys())}")
    arr = np.asarray(payload[key], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{file_path.name} expected 2D array, got shape {arr.shape}")
    if arr.shape[1] < TEP_NUM_INPUT_CHANNELS:
        raise ValueError(f"{file_path.name} has only {arr.shape[1]} columns, expected at least 53.")
    arr = arr[:, :TEP_NUM_INPUT_CHANNELS]
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"{file_path.name} contains NaN/Inf values.")
    return arr


def _split_tep_normal_segment(
    arr: np.ndarray,
    seq_len: int,
    val_ratio: float,
    gap: bool,
    min_train_windows: int,
    file_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    total_steps = len(arr)
    n_val = max(int(total_steps * float(val_ratio)), int(seq_len))
    gap_size = int(seq_len) if gap else 0
    n_train = total_steps - n_val - gap_size
    # Require enough raw timesteps to realize at least `min_train_windows`
    # legal windows, rather than multiplying windows by seq_len.
    min_train_steps = int(seq_len) + max(0, int(min_train_windows) - 1)
    if n_train < min_train_steps:
        raise ValueError(
            f"TEP normal file {file_name} is too short for seq_len={seq_len}, "
            f"val_ratio={val_ratio}, gap={gap_size}, min_train_windows={min_train_windows}."
        )
    train_seg = np.asarray(arr[:n_train], dtype=np.float32)
    val_seg = np.asarray(arr[n_train + gap_size :], dtype=np.float32)
    if len(val_seg) < int(seq_len):
        raise ValueError(f"TEP validation segment from {file_name} is shorter than seq_len={seq_len}.")
    print(
        f"[TEP] {file_name}: total={total_steps}, train={len(train_seg)}, gap={gap_size}, val={len(val_seg)}"
    )
    return train_seg, val_seg


def _validate_tep_fault_length(num_rows: int, seq_len: int, file_name: str) -> None:
    if int(num_rows) < int(seq_len):
        raise ValueError(
            f"TEP fault file {file_name} has {int(num_rows)} rows, shorter than seq_len={int(seq_len)}."
        )


def _load_tep_raw_dataset_bundle(config: CoReMADConfig) -> RawDatasetBundle:
    file_paths, normal_files, fault_files = resolve_tep_files(
        config.data_root,
        protocol=config.tep_protocol,
    )
    normal_train_segments: list[np.ndarray] = []
    normal_val_segments: list[np.ndarray] = []
    train_segment_names: list[str] = []
    val_segment_names: list[str] = []
    test_sequences: list[np.ndarray] = []
    test_sequence_labels: list[int] = []
    test_sequence_names: list[str] = []
    sequence_lengths: dict[str, int] = {}

    for file_name in normal_files:
        arr = _load_tep_mat(file_paths[file_name])
        sequence_lengths[Path(file_name).stem] = int(len(arr))
        train_seg, val_seg = _split_tep_normal_segment(
            arr,
            seq_len=config.seq_len,
            val_ratio=config.val_ratio,
            gap=config.val_gap,
            min_train_windows=config.val_min_train_windows,
            file_name=file_name,
        )
        normal_train_segments.append(train_seg)
        normal_val_segments.append(val_seg)
        train_segment_names.append(file_name[:-4])
        val_segment_names.append(file_name[:-4] + "_val")

    for file_name in fault_files:
        arr = _load_tep_mat(file_paths[file_name])
        _validate_tep_fault_length(len(arr), config.seq_len, file_name)
        sequence_lengths[Path(file_name).stem] = int(len(arr))
        test_sequences.append(arr)
        test_sequence_labels.append(1)
        test_sequence_names.append(file_name[:-4])

    train, train_ranges = _concat_segments(normal_train_segments)
    val, val_ranges = _concat_segments(normal_val_segments)
    test = _empty_feature_array(train.shape[1])
    test_labels = np.empty(0, dtype=np.float32)
    _validate_loaded_arrays(train, test, test_labels, "TEP")
    mode_ids = sorted({_tep_mode_from_file_name(name) for name in normal_files})
    file_fault_ids = sorted({_tep_file_fault_from_file_name(name) for name in fault_files})
    fault_lengths = [sequence_lengths[Path(name).stem] for name in fault_files]
    metadata = {
        "tep_protocol": str(config.tep_protocol),
        "mode_ids": mode_ids,
        "file_fault_ids": file_fault_ids,
        "official_idv_ids": sorted(tep_file_fault_to_idv(value) for value in file_fault_ids),
        "normal_files": [Path(name).stem for name in normal_files],
        "fault_files": [Path(name).stem for name in fault_files],
        "sequence_lengths": sequence_lengths,
        "n_normal_sequences": len(normal_files),
        "n_fault_sequences": len(fault_files),
        "n_full_length_fault_sequences": int(sum(length == 7201 for length in fault_lengths)),
        "n_short_fault_sequences": int(sum(length < 7201 for length in fault_lengths)),
        "official_fault_mapping": "IDV = 29 - d for d01...d28",
    }
    print(
        f"[TEP] loaded root={config.data_root} protocol={config.tep_protocol} "
        f"train={train.shape} val={val.shape} "
        f"sequence_eval_only=True n_test_sequences={len(test_sequence_labels)}"
    )
    return RawDatasetBundle(
        train=train,
        val=val,
        test=test,
        test_labels=test_labels,
        train_segment_ranges=train_ranges,
        val_segment_ranges=val_ranges,
        train_segment_names=list(train_segment_names),
        val_segment_names=list(val_segment_names),
        test_sequence_labels=np.asarray(test_sequence_labels, dtype=np.int32),
        test_sequence_names=list(test_sequence_names),
        test_sequences=[np.asarray(segment, dtype=np.float32) for segment in test_sequences],
        dataset_metadata=metadata,
    )


def load_raw_dataset(dataset: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detect_root = _resolve_detect_root(data_root)
    csv_path, train_lens = _resolve_detect_dataset_file(dataset, detect_root)
    dataset_name = csv_path.stem
    cache_path = _cache_path_for_dataset(detect_root, csv_path)

    cached = _try_load_cached_detect_dataset(cache_path, csv_path, train_lens, dataset_name)
    if cached is not None:
        return cached

    return _build_detect_dataset_cache(
        csv_path=csv_path,
        cache_path=cache_path,
        train_lens=train_lens,
        dataset_name=dataset_name,
    )


def load_raw_dataset_bundle(
    dataset: str,
    data_root: str | Path,
    config: Optional[CoReMADConfig] = None,
) -> RawDatasetBundle:
    dataset_key = str(dataset).strip().upper()
    data_format = str(config.data_format if config is not None else "auto").strip().lower()
    if data_format == "tsb_ad":
        return _load_tsb_ad_raw_dataset_bundle(dataset, data_root)
    if data_format == "tep" or (data_format == "auto" and dataset_key == "TEP"):
        tep_config = config or CoReMADConfig(dataset="TEP", data_root=str(data_root))
        return _load_tep_raw_dataset_bundle(tep_config)
    if data_format not in {"auto", "detect"}:
        raise ValueError(f"Unsupported data_format={data_format!r} for dataset {dataset!r}.")
    train, test, labels = load_raw_dataset(dataset, data_root)
    return RawDatasetBundle(train=train, test=test, test_labels=labels)


def build_data_bundle(config: CoReMADConfig) -> DataBundle:
    raw = load_raw_dataset_bundle(config.dataset, config.data_root, config=config)
    if raw.val is not None:
        normalizer = TimeSeriesNormalizer()
        train_stage_a = normalizer.fit_transform(raw.train)
        train_full = normalizer.transform(raw.train)
        val_stage_a = normalizer.transform(raw.val)
        test_norm = normalizer.transform(raw.test)
        return DataBundle(
            train_full=train_full,
            train_stage_a=train_stage_a,
            val_stage_a=val_stage_a,
            train_stage_a_start_indices=None,
            val_stage_a_start_indices=None,
            train_full_segment_ranges=raw.train_segment_ranges,
            train_stage_a_segment_ranges=raw.train_segment_ranges,
            val_stage_a_segment_ranges=raw.val_segment_ranges,
            test_segment_ranges=raw.test_segment_ranges,
            test=test_norm,
            test_labels=raw.test_labels,
            normalizer=normalizer,
        )
    if config.val_split_mode == "interleaved":
        train_starts, val_starts = _split_interleaved_windows(
            raw.train,
            ratio=config.val_ratio,
            seq_len=config.seq_len,
            stride=config.train_stride,
            min_train_windows=config.val_min_train_windows,
        )
        normalizer = TimeSeriesNormalizer()
        normalizer_fit_source = raw.train
        train_point_mask = _build_point_mask_from_start_indices(
            total_steps=len(raw.train),
            seq_len=config.seq_len,
            start_indices=train_starts,
        )
        if train_point_mask is not None:
            normalizer_fit_source = raw.train[train_point_mask]
            print(
                f"[Split] interleaved normalization uses train-only points: "
                f"{int(train_point_mask.sum())}/{len(raw.train)}"
            )
        normalizer.fit(normalizer_fit_source)
        train_full = normalizer.transform(raw.train)
        test_norm = normalizer.transform(raw.test)
        return DataBundle(
            train_full=train_full,
            train_stage_a=train_full,
            val_stage_a=None if val_starts is None else train_full,
            train_stage_a_start_indices=train_starts,
            val_stage_a_start_indices=val_starts,
            train_full_segment_ranges=raw.train_segment_ranges,
            train_stage_a_segment_ranges=raw.train_segment_ranges,
            val_stage_a_segment_ranges=raw.train_segment_ranges,
            test_segment_ranges=raw.test_segment_ranges,
            test=test_norm,
            test_labels=raw.test_labels,
            normalizer=normalizer,
        )

    train_stage_a_raw, val_stage_a_raw = _split_tail(
        raw.train,
        ratio=config.val_ratio,
        seq_len=config.seq_len,
        gap=config.val_gap,
        min_train_windows=config.val_min_train_windows,
    )
    normalizer = TimeSeriesNormalizer()
    train_stage_a = normalizer.fit_transform(train_stage_a_raw)
    train_full = normalizer.transform(raw.train)
    val_stage_a = None if val_stage_a_raw is None else normalizer.transform(val_stage_a_raw)
    test_norm = normalizer.transform(raw.test)
    assert not np.isnan(train_stage_a).any(), "train_stage_a contains NaN"
    assert not np.isnan(train_full).any(), "train_full contains NaN"
    if val_stage_a is not None:
        assert not np.isnan(val_stage_a).any(), "val_stage_a contains NaN"
    assert not np.isnan(test_norm).any(), "test_norm contains NaN"
    return DataBundle(
        train_full=train_full,
        train_stage_a=train_stage_a,
        val_stage_a=val_stage_a,
        train_stage_a_start_indices=None,
        val_stage_a_start_indices=None,
        train_full_segment_ranges=raw.train_segment_ranges,
        train_stage_a_segment_ranges=raw.train_segment_ranges,
        val_stage_a_segment_ranges=None,
        test_segment_ranges=raw.test_segment_ranges,
        test=test_norm,
        test_labels=raw.test_labels,
        normalizer=normalizer,
    )


def transform_raw_bundle(raw: RawDatasetBundle, normalizer: TimeSeriesNormalizer, config: CoReMADConfig) -> DataBundle:
    if not normalizer.is_fitted:
        raise RuntimeError("Normalizer must be fitted before transforming raw bundle.")
    if raw.val is not None:
        train_stage_a = normalizer.transform(raw.train)
        train_full = normalizer.transform(raw.train)
        val_stage_a = normalizer.transform(raw.val)
        test_norm = normalizer.transform(raw.test)
        return DataBundle(
            train_full=train_full,
            train_stage_a=train_stage_a,
            val_stage_a=val_stage_a,
            train_stage_a_start_indices=None,
            val_stage_a_start_indices=None,
            train_full_segment_ranges=raw.train_segment_ranges,
            train_stage_a_segment_ranges=raw.train_segment_ranges,
            val_stage_a_segment_ranges=raw.val_segment_ranges,
            test_segment_ranges=raw.test_segment_ranges,
            test=test_norm,
            test_labels=raw.test_labels,
            normalizer=normalizer,
        )
    if config.val_split_mode == "interleaved":
        train_starts, val_starts = _split_interleaved_windows(
            raw.train,
            ratio=config.val_ratio,
            seq_len=config.seq_len,
            stride=config.train_stride,
            min_train_windows=config.val_min_train_windows,
        )
        train_full = normalizer.transform(raw.train)
        test_norm = normalizer.transform(raw.test)
        return DataBundle(
            train_full=train_full,
            train_stage_a=train_full,
            val_stage_a=None if val_starts is None else train_full,
            train_stage_a_start_indices=train_starts,
            val_stage_a_start_indices=val_starts,
            train_full_segment_ranges=raw.train_segment_ranges,
            train_stage_a_segment_ranges=raw.train_segment_ranges,
            val_stage_a_segment_ranges=raw.train_segment_ranges,
            test_segment_ranges=raw.test_segment_ranges,
            test=test_norm,
            test_labels=raw.test_labels,
            normalizer=normalizer,
        )

    train_stage_a_raw, val_stage_a_raw = _split_tail(
        raw.train,
        ratio=config.val_ratio,
        seq_len=config.seq_len,
        gap=config.val_gap,
        min_train_windows=config.val_min_train_windows,
    )
    train_stage_a = normalizer.transform(train_stage_a_raw)
    train_full = normalizer.transform(raw.train)
    val_stage_a = None if val_stage_a_raw is None else normalizer.transform(val_stage_a_raw)
    test_norm = normalizer.transform(raw.test)
    assert not np.isnan(train_stage_a).any(), "train_stage_a contains NaN"
    assert not np.isnan(train_full).any(), "train_full contains NaN"
    if val_stage_a is not None:
        assert not np.isnan(val_stage_a).any(), "val_stage_a contains NaN"
    assert not np.isnan(test_norm).any(), "test_norm contains NaN"
    return DataBundle(
        train_full=train_full,
        train_stage_a=train_stage_a,
        val_stage_a=val_stage_a,
        train_stage_a_start_indices=None,
        val_stage_a_start_indices=None,
        train_full_segment_ranges=raw.train_segment_ranges,
        train_stage_a_segment_ranges=raw.train_segment_ranges,
        val_stage_a_segment_ranges=None,
        test_segment_ranges=raw.test_segment_ranges,
        test=test_norm,
        test_labels=raw.test_labels,
        normalizer=normalizer,
    )


def build_loader(
    data: np.ndarray,
    labels: Optional[np.ndarray],
    seq_len: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_windows: int = 0,
    drop_last: bool = False,
    start_indices: Optional[np.ndarray] = None,
    segment_ranges: Optional[np.ndarray] = None,
) -> DataLoader:
    effective_start_indices = start_indices
    if effective_start_indices is None and segment_ranges is not None:
        effective_start_indices = _compute_start_indices_from_segment_ranges(
            seq_len=seq_len,
            stride=stride,
            segment_ranges=segment_ranges,
            max_windows=max_windows,
        )
    dataset = SlidingWindowDataset(
        data=data,
        labels=labels,
        seq_len=seq_len,
        stride=stride,
        max_windows=max_windows,
        start_indices=effective_start_indices,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        **loader_kwargs,
    )
