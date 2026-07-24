from __future__ import annotations

from collections.abc import Sequence

import numpy as np


Event = tuple[int, int]


def find_positive_events(labels: np.ndarray) -> list[Event]:
    values = np.asarray(labels).reshape(-1) > 0
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def split_events_into_folds(
    events: Sequence[Event],
    n_folds: int,
    seed: int,
) -> list[list[Event]]:
    n_folds = int(n_folds)
    if n_folds <= 1:
        raise ValueError("n_folds must be greater than one.")
    ordered = [tuple(map(int, event)) for event in events]
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(ordered))
    folds: list[list[Event]] = [[] for _ in range(n_folds)]
    for position, event_index in enumerate(permutation.tolist()):
        folds[position % n_folds].append(ordered[event_index])
    return [sorted(fold) for fold in folds]


def event_evaluation_mask(length: int, excluded_events: Sequence[Event]) -> np.ndarray:
    mask = np.ones(int(length), dtype=bool)
    for start, end in excluded_events:
        if start < 0 or end < start or end > length:
            raise ValueError(f"Invalid event range ({start}, {end}) for length {length}.")
        mask[int(start) : int(end)] = False
    return mask


def event_window_starts(
    events: Sequence[Event],
    data_length: int,
    seq_len: int,
    stride: int = 1,
    segment_ranges: np.ndarray | None = None,
) -> np.ndarray:
    valid_starts: set[int] = set()
    max_start = int(data_length) - int(seq_len)
    if max_start < 0:
        return np.empty(0, dtype=np.int64)
    ranges = (
        np.asarray([[0, int(data_length)]], dtype=np.int64)
        if segment_ranges is None
        else np.asarray(segment_ranges, dtype=np.int64)
    )
    for event_start, event_end in events:
        containing = ranges[
            (ranges[:, 0] <= int(event_start)) & (ranges[:, 1] >= int(event_end))
        ]
        if len(containing) != 1:
            raise ValueError(
                f"Event ({event_start}, {event_end}) is not contained in one segment."
            )
        segment_start, segment_end = containing[0].tolist()
        first = max(int(segment_start), int(event_start) - int(seq_len) + 1)
        last = min(int(segment_end) - int(seq_len), int(event_end) - 1)
        valid_starts.update(range(first, last + 1, max(1, int(stride))))
    return np.asarray(sorted(valid_starts), dtype=np.int64)


def filter_event_disjoint_window_starts(
    candidate_starts: np.ndarray,
    labels: np.ndarray,
    injection_events: Sequence[Event],
    seq_len: int,
) -> np.ndarray:
    labels_positive = np.asarray(labels).reshape(-1) > 0
    injection_mask = ~event_evaluation_mask(len(labels_positive), injection_events)
    allowed = []
    for start in np.asarray(candidate_starts, dtype=np.int64).tolist():
        end = int(start) + int(seq_len)
        window_foreign_anomaly = labels_positive[int(start) : end] & ~injection_mask[
            int(start) : end
        ]
        if not window_foreign_anomaly.any():
            allowed.append(int(start))
    return np.asarray(allowed, dtype=np.int64)


def sample_contamination_starts(
    candidate_starts: np.ndarray,
    clean_window_count: int,
    contamination_ratio: float,
    seed: int,
) -> np.ndarray:
    return sample_nested_contamination_starts(
        candidate_starts,
        clean_window_count=clean_window_count,
        contamination_ratios=[contamination_ratio],
        seed=seed,
    )[float(contamination_ratio)]


def build_nested_candidate_order(
    candidate_starts: np.ndarray,
    *,
    required_count: int,
    seed: int,
    allow_replacement: bool = False,
) -> np.ndarray:
    candidates = np.asarray(candidate_starts, dtype=np.int64)
    required_count = int(required_count)
    if required_count < 0:
        raise ValueError("required_count must be non-negative.")
    if required_count == 0:
        return np.empty(0, dtype=np.int64)
    if len(candidates) == 0:
        raise ValueError("No event windows are available for contamination.")
    rng = np.random.default_rng(int(seed))
    if required_count <= len(candidates):
        return candidates[rng.permutation(len(candidates))][:required_count]
    if not allow_replacement:
        raise ValueError(
            f"Requested {required_count} injected windows but only "
            f"{len(candidates)} event windows exist."
        )
    chunks = []
    remaining = required_count
    while remaining > 0:
        permutation = candidates[rng.permutation(len(candidates))]
        take = min(remaining, len(permutation))
        chunks.append(permutation[:take])
        remaining -= take
    return np.concatenate(chunks).astype(np.int64, copy=False)


def sample_nested_contamination_starts(
    candidate_starts: np.ndarray,
    clean_window_count: int,
    contamination_ratios: Sequence[float],
    seed: int,
    allow_replacement: bool = False,
) -> dict[float, np.ndarray]:
    ratios = [float(value) for value in contamination_ratios]
    if len(set(ratios)) != len(ratios):
        raise ValueError("contamination_ratios must be unique.")
    for ratio in ratios:
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError("contamination_ratio must be in [0, 1).")
    candidates = np.asarray(candidate_starts, dtype=np.int64)
    targets = {
        ratio: (
            0
            if ratio == 0.0
            else int(round(int(clean_window_count) * ratio / (1.0 - ratio)))
        )
        for ratio in ratios
    }
    ordered = build_nested_candidate_order(
        candidates,
        required_count=max(targets.values(), default=0),
        seed=seed,
        allow_replacement=allow_replacement,
    )
    result: dict[float, np.ndarray] = {}
    for ratio in sorted(ratios):
        if ratio == 0.0:
            result[ratio] = np.empty(0, dtype=np.int64)
            continue
        target = targets[ratio]
        result[ratio] = np.sort(ordered[:target].astype(np.int64))
    return result
