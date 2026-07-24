from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _as_window_batch(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected windows with shape [N, L, C], got {values.shape}.")
    if values.shape[0] == 0 or values.shape[1] < 2 or values.shape[2] == 0:
        raise ValueError("Windows must contain at least one sample, two time steps, and one channel.")
    return values


def extract_state_proxy_features(
    windows: np.ndarray,
    low_frequency_bins: int = 3,
) -> np.ndarray:
    """Return per-channel level, trend, and low-frequency energy proxies."""
    values = _as_window_batch(windows)
    n_steps = values.shape[1]
    time = np.arange(n_steps, dtype=np.float64)
    centered_time = time - time.mean()
    time_energy = np.square(centered_time).sum()

    level = values.mean(axis=1)
    trend = np.einsum(
        "nlc,l->nc",
        values - level[:, None, :],
        centered_time,
    ) / max(time_energy, np.finfo(np.float64).eps)

    centered = values - level[:, None, :]
    spectrum = np.fft.rfft(centered, axis=1)
    available_bins = max(0, spectrum.shape[1] - 1)
    use_bins = min(max(1, int(low_frequency_bins)), available_bins)
    if use_bins == 0:
        low_frequency_energy = np.zeros_like(level)
    else:
        low_frequency_energy = np.square(np.abs(spectrum[:, 1 : use_bins + 1, :])).mean(axis=1)
        low_frequency_energy /= float(n_steps * n_steps)
    return np.concatenate([level, trend, low_frequency_energy], axis=1)


def normalize_proxy_features(
    train_features: np.ndarray,
    values: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reference = np.asarray(train_features, dtype=np.float64)
    target = np.asarray(values, dtype=np.float64)
    if reference.ndim != 2 or target.ndim != 2 or reference.shape[1] != target.shape[1]:
        raise ValueError("Train and target proxy features must be aligned 2D arrays.")
    center = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.where(scale > eps, scale, 1.0)
    return (target - center) / scale, {"center": center, "scale": scale}


def paired_proxy_distances(
    query_features: np.ndarray,
    neighbor_features: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    query = np.asarray(query_features, dtype=np.float64)
    neighbor = np.asarray(neighbor_features, dtype=np.float64)
    if query.shape != neighbor.shape or query.ndim != 2:
        raise ValueError("Query and neighbor proxy features must have the same [N, D] shape.")
    l1 = np.abs(query - neighbor).mean(axis=1)
    query_centered = query - query.mean(axis=1, keepdims=True)
    neighbor_centered = neighbor - neighbor.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(query_centered, axis=1) * np.linalg.norm(
        neighbor_centered, axis=1
    )
    correlation = np.sum(query_centered * neighbor_centered, axis=1) / np.maximum(
        denominator, eps
    )
    return {
        "proxy_l1": l1,
        "proxy_correlation_distance": 1.0 - np.clip(correlation, -1.0, 1.0),
    }


def jaccard_index(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = {int(value) for value in left if int(value) >= 0}
    right_set = {int(value) for value in right if int(value) >= 0}
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def lagged_jaccard(neighbor_sets: np.ndarray, lag: int = 1) -> np.ndarray:
    values = np.asarray(neighbor_sets)
    if values.ndim != 2:
        raise ValueError("neighbor_sets must have shape [N, K].")
    lag = int(lag)
    if lag <= 0 or lag >= values.shape[0]:
        raise ValueError("lag must be positive and smaller than the number of rows.")
    return np.asarray(
        [
            jaccard_index(values[index], values[index + lag])
            for index in range(values.shape[0] - lag)
        ],
        dtype=np.float64,
    )


def block_bootstrap_mean(
    values: np.ndarray,
    block_size: int,
    n_bootstrap: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("At least one finite value is required.")
    block_size = max(1, min(int(block_size), samples.size))
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    rng = np.random.default_rng(int(seed))
    starts = np.arange(samples.size)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    blocks_per_draw = int(np.ceil(samples.size / block_size))
    offsets = np.arange(block_size)
    for draw_index in range(n_bootstrap):
        selected_starts = rng.choice(starts, size=blocks_per_draw, replace=True)
        indices = (selected_starts[:, None] + offsets[None, :]) % samples.size
        draws[draw_index] = samples[indices.reshape(-1)[: samples.size]].mean()
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(samples.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "n": int(samples.size),
        "block_size": int(block_size),
        "n_bootstrap": n_bootstrap,
        "seed": int(seed),
    }
