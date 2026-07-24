from __future__ import annotations

import numpy as np

from scripts.rebuttal.muqn.proxy_metrics import (
    block_bootstrap_mean,
    extract_state_proxy_features,
    jaccard_index,
    lagged_jaccard,
    normalize_proxy_features,
)


def test_state_proxy_features_capture_level_trend_and_energy() -> None:
    time = np.arange(8, dtype=np.float64)
    windows = np.stack(
        [
            np.stack([time, 2.0 * time], axis=1),
            np.stack([time + 10.0, 2.0 * time + 5.0], axis=1),
        ]
    )
    features = extract_state_proxy_features(windows)

    assert features.shape == (2, 6)
    assert np.allclose(features[1, :2] - features[0, :2], [10.0, 5.0])
    assert np.allclose(features[0, 2:4], features[1, 2:4])


def test_normalization_uses_train_reference_statistics() -> None:
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    values = np.array([[1.0, 3.0]])
    normalized, stats = normalize_proxy_features(train, values)

    assert np.allclose(normalized, 0.0)
    assert np.allclose(stats["center"], [1.0, 3.0])


def test_jaccard_and_lagged_jaccard_ignore_padding_ids() -> None:
    assert jaccard_index([1, 2, -1], [2, 3, -1]) == 1.0 / 3.0
    sets = np.array([[1, 2, -1], [2, 1, -1], [9, 10, -1]])
    values = lagged_jaccard(sets, lag=1)
    assert np.allclose(values, [1.0, 0.0])


def test_block_bootstrap_is_deterministic_and_contains_constant_mean() -> None:
    summary_a = block_bootstrap_mean(
        np.ones(20),
        block_size=4,
        n_bootstrap=100,
        seed=7,
    )
    summary_b = block_bootstrap_mean(
        np.ones(20),
        block_size=4,
        n_bootstrap=100,
        seed=7,
    )

    assert summary_a == summary_b
    assert summary_a["mean"] == 1.0
    assert summary_a["ci_low"] == 1.0
    assert summary_a["ci_high"] == 1.0
