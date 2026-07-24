from __future__ import annotations

import numpy as np

from scripts.rebuttal.muqn.contamination import (
    build_nested_candidate_order,
    event_evaluation_mask,
    event_window_starts,
    filter_event_disjoint_window_starts,
    find_positive_events,
    sample_nested_contamination_starts,
    split_events_into_folds,
)


def test_find_positive_events_returns_half_open_ranges() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1])
    assert find_positive_events(labels) == [(1, 3), (4, 5), (6, 9)]


def test_event_folds_are_deterministic_and_disjoint() -> None:
    events = [(1, 3), (4, 5), (6, 9), (11, 14)]
    folds_a = split_events_into_folds(events, n_folds=3, seed=7)
    folds_b = split_events_into_folds(events, n_folds=3, seed=7)
    assert folds_a == folds_b
    assert sorted(event for fold in folds_a for event in fold) == events


def test_evaluation_mask_excludes_only_injected_events() -> None:
    mask = event_evaluation_mask(10, [(2, 4), (7, 9)])
    assert mask.tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
    ]


def test_contamination_windows_do_not_include_heldout_events() -> None:
    labels = np.zeros(20, dtype=np.int8)
    labels[5:7] = 1
    labels[9:11] = 1
    candidates = np.array([2, 4, 5, 6])
    filtered = filter_event_disjoint_window_starts(
        candidates,
        labels,
        injection_events=[(5, 7)],
        seq_len=5,
    )
    assert filtered.tolist() == [2, 4]


def test_event_windows_respect_segment_boundaries() -> None:
    starts = event_window_starts(
        events=[(8, 10)],
        data_length=20,
        seq_len=5,
        segment_ranges=np.array([[0, 10], [10, 20]]),
    )
    assert starts.tolist() == [4, 5]


def test_contamination_samples_are_nested_across_ratios() -> None:
    samples = sample_nested_contamination_starts(
        np.arange(1000, dtype=np.int64),
        clean_window_count=100,
        contamination_ratios=[0.01, 0.03, 0.05, 0.10],
        seed=42,
    )
    previous: set[int] = set()
    for ratio in (0.01, 0.03, 0.05, 0.10):
        current = set(samples[ratio].tolist())
        assert previous <= current
        previous = current


def test_replacement_stream_is_deterministic_when_unique_pool_is_too_small() -> None:
    candidates = np.asarray([10, 20, 30], dtype=np.int64)
    first = build_nested_candidate_order(
        candidates,
        required_count=8,
        seed=42,
        allow_replacement=True,
    )
    second = build_nested_candidate_order(
        candidates,
        required_count=8,
        seed=42,
        allow_replacement=True,
    )
    assert np.array_equal(first, second)
    assert len(first) == 8
    assert set(first.tolist()) == set(candidates.tolist())
