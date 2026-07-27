from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rebuttal.muqn.analyze_purification_audit import (
    analyze_experiment,
    recover_injected_provenance_offset,
)
from scripts.rebuttal.muqn.collect_purification_sweep import (
    aggregate_e12_rows,
    parse_args,
    validate_e12_rows,
)


def test_recovers_provenance_offset_from_state_tail() -> None:
    protocol = {
        "clean_window_count": 2,
        "injected_provenance_offset": None,
        "injected_test_window_starts": [10, 20, 30],
    }
    state_starts = np.asarray([0, 4, 110, 120, 130], dtype=np.int64)

    offset, source = recover_injected_provenance_offset(protocol, state_starts)

    assert offset == 100
    assert source == "recovered_from_memory_audit"


def test_rejects_inconsistent_recovered_offsets() -> None:
    protocol = {
        "clean_window_count": 2,
        "injected_provenance_offset": None,
        "injected_test_window_starts": [10, 20, 30],
    }
    state_starts = np.asarray([0, 4, 110, 121, 130], dtype=np.int64)

    with pytest.raises(ValueError, match="single provenance offset"):
        recover_injected_provenance_offset(protocol, state_starts)


def _write_synthetic_e10(experiment_dir: Path) -> None:
    audit_dir = experiment_dir / "memory_audit"
    audit_dir.mkdir(parents=True)
    protocol = {
        "schema_version": 1,
        "clean_window_count": 2,
        "injected_provenance_offset": None,
        "injected_window_count": 4,
        "injection_events": [[10, 48]],
        "injected_test_window_starts": [10, 20, 30, 40],
    }
    (experiment_dir / "contamination_protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    np.savez(
        audit_dir / "memory_audit_state.npz",
        window_id=np.arange(6, dtype=np.int64),
        raw_start=np.asarray([0, 4, 110, 120, 130, 140], dtype=np.int64),
        prototype_distance=np.asarray(
            [0.1, 0.9, 0.2, 0.3, 0.4, 0.5],
            dtype=np.float64,
        ),
    )
    np.savez(
        audit_dir / "memory_audit_scale8.npz",
        patch_size=np.asarray(8, dtype=np.int64),
        window_id=np.arange(6, dtype=np.int64),
        raw_start=np.asarray([0, 4, 110, 120, 130, 140], dtype=np.int64),
        completion_score=np.asarray(
            [0.5, 0.6, 0.1, 0.2, 0.8, 0.9],
            dtype=np.float64,
        ),
        kept_after_purification=np.asarray(
            [True, True, True, False, False, False],
            dtype=bool,
        ),
        kept_after_coreset=np.asarray(
            [True, False, True, False, False, False],
            dtype=bool,
        ),
    )


def test_analyzes_recovered_low_error_anomaly_quantiles(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "e10"
    _write_synthetic_e10(experiment_dir)

    payload = analyze_experiment(
        experiment_dir,
        rare_normal_quantile=0.90,
        low_error_quantile=0.25,
        high_error_quantile=0.75,
    )

    assert payload["injected_provenance_offset"] == 100
    assert payload["provenance_source"] == "recovered_from_memory_audit"
    anomaly_rows = {
        row["error_band"]: row
        for row in payload["groups"]
        if row["group"] == "injected_anomaly"
    }
    assert anomaly_rows["all"]["count"] == 4
    assert anomaly_rows["low_q25"]["count"] == 1
    assert anomaly_rows["low_q25"]["purification_retention"] == 1.0
    assert anomaly_rows["mid_q25_q75"]["count"] == 2
    assert anomaly_rows["high_q75"]["count"] == 1


def test_e12_validation_rejects_zero_denominator() -> None:
    rows = [
        {
            "dataset": "MSL",
            "patch_size": 8,
            "group": "injected_anomaly",
            "error_band": "low_q25",
            "count": 0,
            "purification_retention": float("nan"),
            "purification_removal": float("nan"),
            "final_retention": float("nan"),
        }
    ]

    with pytest.raises(ValueError, match="zero denominator"):
        validate_e12_rows(rows)


def test_e12_fold_aggregation_is_patch_weighted() -> None:
    base = {
        "dataset": "MSL",
        "contamination_ratio": 0.01,
        "clean_ratio": 0.02,
        "patch_size": 8,
        "error_band": "low_q25",
        "error_quantile_low": 0.0,
        "error_quantile_high": 0.25,
    }
    rows = [
        {
            **base,
            "fold": 0,
            "count": 10,
            "purification_kept_count": 10,
            "final_kept_count": 5,
        },
        {
            **base,
            "fold": 1,
            "count": 30,
            "purification_kept_count": 0,
            "final_kept_count": 0,
        },
    ]

    summary = aggregate_e12_rows(rows)

    assert len(summary) == 1
    assert summary[0]["count"] == 40
    assert summary[0]["purification_retention"] == 0.25
    assert summary[0]["final_retention"] == 0.125


def test_collector_can_skip_unrelated_e11_inputs(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output-dir",
            str(tmp_path / "corrected"),
            "--skip-e11",
        ]
    )

    assert args.skip_e11 is True
