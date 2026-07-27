from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize memory purification retention for rare normal and injected patches."
    )
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--rare_normal_quantile", type=float, default=0.90)
    parser.add_argument("--low_error_quantile", type=float, default=0.25)
    parser.add_argument("--high_error_quantile", type=float, default=0.75)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def _validate_quantiles(
    rare_normal_quantile: float,
    low_error_quantile: float,
    high_error_quantile: float,
) -> None:
    if not 0.0 < rare_normal_quantile < 1.0:
        raise ValueError("rare_normal_quantile must be in (0, 1)")
    if not 0.0 < low_error_quantile < high_error_quantile < 1.0:
        raise ValueError(
            "low_error_quantile and high_error_quantile must satisfy "
            "0 < low < high < 1"
        )


def _is_contamination_protocol(protocol: dict) -> bool:
    return bool(
        protocol.get("injection_events")
        or protocol.get("injected_test_window_starts")
        or protocol.get("injected_window_count")
    )


def recover_injected_provenance_offset(
    protocol: dict,
    state_starts: np.ndarray,
) -> tuple[int | None, str]:
    """Return a validated provenance offset without mutating the protocol."""
    explicit = protocol.get("injected_provenance_offset")
    injected_starts = np.asarray(
        protocol.get("injected_test_window_starts", []),
        dtype=np.int64,
    )
    if not _is_contamination_protocol(protocol):
        return None, "not_applicable"
    if injected_starts.size == 0:
        raise ValueError(
            "Contamination protocol has no injected_test_window_starts; "
            "injected provenance cannot be recovered."
        )

    clean_count_value = protocol.get("clean_window_count")
    clean_count = (
        int(clean_count_value)
        if clean_count_value is not None
        else int(len(state_starts) - len(injected_starts))
    )
    if clean_count < 0 or clean_count + len(injected_starts) != len(state_starts):
        raise ValueError(
            "Memory-audit state length does not match clean_window_count plus "
            "injected_test_window_starts."
        )
    injected_state_starts = np.asarray(state_starts[clean_count:], dtype=np.int64)
    offsets = injected_state_starts - injected_starts
    if offsets.size == 0 or not np.all(offsets == offsets[0]):
        raise ValueError("Injected state rows do not imply a single provenance offset.")
    recovered = int(offsets[0])
    if explicit is not None:
        explicit = int(explicit)
        if explicit != recovered:
            raise ValueError(
                f"Protocol provenance offset {explicit} disagrees with "
                f"memory-audit offset {recovered}."
            )
        return explicit, "protocol_validated_against_memory_audit"
    return recovered, "recovered_from_memory_audit"


def _summarize_group(
    *,
    patch_size: int,
    group: str,
    error_band: str,
    mask: np.ndarray,
    completion: np.ndarray,
    kept_after_purification: np.ndarray,
    kept_after_coreset: np.ndarray,
    error_quantile_low: float | None = None,
    error_quantile_high: float | None = None,
    error_threshold_low: float | None = None,
    error_threshold_high: float | None = None,
) -> dict:
    count = int(mask.sum())
    purification_kept_count = int(kept_after_purification[mask].sum())
    final_kept_count = int(kept_after_coreset[mask].sum())
    purification_retention = (
        float(purification_kept_count / count) if count else float("nan")
    )
    final_retention = float(final_kept_count / count) if count else float("nan")
    finite_completion = completion[mask & np.isfinite(completion)]
    return {
        "patch_size": patch_size,
        "group": group,
        "error_band": error_band,
        "error_quantile_low": error_quantile_low,
        "error_quantile_high": error_quantile_high,
        "error_threshold_low": error_threshold_low,
        "error_threshold_high": error_threshold_high,
        "count": count,
        "purification_kept_count": purification_kept_count,
        "final_kept_count": final_kept_count,
        "purification_retention": purification_retention,
        "purification_removal": (
            1.0 - purification_retention
            if np.isfinite(purification_retention)
            else float("nan")
        ),
        "final_retention": final_retention,
        "completion_mean": (
            float(finite_completion.mean()) if finite_completion.size else float("nan")
        ),
        "completion_p95": (
            float(np.quantile(finite_completion, 0.95))
            if finite_completion.size
            else float("nan")
        ),
    }


def analyze_experiment(
    experiment_dir: Path,
    *,
    rare_normal_quantile: float = 0.90,
    low_error_quantile: float = 0.25,
    high_error_quantile: float = 0.75,
) -> dict:
    _validate_quantiles(
        rare_normal_quantile,
        low_error_quantile,
        high_error_quantile,
    )
    experiment_dir = experiment_dir.resolve()
    audit_dir = experiment_dir / "memory_audit"
    state_path = audit_dir / "memory_audit_state.npz"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing {state_path}; rebuild Stage B with --memory_audit_mode full."
        )
    protocol_path = experiment_dir / "contamination_protocol.json"
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else {}
    )

    with np.load(state_path, allow_pickle=False) as state:
        state_window_ids = np.asarray(state["window_id"], dtype=np.int64)
        state_starts = np.asarray(state["raw_start"], dtype=np.int64)
        state_rarity = np.asarray(state["prototype_distance"], dtype=np.float64)
    provenance_offset, provenance_source = recover_injected_provenance_offset(
        protocol,
        state_starts,
    )
    normal_state_mask = (
        np.ones(len(state_window_ids), dtype=bool)
        if provenance_offset is None
        else state_starts < provenance_offset
    )
    finite_normal_rarity = state_rarity[normal_state_mask & np.isfinite(state_rarity)]
    if finite_normal_rarity.size == 0:
        raise ValueError("No finite normal prototype distances are available.")
    rarity_threshold = float(np.quantile(finite_normal_rarity, rare_normal_quantile))
    rare_window_ids = set(
        state_window_ids[
            normal_state_mask
            & np.isfinite(state_rarity)
            & (state_rarity >= rarity_threshold)
        ].tolist()
    )

    summaries = []
    scale_paths = sorted(audit_dir.glob("memory_audit_scale*.npz"))
    if not scale_paths:
        raise FileNotFoundError(f"No scale audit files found under {audit_dir}")
    for path in scale_paths:
        with np.load(path, allow_pickle=False) as audit:
            patch_size = int(np.asarray(audit["patch_size"]).item())
            window_ids = np.asarray(audit["window_id"], dtype=np.int64)
            raw_starts = np.asarray(audit["raw_start"], dtype=np.int64)
            completion = np.asarray(audit["completion_score"], dtype=np.float64)
            kept_clean = np.asarray(audit["kept_after_purification"], dtype=bool)
            kept_final = np.asarray(audit["kept_after_coreset"], dtype=bool)

        lengths = {
            len(window_ids),
            len(raw_starts),
            len(completion),
            len(kept_clean),
            len(kept_final),
        }
        if len(lengths) != 1:
            raise ValueError(f"Length mismatch in {path}")
        rare_normal_mask = np.asarray(
            [int(window_id) in rare_window_ids for window_id in window_ids],
            dtype=bool,
        )
        injected_source_mask = (
            np.zeros(len(raw_starts), dtype=bool)
            if provenance_offset is None
            else raw_starts >= provenance_offset
        )
        injected_anomaly_mask = np.zeros(len(raw_starts), dtype=bool)
        if provenance_offset is not None:
            test_patch_starts = raw_starts - provenance_offset
            for event_start, event_end in protocol.get("injection_events", []):
                injected_anomaly_mask |= (
                    injected_source_mask
                    & (test_patch_starts < int(event_end))
                    & (test_patch_starts + patch_size > int(event_start))
                )
        injected_context_mask = injected_source_mask & ~injected_anomaly_mask
        ordinary_normal_mask = ~rare_normal_mask & ~injected_source_mask
        for group, mask in (
            ("ordinary_normal", ordinary_normal_mask),
            ("rare_normal", rare_normal_mask),
            ("injected_context", injected_context_mask),
        ):
            summaries.append(
                _summarize_group(
                    patch_size=patch_size,
                    group=group,
                    error_band="all",
                    mask=mask,
                    completion=completion,
                    kept_after_purification=kept_clean,
                    kept_after_coreset=kept_final,
                )
            )

        if provenance_offset is None:
            continue
        anomaly_count = int(injected_anomaly_mask.sum())
        if anomaly_count == 0:
            raise ValueError(
                f"No injected anomaly patches were identified in {path}; "
                "E12 would have a zero denominator."
            )
        anomaly_completion = completion[injected_anomaly_mask]
        if not np.isfinite(anomaly_completion).all():
            raise ValueError(
                f"Injected anomaly completion scores are non-finite in {path}."
            )
        low_threshold = float(np.quantile(anomaly_completion, low_error_quantile))
        high_threshold = float(np.quantile(anomaly_completion, high_error_quantile))
        error_bands = (
            (
                "all",
                injected_anomaly_mask,
                0.0,
                1.0,
                float(anomaly_completion.min()),
                float(anomaly_completion.max()),
            ),
            (
                f"low_q{int(low_error_quantile * 100)}",
                injected_anomaly_mask & (completion <= low_threshold),
                0.0,
                low_error_quantile,
                float(anomaly_completion.min()),
                low_threshold,
            ),
            (
                f"mid_q{int(low_error_quantile * 100)}_q{int(high_error_quantile * 100)}",
                injected_anomaly_mask
                & (completion > low_threshold)
                & (completion <= high_threshold),
                low_error_quantile,
                high_error_quantile,
                low_threshold,
                high_threshold,
            ),
            (
                f"high_q{int(high_error_quantile * 100)}",
                injected_anomaly_mask & (completion > high_threshold),
                high_error_quantile,
                1.0,
                high_threshold,
                float(anomaly_completion.max()),
            ),
        )
        for band, mask, q_low, q_high, threshold_low, threshold_high in error_bands:
            summaries.append(
                _summarize_group(
                    patch_size=patch_size,
                    group="injected_anomaly",
                    error_band=band,
                    mask=mask,
                    completion=completion,
                    kept_after_purification=kept_clean,
                    kept_after_coreset=kept_final,
                    error_quantile_low=q_low,
                    error_quantile_high=q_high,
                    error_threshold_low=threshold_low,
                    error_threshold_high=threshold_high,
                )
            )

    return {
        "schema_version": 2,
        "experiment_dir": str(experiment_dir),
        "rare_normal_quantile": float(rare_normal_quantile),
        "low_error_quantile": float(low_error_quantile),
        "high_error_quantile": float(high_error_quantile),
        "rarity_threshold": rarity_threshold,
        "injected_provenance_offset": provenance_offset,
        "provenance_source": provenance_source,
        "groups": summaries,
    }


def _write_payload(
    payload: dict,
    output_dir: Path,
    protocol_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "purification_audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summaries = payload["groups"]
    csv_path = output_dir / "purification_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    if payload["provenance_source"] == "recovered_from_memory_audit":
        original_bytes = protocol_path.read_bytes()
        original_protocol = json.loads(original_bytes.decode("utf-8"))
        recovered_protocol = {
            **original_protocol,
            "injected_provenance_offset": payload["injected_provenance_offset"],
            "provenance_recovery": {
                "method": (
                    "memory_audit_state tail raw_start minus "
                    "injected_test_window_starts"
                ),
                "original_protocol_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "source_protocol": str(protocol_path),
            },
        }
        (output_dir / "recovered_contamination_protocol.json").write_text(
            json.dumps(recovered_protocol, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"[muQn] wrote {json_path}")
    print(f"[muQn] wrote {csv_path}")


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    payload = analyze_experiment(
        experiment_dir,
        rare_normal_quantile=args.rare_normal_quantile,
        low_error_quantile=args.low_error_quantile,
        high_error_quantile=args.high_error_quantile,
    )
    output_dir = args.output_dir or (experiment_dir / "rebuttal_muqn" / "purification")
    _write_payload(
        payload,
        output_dir,
        experiment_dir / "contamination_protocol.json",
    )


if __name__ == "__main__":
    main()
