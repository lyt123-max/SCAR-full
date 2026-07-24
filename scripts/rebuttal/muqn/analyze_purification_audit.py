from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize memory purification retention for rare normal and injected patches."
    )
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--rare_normal_quantile", type=float, default=0.90)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def _rate(mask: np.ndarray, selected: np.ndarray) -> tuple[int, float]:
    count = int(mask.sum())
    if count == 0:
        return 0, float("nan")
    return count, float(selected[mask].mean())


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
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
    provenance_offset = protocol.get("injected_provenance_offset")

    with np.load(state_path, allow_pickle=False) as state:
        state_window_ids = np.asarray(state["window_id"], dtype=np.int64)
        state_starts = np.asarray(state["raw_start"], dtype=np.int64)
        state_rarity = np.asarray(state["prototype_distance"], dtype=np.float64)
    if provenance_offset is None:
        normal_state_mask = np.ones(len(state_window_ids), dtype=bool)
    else:
        normal_state_mask = state_starts < int(provenance_offset)
    rarity_threshold = float(
        np.quantile(state_rarity[normal_state_mask], args.rare_normal_quantile)
    )
    rare_window_ids = set(
        state_window_ids[normal_state_mask & (state_rarity >= rarity_threshold)].tolist()
    )

    summaries = []
    for path in sorted(audit_dir.glob("memory_audit_scale*.npz")):
        if path.name == "memory_audit_state.npz":
            continue
        with np.load(path, allow_pickle=False) as audit:
            patch_size = int(np.asarray(audit["patch_size"]).item())
            window_ids = np.asarray(audit["window_id"], dtype=np.int64)
            raw_starts = np.asarray(audit["raw_start"], dtype=np.int64)
            completion = np.asarray(audit["completion_score"], dtype=np.float64)
            kept_clean = np.asarray(audit["kept_after_purification"], dtype=bool)
            kept_final = np.asarray(audit["kept_after_coreset"], dtype=bool)

        rare_normal_mask = np.asarray(
            [int(window_id) in rare_window_ids for window_id in window_ids],
            dtype=bool,
        )
        injected_source_mask = (
            np.zeros(len(raw_starts), dtype=bool)
            if provenance_offset is None
            else raw_starts >= int(provenance_offset)
        )
        injected_anomaly_mask = np.zeros(len(raw_starts), dtype=bool)
        if provenance_offset is not None:
            test_patch_starts = raw_starts - int(provenance_offset)
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
            ("injected_anomaly", injected_anomaly_mask),
        ):
            count, purification_retention = _rate(mask, kept_clean)
            _, final_retention = _rate(mask, kept_final)
            finite_completion = completion[mask & np.isfinite(completion)]
            summaries.append(
                {
                    "patch_size": patch_size,
                    "group": group,
                    "count": count,
                    "purification_retention": purification_retention,
                    "purification_removal": (
                        1.0 - purification_retention
                        if np.isfinite(purification_retention)
                        else float("nan")
                    ),
                    "final_retention": final_retention,
                    "completion_mean": (
                        float(finite_completion.mean())
                        if finite_completion.size
                        else float("nan")
                    ),
                    "completion_p95": (
                        float(np.quantile(finite_completion, 0.95))
                        if finite_completion.size
                        else float("nan")
                    ),
                }
            )

    output_dir = args.output_dir or (experiment_dir / "rebuttal_muqn" / "purification")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment_dir": str(experiment_dir),
        "rare_normal_quantile": float(args.rare_normal_quantile),
        "rarity_threshold": rarity_threshold,
        "injected_provenance_offset": provenance_offset,
        "groups": summaries,
    }
    json_path = output_dir / "purification_audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    csv_path = output_dir / "purification_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[muQn] wrote {json_path}")
    print(f"[muQn] wrote {csv_path}")


if __name__ == "__main__":
    main()
