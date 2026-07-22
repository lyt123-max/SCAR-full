from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tep_common import (
    empirical_percentile,
    load_sequence_scores_with_meta,
    load_train_state_meta,
    load_window_logs,
    np_to_python,
)


EVIDENCE_KEYS = ["memory_distance", "state_novelty", "completion_scale8", "completion_scale32"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute offline TEP mechanism metrics from exported logs.")
    parser.add_argument("--experiment_dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--short_dir", type=Path, default=None)
    parser.add_argument("--long_dir", type=Path, default=None)
    parser.add_argument("--multi_dir", type=Path, default=None)
    parser.add_argument("--log_subdir", type=str, default="tep_mechanism")
    parser.add_argument("--smc_k", type=int, default=10)
    parser.add_argument("--smr_k", type=int, default=10)
    parser.add_argument("--fpr_quantile", type=float, default=0.95)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def _pairwise_knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n <= 1:
        return np.empty((n, 0), dtype=np.int64)
    k_eff = min(max(1, int(k)), n - 1)
    dists = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(dists, np.inf)
    part = np.argpartition(dists, kth=np.arange(k_eff), axis=1)[:, :k_eff]
    part_dists = np.take_along_axis(dists, part, axis=1)
    order = np.argsort(part_dists, axis=1)
    return np.take_along_axis(part, order, axis=1)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def _normalized_entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(float(probs.size)))


def _sequence_reference_key(score_key: str) -> str:
    return "final" if score_key == "selected" else score_key


def _build_fault_sequence_rows(
    audit_logs: dict[str, np.ndarray],
    fault_sequence_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audit_reference = {
        "selected": np.asarray(audit_logs["final"], dtype=np.float64),
        "memory_distance": np.asarray(audit_logs["memory_distance"], dtype=np.float64),
        "state_novelty": np.asarray(audit_logs["state_novelty"], dtype=np.float64),
        "completion_scale8": np.asarray(audit_logs["completion_scale8"], dtype=np.float64),
        "completion_scale32": np.asarray(audit_logs["completion_scale32"], dtype=np.float64),
    }
    rows: list[dict[str, Any]] = []
    for record in fault_sequence_records:
        row = dict(record)
        if "memory_distance" not in row and "knn_distance" in row:
            row["memory_distance"] = float(row["knn_distance"])
        for score_key, reference in audit_reference.items():
            value = float(record.get(score_key, float("nan")))
            if score_key == "memory_distance" and not np.isfinite(value):
                value = float(record.get("knn_distance", float("nan")))
            calibrated = float(empirical_percentile(reference, np.asarray([value], dtype=np.float64))[0])
            row[f"{score_key}_calibrated"] = calibrated
        evidence_scores = {
            key: float(row.get(f"{key}_calibrated", float("nan")))
            for key in EVIDENCE_KEYS
        }
        finite_items = [(key, value) for key, value in evidence_scores.items() if np.isfinite(value)]
        row["dominant_evidence"] = max(finite_items, key=lambda item: item[1])[0] if finite_items else "unknown"
        rows.append(row)
    return rows


def _compute_prototype_metrics(train_state_meta: dict[str, np.ndarray]) -> tuple[float, float]:
    prototype_label = np.asarray(train_state_meta["prototype_label"], dtype=np.int32)
    mode_id = np.asarray(train_state_meta["mode_id"], dtype=np.int32)
    valid_mask = prototype_label >= 0
    if not np.any(valid_mask):
        return float("nan"), float("nan")

    weights: list[float] = []
    purities: list[float] = []
    entropies: list[float] = []
    for current_proto in sorted(np.unique(prototype_label[valid_mask]).tolist()):
        group_modes = mode_id[prototype_label == current_proto]
        if group_modes.size == 0:
            continue
        counts = np.asarray([np.sum(group_modes == mode) for mode in sorted(np.unique(group_modes).tolist())], dtype=np.float64)
        probabilities = counts / max(counts.sum(), 1e-12)
        purities.append(float(np.max(probabilities)))
        entropies.append(_normalized_entropy(probabilities))
        weights.append(float(group_modes.size))
    if not weights:
        return float("nan"), float("nan")
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / max(weights_arr.sum(), 1e-12)
    purity = float(np.sum(weights_arr * np.asarray(purities, dtype=np.float64)))
    entropy = float(np.sum(weights_arr * np.asarray(entropies, dtype=np.float64)))
    return purity, entropy


def compute_single_metrics(
    audit_logs: dict[str, np.ndarray],
    fault_logs: dict[str, np.ndarray],
    train_state_meta: dict[str, np.ndarray],
    fault_sequence_records: list[dict[str, Any]],
    smc_k: int,
    smr_k: int,
    fpr_quantile: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit_mode_id = np.asarray(audit_logs["mode_id"], dtype=np.int32)
    audit_state = np.asarray(audit_logs["state_vec"], dtype=np.float64)
    audit_final = np.asarray(audit_logs["final"], dtype=np.float64)
    audit_memory = np.asarray(audit_logs["memory_distance"], dtype=np.float64)

    fault_mode_id = np.asarray(fault_logs["mode_id"], dtype=np.int32)
    fault_memory = np.asarray(fault_logs["memory_distance"], dtype=np.float64)
    fault_topk_mode_ids = np.asarray(fault_logs["topk_neighbor_mode_ids"], dtype=np.int32)

    knn_idx = _pairwise_knn_indices(audit_state, smc_k)
    smc_values = [
        float(np.mean(audit_mode_id[row_neighbors] == audit_mode_id[row_idx]))
        for row_idx, row_neighbors in enumerate(knn_idx)
        if row_neighbors.size > 0
    ]
    smc_at_k = _safe_mean(smc_values)

    mode_means = []
    within_terms = []
    global_mean = np.mean(audit_state, axis=0) if len(audit_state) else np.zeros(0, dtype=np.float64)
    for current_mode in sorted(np.unique(audit_mode_id).tolist()):
        group = audit_state[audit_mode_id == current_mode]
        if len(group) == 0:
            continue
        group_mean = np.mean(group, axis=0)
        mode_means.append(float(np.sum((group_mean - global_mean) ** 2)))
        within_terms.append(float(np.mean(np.sum((group - group_mean) ** 2, axis=1))))
    sfr = float(np.mean(mode_means) / max(np.mean(within_terms), 1e-12)) if mode_means and within_terms else float("nan")

    global_threshold = float(np.quantile(audit_final, fpr_quantile)) if len(audit_final) else float("nan")
    per_mode_fpr = {}
    for current_mode in sorted(np.unique(audit_mode_id).tolist()):
        group_mask = audit_mode_id == current_mode
        if not np.any(group_mask):
            continue
        per_mode_fpr[str(current_mode)] = float(np.mean(audit_final[group_mask] > global_threshold))
    mode_fpr_std = float(np.std(list(per_mode_fpr.values()))) if per_mode_fpr else float("nan")

    smr_values = []
    cross_mode_margin_terms = []
    valid_neighbor_mask = fault_topk_mode_ids >= 0
    fault_topk_distances = np.asarray(fault_logs["topk_neighbor_distances"], dtype=np.float64)
    for row_idx in range(len(fault_mode_id)):
        valid_modes = fault_topk_mode_ids[row_idx][valid_neighbor_mask[row_idx]][:smr_k]
        valid_dists = fault_topk_distances[row_idx][valid_neighbor_mask[row_idx]][:smr_k]
        if valid_modes.size == 0:
            continue
        same_mask = valid_modes == fault_mode_id[row_idx]
        smr_values.append(float(np.mean(same_mask)))
        if np.any(same_mask) and np.any(~same_mask):
            same_mean = float(np.mean(valid_dists[same_mask]))
            cross_mean = float(np.mean(valid_dists[~same_mask]))
            cross_mode_margin_terms.append(cross_mean - same_mean)
    smr_at_k = _safe_mean(smr_values)
    cross_mode_margin = _safe_mean(cross_mode_margin_terms)

    per_mode_gap = {}
    delta_terms = []
    for current_mode in sorted(np.unique(np.concatenate([audit_mode_id, fault_mode_id], axis=0)).tolist()):
        mode_normal = audit_memory[audit_mode_id == current_mode]
        mode_fault = fault_memory[fault_mode_id == current_mode]
        if len(mode_normal) == 0 or len(mode_fault) == 0:
            continue
        gap = float(np.mean(mode_fault) - np.mean(mode_normal))
        per_mode_gap[str(current_mode)] = gap
        delta_terms.append(gap)
    delta_mem_mode = _safe_mean(delta_terms)

    q95 = 0.95
    q99 = 0.99
    threshold95 = float(np.quantile(audit_final, q95)) if len(audit_final) else float("nan")
    threshold99 = float(np.quantile(audit_final, q99)) if len(audit_final) else float("nan")
    ee95_terms = []
    ee99_terms = []
    per_mode_tail95 = {}
    per_mode_tail99 = {}
    for current_mode in sorted(np.unique(audit_mode_id).tolist()):
        group_mask = audit_mode_id == current_mode
        if not np.any(group_mask):
            continue
        tail95 = float(np.mean(audit_final[group_mask] > threshold95))
        tail99 = float(np.mean(audit_final[group_mask] > threshold99))
        per_mode_tail95[str(current_mode)] = tail95
        per_mode_tail99[str(current_mode)] = tail99
        ee95_terms.append(abs(tail95 - (1.0 - q95)))
        ee99_terms.append(abs(tail99 - (1.0 - q99)))

    q_grid = np.linspace(0.90, 0.99, 10)
    exceedance_curve = {}
    for q in q_grid.tolist():
        threshold = float(np.quantile(audit_final, q)) if len(audit_final) else float("nan")
        exceedance_curve[f"{q:.2f}"] = {
            str(current_mode): float(np.mean(audit_final[audit_mode_id == current_mode] > threshold))
            for current_mode in sorted(np.unique(audit_mode_id).tolist())
            if np.any(audit_mode_id == current_mode)
        }

    fault_sequence_rows = _build_fault_sequence_rows(audit_logs, fault_sequence_records)
    per_fault_consistency = {}
    fault_std_terms = []
    evidence_agreement_terms = []
    for current_fault in sorted({int(row["fault_id"]) for row in fault_sequence_rows}):
        group = [row for row in fault_sequence_rows if int(row["fault_id"]) == current_fault]
        calibrated_scores = [float(row["selected_calibrated"]) for row in group if np.isfinite(float(row["selected_calibrated"]))]
        if len(calibrated_scores) >= 2:
            std_val = float(np.std(np.asarray(calibrated_scores, dtype=np.float64)))
            per_fault_consistency[str(current_fault)] = std_val
            fault_std_terms.append(std_val)
        dominant = [str(row["dominant_evidence"]) for row in group if str(row["dominant_evidence"]) != "unknown"]
        if len(dominant) >= 2:
            agree = 0.0
            total = 0.0
            for i in range(len(dominant)):
                for j in range(i + 1, len(dominant)):
                    total += 1.0
                    agree += float(dominant[i] == dominant[j])
            if total > 0:
                evidence_agreement_terms.append(agree / total)

    proto_purity, proto_entropy = _compute_prototype_metrics(train_state_meta)

    metrics = {
        "SMC@K": smc_at_k,
        "SFR": sfr,
        "Mode-FPR-Std": mode_fpr_std,
        "SMR@K": smr_at_k,
        "delta_mem_mode": delta_mem_mode,
        "EE95": _safe_mean(ee95_terms),
        "Tail@0.99_error": _safe_mean(ee99_terms),
        "Fault-Consistency-Std": _safe_mean(fault_std_terms),
        "Evidence-Dom-Consistency": _safe_mean(evidence_agreement_terms),
        "Proto-Purity": proto_purity,
        "Proto-Entropy": proto_entropy,
        "Cross-mode-margin": cross_mode_margin,
        "normal_global_threshold_q95": threshold95,
        "normal_global_threshold_q99": threshold99,
        "per_mode_fpr": per_mode_fpr,
        "per_mode_mem_gap": per_mode_gap,
        "per_mode_tail95": per_mode_tail95,
        "per_mode_tail99": per_mode_tail99,
        "per_fault_consistency_std": per_fault_consistency,
        "exceedance_curve": exceedance_curve,
        "n_audit_normal_windows": int(len(audit_final)),
        "n_fault_windows": int(len(fault_memory)),
        "n_fault_sequences": int(len(fault_sequence_rows)),
        "fault_ids_present": sorted({int(row["fault_id"]) for row in fault_sequence_rows}),
    }
    return metrics, fault_sequence_rows


def _load_experiment_payload(exp_dir: Path, log_subdir: str) -> dict[str, Any]:
    log_dir = exp_dir / log_subdir
    return {
        "audit_logs": load_window_logs(log_dir, prefix="audit_normal_window_logs"),
        "fault_logs": load_window_logs(log_dir, prefix="fault_window_logs"),
        "train_state_meta": load_train_state_meta(log_dir),
        "fault_sequence_records": load_sequence_scores_with_meta(log_dir, filename="fault_sequence_scores_with_meta.json"),
    }


def compute_a4_metrics(
    short_payload: dict[str, Any],
    long_payload: dict[str, Any],
    multi_payload: dict[str, Any],
) -> dict[str, Any]:
    def build_fault_table(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
        rows = _build_fault_sequence_rows(payload["audit_logs"], payload["fault_sequence_records"])
        out: dict[str, dict[str, float]] = {}
        for row in rows:
            out[str(row["sequence_name"])] = {
                "mode_id": int(row["mode_id"]),
                "fault_id": int(row["fault_id"]),
                "selected": float(row["selected"]),
                "calibrated_sequence_score": float(row["selected_calibrated"]),
            }
        return out

    short_table = build_fault_table(short_payload)
    long_table = build_fault_table(long_payload)
    multi_table = build_fault_table(multi_payload)
    common_names = sorted(set(short_table) & set(long_table) & set(multi_table))
    rows = []
    gains = []
    multi_best = []
    heatmap = {}
    for name in common_names:
        short_score = short_table[name]["calibrated_sequence_score"]
        long_score = long_table[name]["calibrated_sequence_score"]
        multi_score = multi_table[name]["calibrated_sequence_score"]
        gain = float(multi_score - max(short_score, long_score))
        is_best = float(multi_score >= max(short_score, long_score))
        row = {
            "sequence_name": name,
            "mode_id": int(multi_table[name]["mode_id"]),
            "fault_id": int(multi_table[name]["fault_id"]),
            "short_sequence_score": float(short_table[name]["selected"]),
            "long_sequence_score": float(long_table[name]["selected"]),
            "multi_sequence_score": float(multi_table[name]["selected"]),
            "short_calibrated": float(short_score),
            "long_calibrated": float(long_score),
            "multi_calibrated": float(multi_score),
            "complementarity_gain": gain,
            "multi_best": is_best,
        }
        rows.append(row)
        gains.append(gain)
        multi_best.append(is_best)
        heatmap[f"mode{row['mode_id']}_fault{row['fault_id']:02d}"] = gain
    return {
        "CG_mean": _safe_mean(gains),
        "%_multi_best": _safe_mean(multi_best),
        "n_fault_sequences": len(common_names),
        "fault_rows": rows,
        "gain_heatmap": heatmap,
    }


def main() -> None:
    args = parse_args()
    experiment_dirs = list(args.experiment_dirs)
    trio_dirs = [args.short_dir, args.long_dir, args.multi_dir]
    for path in trio_dirs:
        if path is not None:
            experiment_dirs.append(path)
    experiment_dirs = [path for path in experiment_dirs if path is not None]
    if not experiment_dirs:
        raise ValueError("Please provide --experiment_dirs or the A4 trio (--short_dir/--long_dir/--multi_dir).")

    unique_dirs = []
    seen = set()
    for path in experiment_dirs:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_dirs.append(path)
    experiment_dirs = unique_dirs

    output_dir = args.output_dir or (experiment_dirs[0] / args.log_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"experiments": {}}
    cached_payloads: dict[str, dict[str, Any]] = {}
    for exp_dir in experiment_dirs:
        payload = _load_experiment_payload(exp_dir, args.log_subdir)
        cached_payloads[str(exp_dir.resolve())] = payload
        metrics, fault_rows = compute_single_metrics(
            audit_logs=payload["audit_logs"],
            fault_logs=payload["fault_logs"],
            train_state_meta=payload["train_state_meta"],
            fault_sequence_records=payload["fault_sequence_records"],
            smc_k=args.smc_k,
            smr_k=args.smr_k,
            fpr_quantile=args.fpr_quantile,
        )
        metrics["fault_sequence_rows"] = fault_rows
        summary["experiments"][exp_dir.name] = metrics

    if args.short_dir and args.long_dir and args.multi_dir:
        summary["A4"] = compute_a4_metrics(
            short_payload=cached_payloads[str(args.short_dir.resolve())],
            long_payload=cached_payloads[str(args.long_dir.resolve())],
            multi_payload=cached_payloads[str(args.multi_dir.resolve())],
        )

    json_path = output_dir / "mechanism_metrics.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=np_to_python),
        encoding="utf-8",
    )
    print(f"[TEP Metrics] wrote json: {json_path}")

    csv_lines = [
        "experiment,SMC@K,SFR,Mode-FPR-Std,SMR@K,delta_mem_mode,EE95,Tail@0.99_error,Fault-Consistency-Std,Evidence-Dom-Consistency,Proto-Purity,Proto-Entropy,Cross-mode-margin"
    ]
    for exp_name, metrics in summary["experiments"].items():
        csv_lines.append(
            ",".join(
                [
                    exp_name,
                    f"{float(metrics['SMC@K']):.10f}",
                    f"{float(metrics['SFR']):.10f}",
                    f"{float(metrics['Mode-FPR-Std']):.10f}",
                    f"{float(metrics['SMR@K']):.10f}",
                    f"{float(metrics['delta_mem_mode']):.10f}",
                    f"{float(metrics['EE95']):.10f}",
                    f"{float(metrics['Tail@0.99_error']):.10f}",
                    f"{float(metrics['Fault-Consistency-Std']):.10f}",
                    f"{float(metrics['Evidence-Dom-Consistency']):.10f}",
                    f"{float(metrics['Proto-Purity']):.10f}",
                    f"{float(metrics['Proto-Entropy']):.10f}",
                    f"{float(metrics['Cross-mode-margin']):.10f}",
                ]
            )
        )
    if "A4" in summary:
        csv_lines.append("")
        csv_lines.append("A4_metric,value")
        csv_lines.append(f"CG_mean,{float(summary['A4']['CG_mean']):.10f}")
        csv_lines.append(f"%_multi_best,{float(summary['A4']['%_multi_best']):.10f}")
    csv_path = output_dir / "mechanism_metrics.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"[TEP Metrics] wrote csv: {csv_path}")

    for exp_name, metrics in summary["experiments"].items():
        rows_path = output_dir / f"{exp_name}_fault_sequence_rows.json"
        rows_path.write_text(
            json.dumps(metrics["fault_sequence_rows"], indent=2, ensure_ascii=False, default=np_to_python),
            encoding="utf-8",
        )
        print(f"[TEP Metrics] wrote fault sequence rows: {rows_path}")

    if "A4" in summary:
        rows_path = output_dir / "a4_fault_rows.json"
        rows_path.write_text(
            json.dumps(summary["A4"]["fault_rows"], indent=2, ensure_ascii=False, default=np_to_python),
            encoding="utf-8",
        )
        print(f"[TEP Metrics] wrote A4 fault rows: {rows_path}")


if __name__ == "__main__":
    main()
