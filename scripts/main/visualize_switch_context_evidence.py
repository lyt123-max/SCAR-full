from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEP_SCRIPT_DIR = ROOT / "scripts" / "tep"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEP_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TEP_SCRIPT_DIR))

from coremad import CoReMADConfig, CoReMADTrainer
from coremad.data import build_loader, load_raw_dataset_bundle
from export_mechanism_logs import _collect_window_records
from tep_common import (
    get_loader_starts,
    load_window_logs,
    metadata_for_starts,
    np_to_python,
    parse_tep_name,
)


SWITCH_SCORE_KEYS = [
    "memory_distance",
    "state_novelty",
    "completion_scale8",
    "completion_scale32",
    "cdf_mean",
    "final",
]
CONTEXT_SCORE_KEYS = [
    ("cdf_mean_score", "Fused CDF-mean"),
    ("completion_scale8", "Completion short"),
    ("completion_scale32", "Completion long"),
    ("knn_distance", "Patch memory"),
    ("state_novelty", "State novelty"),
]
MODE_COLORS = {
    1: "#2F4858",
    2: "#F26419",
    3: "#F6AE2D",
    4: "#86BBD8",
    5: "#7D5A50",
    6: "#758E4F",
}
EVIDENCE_COLORS = {
    "Fused CDF-mean": "#2F4858",
    "Completion short": "#F26419",
    "Completion long": "#86BBD8",
    "Patch memory": "#758E4F",
    "State novelty": "#7D5A50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual evidence for switch false alarms and contextual anomaly misses."
    )
    parser.add_argument(
        "--tep-exp-dir",
        type=Path,
        default=ROOT / "TEP-abalation" / "tep_ablation_full_20260418_193550",
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=ROOT / "main-result" / "synthetic",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "main-result" / "figures" / "switch_context_evidence",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--switch-radius", type=int, default=0, help="Defaults to TEP seq_len.")
    parser.add_argument("--switch-bin-width", type=int, default=8)
    parser.add_argument("--switch-force", action="store_true")
    parser.add_argument("--skip-switch", action="store_true")
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--context-score-key", type=str, default="cdf_mean_score")
    parser.add_argument("--context-case-pad", type=int, default=160)
    parser.add_argument("--context-quantiles", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[EvidenceViz] wrote {path}")


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        field_set: list[str] = []
        for row in rows:
            for key in row:
                if key not in field_set:
                    field_set.append(key)
        fieldnames = field_set
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[EvidenceViz] wrote {path}")


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(arr, kernel, mode="same")


def finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    vals = np.asarray(values, dtype=np.float64)
    if ref.size == 0:
        return np.full(vals.shape, np.nan, dtype=np.float64)
    return np.searchsorted(ref, vals, side="right").astype(np.float64) / float(ref.size)


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size == 0:
        return []
    diff = np.diff(np.r_[False, arr, False].astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def load_coremad_runtime(exp_dir: Path, device: str, batch_size: int, num_workers: int) -> dict[str, Any]:
    config = CoReMADConfig.load(exp_dir / "config.json")
    config.artifact_root = str(exp_dir.parent)
    config.experiment_name = exp_dir.name
    if str(config.dataset).upper() == "TEP" and not Path(config.data_root).exists():
        local_tep_root = ROOT / "TEP-DATA"
        if local_tep_root.exists():
            config.data_root = str(local_tep_root)
    elif not Path(config.data_root).exists():
        local_detect_root = ROOT / "dataset" / "anomaly_detect"
        if local_detect_root.exists():
            config.data_root = str(local_detect_root)
    config.device = device
    config.faiss_use_gpu = False
    config.test_batch_size = int(batch_size)
    config.memory_batch_size = int(batch_size)
    config.num_workers = int(num_workers)

    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    memory = trainer.load_memory_bank()
    raw_bundle = trainer._load_raw_bundle()
    cdf_fusion = trainer.load_cdf_fusion(optional=False)
    zscore_fusion = trainer.load_zscore_fusion(optional=False)
    return {
        "config": config,
        "trainer": trainer,
        "model": model,
        "normalizer": normalizer,
        "memory": memory,
        "raw_bundle": raw_bundle,
        "cdf_fusion": cdf_fusion,
        "zscore_fusion": zscore_fusion,
    }


def center_meta_for_start(
    start: int,
    seq_len: int,
    ranges: np.ndarray,
    names: list[str],
) -> dict[str, object]:
    center = int(start) + int(seq_len) // 2
    ranges_arr = np.asarray(ranges, dtype=np.int64)
    for (lo, hi), name in zip(ranges_arr.tolist(), names):
        if int(lo) <= center < int(hi):
            meta = parse_tep_name(str(name))
            return {
                "mode_id": int(meta["mode_id"]),
                "file_id": str(meta["file_id"]),
                "sequence_name": str(meta["name"]),
                "is_normal": int(meta["is_normal"]),
                "fault_id": int(meta["fault_id"]),
            }
    distances = np.minimum(np.abs(center - ranges_arr[:, 0]), np.abs(center - ranges_arr[:, 1]))
    nearest_idx = int(np.argmin(distances))
    meta = parse_tep_name(str(names[nearest_idx]))
    return {
        "mode_id": int(meta["mode_id"]),
        "file_id": str(meta["file_id"]),
        "sequence_name": str(meta["name"]),
        "is_normal": int(meta["is_normal"]),
        "fault_id": int(meta["fault_id"]),
    }


def load_or_compute_switch_payload(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_dir
    cache_path = out_dir / "tep_switch_window_logs.npz"
    meta_path = out_dir / "tep_switch_window_logs_meta.json"
    if cache_path.exists() and meta_path.exists() and not args.switch_force:
        payload = dict(np.load(cache_path, allow_pickle=True))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["meta"] = meta
        return payload

    print(f"[EvidenceViz] loading TEP runtime: {args.tep_exp_dir}")
    ctx = load_coremad_runtime(
        exp_dir=args.tep_exp_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    config = ctx["config"]
    trainer = ctx["trainer"]
    raw_bundle = ctx["raw_bundle"]
    if raw_bundle.val is None or raw_bundle.val_segment_ranges is None or raw_bundle.val_segment_names is None:
        raise RuntimeError("TEP switch visualization requires normal validation segments.")

    train_starts = get_loader_starts(
        data=raw_bundle.train,
        config=config,
        stride=config.memory_build_stride,
        max_windows=config.max_train_windows,
        segment_ranges=raw_bundle.train_segment_ranges,
    )
    train_meta = metadata_for_starts(
        starts=train_starts,
        ranges=raw_bundle.train_segment_ranges,
        names=raw_bundle.train_segment_names or [],
    )
    if len(train_starts) != int(ctx["memory"].state_bank.size(0)):
        raise RuntimeError(
            f"State bank/window count mismatch: starts={len(train_starts)} "
            f"state_bank={int(ctx['memory'].state_bank.size(0))}"
        )

    val_norm = ctx["normalizer"].transform(raw_bundle.val)
    loader = build_loader(
        data=val_norm,
        labels=None,
        seq_len=config.seq_len,
        stride=config.test_stride,
        batch_size=config.test_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        max_windows=0,
        drop_last=False,
        segment_ranges=None,
    )
    print(
        f"[EvidenceViz] scoring stitched TEP validation normal sequence: "
        f"windows={len(loader.dataset)} seq_len={config.seq_len}"
    )
    records = _collect_window_records(
        trainer=trainer,
        model=ctx["model"],
        memory=ctx["memory"],
        cdf_fusion=ctx["cdf_fusion"],
        zscore_fusion=ctx["zscore_fusion"],
        loader=loader,
        metadata_for_start=lambda start: center_meta_for_start(
            start,
            config.seq_len,
            raw_bundle.val_segment_ranges,
            raw_bundle.val_segment_names or [],
        ),
        train_meta=train_meta,
        window_aggregation="p95",
        top_k=int(config.top_K),
        record_offset=0,
    )
    starts = np.asarray([record["start"] for record in records], dtype=np.int64)
    topk_modes = np.stack(
        [np.asarray(record["topk_neighbor_mode_ids"], dtype=np.int32) for record in records],
        axis=0,
    )
    output: dict[str, Any] = {
        "starts": starts,
        "centers": starts.astype(np.float64) + float(config.seq_len) / 2.0,
        "mode_id": np.asarray([record["mode_id"] for record in records], dtype=np.int32),
        "topk_neighbor_mode_ids": topk_modes,
        "segment_ranges": np.asarray(raw_bundle.val_segment_ranges, dtype=np.int64),
        "segment_names": np.asarray(raw_bundle.val_segment_names or [], dtype=object),
    }
    for key in SWITCH_SCORE_KEYS:
        output[key] = np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64)
    boundaries = np.asarray(raw_bundle.val_segment_ranges[:-1, 1], dtype=np.float64)
    boundary_labels = []
    for idx, boundary in enumerate(boundaries.tolist()):
        left = parse_tep_name(str(raw_bundle.val_segment_names[idx]))["mode_id"]
        right = parse_tep_name(str(raw_bundle.val_segment_names[idx + 1]))["mode_id"]
        boundary_labels.append(f"m{left}->m{right}")
    output["boundaries"] = boundaries
    output["boundary_labels"] = np.asarray(boundary_labels, dtype=object)
    output["seq_len"] = np.asarray(int(config.seq_len), dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **output)
    meta = {
        "tep_exp_dir": str(args.tep_exp_dir),
        "score_key": "final",
        "window_aggregation": "p95",
        "top_k": int(config.top_K),
        "n_windows": int(len(starts)),
        "seq_len": int(config.seq_len),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    output["meta"] = meta
    print(f"[EvidenceViz] wrote {cache_path}")
    print(f"[EvidenceViz] wrote {meta_path}")
    return output


def switch_masks(payload: dict[str, Any], radius: int) -> dict[str, np.ndarray]:
    starts = np.asarray(payload["starts"], dtype=np.float64)
    centers = np.asarray(payload["centers"], dtype=np.float64)
    boundaries = np.asarray(payload["boundaries"], dtype=np.float64)
    seq_len = int(np.asarray(payload["seq_len"]).item())
    if radius <= 0:
        radius = seq_len
    if boundaries.size == 0:
        false_mask = np.zeros(starts.shape, dtype=bool)
        return {"crossing": false_mask, "near": false_mask, "interior": np.ones(starts.shape, dtype=bool)}
    crossing = np.zeros(starts.shape, dtype=bool)
    nearest_dist = np.full(starts.shape, np.inf, dtype=np.float64)
    for boundary in boundaries.tolist():
        crossing |= (starts < boundary) & ((starts + seq_len) > boundary)
        dist = np.abs(centers - boundary)
        nearest_dist = np.minimum(nearest_dist, dist)
    near = nearest_dist <= float(radius)
    interior = nearest_dist > float(2 * radius)
    if not np.any(interior):
        interior = ~near
    return {
        "crossing": crossing,
        "near": near,
        "interior": interior,
        "nearest_dist": nearest_dist,
    }


def build_switch_metrics(
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    audit_logs = load_window_logs(args.tep_exp_dir / "tep_mechanism", prefix="audit_normal_window_logs")
    reference = np.asarray(audit_logs["final"], dtype=np.float64)
    scores = np.asarray(payload["final"], dtype=np.float64)
    masks = switch_masks(payload, args.switch_radius)
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for q in (0.90, 0.95, 0.99):
        threshold = finite_quantile(reference, q)
        thresholds[f"q{int(q * 100)}"] = threshold
        for group, mask in (
            ("interior", masks["interior"]),
            ("near_boundary", masks["near"]),
            ("switch_crossing", masks["crossing"]),
        ):
            group_scores = scores[mask]
            rows.append(
                {
                    "quantile": q,
                    "threshold": threshold,
                    "group": group,
                    "n_windows": int(group_scores.size),
                    "fpr": float(np.mean(group_scores > threshold)) if group_scores.size else float("nan"),
                    "mean_score": float(np.mean(group_scores)) if group_scores.size else float("nan"),
                    "p95_score": finite_quantile(group_scores, 0.95),
                }
            )
    return rows, thresholds


def plot_switch_timeline(payload: dict[str, Any], thresholds: dict[str, float], out_dir: Path) -> None:
    centers = np.asarray(payload["centers"], dtype=np.float64)
    scores = np.asarray(payload["final"], dtype=np.float64)
    seq_len = int(np.asarray(payload["seq_len"]).item())
    boundaries = np.asarray(payload["boundaries"], dtype=np.float64)
    boundary_labels = [str(x) for x in np.asarray(payload["boundary_labels"], dtype=object).tolist()]
    segment_ranges = np.asarray(payload["segment_ranges"], dtype=np.int64)
    segment_names = [str(x) for x in np.asarray(payload["segment_names"], dtype=object).tolist()]

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    for (lo, hi), name in zip(segment_ranges.tolist(), segment_names):
        mode = int(parse_tep_name(name)["mode_id"])
        ax.axvspan(lo, hi, color=MODE_COLORS.get(mode, "#cccccc"), alpha=0.08, linewidth=0)
        ax.text((lo + hi) / 2.0, 1.02, f"mode {mode}", ha="center", va="bottom", transform=ax.get_xaxis_transform())
    for boundary, label in zip(boundaries.tolist(), boundary_labels):
        ax.axvspan(boundary - seq_len / 2, boundary + seq_len / 2, color="#F26419", alpha=0.10, linewidth=0)
        ax.axvline(boundary, color="#F26419", linewidth=1.2)
        ax.text(boundary, 0.97, label, rotation=90, ha="right", va="top", transform=ax.get_xaxis_transform())

    ax.plot(centers, scores, color="#2F4858", alpha=0.28, linewidth=0.8, label="window score")
    ax.plot(centers, rolling_mean(scores, 25), color="#2F4858", linewidth=2.0, label="rolling mean")
    if np.isfinite(thresholds.get("q95", np.nan)):
        ax.axhline(thresholds["q95"], color="#758E4F", linestyle="--", linewidth=1.2, label="normal q95")
    if np.isfinite(thresholds.get("q99", np.nan)):
        ax.axhline(thresholds["q99"], color="#B23A48", linestyle=":", linewidth=1.5, label="normal q99")
    ax.set_title("Normal-only TEP stitched mode-boundary stress test")
    ax.set_xlabel("stitched validation-normal time index")
    ax.set_ylabel("p95 window final score")
    ax.set_ylim(max(0.0, np.nanmin(scores) - 0.02), min(1.05, np.nanmax(scores) + 0.04))
    ax.legend(loc="lower right", ncol=4)
    save_figure(fig, out_dir / "tep_switch_timeline.png")


def plot_switch_aligned(payload: dict[str, Any], thresholds: dict[str, float], radius: int, bin_width: int, out_dir: Path) -> None:
    centers = np.asarray(payload["centers"], dtype=np.float64)
    scores = np.asarray(payload["final"], dtype=np.float64)
    seq_len = int(np.asarray(payload["seq_len"]).item())
    radius = int(radius) if int(radius) > 0 else seq_len
    boundaries = np.asarray(payload["boundaries"], dtype=np.float64)
    if boundaries.size == 0:
        return

    rel_values = []
    score_values = []
    for boundary in boundaries.tolist():
        rel = centers - boundary
        mask = np.abs(rel) <= float(radius)
        rel_values.append(rel[mask])
        score_values.append(scores[mask])
    rel_all = np.concatenate(rel_values, axis=0)
    score_all = np.concatenate(score_values, axis=0)
    bins = np.arange(-radius, radius + bin_width, bin_width, dtype=np.float64)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    means = np.full(bin_centers.shape, np.nan, dtype=np.float64)
    lo = np.full(bin_centers.shape, np.nan, dtype=np.float64)
    hi = np.full(bin_centers.shape, np.nan, dtype=np.float64)
    for idx in range(len(bin_centers)):
        mask = (rel_all >= bins[idx]) & (rel_all < bins[idx + 1])
        vals = score_all[mask]
        if vals.size:
            means[idx] = float(np.mean(vals))
            lo[idx] = finite_quantile(vals, 0.25)
            hi[idx] = finite_quantile(vals, 0.75)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(rel_all, score_all, s=10, color="#2F4858", alpha=0.12, edgecolor="none", label="aligned windows")
    ax.plot(bin_centers, means, color="#F26419", linewidth=2.3, label="binned mean")
    ax.fill_between(bin_centers, lo, hi, color="#F26419", alpha=0.15, label="IQR")
    ax.axvline(0.0, color="#1A1A1A", linewidth=1.2)
    ax.axvspan(-seq_len / 2, seq_len / 2, color="#F6AE2D", alpha=0.13, linewidth=0, label="crossing zone")
    if np.isfinite(thresholds.get("q95", np.nan)):
        ax.axhline(thresholds["q95"], color="#758E4F", linestyle="--", linewidth=1.2, label="normal q95")
    if np.isfinite(thresholds.get("q99", np.nan)):
        ax.axhline(thresholds["q99"], color="#B23A48", linestyle=":", linewidth=1.5, label="normal q99")
    ax.set_title("Event-aligned normal mode-boundary scores (stitched stress test)")
    ax.set_xlabel("window center offset from mode boundary")
    ax.set_ylabel("p95 window final score")
    ax.legend(loc="best", ncol=2)
    save_figure(fig, out_dir / "tep_switch_aligned_scores.png")


def plot_switch_fpr(rows: list[dict[str, Any]], out_dir: Path) -> None:
    groups = ["interior", "near_boundary", "switch_crossing"]
    group_labels = ["Interior", "Near boundary", "Switch-crossing"]
    quantiles = [0.95, 0.99]
    values = np.zeros((len(groups), len(quantiles)), dtype=np.float64)
    for gi, group in enumerate(groups):
        for qi, q in enumerate(quantiles):
            matched = [row for row in rows if row["group"] == group and abs(float(row["quantile"]) - q) < 1e-9]
            values[gi, qi] = float(matched[0]["fpr"]) if matched else np.nan

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.34
    colors = ["#758E4F", "#B23A48"]
    for qi, q in enumerate(quantiles):
        bars = ax.bar(
            x + (qi - 0.5) * width,
            values[:, qi],
            width=width,
            label=f"FPR@q{int(q * 100)}",
            color=colors[qi],
            alpha=0.88,
        )
        for bar, value in zip(bars, values[:, qi]):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.set_xticks(x, group_labels)
    ax.set_ylabel("normal windows above audit-normal threshold")
    ax.set_ylim(0.0, max(0.08, np.nanmax(values) * 1.25 if np.isfinite(values).any() else 0.08))
    ax.set_title("Switch-FPR on normal-only stitched mode boundaries")
    ax.legend(loc="upper left")
    save_figure(fig, out_dir / "tep_switch_fpr_bars.png")


def plot_switch_retrieval_composition(payload: dict[str, Any], radius: int, out_dir: Path) -> None:
    masks = switch_masks(payload, radius)
    topk = np.asarray(payload["topk_neighbor_mode_ids"], dtype=np.int32)
    groups = [
        ("Interior", masks["interior"]),
        ("Near boundary", masks["near"]),
        ("Switch-crossing", masks["crossing"]),
    ]
    mode_order = sorted(int(mode) for mode in np.unique(topk[topk >= 0]).tolist())
    values = np.zeros((len(groups), len(mode_order)), dtype=np.float64)
    for gi, (_, mask) in enumerate(groups):
        modes = topk[mask].reshape(-1)
        modes = modes[modes >= 0]
        denom = max(1, modes.size)
        for mi, mode in enumerate(mode_order):
            values[gi, mi] = float(np.sum(modes == mode)) / float(denom)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(groups), dtype=np.float64)
    bottom = np.zeros(len(groups), dtype=np.float64)
    for mi, mode in enumerate(mode_order):
        vals = values[:, mi]
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=MODE_COLORS.get(mode, "#888888"),
            alpha=0.88,
            label=f"retrieved mode {mode}",
        )
        for gi, val in enumerate(vals):
            if val >= 0.06:
                ax.text(gi, bottom[gi] + val / 2, f"{val:.0%}", ha="center", va="center", fontsize=9)
        bottom += vals
    ax.set_xticks(x, [name for name, _ in groups])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("top-K neighbor mode fraction")
    ax.set_title("Retrieved normal-neighbor composition around mode boundaries")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    save_figure(fig, out_dir / "tep_switch_retrieval_composition.png")


def run_switch_visualization(args: argparse.Namespace) -> None:
    payload = load_or_compute_switch_payload(args)
    rows, thresholds = build_switch_metrics(args, payload)
    write_rows(args.output_dir / "tep_switch_fpr_metrics.csv", rows)
    summary = {
        "thresholds": thresholds,
        "metrics": rows,
        "note": (
            "Switch windows are normal-only stitched TEP validation windows that cross adjacent mode-file "
            "boundaries. This is a discontinuity stress test, not a real smooth operating-condition transition."
        ),
    }
    (args.output_dir / "tep_switch_summary.json").write_text(
        json.dumps(np_to_python(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[EvidenceViz] wrote {args.output_dir / 'tep_switch_summary.json'}")
    plot_switch_timeline(payload, thresholds, args.output_dir)
    plot_switch_aligned(payload, thresholds, args.switch_radius, args.switch_bin_width, args.output_dir)
    plot_switch_fpr(rows, args.output_dir)
    plot_switch_retrieval_composition(payload, args.switch_radius, args.output_dir)


def discover_context_experiments(synthetic_root: Path) -> list[Path]:
    candidates = sorted(synthetic_root.glob("synthetic_con*_baseline_*"))
    return [path for path in candidates if (path / "test_diagnostic_scores.npz").exists()]


def load_context_payload(exp_dir: Path) -> dict[str, Any]:
    config = CoReMADConfig.load(exp_dir / "config.json")
    if not Path(config.data_root).exists():
        local_detect_root = ROOT / "dataset" / "anomaly_detect"
        if local_detect_root.exists():
            config.data_root = str(local_detect_root)
    raw = load_raw_dataset_bundle(config.dataset, config.data_root, config=config)
    diag = np.load(exp_dir / "test_diagnostic_scores.npz", allow_pickle=True)
    labels = np.asarray(diag["labels"], dtype=np.int32)
    test = np.asarray(raw.test, dtype=np.float64)
    if len(test) != len(labels):
        raise RuntimeError(f"{exp_dir}: raw test length {len(test)} != diagnostic labels {len(labels)}")
    return {
        "exp_dir": exp_dir,
        "dataset": str(config.dataset),
        "test": test,
        "labels": labels,
        "diag": {key: np.asarray(diag[key]) for key in diag.files},
    }


def context_metric_rows(
    payloads: list[dict[str, Any]],
    quantiles: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        labels = np.asarray(payload["labels"], dtype=np.int32) > 0
        segments = contiguous_segments(labels)
        for key, label in CONTEXT_SCORE_KEYS:
            if key not in payload["diag"]:
                continue
            score = np.asarray(payload["diag"][key], dtype=np.float64)
            normal_score = score[~labels]
            anomaly_score = score[labels]
            for q in quantiles:
                threshold = finite_quantile(normal_score, q)
                pred = score > threshold
                event_hits = [bool(np.any(pred[start:end])) for start, end in segments]
                rows.append(
                    {
                        "dataset": payload["dataset"],
                        "experiment": payload["exp_dir"].name,
                        "score_key": key,
                        "score_label": label,
                        "quantile": float(q),
                        "threshold": threshold,
                        "normal_fpr": float(np.mean(pred[~labels])) if np.any(~labels) else float("nan"),
                        "point_recall": float(np.mean(pred[labels])) if anomaly_score.size else float("nan"),
                        "point_miss_rate": float(1.0 - np.mean(pred[labels])) if anomaly_score.size else float("nan"),
                        "event_recall": float(np.mean(event_hits)) if event_hits else float("nan"),
                        "event_miss_rate": float(1.0 - np.mean(event_hits)) if event_hits else float("nan"),
                        "n_anomaly_points": int(anomaly_score.size),
                        "n_anomaly_events": int(len(segments)),
                    }
                )
    return rows


def plot_context_recall_heatmap(rows: list[dict[str, Any]], q: float, out_dir: Path) -> None:
    filtered = [row for row in rows if abs(float(row["quantile"]) - q) < 1e-9]
    datasets = sorted({str(row["dataset"]) for row in filtered})
    labels = [label for key, label in CONTEXT_SCORE_KEYS if any(row["score_key"] == key for row in filtered)]
    heat = np.full((len(datasets), len(labels)), np.nan, dtype=np.float64)
    for row in filtered:
        di = datasets.index(str(row["dataset"]))
        li = labels.index(str(row["score_label"]))
        heat[di, li] = float(row["point_recall"])

    fig, ax = plt.subplots(figsize=(1.55 * len(labels) + 2.5, 1.05 * len(datasets) + 2.2))
    im = ax.imshow(heat, cmap="YlGnBu", vmin=0.0, vmax=max(0.5, np.nanmax(heat) if np.isfinite(heat).any() else 0.5))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(datasets)), datasets)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.1%}", ha="center", va="center", fontsize=9, color="#1A1A1A")
    ax.set_title(f"Contextual anomaly point recall at normal q{int(q * 100)}")
    fig.colorbar(im, ax=ax, shrink=0.86, label="point recall")
    save_figure(fig, out_dir / f"contextual_recall_at_q{int(q * 100)}_heatmap.png")


def select_context_segment(labels: np.ndarray, score: np.ndarray, mode: str) -> tuple[int, int]:
    segments = contiguous_segments(labels > 0)
    if not segments:
        return 0, min(len(labels), 1)
    ranked = []
    for start, end in segments:
        segment_score = np.asarray(score[start:end], dtype=np.float64)
        peak = float(np.nanmax(segment_score)) if segment_score.size else float("nan")
        mean = float(np.nanmean(segment_score)) if segment_score.size else float("nan")
        ranked.append((peak, mean, start, end))
    ranked = [item for item in ranked if np.isfinite(item[0])]
    if not ranked:
        return segments[0]
    if mode == "hard":
        chosen = sorted(ranked, key=lambda item: (item[0], item[1], item[2]))[0]
    else:
        chosen = sorted(ranked, key=lambda item: (-item[0], -item[1], item[2]))[0]
    return int(chosen[2]), int(chosen[3])


def plot_context_case_gallery(
    payloads: list[dict[str, Any]],
    score_key: str,
    case_mode: str,
    pad: int,
    out_dir: Path,
) -> None:
    if not payloads:
        return
    fig, axes = plt.subplots(
        len(payloads),
        2,
        figsize=(12.0, 3.9 * len(payloads)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.05, 1.35]},
    )
    for row_idx, payload in enumerate(payloads):
        labels = np.asarray(payload["labels"], dtype=np.int32)
        if score_key not in payload["diag"]:
            raise KeyError(f"{payload['exp_dir']} missing score key {score_key!r}")
        chosen_score = np.asarray(payload["diag"][score_key], dtype=np.float64)
        start, end = select_context_segment(labels, chosen_score, mode=case_mode)
        center = (start + end) // 2
        lo = max(0, center - int(pad))
        hi = min(len(labels), center + int(pad))
        x = np.arange(lo, hi)
        local_labels = labels[lo:hi] > 0
        test = np.asarray(payload["test"], dtype=np.float64)
        ax_signal = axes[row_idx, 0]
        ax_score = axes[row_idx, 1]

        n_channels = min(3, test.shape[1])
        for channel in range(n_channels):
            vals = test[lo:hi, channel]
            vals = (vals - np.nanmean(vals)) / max(float(np.nanstd(vals)), 1e-8)
            ax_signal.plot(x, vals + channel * 3.0, linewidth=1.2, label=f"ch{channel}")
        for seg_start, seg_end in contiguous_segments(local_labels):
            ax_signal.axvspan(lo + seg_start, lo + seg_end, color="#F26419", alpha=0.22, linewidth=0)
            ax_score.axvspan(lo + seg_start, lo + seg_end, color="#F26419", alpha=0.16, linewidth=0)
        ax_signal.set_title(f"{payload['dataset']} - {case_mode} contextual case")
        ax_signal.set_xlabel("test time index")
        ax_signal.set_ylabel("local z-score + offset")
        ax_signal.legend(loc="upper right", ncol=n_channels)

        for key, label in CONTEXT_SCORE_KEYS:
            if key not in payload["diag"]:
                continue
            score = np.asarray(payload["diag"][key], dtype=np.float64)
            normal_ref = score[labels == 0]
            pct = empirical_percentile(normal_ref, score[lo:hi])
            ax_score.plot(
                x,
                pct,
                linewidth=1.45 if key != score_key else 2.2,
                alpha=0.95,
                label=label,
                color=EVIDENCE_COLORS.get(label),
            )
        ax_score.axhline(0.95, color="#B23A48", linestyle="--", linewidth=1.2, label="normal q95")
        ax_score.axhline(0.99, color="#1A1A1A", linestyle=":", linewidth=1.0, label="normal q99")
        selected_pct = empirical_percentile(chosen_score[labels == 0], chosen_score[start:end])
        peak_pct = float(np.nanmax(selected_pct)) if selected_pct.size else float("nan")
        ax_score.text(
            0.02,
            0.95,
            f"selected event peak percentile: {peak_pct:.1%}",
            transform=ax_score.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        ax_score.set_ylim(0.0, 1.03)
        ax_score.set_title("Evidence percentiles against normal test points")
        ax_score.set_xlabel("test time index")
        ax_score.set_ylabel("empirical normal percentile")
        ax_score.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / f"contextual_{case_mode}_case_gallery.png")


def plot_context_distribution(payloads: list[dict[str, Any]], out_dir: Path) -> None:
    if not payloads:
        return
    fig, axes = plt.subplots(1, len(payloads), figsize=(5.6 * len(payloads), 4.3), squeeze=False)
    labels_for_axis = [label for _, label in CONTEXT_SCORE_KEYS]
    for col_idx, payload in enumerate(payloads):
        ax = axes[0, col_idx]
        labels = np.asarray(payload["labels"], dtype=np.int32) > 0
        positions = []
        data = []
        colors = []
        tick_positions = []
        tick_labels = []
        pos = 1
        for key, label in CONTEXT_SCORE_KEYS:
            if key not in payload["diag"]:
                continue
            score = np.asarray(payload["diag"][key], dtype=np.float64)
            pct = empirical_percentile(score[~labels], score)
            data.extend([pct[~labels], pct[labels]])
            positions.extend([pos, pos + 0.35])
            colors.extend(["#D8E2DC", "#F26419"])
            tick_positions.append(pos + 0.175)
            tick_labels.append(label)
            pos += 1.1
        parts = ax.violinplot(data, positions=positions, widths=0.28, showmedians=True)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_alpha(0.78)
        ax.axhline(0.95, color="#B23A48", linestyle="--", linewidth=1.1)
        ax.set_xticks(tick_positions, tick_labels, rotation=24, ha="right")
        ax.set_ylim(0.0, 1.03)
        ax.set_title(payload["dataset"])
        ax.set_ylabel("empirical normal percentile")
        ax.text(0.02, 0.04, "left: normal  right: anomaly", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / "contextual_score_distribution_shift.png")


def run_context_visualization(args: argparse.Namespace) -> None:
    exp_dirs = discover_context_experiments(args.synthetic_root)
    if not exp_dirs:
        raise FileNotFoundError(f"No synthetic_con experiments found under {args.synthetic_root}")
    payloads = [load_context_payload(path) for path in exp_dirs]
    rows = context_metric_rows(payloads, args.context_quantiles)
    write_rows(args.output_dir / "contextual_recall_metrics.csv", rows)
    (args.output_dir / "contextual_recall_summary.json").write_text(
        json.dumps(np_to_python({"metrics": rows}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[EvidenceViz] wrote {args.output_dir / 'contextual_recall_summary.json'}")
    for q in args.context_quantiles:
        if abs(float(q) - 0.95) < 1e-9 or len(args.context_quantiles) <= 3:
            plot_context_recall_heatmap(rows, float(q), args.output_dir)
    plot_context_case_gallery(
        payloads=payloads,
        score_key=args.context_score_key,
        case_mode="detected",
        pad=args.context_case_pad,
        out_dir=args.output_dir,
    )
    plot_context_case_gallery(
        payloads=payloads,
        score_key=args.context_score_key,
        case_mode="hard",
        pad=args.context_case_pad,
        out_dir=args.output_dir,
    )
    plot_context_distribution(payloads, args.output_dir)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    if not args.skip_switch:
        run_switch_visualization(args)
    if not args.skip_context:
        run_context_visualization(args)


if __name__ == "__main__":
    main()
