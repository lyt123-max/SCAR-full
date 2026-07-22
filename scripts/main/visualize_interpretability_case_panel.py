from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad import CoReMADConfig, CoReMADTrainer


FUSED_SCORE_CANDIDATES = [
    "final_fused_score",
    "cdf_mean_soft_support_score",
    "cdf_mean_score",
    "cdf_softmax_score",
    "raw_max_score",
    "zscore_mean_score",
]
COMPONENT_ORDER = [
    "completion_scale8",
    "completion_scale32",
    "knn_distance",
    "soft_support_score",
    "state_novelty",
]
FINAL_FUSION_COMPONENT_ORDER = [
    "completion_scale8",
    "completion_scale32",
    "knn_distance",
    "state_novelty",
]
COMPONENT_STYLE = {
    "completion_scale8": ("Short pred", "#f97316"),
    "completion_scale32": ("Long pred", "#60a5fa"),
    "knn_distance": ("Memory", "#84a559"),
    "soft_support_score": ("Soft support", "#8b5e3c"),
    "state_novelty": ("State novelty", "#ef4444"),
}
ROW_A_COLORS = ["#5B8FD1", "#FF8A3D", "#4CAF50", "#F24C4C", "#B388EB", "#9C6B5A"]
ROW_LABELS = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)",
]
TICK_LABEL_SIZE = 9.6
LEGEND_FONT_SIZE = 9.2
HEATMAP_TICK_SIZE = 5.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a paper-style multi-case interpretability panel for CoReM-AD."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "main-result" / "figures" / "interpretability_case_panel" / "demo_case_manifest.json",
        help="JSON manifest that lists the cases to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "main-result" / "figures" / "interpretability_case_panel",
        help="Directory used for the exported PNG, PDF, and summary JSON.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="coremad_interpretability_case_panel",
        help="Base name of the exported figure files.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Model device used to decode the selected windows.")
    parser.add_argument("--channels", type=int, default=10, help="How many raw channels to display in Row (a).")
    parser.add_argument(
        "--relation-channels",
        type=int,
        default=10,
        help="How many channels to use for the relation drift heatmaps in Row (e).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.10,
        help="Softmax temperature used to convert evidence percentiles into relative weights.",
    )
    parser.add_argument(
        "--composition-mode",
        type=str,
        default="stacked",
        choices=["stacked", "lines"],
        help="How to render Row (c): stacked area contributions or multi-line evidence curves.",
    )
    parser.add_argument(
        "--auto-prototypes",
        action="store_true",
        help="Ignore the manifest and automatically mine four representative anomaly prototypes per dataset.",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=REPO_ROOT / "main-result",
        help="Directory scanned for experiment folders when --auto-prototypes is enabled.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MSL", "PSM", "SMAP", "SMD", "SWAT"],
        help="Dataset names to render in automatic prototype mode.",
    )
    parser.add_argument(
        "--prototypes-per-dataset",
        type=int,
        default=4,
        help="How many automatically discovered prototypes to render for each dataset.",
    )
    parser.add_argument(
        "--cluster-count-mode",
        type=str,
        default="auto",
        choices=["auto", "fixed"],
        help="How to choose the internal cluster count in automatic prototype mode.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum cluster count considered in automatic prototype mode.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=4,
        help="Maximum cluster count considered in automatic prototype mode.",
    )
    parser.add_argument(
        "--min-alignment-gain",
        type=float,
        default=0.0,
        help="Optional minimum label-vs-outside score gain. Use 0 to disable hard filtering and rely on alignment-priority ranking.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=48,
        help="Maximum number of anomaly windows considered per dataset before clustering.",
    )
    parser.add_argument(
        "--candidate-separation",
        type=int,
        default=0,
        help="Minimum timestep separation between candidate anomaly centers. Defaults to seq_len // 2.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def patch_data_root(config: CoReMADConfig) -> CoReMADConfig:
    data_root = Path(str(config.data_root))
    if data_root.exists():
        return config
    if str(config.dataset).upper() == "TEP":
        local_root = REPO_ROOT / "TEP-DATA"
    else:
        local_root = REPO_ROOT / "dataset" / "anomaly_detect"
    if local_root.exists():
        config.data_root = str(local_root)
    return config


def load_runtime(experiment_dir: Path, device: str) -> dict[str, Any]:
    config = CoReMADConfig.load(experiment_dir / "config.json")
    config.artifact_root = str(experiment_dir.parent)
    config.experiment_name = experiment_dir.name
    config.device = device
    config.faiss_use_gpu = False
    config = patch_data_root(config)

    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    raw_bundle = trainer._load_raw_bundle()
    score_npz = np.load(experiment_dir / "test_diagnostic_scores.npz")
    score_arrays = {key: np.asarray(score_npz[key]) for key in score_npz.files}
    labels = np.asarray(score_arrays.get("labels", raw_bundle.test_labels), dtype=np.int32)
    return {
        "experiment_dir": experiment_dir,
        "config": config,
        "trainer": trainer,
        "model": model,
        "normalizer": normalizer,
        "raw_test": np.asarray(raw_bundle.test, dtype=np.float64),
        "labels": labels,
        "score_arrays": score_arrays,
        "percentile_cache": {},
    }


def choose_score_key(score_arrays: dict[str, np.ndarray], requested: str | None) -> str:
    if requested and requested in score_arrays:
        return requested
    for key in FUSED_SCORE_CANDIDATES:
        if key not in score_arrays:
            continue
        values = np.asarray(score_arrays[key], dtype=np.float64)
        if np.nanstd(values) > 1e-8:
            return key
    for key, values in score_arrays.items():
        if key == "labels":
            continue
        if np.issubdtype(np.asarray(values).dtype, np.number):
            return key
    raise RuntimeError("No numeric score array is available in test_diagnostic_scores.npz.")


def available_component_keys(
    score_arrays: dict[str, np.ndarray],
    configured_keys: list[str] | None = None,
) -> list[str]:
    if configured_keys:
        ordered = [str(key) for key in configured_keys if key in score_arrays]
        if ordered:
            return ordered
    # Fall back to the legacy order, but keep soft support out unless the
    # experiment truly fused it.
    return [key for key in FINAL_FUSION_COMPONENT_ORDER if key in score_arrays]


def fusion_component_keys(runtime: dict[str, Any]) -> list[str]:
    config = runtime["config"]
    score_arrays = runtime["score_arrays"]
    return available_component_keys(score_arrays, configured_keys=list(config.fusion_score_names()))


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]
    vals = np.asarray(values, dtype=np.float64)
    if ref.size == 0:
        return np.full(vals.shape, np.nan, dtype=np.float64)
    ref = np.sort(ref)
    positions = np.searchsorted(ref, vals, side="right")
    return positions.astype(np.float64) / float(ref.size)


def runtime_percentile(runtime: dict[str, Any], key: str) -> np.ndarray:
    cache = runtime["percentile_cache"]
    if key in cache:
        return cache[key]
    score_arrays = runtime["score_arrays"]
    if key not in score_arrays:
        raise KeyError(f"Score key '{key}' is missing from {runtime['experiment_dir'] / 'test_diagnostic_scores.npz'}")
    values = np.asarray(score_arrays[key], dtype=np.float64)
    labels = np.asarray(runtime["labels"], dtype=np.int32)
    normal_ref = values[labels == 0] if np.any(labels == 0) else values
    cache[key] = empirical_percentile(normal_ref, values)
    return cache[key]


def clamp_window_start(total: int, seq_len: int, center: int) -> int:
    max_start = max(0, int(total) - int(seq_len))
    start = int(center) - int(seq_len) // 2
    return max(0, min(max_start, start))


def component_keys_for_case(runtime: dict[str, Any], start: int, end: int) -> list[str]:
    del start, end
    return fusion_component_keys(runtime)


def safe_corrcoef(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if arr.shape[0] < 3:
        return np.eye(arr.shape[1], dtype=np.float64)
    centered = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z = centered / scale
    corr = (z.T @ z) / max(1, int(z.shape[0] - 1))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size == 0:
        return []
    diff = np.diff(np.r_[False, arr, False].astype(np.int32))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts.tolist(), ends.tolist())]


def resolve_event_span(labels_window: np.ndarray | None, score_window: np.ndarray) -> tuple[int, int]:
    total = int(len(score_window))
    max_len = max(12, total // 3)
    if labels_window is not None and np.any(labels_window > 0):
        segments = contiguous_segments(labels_window > 0)
        segments = sorted(
            segments,
            key=lambda item: (
                float(np.nanmean(score_window[item[0] : item[1]])),
                float(np.nanmax(score_window[item[0] : item[1]])),
                item[1] - item[0],
            ),
            reverse=True,
        )
        event_start, event_end = segments[0]
        if event_end - event_start > max_len:
            local_peak = int(np.nanargmax(score_window[event_start:event_end])) + int(event_start)
            half_width = max_len // 2
            event_start = max(int(event_start), int(local_peak) - half_width)
            event_end = min(int(segments[0][1]), event_start + max_len)
            event_start = max(int(segments[0][0]), event_end - max_len)
        return int(event_start), int(event_end)
    peak = int(np.nanargmax(score_window))
    half_width = max(6, total // 12)
    return max(0, peak - half_width), min(total, peak + half_width + 1)


def resolve_baseline_span(total: int, event_start: int, event_end: int) -> tuple[int, int]:
    span = max(8, int(event_end - event_start))
    base_end = max(1, int(event_start))
    base_start = max(0, base_end - span)
    if base_end - base_start >= 6:
        return base_start, base_end
    alt_start = min(total - span, int(event_end))
    alt_start = max(0, alt_start)
    alt_end = min(total, alt_start + span)
    return alt_start, alt_end


def compact_index_text(value: int) -> str:
    value = int(value)
    if abs(value) >= 10000:
        compact = f"{value / 1000.0:.1f}".rstrip("0").rstrip(".")
        return f"{compact}k"
    return f"{value:,}"


def compact_span_text(start: int, end: int) -> str:
    return f"{compact_index_text(start)}-{compact_index_text(end)}"


def compact_component_text(text: str) -> str:
    compact = str(text)
    replacements = {
        "Short pred": "S-pred",
        "Long pred": "L-pred",
        "Memory": "Mem",
        "State novelty": "State",
        "Support": "Supp",
    }
    for source, target in replacements.items():
        compact = compact.replace(source, target)
    return compact


def reviewer_span_text(start: int, end: int) -> str:
    return f"Window {int(start):,}-{int(end):,}"


def reviewer_component_text(text: str) -> str:
    readable = str(text)
    replacements = {
        "Short pred": "short-horizon prediction",
        "Long pred": "long-horizon prediction",
        "Memory": "memory retrieval",
        "State novelty": "state novelty",
        "Support": "prototype support",
        "+": " + ",
    }
    for source, target in replacements.items():
        readable = readable.replace(source, target)
    return readable


def reviewer_morphology_text(name: str) -> str:
    mapping = {
        "relation shift": "cross-channel relation shift",
        "broad drift": "broad contextual drift",
        "persistent": "persistent anomaly",
        "volatile": "volatile fluctuation",
        "burst": "transient burst",
        "ramp": "gradual ramp-up",
        "mixed": "mixed anomaly pattern",
    }
    return mapping.get(str(name), str(name))


def choose_channels(residual_norm: np.ndarray, count: int) -> list[int]:
    energy = np.mean(np.abs(np.asarray(residual_norm, dtype=np.float64)), axis=0)
    order = np.argsort(energy)[::-1]
    top_k = max(1, min(int(count), residual_norm.shape[1]))
    return [int(idx) for idx in order[:top_k]]


def local_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (arr - mean) / std


def smooth_columns(values: np.ndarray, kernel_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or kernel_size <= 1 or arr.shape[0] <= 2:
        return arr.copy()
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(1, min(int(kernel_size), int(arr.shape[0])))
    if kernel_size <= 1:
        return arr.copy()
    pad = kernel_size // 2
    kernel = np.full(kernel_size, 1.0 / float(kernel_size), dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    for col_idx in range(arr.shape[1]):
        padded = np.pad(arr[:, col_idx], (pad, pad), mode="edge")
        out[:, col_idx] = np.convolve(padded, kernel, mode="valid")
    return out


def robust_minmax(values: np.ndarray, lower_q: float = 2.0, upper_q: float = 98.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    lo = float(np.quantile(finite, lower_q / 100.0))
    hi = float(np.quantile(finite, upper_q / 100.0))
    if hi - lo < 1e-12:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def smooth_vector(values: np.ndarray, kernel_size: int) -> np.ndarray:
    return smooth_columns(np.asarray(values, dtype=np.float64)[:, None], kernel_size)[:, 0]


def boundary_anchor_starts(total: int, seq_len: int, seg_start: int, seg_end: int, peak_idx: int | None = None) -> list[int]:
    anchors = [
        int(seg_start) - (3 * int(seq_len)) // 4,
        int(seg_start) - int(seq_len) // 2,
        int(seg_start) - int(seq_len) // 4,
        int(seg_end) - (3 * int(seq_len)) // 4,
        int(seg_end) - int(seq_len) // 2,
        int(seg_end) - int(seq_len) // 4,
        ((int(seg_start) + int(seg_end)) // 2) - int(seq_len) // 2,
    ]
    if peak_idx is not None:
        anchors.extend(
            [
                int(peak_idx) - (3 * int(seq_len)) // 4,
                int(peak_idx) - int(seq_len) // 2,
                int(peak_idx) - int(seq_len) // 4,
            ]
        )
    deduped: list[int] = []
    seen: set[int] = set()
    for anchor in anchors:
        start = clamp_window_start(total=total, seq_len=seq_len, center=anchor + seq_len // 2)
        if start not in seen:
            seen.add(start)
            deduped.append(int(start))
    return deduped


def alignment_proxy_score(labels_window: np.ndarray, score_window: np.ndarray) -> float:
    label_mask = np.asarray(labels_window, dtype=np.int32) > 0
    values = np.asarray(score_window, dtype=np.float64)
    label_fraction = float(np.mean(label_mask))
    if label_fraction <= 0.0 or label_fraction >= 1.0:
        return -1e6
    inside = values[label_mask]
    outside = values[~label_mask]
    if inside.size == 0 or outside.size == 0:
        return -1e6
    inside_mean = float(np.mean(inside))
    outside_mean = float(np.mean(outside))
    gain = inside_mean - outside_mean
    contrast = float(np.max(inside)) - float(np.max(outside))
    # Prefer windows where the anomaly occupies a meaningful but not dominant
    # fraction of the window, so the outside region remains interpretable.
    fraction_penalty = abs(label_fraction - 0.35)
    return (1.25 * gain) + (0.55 * contrast) + (0.20 * inside_mean) - (0.25 * outside_mean) - (0.30 * fraction_penalty)


def shannon_entropy(probs: np.ndarray) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 1e-12:
        return 0.0
    arr = arr / total
    mask = arr > 1e-12
    if not np.any(mask):
        return 0.0
    return float(-(arr[mask] * np.log(arr[mask])).sum())


def collect_candidate_starts(
    runtime: dict[str, Any],
    score_key: str,
    max_candidates: int,
    min_separation: int,
) -> list[int]:
    seq_len = int(runtime["config"].seq_len)
    labels = np.asarray(runtime["labels"], dtype=np.int32)
    total = int(len(labels))
    raw_scores = np.asarray(runtime["score_arrays"][score_key], dtype=np.float64)
    smooth_scores = smooth_vector(runtime_percentile(runtime, score_key), kernel_size=7)

    if np.any(labels > 0):
        scored_starts: list[tuple[float, int]] = []
        segments = contiguous_segments(labels > 0)
        for seg_start, seg_end in segments:
            local_peak = int(seg_start) + int(np.nanargmax(smooth_scores[seg_start:seg_end]))
            for start in boundary_anchor_starts(total=total, seq_len=seq_len, seg_start=seg_start, seg_end=seg_end, peak_idx=local_peak):
                end = start + seq_len
                proxy = alignment_proxy_score(labels[start:end], smooth_scores[start:end])
                if not np.isfinite(proxy):
                    continue
                scored_starts.append((float(proxy), int(start)))
        if not scored_starts:
            candidate_idx = np.flatnonzero(labels > 0)
            order = candidate_idx[np.argsort(smooth_scores[candidate_idx])[::-1]]
            candidate_starts_ranked = [clamp_window_start(total=total, seq_len=seq_len, center=int(center)) for center in order.tolist()]
        else:
            # Keep one best score per unique start, then rank by proxy.
            best_by_start: dict[int, float] = {}
            for proxy, start in scored_starts:
                best_by_start[start] = max(float(proxy), float(best_by_start.get(start, -1e9)))
            ranked_pairs = sorted(best_by_start.items(), key=lambda item: (item[1], -item[0]), reverse=True)
            candidate_starts_ranked = [int(start) for start, _ in ranked_pairs]
    else:
        threshold = float(np.nanquantile(smooth_scores, 0.995))
        candidate_idx = np.flatnonzero(smooth_scores >= threshold)
        if candidate_idx.size == 0:
            candidate_idx = np.argsort(smooth_scores)[::-1][: max(12, max_candidates * 2)]
        order = candidate_idx[np.argsort(smooth_scores[candidate_idx])[::-1]]
        candidate_starts_ranked = [clamp_window_start(total=total, seq_len=seq_len, center=int(center)) for center in order.tolist()]

    min_sep = max(4, int(min_separation))
    starts: list[int] = []
    centers: list[int] = []
    for start in candidate_starts_ranked:
        start = int(start)
        center = int(start + seq_len // 2)
        if any(abs(start - prev) < max(4, min_sep // 2) for prev in starts):
            continue
        window = raw_scores[start : start + seq_len]
        if not np.any(np.isfinite(window)):
            continue
        centers.append(center)
        starts.append(start)
        if len(starts) >= int(max_candidates):
            break
    return starts


def summarize_candidate_window(
    runtime: dict[str, Any],
    start: int,
    score_key: str,
    temperature: float,
) -> dict[str, Any]:
    seq_len = int(runtime["config"].seq_len)
    end = int(start) + seq_len
    score_arrays = runtime["score_arrays"]
    raw_window = np.asarray(runtime["raw_test"][start:end], dtype=np.float64)
    labels_window = np.asarray(runtime["labels"][start:end], dtype=np.int32)

    fused_curve = runtime_percentile(runtime, score_key)[start:end]
    event_start, event_end = resolve_event_span(labels_window, fused_curve)
    event_end = max(event_start + 1, event_end)
    base_start, base_end = resolve_baseline_span(seq_len, event_start, event_end)
    base_end = max(base_start + 1, base_end)

    fused_plot_curve = smooth_vector(np.asarray(fused_curve, dtype=np.float64), 5)
    component_keys = fusion_component_keys(runtime)
    component_plot_curves = {
        key: smooth_vector(np.asarray(runtime_percentile(runtime, key)[start:end], dtype=np.float64), 5)
        for key in component_keys
    }
    if component_keys:
        stacked = np.stack([component_plot_curves[key] for key in component_keys], axis=0)
        weight_curves = softmax_weights(stacked, temperature=temperature)
        weight_curves = smooth_columns(weight_curves.T, 7).T
        weight_curves = np.clip(weight_curves, 0.0, None)
        weight_curves = weight_curves / np.maximum(weight_curves.sum(axis=0, keepdims=True), 1e-12)
    else:
        weight_curves = np.zeros((0, seq_len), dtype=np.float64)

    event_slice = slice(event_start, event_end)
    base_slice = slice(base_start, base_end)
    event_curve = fused_plot_curve[event_slice]
    base_curve = fused_plot_curve[base_slice]
    if labels_window is not None and np.any(labels_window > 0):
        label_mask = np.asarray(labels_window > 0, dtype=bool)
    else:
        label_mask = np.zeros(seq_len, dtype=bool)
        label_mask[event_slice] = True
    inside_curve = fused_plot_curve[label_mask]
    outside_curve = fused_plot_curve[~label_mask]
    raw_local_z = local_zscore(raw_window)
    event_abs = np.abs(raw_local_z[event_slice]).mean(axis=0)
    base_abs = np.abs(raw_local_z[base_slice]).mean(axis=0)
    activity_ratio = event_abs / np.maximum(base_abs, 0.25)
    channel_activation_ratio = float(np.mean(activity_ratio >= 1.25))
    channel_mass = event_abs / max(float(event_abs.sum()), 1e-12)
    channel_concentration = float(np.sum(channel_mass**2))

    rel_count = min(10, raw_window.shape[1])
    rel_channels = np.argsort(event_abs)[::-1][:rel_count]
    base_corr = safe_corrcoef(raw_window[base_slice][:, rel_channels])
    event_corr = safe_corrcoef(raw_window[event_slice][:, rel_channels])
    relation_shift = float(np.mean(np.abs(event_corr - base_corr))) if base_corr.size else 0.0

    component_share_map = {key: 0.0 for key in FINAL_FUSION_COMPONENT_ORDER}
    if component_keys:
        event_weights = weight_curves[:, event_slice]
        event_contrib = event_weights * event_curve[None, :]
        share_vec = event_contrib.mean(axis=1)
        if float(share_vec.sum()) > 1e-12:
            share_vec = share_vec / float(share_vec.sum())
        for key, share in zip(component_keys, share_vec.tolist()):
            component_share_map[key] = float(share)
        dominant_key = component_keys[int(np.argmax(share_vec))]
    else:
        share_vec = np.zeros((0,), dtype=np.float64)
        dominant_key = "score"

    peakiness = float(np.max(event_curve) / max(float(np.mean(event_curve)), 1e-6))
    duration_ratio = float((event_end - event_start) / float(seq_len))
    plateau_ratio = float(np.mean(event_curve >= 0.70 * max(float(np.max(event_curve)), 1e-6)))
    ramp_delta = float(event_curve[-1] - event_curve[0]) if event_curve.size >= 2 else 0.0
    volatility = float(np.std(event_curve))
    event_score_mean = float(np.mean(event_curve))
    event_score_max = float(np.max(event_curve))
    base_score_mean = float(np.mean(base_curve)) if base_curve.size else 0.0
    label_score_mean = float(np.mean(inside_curve)) if inside_curve.size else event_score_mean
    non_label_score_mean = float(np.mean(outside_curve)) if outside_curve.size else base_score_mean
    alignment_gain = label_score_mean - non_label_score_mean
    label_peak = float(np.max(inside_curve)) if inside_curve.size else event_score_max
    entropy_keys = component_keys if component_keys else list(FINAL_FUSION_COMPONENT_ORDER)
    component_entropy = shannon_entropy(np.asarray([component_share_map[key] for key in entropy_keys], dtype=np.float64))

    feature_vector = np.asarray(
        [
            duration_ratio,
            peakiness,
            plateau_ratio,
            ramp_delta,
            volatility,
            event_score_mean,
            event_score_max,
            base_score_mean,
            label_score_mean,
            non_label_score_mean,
            alignment_gain,
            label_peak,
            channel_activation_ratio,
            channel_concentration,
            relation_shift,
            component_entropy,
            *[component_share_map.get(key, 0.0) for key in FINAL_FUSION_COMPONENT_ORDER],
        ],
        dtype=np.float64,
    )
    return {
        "start": int(start),
        "end": int(end),
        "score_key": str(score_key),
        "event_span": [int(event_start), int(event_end)],
        "baseline_span": [int(base_start), int(base_end)],
        "feature_vector": feature_vector,
        "dominant_component": str(dominant_key),
        "component_share_map": component_share_map,
        "duration_ratio": duration_ratio,
        "peakiness": peakiness,
        "plateau_ratio": plateau_ratio,
        "ramp_delta": ramp_delta,
        "volatility": volatility,
        "event_score_mean": event_score_mean,
        "event_score_max": event_score_max,
        "label_score_mean": label_score_mean,
        "non_label_score_mean": non_label_score_mean,
        "alignment_gain": alignment_gain,
        "label_peak": label_peak,
        "channel_activation_ratio": channel_activation_ratio,
        "channel_concentration": channel_concentration,
        "relation_shift": relation_shift,
    }


def robust_standardize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("robust_standardize expects a 2D matrix.")
    median = np.median(arr, axis=0, keepdims=True)
    q1 = np.quantile(arr, 0.25, axis=0, keepdims=True)
    q3 = np.quantile(arr, 0.75, axis=0, keepdims=True)
    scale = q3 - q1
    fallback = np.std(arr, axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, fallback, scale)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (arr - median) / scale


def kmeans_plus_plus_init(values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = values.shape[0]
    centers = [values[int(rng.integers(0, n))]]
    while len(centers) < k:
        stacked = np.stack(centers, axis=0)
        dist_sq = np.min(np.sum((values[:, None, :] - stacked[None, :, :]) ** 2, axis=2), axis=1)
        total = float(dist_sq.sum())
        if total <= 1e-12:
            centers.append(values[int(rng.integers(0, n))])
            continue
        probs = dist_sq / total
        idx = int(rng.choice(n, p=probs))
        centers.append(values[idx])
    return np.stack(centers, axis=0)


def run_kmeans(values: np.ndarray, k: int, seed: int = 7, n_init: int = 10, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = robust_standardize(values)
    n = x.shape[0]
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia: float | None = None

    for _ in range(max(1, n_init)):
        centers = kmeans_plus_plus_init(x, k=k, rng=rng)
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            dist_sq = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dist_sq, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            new_centers = centers.copy()
            for cluster_id in range(k):
                members = x[labels == cluster_id]
                if members.size == 0:
                    farthest = int(np.argmax(np.min(dist_sq, axis=1)))
                    new_centers[cluster_id] = x[farthest]
                else:
                    new_centers[cluster_id] = members.mean(axis=0)
            centers = new_centers
        final_dist_sq = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(final_dist_sq, axis=1)))
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    assert best_labels is not None and best_centers is not None
    return x, best_labels, best_centers


def pairwise_l2(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    diff = arr[:, None, :] - arr[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))


def silhouette_score(values: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = int(x.shape[0])
    unique = np.unique(y)
    if n <= 2 or unique.size <= 1 or unique.size >= n:
        return -1.0
    dists = pairwise_l2(x)
    scores = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        own = int(y[idx])
        same_mask = y == own
        same_mask[idx] = False
        if np.any(same_mask):
            a = float(np.mean(dists[idx, same_mask]))
        else:
            scores[idx] = 0.0
            continue
        b = min(float(np.mean(dists[idx, y == other])) for other in unique.tolist() if int(other) != own)
        denom = max(a, b, 1e-12)
        scores[idx] = (b - a) / denom
    return float(np.mean(scores))


def choose_cluster_count_auto(
    feature_matrix: np.ndarray,
    display_count: int,
    min_clusters: int,
    max_clusters: int,
) -> tuple[int, list[dict[str, Any]]]:
    n = int(feature_matrix.shape[0])
    if n <= 1:
        return 1, []
    min_k = max(2, int(min_clusters))
    max_k = max(min_k, min(int(max_clusters), int(display_count), max(2, n - 1)))
    min_k = min(min_k, max_k)
    diagnostics: list[dict[str, Any]] = []
    best_k = min_k
    best_score: float | None = None
    for k in range(min_k, max_k + 1):
        x, labels, _ = run_kmeans(feature_matrix, k=k)
        unique, counts = np.unique(labels, return_counts=True)
        singletons = int(np.sum(counts <= 1))
        silhouette = silhouette_score(x, labels)
        singleton_penalty = 0.08 * (singletons / max(1, len(unique)))
        adjusted = float(silhouette - singleton_penalty)
        diagnostics.append(
            {
                "k": int(k),
                "silhouette": float(silhouette),
                "singleton_clusters": int(singletons),
                "adjusted_score": adjusted,
                "cluster_sizes": [int(v) for v in counts.tolist()],
            }
        )
        if best_score is None or adjusted > best_score + 1e-6 or (
            abs(adjusted - best_score) <= 0.015 and k < best_k
        ):
            best_score = adjusted
            best_k = int(k)
    return best_k, diagnostics


def filter_candidates_by_alignment(
    candidates: list[dict[str, Any]],
    min_alignment_gain: float,
    target_count: int,
    preferred_pool_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        return [], {
            "min_alignment_gain": float(min_alignment_gain),
            "candidate_count_before": 0,
            "candidate_count_after": 0,
            "selection_mode": "empty",
        }
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["alignment_gain"]),
            float(item["non_label_score_mean"]),
            -float(item["label_score_mean"]),
            -float(item["label_peak"]),
            -float(item["event_score_max"]),
            int(item["start"]),
        ),
    )
    if float(min_alignment_gain) > 0.0:
        eligible = [
            item
            for item in ranked
            if float(item["alignment_gain"]) >= float(min_alignment_gain)
        ]
        if len(eligible) >= max(1, int(target_count)):
            source = eligible
            selection_mode = "threshold_then_rank"
        else:
            source = ranked
            selection_mode = "ranking_fallback"
    else:
        source = ranked
        selection_mode = "ranking_only"

    pool_size = min(
        len(source),
        max(int(target_count), int(preferred_pool_size)),
    )
    kept = source[:pool_size]
    summary = {
        "min_alignment_gain": float(min_alignment_gain),
        "candidate_count_before": int(len(candidates)),
        "candidate_count_after": int(len(kept)),
        "selection_mode": selection_mode,
        "preferred_pool_size": int(preferred_pool_size),
        "kept_alignment_range": [
            float(min(float(item["alignment_gain"]) for item in kept)),
            float(max(float(item["alignment_gain"]) for item in kept)),
        ],
    }
    return kept, summary


def fill_with_diverse_candidates(
    feature_matrix: np.ndarray,
    selected: list[int],
    need: int,
    strength: np.ndarray,
) -> list[int]:
    x = robust_standardize(feature_matrix)
    selected = list(selected)
    if not selected:
        selected.append(int(np.argmax(strength)))
    while len(selected) < need:
        remaining = [idx for idx in range(x.shape[0]) if idx not in selected]
        if not remaining:
            break
        scored = []
        for idx in remaining:
            min_dist = min(float(np.sum((x[idx] - x[chosen]) ** 2)) for chosen in selected)
            scored.append((float(strength[idx]), min_dist, idx))
        scored.sort(reverse=True)
        selected.append(int(scored[0][2]))
    return selected


def component_descriptor(component_share_map: dict[str, float]) -> str:
    aliases = {
        "completion_scale8": "Short pred",
        "completion_scale32": "Long pred",
        "knn_distance": "Memory",
        "soft_support_score": "Support",
        "state_novelty": "State",
    }
    ranked = sorted(component_share_map.items(), key=lambda item: float(item[1]), reverse=True)
    ranked = [(key, float(value)) for key, value in ranked if value > 1e-4]
    if not ranked:
        return "Score-led"
    top_key, top_value = ranked[0]
    top_label = aliases.get(top_key, top_key)
    if len(ranked) >= 2 and ranked[1][1] >= 0.72 * top_value and ranked[1][1] >= 0.18:
        second_label = aliases.get(ranked[1][0], ranked[1][0])
        return f"{top_label}+{second_label}"
    return top_label


def short_morphology_label(name: str) -> str:
    mapping = {
        "relation shift": "rel. shift",
        "broad drift": "drift",
        "persistent": "persist.",
        "volatile": "volat.",
        "burst": "burst",
        "ramp": "ramp",
        "mixed": "mixed",
    }
    return mapping.get(str(name), str(name))


def morphology_descriptor(metrics: dict[str, float]) -> str:
    if float(metrics["relation_shift"]) >= 0.34:
        return "relation shift"
    if float(metrics["duration_ratio"]) <= 0.14 and float(metrics["peakiness"]) >= 1.45:
        return "burst"
    if float(metrics["ramp_delta"]) >= 0.22:
        return "ramp"
    if float(metrics["duration_ratio"]) >= 0.38 or float(metrics["plateau_ratio"]) >= 0.42:
        return "persistent"
    if float(metrics["channel_activation_ratio"]) >= 0.42 and float(metrics["channel_concentration"]) <= 0.18:
        return "broad drift"
    if float(metrics["volatility"]) >= 0.24:
        return "volatile"
    return "mixed"


def mine_auto_case_specs(
    runtime: dict[str, Any],
    prototypes_per_dataset: int,
    max_candidates: int,
    min_separation: int,
    temperature: float,
    cluster_count_mode: str,
    min_clusters: int,
    max_clusters: int,
    min_alignment_gain: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset = str(runtime["config"].dataset)
    score_key = choose_score_key(runtime["score_arrays"], requested=None)
    candidate_starts = collect_candidate_starts(
        runtime=runtime,
        score_key=score_key,
        max_candidates=max_candidates,
        min_separation=min_separation,
    )
    candidates = [
        summarize_candidate_window(runtime=runtime, start=start, score_key=score_key, temperature=temperature)
        for start in candidate_starts
    ]
    desired_case_count = max(1, int(prototypes_per_dataset))
    preferred_pool_size = min(len(candidates), max(desired_case_count * 3, 12))
    aligned_candidates, alignment_summary = filter_candidates_by_alignment(
        candidates=candidates,
        min_alignment_gain=min_alignment_gain,
        target_count=desired_case_count,
        preferred_pool_size=preferred_pool_size,
    )
    target_case_count = min(desired_case_count, int(len(aligned_candidates)))
    if target_case_count <= 0:
        raise RuntimeError(f"Dataset {dataset} yielded no valid candidates after alignment filtering.")

    feature_matrix = np.stack([item["feature_vector"] for item in aligned_candidates], axis=0)
    strength = np.asarray(
        [
            0.55 * float(item["label_peak"])
            + 0.35 * max(float(item["alignment_gain"]), 0.0)
            + 0.10 * float(item["event_score_max"])
            for item in aligned_candidates
        ],
        dtype=np.float64,
    )
    cluster_diagnostics: list[dict[str, Any]] = []
    effective_max_clusters = max(1, min(int(max_clusters), int(target_case_count), int(len(aligned_candidates))))
    effective_min_clusters = max(1, min(int(min_clusters), int(effective_max_clusters)))
    if str(cluster_count_mode).lower() == "auto":
        cluster_count, cluster_diagnostics = choose_cluster_count_auto(
            feature_matrix=feature_matrix,
            display_count=target_case_count,
            min_clusters=effective_min_clusters,
            max_clusters=effective_max_clusters,
        )
    else:
        cluster_count = max(1, min(int(target_case_count), len(aligned_candidates)))
    x, labels, centers = run_kmeans(feature_matrix, k=cluster_count)
    selected_indices: list[int] = []
    selected_roles: list[str] = []
    cluster_summaries: list[dict[str, Any]] = []
    for cluster_id in range(int(np.max(labels)) + 1):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0:
            continue
        dist_sq = np.sum((x[members] - centers[cluster_id][None, :]) ** 2, axis=1)
        ranked_members = sorted(
            zip(dist_sq.tolist(), members.tolist()),
            key=lambda item: (
                -float(aligned_candidates[item[1]]["alignment_gain"]),
                -float(aligned_candidates[item[1]]["label_peak"]),
                item[0],
                -float(aligned_candidates[item[1]]["event_score_max"]),
            ),
        )
        rep_idx = int(ranked_members[0][1])
        selected_indices.append(rep_idx)
        selected_roles.append("cluster_representative")
        member_component_mean = {
            key: float(np.mean([aligned_candidates[idx]["component_share_map"][key] for idx in members.tolist()]))
            for key in FINAL_FUSION_COMPONENT_ORDER
        }
        metric_mean = {
            key: float(np.mean([aligned_candidates[idx][key] for idx in members.tolist()]))
            for key in (
                "duration_ratio",
                "peakiness",
                "plateau_ratio",
                "ramp_delta",
                "volatility",
                "label_score_mean",
                "non_label_score_mean",
                "alignment_gain",
                "label_peak",
                "channel_activation_ratio",
                "channel_concentration",
                "relation_shift",
            )
        }
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(members.size),
                "representative_index": int(rep_idx),
                "selection_role": "cluster_representative",
                "component_descriptor": component_descriptor(member_component_mean),
                "morphology_descriptor": morphology_descriptor(metric_mean),
                "mean_component_share_map": member_component_mean,
                "mean_metrics": metric_mean,
            }
        )

    if len(selected_indices) < target_case_count:
        filled_indices = fill_with_diverse_candidates(
            feature_matrix=feature_matrix,
            selected=selected_indices,
            need=target_case_count,
            strength=strength,
        )
        extra_indices = [idx for idx in filled_indices if idx not in selected_indices]
        selected_indices = filled_indices
        selected_roles.extend(["diverse_exemplar"] * len(extra_indices))

    cluster_lookup = {item["representative_index"]: item for item in cluster_summaries}
    selected_candidates = [aligned_candidates[idx] for idx in selected_indices]
    selected_cluster_summaries = [
        cluster_lookup.get(
            idx,
            {
                "cluster_id": -1,
                "size": 1,
                "representative_index": int(idx),
                "selection_role": "diverse_exemplar",
                "component_descriptor": component_descriptor(aligned_candidates[idx]["component_share_map"]),
                "morphology_descriptor": morphology_descriptor(aligned_candidates[idx]),
                "mean_component_share_map": aligned_candidates[idx]["component_share_map"],
                "mean_metrics": {
                    key: float(aligned_candidates[idx][key])
                    for key in (
                        "duration_ratio",
                        "peakiness",
                        "plateau_ratio",
                        "ramp_delta",
                        "volatility",
                        "label_score_mean",
                        "non_label_score_mean",
                        "alignment_gain",
                        "label_peak",
                        "channel_activation_ratio",
                        "channel_concentration",
                        "relation_shift",
                    )
                },
            },
        )
        for idx in selected_indices
    ]
    ordering = sorted(
        range(len(selected_candidates)),
        key=lambda i: (
            -max(float(selected_candidates[i]["alignment_gain"]), -1.0),
            -float(selected_candidates[i]["label_peak"]),
            -int(selected_cluster_summaries[i]["size"]),
            -float(selected_candidates[i]["event_score_max"]),
            int(selected_candidates[i]["start"]),
        ),
    )

    case_specs: list[dict[str, Any]] = []
    ordered_cluster_summaries: list[dict[str, Any]] = []
    role_lookup = {idx: role for idx, role in zip(selected_indices, selected_roles)}
    for proto_rank, order_idx in enumerate(ordering, start=1):
        candidate = selected_candidates[order_idx]
        cluster_summary = selected_cluster_summaries[order_idx]
        selection_role = str(role_lookup.get(int(cluster_summary["representative_index"]), cluster_summary.get("selection_role", "cluster_representative")))
        reviewer_span = reviewer_span_text(int(candidate["start"]), int(candidate["end"]))
        reviewer_desc = (
            f"{reviewer_component_text(cluster_summary['component_descriptor']).capitalize()} dominant; "
            f"{reviewer_morphology_text(cluster_summary['morphology_descriptor'])}"
        )
        subtitle = f"{reviewer_span}\n{reviewer_desc}"
        case_specs.append(
            {
                "experiment_dir": str(runtime["experiment_dir"]),
                "start": int(candidate["start"]),
                "score_key": score_key,
                "title": f"{dataset} C{proto_rank}",
                "subtitle": subtitle,
            }
        )
        cluster_summary = dict(cluster_summary)
        cluster_summary["display_rank"] = int(proto_rank)
        cluster_summary["selection_role"] = selection_role
        cluster_summary["start"] = int(candidate["start"])
        cluster_summary["end"] = int(candidate["end"])
        cluster_summary["score_key"] = str(score_key)
        cluster_summary["title"] = case_specs[-1]["title"]
        cluster_summary["subtitle"] = case_specs[-1]["subtitle"]
        ordered_cluster_summaries.append(cluster_summary)
    mining_summary = {
        "cluster_count_mode": str(cluster_count_mode),
        "selected_cluster_count": int(cluster_count),
        "display_case_count_requested": int(prototypes_per_dataset),
        "display_case_count_actual": int(target_case_count),
        "cluster_count_search": cluster_diagnostics,
        "candidate_count_before_alignment_priority": int(len(candidates)),
        "candidate_count_after_alignment_priority": int(len(aligned_candidates)),
        "alignment_priority": alignment_summary,
    }
    return case_specs, ordered_cluster_summaries, mining_summary


def find_dataset_experiment_dirs(scan_root: Path, datasets: list[str]) -> dict[str, Path]:
    wanted = {str(name).upper(): str(name).upper() for name in datasets}
    grouped: dict[str, list[Path]] = {}
    for path in sorted(scan_root.iterdir()):
        if not path.is_dir():
            continue
        cfg_path = path / "config.json"
        npz_path = path / "test_diagnostic_scores.npz"
        if not (cfg_path.exists() and npz_path.exists()):
            continue
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        dataset = str(config.get("dataset", "")).upper()
        if dataset in wanted:
            grouped.setdefault(dataset, []).append(path)

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for dataset in wanted:
        candidates = grouped.get(dataset, [])
        if not candidates:
            missing.append(dataset)
            continue
        candidates = sorted(candidates, key=lambda path: (path.name, str(path)))
        resolved[dataset] = candidates[-1]
    if missing:
        raise FileNotFoundError(
            "Automatic prototype mode could not find experiment folders for: " + ", ".join(sorted(missing))
        )
    return resolved

def build_row_a_payload(
    raw_window: np.ndarray,
    residual_norm: np.ndarray,
    highlight_count: int,
) -> dict[str, Any]:
    n_channels = int(raw_window.shape[1])
    if n_channels >= 30:
        highlight_n = min(max(6, highlight_count), 8, n_channels)
    else:
        highlight_n = min(max(6, highlight_count - 1), 8, n_channels)
    highlight_channels = choose_channels(residual_norm, highlight_n)

    if n_channels >= 30:
        all_channels = list(range(n_channels))
        dense_bg = smooth_columns(local_zscore(raw_window[:, all_channels]), kernel_size=3) * 0.42
        dense_fg = smooth_columns(local_zscore(raw_window[:, highlight_channels]), kernel_size=5) * 0.62
        return {
            "style": "dense",
            "highlight_channels": highlight_channels,
            "highlight_values": dense_fg,
            "background_channels": all_channels,
            "background_values": dense_bg,
            "ylabel": "z",
        }

    stack_count = min(n_channels, max(highlight_n + 8, 14))
    stack_channels = choose_channels(residual_norm, stack_count)
    stack_values = smooth_columns(local_zscore(raw_window[:, stack_channels]), kernel_size=5) * 0.62
    return {
        "style": "stacked",
        "highlight_channels": highlight_channels,
        "stack_channels": stack_channels,
        "stack_values": stack_values,
        "ylabel": "z+off",
    }


def softmax_weights(curves: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(curves, dtype=np.float64) / max(float(temperature), 1e-6)
    logits = logits - np.nanmax(logits, axis=0, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    return weights


def top_changed_pairs(delta: np.ndarray, k: int = 3) -> list[tuple[int, int]]:
    if delta.size == 0 or delta.shape[0] <= 1:
        return []
    upper = np.triu_indices(delta.shape[0], k=1)
    values = np.abs(delta[upper])
    if values.size == 0:
        return []
    order = np.argsort(values)[::-1][:k]
    return [(int(upper[0][idx]), int(upper[1][idx])) for idx in order.tolist()]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 11.0,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": ":",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.frameon": True,
        }
    )


def build_case_payload(
    runtime: dict[str, Any],
    case_spec: dict[str, Any],
    channels: int,
    relation_channels: int,
    temperature: float,
) -> dict[str, Any]:
    config = runtime["config"]
    seq_len = int(config.seq_len)
    raw_test = runtime["raw_test"]
    labels = runtime["labels"]
    score_arrays = runtime["score_arrays"]

    start = int(case_spec["start"])
    end = start + seq_len
    if start < 0 or end > len(raw_test):
        raise IndexError(
            f"Case start={start} with seq_len={seq_len} is outside test range [0, {len(raw_test)})."
        )
    score_key = choose_score_key(score_arrays, case_spec.get("score_key"))

    raw_window = raw_test[start:end]
    labels_window = np.asarray(labels[start:end], dtype=np.int32) if labels is not None else None
    x_norm = runtime["normalizer"].transform(raw_window)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(runtime["trainer"].device)
    with torch.no_grad():
        encoded = runtime["model"].encode(x_tensor)

    slow_norm = encoded["slow"].squeeze(0).detach().cpu().numpy().astype(np.float64)
    residual_norm = encoded["residual"].squeeze(0).detach().cpu().numpy().astype(np.float64)
    mean = np.asarray(runtime["normalizer"].mean_, dtype=np.float64)
    scale = np.asarray(runtime["normalizer"].scale_, dtype=np.float64)
    slow_raw = slow_norm * scale[None, :] + mean[None, :]
    residual_raw = residual_norm * scale[None, :]

    display_channels = choose_channels(residual_norm, channels)
    row_a_payload = build_row_a_payload(raw_window=raw_window, residual_norm=residual_norm, highlight_count=channels)
    relation_ch = choose_channels(residual_norm, relation_channels)
    plot_t = np.arange(seq_len, dtype=np.int64)

    component_keys = component_keys_for_case(runtime, start, end)
    component_curves = {
        key: runtime_percentile(runtime, key)[start:end]
        for key in component_keys
    }
    fused_curve = runtime_percentile(runtime, score_key)[start:end]
    event_start, event_end = resolve_event_span(labels_window, fused_curve)
    base_start, base_end = resolve_baseline_span(seq_len, event_start, event_end)
    fused_plot_curve = smooth_columns(np.asarray(fused_curve, dtype=np.float64)[:, None], 5)[:, 0]
    component_plot_curves = {
        key: smooth_columns(np.asarray(component_curves[key], dtype=np.float64)[:, None], 5)[:, 0]
        for key in component_keys
    }
    stacked_components = np.stack([component_plot_curves[key] for key in component_keys], axis=0)
    weight_curves = softmax_weights(stacked_components, temperature=temperature)
    weight_curves = smooth_columns(weight_curves.T, 7).T
    weight_curves = np.clip(weight_curves, 0.0, None)
    weight_curves = weight_curves / np.maximum(weight_curves.sum(axis=0, keepdims=True), 1e-12)
    composition_curves = weight_curves * fused_plot_curve[None, :]

    relation_source = str(case_spec.get("relation_source", "slow")).lower()
    if relation_source == "raw":
        relation_data = raw_window[:, relation_ch]
    elif relation_source == "residual":
        relation_data = residual_raw[:, relation_ch]
    else:
        relation_data = slow_raw[:, relation_ch]
    base_corr = safe_corrcoef(relation_data[base_start:base_end])
    event_corr = safe_corrcoef(relation_data[event_start:event_end])
    delta_corr = event_corr - base_corr

    raw_display = local_zscore(raw_window[:, display_channels])
    return {
        "experiment_dir": str(runtime["experiment_dir"]),
        "dataset": str(config.dataset),
        "title": str(case_spec.get("title", config.dataset)),
        "subtitle": str(case_spec.get("subtitle", "")),
        "start": int(start),
        "end": int(end),
        "seq_len": seq_len,
        "score_key": score_key,
        "plot_t": plot_t,
        "labels_window": labels_window,
        "display_channels": display_channels,
        "raw_display": raw_display,
        "row_a_payload": row_a_payload,
        "fused_curve": fused_curve,
        "fused_plot_curve": fused_plot_curve,
        "component_keys": component_keys,
        "component_curves": component_curves,
        "component_plot_curves": component_plot_curves,
        "weight_curves": weight_curves,
        "composition_curves": composition_curves,
        "event_span": [int(event_start), int(event_end)],
        "baseline_span": [int(base_start), int(base_end)],
        "relation_channels": relation_ch,
        "relation_source": relation_source,
        "relation_base": base_corr,
        "relation_event": event_corr,
        "relation_delta": delta_corr,
        "relation_markers": top_changed_pairs(delta_corr, k=3),
    }


def style_legend_frame(legend: Any | None) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.92)
    frame.set_edgecolor("#d1d5db")
    frame.set_linewidth(0.6)


def style_timeline_axis(ax, show_xlabel: bool = False, show_yticklabels: bool = True) -> None:
    del show_xlabel, show_yticklabels
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#cfcfd4")
    ax.grid(True, axis="both", linestyle=":", linewidth=0.6, alpha=0.22)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, pad=1.5, width=0.6, length=2.5)
    ax.tick_params(axis="x", labelbottom=True)
    ax.tick_params(axis="y", labelleft=True)


def add_event_shading(ax, event_span: tuple[int, int], labels_window: np.ndarray | None) -> None:
    start, end = [int(v) for v in event_span]
    if labels_window is not None and np.any(labels_window > 0):
        mask = np.asarray(labels_window, dtype=np.int32) > 0
        for seg_start, seg_end in contiguous_segments(mask):
            ax.axvspan(int(seg_start), int(seg_end), color="#fecaca", alpha=0.18, linewidth=0.0)
    ax.axvspan(start, end, color="#fca5a5", alpha=0.30, linewidth=0.0)


def draw_column_title(ax, title: str, subtitle: str) -> None:
    ax.text(
        0.5,
        1.275,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.8,
        fontweight="bold",
        color="#000000",
        clip_on=False,
    )
    subtitle_lines = [line.strip() for line in str(subtitle).splitlines() if line.strip()]
    for line_idx, line in enumerate(subtitle_lines[:2]):
        ax.text(
            0.5,
            1.155 - 0.080 * line_idx,
            line,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#000000",
            clip_on=False,
            wrap=True,
            multialignment="center",
        )


def render_figure(
    cases: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
    composition_mode: str,
    figure_title: str | None = None,
) -> None:
    if not cases:
        raise RuntimeError("No cases were provided for rendering.")

    set_plot_style()
    n_cols = len(cases)
    fig = plt.figure(figsize=(5.02 * n_cols, 10.30))
    grid = fig.add_gridspec(
        5,
        n_cols,
        height_ratios=[1.08, 0.82, 0.82, 0.82, 1.02],
        left=0.082,
        right=0.992,
        top=0.855,
        bottom=0.072,
        wspace=0.17,
        hspace=0.12,
    )

    relation_vmax = max(
        max(float(np.nanmax(np.abs(case["relation_delta"]))) for case in cases),
        0.25,
    )
    row_anchor_axes: dict[int, Any] = {}

    for col_idx, case in enumerate(cases):
        event_span = tuple(case["event_span"])
        labels_window = case["labels_window"]

        ax_a = fig.add_subplot(grid[0, col_idx])
        if col_idx == 0:
            row_anchor_axes[0] = ax_a
        draw_column_title(ax_a, case["title"], case["subtitle"])
        add_event_shading(ax_a, event_span, labels_window)
        row_a_payload = case["row_a_payload"]
        if row_a_payload["style"] == "dense":
            bg_values = np.asarray(row_a_payload["background_values"], dtype=np.float64)
            for idx in range(bg_values.shape[1]):
                ax_a.plot(
                    case["plot_t"],
                    bg_values[:, idx],
                    color=ROW_A_COLORS[idx % len(ROW_A_COLORS)],
                    alpha=0.07,
                    linewidth=0.45,
                    zorder=1,
                )
            fg_values = np.asarray(row_a_payload["highlight_values"], dtype=np.float64)
            for idx, channel in enumerate(row_a_payload["highlight_channels"]):
                ax_a.plot(
                    case["plot_t"],
                    fg_values[:, idx],
                    color=ROW_A_COLORS[idx % len(ROW_A_COLORS)],
                    alpha=0.90,
                    linewidth=0.95,
                    label=f"c{int(channel)}",
                    zorder=3,
                )
            y_abs = max(float(np.nanpercentile(np.abs(bg_values), 99.0)), 0.8)
            ax_a.set_ylim(-1.15 * y_abs, 1.15 * y_abs)
        else:
            stack_channels = list(row_a_payload["stack_channels"])
            stack_values = np.asarray(row_a_payload["stack_values"], dtype=np.float64)
            highlight_set = {int(ch) for ch in row_a_payload["highlight_channels"]}
            offsets = np.arange(len(stack_channels), dtype=np.float64) * 1.30
            for idx, channel in enumerate(stack_channels):
                is_highlight = int(channel) in highlight_set
                color = ROW_A_COLORS[idx % len(ROW_A_COLORS)] if is_highlight else "#cbd5e1"
                alpha = 0.95 if is_highlight else 0.60
                linewidth = 1.15 if is_highlight else 0.75
                label = f"c{int(channel)}" if is_highlight else None
                ax_a.plot(
                    case["plot_t"],
                    stack_values[:, idx] + offsets[idx],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=label,
                    zorder=3 if is_highlight else 1,
                )
        ax_a.set_ylabel(row_a_payload["ylabel"] if col_idx == 0 else "")
        handles, labels = ax_a.get_legend_handles_labels()
        if handles:
            legend = ax_a.legend(
                handles,
                labels,
                loc="upper right",
                ncol=2 if len(labels) >= 6 else 1,
                fontsize=LEGEND_FONT_SIZE,
                columnspacing=0.8,
                handlelength=1.4,
                borderpad=0.3,
                labelspacing=0.25,
                handletextpad=0.4,
                frameon=True,
            )
            style_legend_frame(legend)
        style_timeline_axis(ax_a, show_xlabel=False, show_yticklabels=(col_idx == 0))
        ax_a.set_xlim(0, case["seq_len"] - 1)

        ax_b = fig.add_subplot(grid[1, col_idx], sharex=ax_a)
        if col_idx == 0:
            row_anchor_axes[1] = ax_b
        add_event_shading(ax_b, event_span, labels_window)
        ax_b.plot(case["plot_t"], case["fused_plot_curve"], color="#1f2937", linewidth=1.10)
        ax_b.set_ylim(-0.02, 1.02)
        ax_b.set_ylabel("score" if col_idx == 0 else "")
        style_timeline_axis(ax_b, show_xlabel=False, show_yticklabels=(col_idx == 0))

        ax_c = fig.add_subplot(grid[2, col_idx], sharex=ax_a)
        if col_idx == 0:
            row_anchor_axes[2] = ax_c
        add_event_shading(ax_c, event_span, labels_window)
        if composition_mode == "stacked":
            stack_values = np.asarray(case["composition_curves"], dtype=np.float64)
            colors = [COMPONENT_STYLE[key][1] for key in case["component_keys"]]
            labels = [COMPONENT_STYLE[key][0] for key in case["component_keys"]]
            handles = ax_c.stackplot(
                case["plot_t"],
                *stack_values,
                colors=colors,
                alpha=0.78,
                linewidth=0.0,
            )
            ax_c.plot(case["plot_t"], case["fused_plot_curve"], color="#111827", linewidth=0.85, alpha=0.85)
            if col_idx == n_cols - 1:
                legend = ax_c.legend(
                    handles,
                    labels,
                    loc="upper right",
                    fontsize=LEGEND_FONT_SIZE,
                    borderpad=0.3,
                    labelspacing=0.25,
                    handlelength=1.3,
                    handletextpad=0.4,
                    frameon=True,
                )
                style_legend_frame(legend)
        else:
            for key in case["component_keys"]:
                label, color = COMPONENT_STYLE[key]
                ax_c.plot(case["plot_t"], case["component_plot_curves"][key], linewidth=0.95, color=color, label=label)
            if col_idx == n_cols - 1:
                legend = ax_c.legend(
                    loc="upper right",
                    fontsize=LEGEND_FONT_SIZE,
                    borderpad=0.3,
                    labelspacing=0.25,
                    handlelength=1.3,
                    handletextpad=0.4,
                    frameon=True,
                )
                style_legend_frame(legend)
        ax_c.set_ylim(-0.02, 1.02)
        ax_c.set_ylabel(("contrib." if composition_mode == "stacked" else "evidence") if col_idx == 0 else "")
        style_timeline_axis(ax_c, show_xlabel=False, show_yticklabels=(col_idx == 0))

        ax_d = fig.add_subplot(grid[3, col_idx], sharex=ax_a)
        if col_idx == 0:
            row_anchor_axes[3] = ax_d
        add_event_shading(ax_d, event_span, labels_window)
        for idx, key in enumerate(case["component_keys"]):
            label, color = COMPONENT_STYLE[key]
            ax_d.plot(case["plot_t"], case["weight_curves"][idx], linewidth=1.15, color=color, label=label)
        ax_d.set_ylim(-0.02, 1.02)
        ax_d.set_ylabel("weight" if col_idx == 0 else "")
        if col_idx == n_cols - 1:
            legend = ax_d.legend(
                loc="upper right",
                fontsize=LEGEND_FONT_SIZE,
                borderpad=0.3,
                labelspacing=0.25,
                handlelength=1.3,
                handletextpad=0.4,
                frameon=True,
            )
            style_legend_frame(legend)
        style_timeline_axis(ax_d, show_xlabel=True, show_yticklabels=(col_idx == 0))

        subgrid = grid[4, col_idx].subgridspec(1, 3, wspace=0.06)
        heatmaps = [
            ("Base", case["relation_base"], "Blues", -1.0, 1.0),
            ("Event", case["relation_event"], "Reds", -1.0, 1.0),
            ("Delta", case["relation_delta"], "coolwarm", -relation_vmax, relation_vmax),
        ]
        relation_tick_labels = [str(int(channel)) for channel in case["relation_channels"]]
        for heat_idx, (label, matrix, cmap, vmin, vmax) in enumerate(heatmaps):
            ax_h = fig.add_subplot(subgrid[0, heat_idx])
            if col_idx == 0 and heat_idx == 0:
                row_anchor_axes[4] = ax_h
            ax_h.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
            ax_h.set_title(label, fontsize=8, pad=3)
            ax_h.set_xticks(np.arange(len(relation_tick_labels)))
            ax_h.set_yticks(np.arange(len(relation_tick_labels)))
            ax_h.set_xticklabels(relation_tick_labels, rotation=90, fontsize=HEATMAP_TICK_SIZE)
            ax_h.set_yticklabels(relation_tick_labels, fontsize=HEATMAP_TICK_SIZE)
            ax_h.tick_params(axis="x", pad=0.6, length=1.8, width=0.4)
            ax_h.tick_params(axis="y", pad=0.6, length=1.8, width=0.4)
            for spine in ax_h.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_color("#d4d4d8")
            if label == "Delta":
                for row_id, col_id in case["relation_markers"]:
                    ax_h.scatter([col_id, row_id], [row_id, col_id], s=10, c="black", alpha=0.85)

    for row_idx, row_label in enumerate(ROW_LABELS):
        bbox = row_anchor_axes[row_idx].get_position()
        fig.text(
            bbox.x0 - 0.058,
            bbox.y0 + bbox.height / 2.0,
            row_label,
            ha="left",
            va="center",
            fontsize=12.4,
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def load_manifest(path: Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    payload["_manifest_path"] = str(resolved)
    return payload


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)

    if args.auto_prototypes:
        scan_root = resolve_repo_path(args.scan_root)
        dataset_dirs = find_dataset_experiment_dirs(scan_root=scan_root, datasets=list(args.datasets))
        runtime_cache: dict[str, dict[str, Any]] = {}
        global_summary: dict[str, Any] = {
            "mode": "auto_prototypes",
            "scan_root": str(scan_root),
            "output_dir": str(output_dir),
            "datasets": [],
            "temperature": float(args.temperature),
            "channels": int(args.channels),
            "relation_channels": int(args.relation_channels),
            "composition_mode": str(args.composition_mode),
            "prototypes_per_dataset": int(args.prototypes_per_dataset),
            "cluster_count_mode": str(args.cluster_count_mode),
            "min_clusters": int(args.min_clusters),
            "max_clusters": int(args.max_clusters),
            "min_alignment_gain": float(args.min_alignment_gain),
            "max_candidates": int(args.max_candidates),
        }
        for dataset in [str(name).upper() for name in args.datasets]:
            experiment_dir = dataset_dirs[dataset]
            runtime = runtime_cache.get(str(experiment_dir))
            if runtime is None:
                runtime = load_runtime(experiment_dir, device=args.device)
                runtime_cache[str(experiment_dir)] = runtime
            seq_len = int(runtime["config"].seq_len)
            min_separation = int(args.candidate_separation) if int(args.candidate_separation) > 0 else max(8, seq_len // 2)
            case_specs, prototype_summary, mining_summary = mine_auto_case_specs(
                runtime=runtime,
                prototypes_per_dataset=args.prototypes_per_dataset,
                max_candidates=args.max_candidates,
                min_separation=min_separation,
                temperature=args.temperature,
                cluster_count_mode=args.cluster_count_mode,
                min_clusters=args.min_clusters,
                max_clusters=args.max_clusters,
                min_alignment_gain=args.min_alignment_gain,
            )
            cases = [
                build_case_payload(
                    runtime=runtime,
                    case_spec=case_spec,
                    channels=args.channels,
                    relation_channels=args.relation_channels,
                    temperature=args.temperature,
                )
                for case_spec in case_specs
            ]
            output_path = output_dir / f"{dataset.lower()}_{args.output_stem}.png"
            render_figure(
                cases=cases,
                output_path=output_path,
                dpi=args.dpi,
                composition_mode=str(args.composition_mode),
                figure_title=None,
            )
            dataset_summary = {
                "dataset": dataset,
                "experiment_dir": str(experiment_dir),
                "output_png": str(output_path),
                "output_pdf": str(output_path.with_suffix(".pdf")),
                "mining_summary": mining_summary,
                "case_specs": case_specs,
                "prototypes": prototype_summary,
            }
            summary_path = output_dir / f"{dataset.lower()}_{args.output_stem}_summary.json"
            summary_path.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            global_summary["datasets"].append(dataset_summary)
            print(f"[CasePanel] wrote {output_path}")
            print(f"[CasePanel] wrote {output_path.with_suffix('.pdf')}")
            print(f"[CasePanel] wrote {summary_path}")

        global_summary_path = output_dir / f"{args.output_stem}_summary.json"
        global_summary_path.write_text(json.dumps(global_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[CasePanel] wrote {global_summary_path}")
        return

    manifest = load_manifest(args.manifest)
    output_path = output_dir / f"{args.output_stem}.png"

    runtime_cache: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    summary_cases: list[dict[str, Any]] = []
    for case_spec in manifest.get("cases", []):
        experiment_dir = resolve_repo_path(case_spec["experiment_dir"])
        runtime = runtime_cache.get(str(experiment_dir))
        if runtime is None:
            runtime = load_runtime(experiment_dir, device=args.device)
            runtime_cache[str(experiment_dir)] = runtime
        payload = build_case_payload(
            runtime=runtime,
            case_spec=case_spec,
            channels=args.channels,
            relation_channels=args.relation_channels,
            temperature=args.temperature,
        )
        cases.append(payload)
        summary_cases.append(
            {
                "title": payload["title"],
                "subtitle": payload["subtitle"],
                "experiment_dir": payload["experiment_dir"],
                "start": payload["start"],
                "end": payload["end"],
                "score_key": payload["score_key"],
                "component_keys": payload["component_keys"],
                "display_channels": payload["display_channels"],
                "row_a_style": payload["row_a_payload"]["style"],
                "relation_channels": payload["relation_channels"],
                "relation_source": payload["relation_source"],
                "event_span_in_window": payload["event_span"],
                "baseline_span_in_window": payload["baseline_span"],
            }
        )

    render_figure(
        cases=cases,
        output_path=output_path,
        dpi=args.dpi,
        composition_mode=str(args.composition_mode),
        figure_title=manifest.get("figure_title"),
    )
    summary = {
        "manifest_path": manifest["_manifest_path"],
        "output_png": str(output_path),
        "output_pdf": str(output_path.with_suffix(".pdf")),
        "temperature": float(args.temperature),
        "channels": int(args.channels),
        "relation_channels": int(args.relation_channels),
        "composition_mode": str(args.composition_mode),
        "cases": summary_cases,
    }
    summary_path = output_dir / f"{args.output_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[CasePanel] wrote {output_path}")
    print(f"[CasePanel] wrote {output_path.with_suffix('.pdf')}")
    print(f"[CasePanel] wrote {summary_path}")


if __name__ == "__main__":
    main()
