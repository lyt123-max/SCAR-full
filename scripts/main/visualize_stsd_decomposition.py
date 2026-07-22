from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad.config import CoReMADConfig
from coremad.data import build_loader, load_raw_dataset_bundle
from coremad.model import CoReMADModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the STSD decomposition (X, S, R) for a selected window."
    )
    parser.add_argument("--experiment-dir", type=str, required=True, help="Experiment directory with config.json and stage_a.pt.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Which split to visualize.")
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        choices=["normal_steady", "normal_switch", "anomaly_disturbance"],
        help="Auto-select a representative case instead of a manual window.",
    )
    parser.add_argument("--window-index", type=int, default=None, help="Window index in the selected split.")
    parser.add_argument("--start", type=int, default=None, help="Absolute start timestep of the selected window.")
    parser.add_argument("--channels", type=int, nargs="*", default=None, help="Specific channel ids to plot.")
    parser.add_argument("--max-channels", type=int, default=6, help="How many channels to auto-select when --channels is not provided.")
    parser.add_argument("--device", type=str, default=None, help="Override device. Defaults to config.device if available.")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <experiment-dir>/stsd_visualization.")
    parser.add_argument(
        "--edge-trim",
        type=int,
        default=4,
        help="Trim this many timesteps from each window boundary when plotting to reduce edge artifacts.",
    )
    parser.add_argument(
        "--smooth-k",
        type=int,
        default=1,
        help="Moving-average kernel size for aggregate timeline curves. Use 0 or 1 to disable smoothing.",
    )
    parser.add_argument(
        "--peak-stat",
        type=str,
        default="max",
        choices=["max", "q95", "top3mean"],
        help="Peak residual statistic for timeline plots.",
    )
    return parser.parse_args()


def ensure_output_dir(args: argparse.Namespace, experiment_dir: Path) -> Path:
    out_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "stsd_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_stage_a_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Stage-A checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def infer_device(config: CoReMADConfig, override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if str(config.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_split(
    config: CoReMADConfig,
    raw_bundle,
    split: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str] | None, int]:
    if split == "train":
        return (
            raw_bundle.train,
            None,
            raw_bundle.train_segment_ranges,
            raw_bundle.train_segment_names,
            int(config.train_stride),
        )
    return (
        raw_bundle.test,
        np.asarray(raw_bundle.test_labels, dtype=np.int32),
        raw_bundle.test_segment_ranges,
        raw_bundle.test_segment_names,
        int(config.test_stride),
    )


def resolve_window_index(
    start_indices: np.ndarray,
    labels: np.ndarray | None,
    seq_len: int,
    requested_index: int | None,
    requested_start: int | None,
) -> int:
    if requested_start is not None:
        matches = np.where(start_indices == int(requested_start))[0]
        if matches.size == 0:
            raise ValueError(f"start={requested_start} is not a legal window start under the current split/stride.")
        return int(matches[0])
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(start_indices):
            raise IndexError(f"window_index={requested_index} is out of range [0, {len(start_indices) - 1}].")
        return int(requested_index)
    if labels is not None and np.any(labels > 0):
        anomaly_starts = np.where(
            np.asarray([labels[start : start + seq_len].max() > 0 for start in start_indices], dtype=bool)
        )[0]
        if anomaly_starts.size > 0:
            return int(anomaly_starts[0])
    return 0


def inverse_normalize_component(
    values_norm: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    add_mean: bool,
) -> np.ndarray:
    restored = values_norm * scale[None, :]
    if add_mean:
        restored = restored + mean[None, :]
    return restored.astype(np.float64, copy=False)


def resolve_edge_trim(seq_len: int, edge_trim: int) -> int:
    if edge_trim <= 0:
        return 0
    max_trim = max(0, (int(seq_len) - 3) // 2)
    return min(int(edge_trim), max_trim)


def build_time_view(seq_len: int, edge_trim: int) -> tuple[np.ndarray, slice]:
    trim = resolve_edge_trim(seq_len, edge_trim)
    start = trim
    end = int(seq_len) - trim
    return np.arange(start, end, dtype=np.int64), slice(start, end)


def filter_boundaries_for_view(
    boundary_positions: list[int],
    t_values: np.ndarray,
) -> list[int]:
    if t_values.size == 0:
        return []
    left = int(t_values[0])
    right = int(t_values[-1])
    return [int(boundary) for boundary in boundary_positions if left <= int(boundary) <= right]


def find_positive_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return []
    padded = np.pad(mask.astype(np.int32), (1, 1), mode="constant", constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts.tolist(), ends.tolist())]


def draw_label_boundaries(
    ax,
    t_values: np.ndarray,
    label_mask: np.ndarray | None,
    color: str = "#dc2626",
) -> None:
    if label_mask is None:
        return
    intervals = find_positive_intervals(label_mask)
    if not intervals:
        return
    if t_values.size == 0:
        return
    for start_idx, end_idx in intervals:
        start_idx = max(0, min(int(start_idx), len(t_values) - 1))
        end_idx = max(0, min(int(end_idx) - 1, len(t_values) - 1))
        ax.axvline(float(t_values[start_idx]), color=color, linewidth=1.2, linestyle="--", alpha=0.9)
        ax.axvline(float(t_values[end_idx]), color=color, linewidth=1.2, linestyle="--", alpha=0.9)


def smooth_curve(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = int(kernel_size)
    if kernel_size <= 1 or values.size <= 2:
        return values.astype(np.float64, copy=False)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(1, min(kernel_size, int(values.size)))
    if kernel_size <= 1:
        return values.astype(np.float64, copy=False)
    pad = kernel_size // 2
    padded = np.pad(values.astype(np.float64, copy=False), (pad, pad), mode="edge")
    kernel = np.full(kernel_size, 1.0 / float(kernel_size), dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def compute_residual_peak_curve(
    residual_raw: np.ndarray,
    peak_stat: str,
) -> tuple[np.ndarray, str]:
    abs_residual = np.abs(np.asarray(residual_raw, dtype=np.float64))
    if peak_stat == "max":
        return np.max(abs_residual, axis=1), "max(|R|)"
    if peak_stat == "q95":
        return np.quantile(abs_residual, 0.95, axis=1), "q95(|R|)"
    top_k = min(3, abs_residual.shape[1])
    sorted_abs = np.sort(abs_residual, axis=1)
    return np.mean(sorted_abs[:, -top_k:], axis=1), f"top-{top_k} mean(|R|)"


def choose_slow_channel(
    raw_x: np.ndarray,
    slow_raw: np.ndarray,
) -> int:
    raw_x = np.asarray(raw_x, dtype=np.float64)
    slow_raw = np.asarray(slow_raw, dtype=np.float64)
    slow_variation = np.std(slow_raw, axis=0)
    smoothing_gap = np.mean(np.abs(raw_x - slow_raw), axis=0)
    score = slow_variation * (smoothing_gap + 1e-6)
    return int(np.argmax(score))


def choose_channels(
    residual_norm: np.ndarray,
    requested_channels: list[int] | None,
    max_channels: int,
) -> list[int]:
    n_channels = residual_norm.shape[1]
    if requested_channels:
        channels = [int(ch) for ch in requested_channels]
        for channel in channels:
            if channel < 0 or channel >= n_channels:
                raise IndexError(f"Channel {channel} is out of range [0, {n_channels - 1}].")
        return channels
    residual_energy = np.mean(np.abs(residual_norm), axis=0)
    order = np.argsort(residual_energy)[::-1]
    top_k = max(1, min(int(max_channels), n_channels))
    return [int(idx) for idx in order[:top_k]]


def build_fallback_start_indices(
    total_steps: int,
    seq_len: int,
    stride: int,
) -> np.ndarray:
    last_start = int(total_steps) - int(seq_len)
    if last_start < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, last_start + 1, int(stride), dtype=np.int64)


def compute_window_anomaly_counts(
    start_indices: np.ndarray,
    labels: np.ndarray | None,
    seq_len: int,
) -> np.ndarray:
    if labels is None:
        return np.zeros(len(start_indices), dtype=np.int64)
    return np.asarray(
        [int(np.asarray(labels[start : start + seq_len], dtype=np.int32).sum()) for start in start_indices],
        dtype=np.int64,
    )


def find_closest_start_index(start_indices: np.ndarray, target_start: int, mask: np.ndarray) -> int | None:
    candidate_ids = np.where(mask)[0]
    if candidate_ids.size == 0:
        return None
    candidate_starts = start_indices[candidate_ids]
    best_local = int(np.argmin(np.abs(candidate_starts - int(target_start))))
    return int(candidate_ids[best_local])


def find_segment_boundaries_in_window(
    start: int,
    end: int,
    segment_ranges: np.ndarray | None,
) -> list[int]:
    if segment_ranges is None:
        return []
    ranges = np.asarray(segment_ranges, dtype=np.int64)
    boundaries: list[int] = []
    for idx in range(len(ranges) - 1):
        boundary = int(ranges[idx][1])
        if start < boundary < end:
            boundaries.append(boundary - start)
    return boundaries


def interval_segment_names(
    start: int,
    end: int,
    segment_ranges: np.ndarray | None,
    segment_names: list[str] | None,
) -> list[str]:
    if segment_ranges is None or not segment_names:
        return []
    names: list[str] = []
    ranges = np.asarray(segment_ranges, dtype=np.int64)
    for idx, (seg_start, seg_end) in enumerate(ranges.tolist()):
        if start < int(seg_end) and end > int(seg_start):
            names.append(str(segment_names[idx]))
    return names


def auto_case_windows(
    start_indices: np.ndarray,
    labels: np.ndarray | None,
    seq_len: int,
    segment_ranges: np.ndarray | None,
    segment_names: list[str] | None,
) -> dict[str, int]:
    anomaly_counts = compute_window_anomaly_counts(start_indices, labels, seq_len)
    is_normal = anomaly_counts == 0
    cases: dict[str, int] = {}

    if np.any(is_normal):
        if segment_ranges is not None:
            ranges = np.asarray(segment_ranges, dtype=np.int64)
            segment_lengths = ranges[:, 1] - ranges[:, 0]
            order = np.argsort(segment_lengths)[::-1]
            for seg_idx in order.tolist():
                seg_start, seg_end = [int(v) for v in ranges[seg_idx]]
                seg_mask = (
                    (start_indices >= seg_start)
                    & ((start_indices + int(seq_len)) <= seg_end)
                    & is_normal
                )
                if not np.any(seg_mask):
                    continue
                center_start = seg_start + max(0, (seg_end - seg_start - int(seq_len)) // 2)
                chosen = find_closest_start_index(start_indices, center_start, seg_mask)
                if chosen is not None:
                    cases["normal_steady"] = chosen
                    break
        if "normal_steady" not in cases:
            cases["normal_steady"] = int(np.where(is_normal)[0][len(np.where(is_normal)[0]) // 2])

    if segment_ranges is not None and len(np.asarray(segment_ranges)) >= 2:
        ranges = np.asarray(segment_ranges, dtype=np.int64)
        for idx in range(len(ranges) - 1):
            left_end = int(ranges[idx][1])
            right_start = int(ranges[idx + 1][0])
            if left_end != right_start:
                continue
            switch_mask = (
                (start_indices < left_end)
                & ((start_indices + int(seq_len)) > left_end)
                & is_normal
            )
            chosen = find_closest_start_index(start_indices, left_end - int(seq_len) // 2, switch_mask)
            if chosen is not None:
                cases["normal_switch"] = chosen
                break

    if np.any(anomaly_counts > 0):
        anomaly_ids = np.where(anomaly_counts == anomaly_counts.max())[0]
        cases["anomaly_disturbance"] = int(anomaly_ids[0])

    return cases


def plot_channel_panels(
    raw_x: np.ndarray,
    slow_raw: np.ndarray,
    residual_raw: np.ndarray,
    labels: np.ndarray | None,
    channels: list[int],
    boundary_positions: list[int],
    edge_trim: int,
    title: str,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDViz] skip channel plots: matplotlib is not available.")
        return
    seq_len = raw_x.shape[0]
    t_plot, time_slice = build_time_view(seq_len, edge_trim)
    boundaries = filter_boundaries_for_view(boundary_positions, t_plot)
    n_rows = len(channels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.2 * n_rows), squeeze=False, sharex=True)

    label_mask = None
    if labels is not None:
        label_mask = np.asarray(labels, dtype=np.int32).reshape(-1) > 0
    label_mask_plot = None if label_mask is None else label_mask[time_slice]

    for row_idx, channel in enumerate(channels):
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]
        raw_series = raw_x[time_slice, channel]
        slow_series = slow_raw[time_slice, channel]
        residual_series = residual_raw[time_slice, channel]
        if label_mask_plot is not None and np.any(label_mask_plot):
            ax_left.fill_between(
                t_plot,
                float(raw_series.min()),
                float(raw_series.max()),
                where=label_mask_plot,
                color="#fee2e2",
                alpha=0.45,
            )
            ax_right.fill_between(
                t_plot,
                float(residual_series.min()),
                float(residual_series.max()),
                where=label_mask_plot,
                color="#fee2e2",
                alpha=0.45,
            )
        for boundary in boundaries:
            ax_left.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.75)
            ax_right.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.75)
        ax_left.plot(t_plot, raw_series, color="#334155", linewidth=1.3, label="X")
        ax_left.plot(t_plot, slow_series, color="#2563eb", linewidth=1.5, label="S")
        ax_left.set_title(f"Channel {channel}: X vs S")
        ax_left.set_ylabel("Value")
        ax_left.legend(loc="upper right", fontsize=8)

        ax_right.plot(t_plot, residual_series, color="#dc2626", linewidth=1.3, label="R")
        ax_right.axhline(0.0, color="#64748b", linewidth=0.8, linestyle="--")
        ax_right.set_title(f"Channel {channel}: R")
        ax_right.legend(loc="upper right", fontsize=8)
        ax_left.set_xlim(float(t_plot[0]), float(t_plot[-1]))
        ax_right.set_xlim(float(t_plot[0]), float(t_plot[-1]))

    axes[-1, 0].set_xlabel("Timestep in Window")
    axes[-1, 1].set_xlabel("Timestep in Window")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmaps(
    x_norm: np.ndarray,
    slow_norm: np.ndarray,
    residual_norm: np.ndarray,
    boundary_positions: list[int],
    edge_trim: int,
    figure_title: str,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDViz] skip heatmaps: matplotlib is not available.")
        return
    t_plot, time_slice = build_time_view(x_norm.shape[0], edge_trim)
    x_norm_plot = x_norm[time_slice]
    slow_norm_plot = slow_norm[time_slice]
    residual_norm_plot = residual_norm[time_slice]
    boundaries = filter_boundaries_for_view(boundary_positions, t_plot)
    vmax = float(
        np.percentile(
            np.abs(np.concatenate([x_norm_plot, slow_norm_plot, residual_norm_plot], axis=1)),
            99.0,
        )
    )
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), squeeze=False)
    panels = [
        ("X (normalized)", x_norm_plot),
        ("S / slow (normalized)", slow_norm_plot),
        ("R / residual (normalized)", residual_norm_plot),
    ]
    for ax, (panel_title, values) in zip(axes[0], panels):
        im = ax.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            extent=(float(t_plot[0]) - 0.5, float(t_plot[-1]) + 0.5, -0.5, values.shape[1] - 0.5),
        )
        for boundary in boundaries:
            ax.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_title(panel_title)
        ax.set_xlabel("Timestep in Window")
        ax.set_ylabel("Channel")
        ax.set_xlim(float(t_plot[0]) - 0.5, float(t_plot[-1]) + 0.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overview_timeline(
    raw_x: np.ndarray,
    slow_raw: np.ndarray,
    residual_raw: np.ndarray,
    labels: np.ndarray | None,
    boundary_positions: list[int],
    edge_trim: int,
    smooth_k: int,
    peak_stat: str,
    figure_title: str,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDViz] skip overview timeline: matplotlib is not available.")
        return
    t_plot, time_slice = build_time_view(raw_x.shape[0], edge_trim)
    boundaries = filter_boundaries_for_view(boundary_positions, t_plot)
    x_mean_abs = smooth_curve(np.mean(np.abs(raw_x), axis=1), smooth_k)
    s_mean_abs = smooth_curve(np.mean(np.abs(slow_raw), axis=1), smooth_k)
    r_mean_abs = smooth_curve(np.mean(np.abs(residual_raw), axis=1), smooth_k)
    r_peak_abs, peak_label = compute_residual_peak_curve(residual_raw, peak_stat)
    r_peak_abs = smooth_curve(r_peak_abs, smooth_k)
    x_mean_abs = x_mean_abs[time_slice]
    s_mean_abs = s_mean_abs[time_slice]
    r_mean_abs = r_mean_abs[time_slice]
    r_peak_abs = r_peak_abs[time_slice]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True, squeeze=False)
    ax_top = axes[0, 0]
    ax_bottom = axes[1, 0]

    label_mask = None if labels is None else (np.asarray(labels, dtype=np.int32).reshape(-1) > 0)
    label_mask_plot = None if label_mask is None else label_mask[time_slice]
    if label_mask_plot is not None and np.any(label_mask_plot):
        ax_top.fill_between(
            t_plot,
            0.0,
            float(max(x_mean_abs.max(), s_mean_abs.max(), r_mean_abs.max())) * 1.05,
            where=label_mask_plot,
            color="#fee2e2",
            alpha=0.42,
        )
        ax_bottom.fill_between(
            t_plot,
            0.0,
            float(r_peak_abs.max()) * 1.05,
            where=label_mask_plot,
            color="#fee2e2",
            alpha=0.42,
        )

    for boundary in boundaries:
        ax_top.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)
        ax_bottom.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)

    ax_top.plot(t_plot, x_mean_abs, color="#334155", linewidth=1.35, label="mean(|X|)")
    ax_top.plot(t_plot, s_mean_abs, color="#2563eb", linewidth=1.35, label="mean(|S|)")
    ax_top.plot(t_plot, r_mean_abs, color="#dc2626", linewidth=1.45, label="mean(|R|)")
    ax_top.set_ylabel("Mean Absolute Value")
    ax_top.set_title("Overall Energy Trend")
    ax_top.legend(loc="upper right", fontsize=8)

    ax_bottom.plot(t_plot, r_peak_abs, color="#b91c1c", linewidth=1.45, label=peak_label)
    ax_bottom.plot(t_plot, r_mean_abs, color="#f97316", linewidth=1.15, linestyle="--", label="mean(|R|)")
    ax_bottom.set_ylabel("Residual Magnitude")
    ax_bottom.set_xlabel("Timestep in Window")
    ax_bottom.set_title("Residual Strength Over Time")
    ax_bottom.legend(loc="upper right", fontsize=8)
    ax_bottom.set_xlim(float(t_plot[0]), float(t_plot[-1]))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_case_comparison_panel(
    case_payloads: list[dict[str, object]],
    edge_trim: int,
    smooth_k: int,
    peak_stat: str,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDViz] skip case comparison panel: matplotlib is not available.")
        return
    if not case_payloads:
        return

    n_rows = len(case_payloads)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14.5, 3.6 * n_rows), squeeze=False)
    for row_idx, payload in enumerate(case_payloads):
        case_title = str(payload["case_title"])
        residual_norm = np.asarray(payload["residual_norm"], dtype=np.float64)
        residual_raw = np.asarray(payload["residual_raw"], dtype=np.float64)
        labels = payload.get("labels")
        label_mask = None if labels is None else (np.asarray(labels, dtype=np.int32).reshape(-1) > 0)
        t_plot, time_slice = build_time_view(residual_raw.shape[0], edge_trim)
        boundaries = filter_boundaries_for_view([int(v) for v in payload.get("boundary_positions", [])], t_plot)
        residual_norm_plot = residual_norm[time_slice]
        residual_raw_plot = residual_raw[time_slice]
        label_mask_plot = None if label_mask is None else label_mask[time_slice]

        ax_heat = axes[row_idx, 0]
        vmax = float(np.percentile(np.abs(residual_norm_plot), 99.0))
        vmax = max(vmax, 1e-6)
        im = ax_heat.imshow(
            residual_norm_plot.T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            extent=(float(t_plot[0]) - 0.5, float(t_plot[-1]) + 0.5, -0.5, residual_norm_plot.shape[1] - 0.5),
        )
        for boundary in boundaries:
            ax_heat.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)
        ax_heat.set_title(f"{case_title}: R heatmap")
        ax_heat.set_ylabel("Channel")
        ax_heat.set_xlabel("Timestep")
        ax_heat.set_xlim(float(t_plot[0]) - 0.5, float(t_plot[-1]) + 0.5)
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        ax_curve = axes[row_idx, 1]
        r_mean_abs = smooth_curve(np.mean(np.abs(residual_raw), axis=1), smooth_k)[time_slice]
        r_peak_abs, peak_label = compute_residual_peak_curve(residual_raw, peak_stat)
        r_peak_abs = smooth_curve(r_peak_abs, smooth_k)[time_slice]
        if label_mask_plot is not None and np.any(label_mask_plot):
            ax_curve.fill_between(
                t_plot,
                0.0,
                float(r_peak_abs.max()) * 1.05,
                where=label_mask_plot,
                color="#fee2e2",
                alpha=0.42,
            )
        for boundary in boundaries:
            ax_curve.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)
        ax_curve.plot(t_plot, r_mean_abs, color="#f97316", linewidth=1.25, label="mean(|R|)")
        ax_curve.plot(t_plot, r_peak_abs, color="#b91c1c", linewidth=1.45, label=peak_label)
        ax_curve.set_title(f"{case_title}: residual timeline")
        ax_curve.set_ylabel("Residual Magnitude")
        ax_curve.set_xlabel("Timestep")
        ax_curve.legend(loc="upper right", fontsize=8)
        ax_curve.set_xlim(float(t_plot[0]), float(t_plot[-1]))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_noncollapse_evidence_panel(
    case_payloads: list[dict[str, object]],
    edge_trim: int,
    smooth_k: int,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDViz] skip non-collapse evidence panel: matplotlib is not available.")
        return
    if not case_payloads:
        return

    by_name = {str(payload["case_name"]): payload for payload in case_payloads}
    selected_payloads: list[dict[str, object]] = []
    for name in ("normal_steady", "anomaly_disturbance"):
        payload = by_name.get(name)
        if payload is not None:
            selected_payloads.append(payload)
    if not selected_payloads:
        selected_payloads = case_payloads[: min(2, len(case_payloads))]

    fig, axes = plt.subplots(len(selected_payloads), 2, figsize=(13.8, 4.1 * len(selected_payloads)), squeeze=False)
    for row_idx, payload in enumerate(selected_payloads):
        case_title = str(payload["case_title"])
        raw_x = np.asarray(payload["raw_x"], dtype=np.float64)
        slow_raw = np.asarray(payload["slow_raw"], dtype=np.float64)
        residual_raw = np.asarray(payload["residual_raw"], dtype=np.float64)
        residual_norm = np.asarray(payload["residual_norm"], dtype=np.float64)
        labels = payload.get("labels")
        label_mask = None if labels is None else (np.asarray(labels, dtype=np.int32).reshape(-1) > 0)

        t_plot, time_slice = build_time_view(raw_x.shape[0], edge_trim)
        boundaries = filter_boundaries_for_view([int(v) for v in payload.get("boundary_positions", [])], t_plot)
        label_mask_plot = None if label_mask is None else label_mask[time_slice]

        x_mean_abs = smooth_curve(np.mean(np.abs(raw_x), axis=1), smooth_k)[time_slice]
        s_mean_abs = smooth_curve(np.mean(np.abs(slow_raw), axis=1), smooth_k)[time_slice]
        r_mean_abs = smooth_curve(np.mean(np.abs(residual_raw), axis=1), smooth_k)[time_slice]
        residual_norm_plot = residual_norm[time_slice]

        ax_curve = axes[row_idx, 0]
        if label_mask_plot is not None and np.any(label_mask_plot):
            ax_curve.fill_between(
                t_plot,
                0.0,
                float(max(x_mean_abs.max(), s_mean_abs.max(), r_mean_abs.max())) * 1.05,
                where=label_mask_plot,
                color="#fee2e2",
                alpha=0.42,
            )
        for boundary in boundaries:
            ax_curve.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)
        ax_curve.plot(t_plot, x_mean_abs, color="#334155", linewidth=1.35, label="mean(|X|)")
        ax_curve.plot(t_plot, s_mean_abs, color="#2563eb", linewidth=1.35, label="mean(|S|)")
        ax_curve.plot(t_plot, r_mean_abs, color="#dc2626", linewidth=1.35, label="mean(|R|)")
        ax_curve.set_title(f"{case_title}: Non-Collapse Magnitudes")
        ax_curve.set_ylabel("Mean Absolute Value")
        ax_curve.set_xlabel("Timestep")
        ax_curve.legend(loc="upper right", fontsize=8)
        ax_curve.set_xlim(float(t_plot[0]), float(t_plot[-1]))

        ax_heat = axes[row_idx, 1]
        vmax = float(np.percentile(np.abs(residual_norm_plot), 99.0))
        vmax = max(vmax, 1e-6)
        im = ax_heat.imshow(
            residual_norm_plot.T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            extent=(float(t_plot[0]) - 0.5, float(t_plot[-1]) + 0.5, -0.5, residual_norm_plot.shape[1] - 0.5),
        )
        for boundary in boundaries:
            ax_heat.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)
        ax_heat.set_title(f"{case_title}: Structured Residual R")
        ax_heat.set_ylabel("Channel")
        ax_heat.set_xlabel("Timestep")
        ax_heat.set_xlim(float(t_plot[0]) - 0.5, float(t_plot[-1]) + 0.5)
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_slow_component_evidence_panel(
    case_payloads: list[dict[str, object]],
    edge_trim: int,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDViz] skip slow-component evidence panel: matplotlib is not available.")
        return
    if not case_payloads:
        return

    by_name = {str(payload["case_name"]): payload for payload in case_payloads}
    selected_payloads: list[dict[str, object]] = []
    for name in ("normal_steady", "normal_switch", "anomaly_disturbance"):
        payload = by_name.get(name)
        if payload is not None and payload not in selected_payloads:
            selected_payloads.append(payload)
    if not selected_payloads:
        selected_payloads = case_payloads[:1]
    selected_payloads = selected_payloads[: min(3, len(selected_payloads))]

    n_cols = len(selected_payloads)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.4 * n_cols, 6.8), squeeze=False, sharex="col")

    for col_idx, payload in enumerate(selected_payloads):
        case_title = str(payload["case_title"])
        raw_x = np.asarray(payload["raw_x"], dtype=np.float64)
        slow_raw = np.asarray(payload["slow_raw"], dtype=np.float64)
        residual_raw = np.asarray(payload["residual_raw"], dtype=np.float64)
        labels = payload.get("labels")
        label_mask = None if labels is None else (np.asarray(labels, dtype=np.int32).reshape(-1) > 0)
        t_plot, time_slice = build_time_view(raw_x.shape[0], edge_trim)
        boundaries = filter_boundaries_for_view([int(v) for v in payload.get("boundary_positions", [])], t_plot)
        label_mask_plot = None if label_mask is None else label_mask[time_slice]
        scale = float(np.sqrt(max(1, raw_x.shape[1])))
        x_env = (np.linalg.norm(raw_x, axis=1) / scale)[time_slice]
        s_env = (np.linalg.norm(slow_raw, axis=1) / scale)[time_slice]
        r_env = (np.linalg.norm(residual_raw, axis=1) / scale)[time_slice]
        dx = np.linalg.norm(np.diff(raw_x, axis=0, prepend=raw_x[:1]), axis=1) / scale
        ds = np.linalg.norm(np.diff(slow_raw, axis=0, prepend=slow_raw[:1]), axis=1) / scale
        dx = dx[time_slice]
        ds = ds[time_slice]

        ax_top = axes[0, col_idx]
        if label_mask_plot is not None and np.any(label_mask_plot):
            y_low = float(min(x_env.min(), s_env.min()))
            y_high = float(max(x_env.max(), s_env.max()))
            ax_top.fill_between(
                t_plot,
                y_low,
                y_high,
                where=label_mask_plot,
                color="#fee2e2",
                alpha=0.42,
            )
        for boundary in boundaries:
            ax_top.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)
        ax_top.plot(t_plot, x_env, color="#334155", linewidth=1.3, label=r"$||X_t||_2 / \sqrt{C}$")
        ax_top.plot(t_plot, s_env, color="#2563eb", linewidth=1.6, label=r"$||S_t||_2 / \sqrt{C}$")
        ax_top.set_title(f"{case_title}: Global Operating Envelope")
        ax_top.set_ylabel("Magnitude")
        ax_top.legend(loc="upper right", fontsize=8)
        ax_top.set_xlim(float(t_plot[0]), float(t_plot[-1]))

        ax_bottom = axes[1, col_idx]
        if label_mask_plot is not None and np.any(label_mask_plot):
            ax_bottom.fill_between(
                t_plot,
                0.0,
                float(max(dx.max(), ds.max(), r_env.max())) * 1.05,
                where=label_mask_plot,
                color="#fee2e2",
                alpha=0.42,
            )
        for boundary in boundaries:
            ax_bottom.axvline(boundary, color="#7c3aed", linewidth=1.0, linestyle="--", alpha=0.8)
        ax_bottom.plot(t_plot, dx, color="#475569", linewidth=1.2, label=r"$||\Delta X_t||_2 / \sqrt{C}$")
        ax_bottom.plot(t_plot, ds, color="#2563eb", linewidth=1.4, label=r"$||\Delta S_t||_2 / \sqrt{C}$")
        ax_bottom.plot(t_plot, r_env, color="#dc2626", linewidth=1.3, label=r"$||R_t||_2 / \sqrt{C}$")
        ax_bottom.set_title(f"{case_title}: Local Fluctuation vs Residual")
        ax_bottom.set_ylabel("Magnitude")
        ax_bottom.set_xlabel("Timestep")
        ax_bottom.legend(loc="upper right", fontsize=8)
        ax_bottom.set_xlim(float(t_plot[0]), float(t_plot[-1]))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_case(
    *,
    case_name: str,
    case_title: str,
    output_prefix: str,
    raw_data: np.ndarray,
    point_labels: np.ndarray | None,
    start: int,
    end: int,
    window_index: int,
    x_norm: np.ndarray,
    slow_norm: np.ndarray,
    residual_norm: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    channels: list[int],
    segment_ranges: np.ndarray | None,
    segment_names: list[str] | None,
    stride: int,
    edge_trim: int,
    smooth_k: int,
    peak_stat: str,
    output_dir: Path,
) -> dict[str, object]:
    labels_window = None if point_labels is None else np.asarray(point_labels[start:end], dtype=np.int32)
    raw_x = np.asarray(raw_data[start:end], dtype=np.float64)
    slow_raw = inverse_normalize_component(slow_norm, mean, scale, add_mean=True)
    residual_raw = inverse_normalize_component(residual_norm, mean, scale, add_mean=False)
    boundaries = find_segment_boundaries_in_window(start, end, segment_ranges)
    overlapped_segments = interval_segment_names(start, end, segment_ranges, segment_names)

    plot_channel_panels(
        raw_x=raw_x,
        slow_raw=slow_raw,
        residual_raw=residual_raw,
        labels=labels_window,
        channels=channels,
        boundary_positions=boundaries,
        edge_trim=edge_trim,
        title=case_title,
        output_path=output_dir / f"{output_prefix}_channels.png",
    )
    plot_heatmaps(
        x_norm=x_norm,
        slow_norm=slow_norm,
        residual_norm=residual_norm,
        boundary_positions=boundaries,
        edge_trim=edge_trim,
        figure_title=case_title,
        output_path=output_dir / f"{output_prefix}_heatmaps.png",
    )
    plot_overview_timeline(
        raw_x=raw_x,
        slow_raw=slow_raw,
        residual_raw=residual_raw,
        labels=labels_window,
        boundary_positions=boundaries,
        edge_trim=edge_trim,
        smooth_k=smooth_k,
        peak_stat=peak_stat,
        figure_title=case_title,
        output_path=output_dir / f"{output_prefix}_overview.png",
    )

    np.savez(
        output_dir / f"{output_prefix}_decomposition.npz",
        raw_x=raw_x.astype(np.float32),
        x_norm=x_norm.astype(np.float32),
        slow_norm=slow_norm.astype(np.float32),
        residual_norm=residual_norm.astype(np.float32),
        slow_raw=slow_raw.astype(np.float32),
        residual_raw=residual_raw.astype(np.float32),
        labels=(labels_window.astype(np.int32) if labels_window is not None else np.zeros(end - start, dtype=np.int32)),
        channels=np.asarray(channels, dtype=np.int32),
        start=np.asarray(start, dtype=np.int64),
        end=np.asarray(end, dtype=np.int64),
        window_index=np.asarray(window_index, dtype=np.int64),
    )

    summary = {
        "case_name": str(case_name),
        "case_title": str(case_title),
        "start": int(start),
        "end": int(end),
        "window_index": int(window_index),
        "stride": int(stride),
        "selected_channels": [int(ch) for ch in channels],
        "segment_boundaries_in_window": [int(boundary) for boundary in boundaries],
        "overlapped_segments": overlapped_segments,
        "anomaly_points_in_window": int(labels_window.sum()) if labels_window is not None else 0,
        "plot_edge_trim": int(resolve_edge_trim(end - start, edge_trim)),
        "plot_smooth_k": int(max(1, smooth_k)),
        "plot_peak_stat": str(peak_stat),
        "slow_mean_abs": float(np.abs(slow_norm).mean()),
        "residual_mean_abs": float(np.abs(residual_norm).mean()),
        "residual_channel_mean_abs": {
            str(int(ch)): float(np.abs(residual_norm[:, ch]).mean())
            for ch in channels
        },
    }
    (output_dir / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[STSDViz] case={case_name} window_index={window_index} "
        f"start={start} end={end} anomaly_points={summary['anomaly_points_in_window']} "
        f"segments={overlapped_segments}"
    )
    return {
        "case_name": str(case_name),
        "case_title": str(case_title),
        "start": int(start),
        "end": int(end),
        "window_index": int(window_index),
        "labels": labels_window,
        "boundary_positions": boundaries,
        "raw_x": raw_x,
        "slow_raw": slow_raw,
        "residual_norm": residual_norm,
        "residual_raw": residual_raw,
    }


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = CoReMADConfig.load(config_path)
    payload = load_stage_a_payload(experiment_dir / "stage_a.pt")
    raw_bundle = load_raw_dataset_bundle(config.dataset, config.data_root, config=config)
    raw_data, point_labels, segment_ranges, segment_names, stride = select_split(config, raw_bundle, args.split)
    normalizer_state = payload.get("normalizer")
    if not isinstance(normalizer_state, dict):
        raise RuntimeError("Stage-A checkpoint is missing the serialized normalizer.")

    mean = np.asarray(normalizer_state["mean"], dtype=np.float64)
    scale = np.asarray(normalizer_state["scale"], dtype=np.float64)
    normalized_data = ((np.asarray(raw_data, dtype=np.float64) - mean[None, :]) / scale[None, :]).astype(np.float32)

    device = infer_device(config, args.device)
    model = CoReMADModel(config).to(device)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load model_state cleanly. missing={missing} unexpected={unexpected}")
    model.eval()

    loader = build_loader(
        data=normalized_data,
        labels=point_labels,
        seq_len=config.seq_len,
        stride=stride,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        max_windows=0,
        drop_last=False,
        segment_ranges=segment_ranges,
    )
    dataset = loader.dataset
    if hasattr(dataset, "start_indices") and dataset.start_indices is not None:
        start_indices = np.asarray(dataset.start_indices, dtype=np.int64)
    else:
        start_indices = build_fallback_start_indices(
            total_steps=len(normalized_data),
            seq_len=int(config.seq_len),
            stride=int(stride),
        )
    if start_indices.size == 0:
        raise RuntimeError("No valid windows available for the selected split.")

    output_dir = ensure_output_dir(args, experiment_dir)
    seq_len = int(config.seq_len)
    if args.window_index is not None or args.start is not None:
        selected_cases = {
            "custom": resolve_window_index(
                start_indices=start_indices,
                labels=point_labels,
                seq_len=seq_len,
                requested_index=args.window_index,
                requested_start=args.start,
            )
        }
    else:
        selected_cases = auto_case_windows(
            start_indices=start_indices,
            labels=point_labels,
            seq_len=seq_len,
            segment_ranges=segment_ranges,
            segment_names=segment_names,
        )
        if args.case is not None:
            if args.case not in selected_cases:
                raise RuntimeError(
                    f"Requested case '{args.case}' is unavailable for split={args.split}. "
                    f"Available cases: {sorted(selected_cases.keys())}"
                )
            selected_cases = {args.case: selected_cases[args.case]}
        if not selected_cases:
            selected_cases = {
                "default": resolve_window_index(
                    start_indices=start_indices,
                    labels=point_labels,
                    seq_len=seq_len,
                    requested_index=None,
                    requested_start=None,
                )
            }

    case_labels = {
        "custom": "Custom Window",
        "default": "Representative Window",
        "normal_steady": "Normal Steady-State Case",
        "normal_switch": "Normal Operating-Mode Switch Case",
        "anomaly_disturbance": "Anomalous Disturbance Case",
    }
    saved_cases: dict[str, dict[str, object]] = {}
    case_payloads_for_panel: list[dict[str, object]] = []
    for case_name, window_index in selected_cases.items():
        item = dataset[int(window_index)]
        start = int(item["start"])
        end = start + seq_len
        x_tensor = item["x"].unsqueeze(0).to(device)
        with torch.no_grad():
            encoded = model.encode(x_tensor)
        x_norm = x_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
        slow_norm = encoded["slow"].squeeze(0).detach().cpu().numpy().astype(np.float64)
        residual_norm = encoded["residual"].squeeze(0).detach().cpu().numpy().astype(np.float64)
        channels = choose_channels(
            residual_norm=residual_norm,
            requested_channels=args.channels,
            max_channels=args.max_channels,
        )
        output_prefix = f"{args.split}_{case_name}_window_{int(window_index):06d}"
        panel_payload = render_case(
            case_name=case_name,
            case_title=case_labels.get(case_name, case_name),
            output_prefix=output_prefix,
            raw_data=raw_data,
            point_labels=point_labels,
            start=start,
            end=end,
            window_index=int(window_index),
            x_norm=x_norm,
            slow_norm=slow_norm,
            residual_norm=residual_norm,
            mean=mean,
            scale=scale,
            channels=channels,
            segment_ranges=segment_ranges,
            segment_names=segment_names,
            stride=int(stride),
            edge_trim=int(args.edge_trim),
            smooth_k=int(args.smooth_k),
            peak_stat=str(args.peak_stat),
            output_dir=output_dir,
        )
        case_payloads_for_panel.append(panel_payload)
        saved_cases[case_name] = {
            "window_index": int(window_index),
            "start": int(start),
            "end": int(end),
            "selected_channels": [int(ch) for ch in channels],
        }

    plot_case_comparison_panel(
        case_payloads=case_payloads_for_panel,
        edge_trim=int(args.edge_trim),
        smooth_k=int(args.smooth_k),
        peak_stat=str(args.peak_stat),
        output_path=output_dir / f"{args.split}_case_comparison_panel.png",
    )
    plot_noncollapse_evidence_panel(
        case_payloads=case_payloads_for_panel,
        edge_trim=int(args.edge_trim),
        smooth_k=int(args.smooth_k),
        output_path=output_dir / f"{args.split}_noncollapse_evidence_panel.png",
    )
    plot_slow_component_evidence_panel(
        case_payloads=case_payloads_for_panel,
        edge_trim=int(args.edge_trim),
        output_path=output_dir / f"{args.split}_slow_component_evidence_panel.png",
    )

    gallery_summary = {
        "experiment_dir": str(experiment_dir),
        "split": str(args.split),
        "stride": int(stride),
        "plot_edge_trim": int(resolve_edge_trim(seq_len, int(args.edge_trim))),
        "plot_smooth_k": int(max(1, int(args.smooth_k))),
        "plot_peak_stat": str(args.peak_stat),
        "available_segment_names": list(segment_names or []),
        "saved_cases": saved_cases,
    }
    (output_dir / f"{args.split}_case_gallery_summary.json").write_text(
        json.dumps(gallery_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[STSDViz] experiment_dir={experiment_dir}")
    print(f"[STSDViz] split={args.split} saved_cases={list(saved_cases.keys())}")
    print(f"[STSDViz] saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
