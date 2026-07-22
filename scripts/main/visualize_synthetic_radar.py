from __future__ import annotations

import json
import sys
import gc
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coremad.config import CoReMADConfig
from coremad.data import load_raw_dataset_bundle
from coremad.trainer import CoReMADTrainer


SUMMARY_PATH = ROOT / "main-result" / "synthetic_main_summary.json"
OUTPUT_DIR = ROOT / "main-result" / "figures"
OUTPUT_JSON = OUTPUT_DIR / "synthetic_radar_averaged_metrics.json"
OUTPUT_EXAMPLE_JSON = OUTPUT_DIR / "synthetic_example_windows.json"
OUTPUT_PNG = OUTPUT_DIR / "synthetic_radar_comparison.png"
OUTPUT_PDF = OUTPUT_DIR / "synthetic_radar_comparison.pdf"
OUTPUT_COMPOSITE_PNG = OUTPUT_DIR / "synthetic_radar_with_examples.png"
OUTPUT_COMPOSITE_PDF = OUTPUT_DIR / "synthetic_radar_with_examples.pdf"

METRICS = [
    ("roc_auc", "AUC-ROC"),
    ("vus_roc", "VUS-ROC"),
    ("r_auc_pr", "R-AUC-PR"),
    ("pr_auc", "AUC-PR"),
    ("vus_pr", "VUS-PR"),
    ("r_auc_roc", "R-AUC-ROC"),
]

DATASET_SPECS = [
    ("Contextual", "synthetic_con"),
    ("Global", "synthetic_glo"),
    ("Seasonal", "synthetic_sea"),
    ("Shapelet", "synthetic_sha"),
    ("Trend", "synthetic_tre"),
    ("Mixture", "synthetic_sub_mix"),
]
DATASET_MARKERS = {
    "Contextual": "o",
    "Global": "s",
    "Seasonal": "^",
    "Shapelet": "D",
    "Trend": "P",
    "Mixture": "X",
}
MECHANISM_PANEL_SPECS = [
    ("Contextual", ["synthetic_con0.0494", "synthetic_con0.072"]),
    ("Shapelet", ["synthetic_sha0.049", "synthetic_sha0.0742"]),
    ("Seasonal", ["synthetic_sea0.0482", "synthetic_sea0.0774"]),
]
MECHANISM_COLUMN_SPECS = [
    ("signal", "Query Signal", "#4A4A4A"),
    ("state_novelty", "Level-1 State Retrieval", "#6F97C7"),
    ("knn_distance", "Level-2 Patch Retrieval", "#D08A57"),
    ("cdf_max_score", "Fused Detection Score", "#B86A82"),
]

GRID_LEVELS = [0.33, 0.67, 1.0]
REFERENCE_NON_HIGHLIGHT_COLORS = [
    "#E18D5C",
    "#98DAB2",
    "#8F7BC1",
    "#8F8F8F",
]
HIGHLIGHT_COLOR = "#7F98D1"
INNER_RED_COLOR = "#C45D76"
DEFAULT_LINEWIDTH = 3.6
HIGHLIGHT_LINEWIDTH = 4.0
DEFAULT_FILL_ALPHA = 0.04
HIGHLIGHT_FILL_ALPHA = 0.05
DISPLAY_RADIUS_FLOOR = 0.08
MARKER_SIZE = 6.0
MARKER_EDGE_WIDTH = 0.7
LEGEND_FONT_SIZE = 13
LEGEND_MARKER_SIZE = 8.0
LEGEND_HANDLE_LENGTH = 3.0
GRID_COLOR = "#D3D3D3"
OUTER_GRID_COLOR = "#BEBEBE"
RADIAL_AXIS_COLOR = "#D8D8D8"
GRID_LINESTYLE = (0, (3, 3))
RADIAL_LINESTYLE = (0, (2, 3))
VERTICAL_LABEL_INSET = 0.02
OUTER_RING_RADIUS = 1.0
OUTER_RING_LABEL_RADIUS = 1.08
OUTER_RING_EDGE_COLOR = "#BDBDBD"
OUTER_RING_LINEWIDTH = 1.2
OUTER_RING_LINESTYLE = (0, (4, 3))
VALUE_LABEL_FONT_SIZE = 10
VALUE_LABEL_RADIAL_OFFSET = 0.04
VALUE_LABEL_OUTER_THRESHOLD = 0.92
VALUE_LABEL_OUTER_RADIAL_OFFSET = -0.05
VALUE_LABEL_TANGENTIAL_STEP = 0.025
EXAMPLE_SIGNAL_COLOR = "#4A4A4A"
EXAMPLE_SIGNAL_LINEWIDTH = 1.3
EXAMPLE_ANOMALY_COLOR = "#F1B6BD"
EXAMPLE_SCORE_FILL = "#B9CCF0"
EXAMPLE_SCORE_LINE = "#7F98D1"
EXAMPLE_PANEL_FACE = "#FCFCFC"
EXAMPLE_PANEL_EDGE = "#6F6F6F"
EXAMPLE_WINDOW_MIN = 180
EXAMPLE_WINDOW_MAX = 320
EXAMPLE_CONTEXT_MARGIN = 110
EXAMPLE_SCORE_SMOOTH_WINDOW = 15
EXAMPLE_PEAK_FILL_HALF_WIDTH = 10
EXAMPLE_YLABEL_FONT_SIZE = 15
EXAMPLE_YLABEL_PAD = 22
EXAMPLE_SCORE_YLABEL_PAD = 18
EXAMPLE_MERGE_GAP = 80
EXAMPLE_MIN_ALIGNMENT = 0.35
EXAMPLE_MIN_SCORE_PEAK = 0.25
RETRIEVAL_COVER_GAP = 24
MECHANISM_LINEWIDTH = 2.0
MECHANISM_MARKER_SIZE = 3.8
MECHANISM_ROW_LABEL_X = -0.18
MECHANISM_CURVE_SMOOTH_WINDOW = 11
RETRIEVAL_QUERY_EDGE = "#5B8FD1"
RETRIEVAL_QUERY_FILL = "#D9E7FB"
SCATTER_BACKGROUND = "#D2D2D2"
SCATTER_COARSE = "#6F97C7"
SCATTER_SUPPORT = "#D89A67"
SCATTER_QUERY = "#C45D76"
HEATMAP_CMAP = "YlOrRd"
SCATTER_MAX_POINTS = 1500
SCATTER_PCA_SAMPLE = 3500
RADAR_LEGEND_EDGE_COLOR = "#D0D0D0"
RADAR_LEGEND_LINEWIDTH = 0.9


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_metrics(summary: dict) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)

    for record in summary["records"]:
        experiment_base = record["experiment_base"]
        metric_block = record["metrics"]["cdf_max"]
        for dataset_name, prefix in DATASET_SPECS:
            if experiment_base.startswith(prefix):
                grouped[dataset_name].append(
                    {
                        "experiment_base": experiment_base,
                        "values": {
                            metric_key: float(metric_block[metric_key])
                            for metric_key, _ in METRICS
                        },
                    }
                )
                break

    aggregated: dict[str, dict] = {}
    for dataset_name, _ in DATASET_SPECS:
        runs = grouped[dataset_name]
        if len(runs) != 2:
            raise ValueError(
                f"Expected 2 runs for {dataset_name}, found {len(runs)}."
            )

        aggregated[dataset_name] = {
            "source_runs": [run["experiment_base"] for run in runs],
            "mean_metrics": {
                metric_key: mean(run["values"][metric_key] for run in runs)
                for metric_key, _ in METRICS
            },
        }

    return aggregated


def normalize_metrics(aggregated: dict) -> tuple[dict, dict]:
    metric_ranges: dict[str, dict[str, float]] = {}
    normalized: dict[str, dict] = {}

    for metric_key, _ in METRICS:
        values = [
            aggregated[dataset_name]["mean_metrics"][metric_key]
            for dataset_name, _ in DATASET_SPECS
        ]
        metric_min = min(values)
        metric_max = max(values)
        metric_ranges[metric_key] = {
            "min": metric_min,
            "max": metric_max,
        }

    for dataset_name, _ in DATASET_SPECS:
        raw_metrics = aggregated[dataset_name]["mean_metrics"]
        normalized_metrics: dict[str, float] = {}
        for metric_key, _ in METRICS:
            metric_min = metric_ranges[metric_key]["min"]
            metric_max = metric_ranges[metric_key]["max"]
            if np.isclose(metric_max, metric_min):
                normalized_value = 0.5
            else:
                normalized_value = (raw_metrics[metric_key] - metric_min) / (
                    metric_max - metric_min
                )
            normalized_metrics[metric_key] = float(normalized_value)

        normalized[dataset_name] = {
            "source_runs": aggregated[dataset_name]["source_runs"],
            "raw_mean_metrics": raw_metrics,
            "normalized_mean_metrics": normalized_metrics,
        }

    return normalized, metric_ranges


def apply_display_radius_floor(normalized: dict) -> dict:
    display_ready: dict[str, dict] = {}

    for dataset_name, _ in DATASET_SPECS:
        normalized_metrics = normalized[dataset_name]["normalized_mean_metrics"]
        display_radius_metrics = {
            metric_key: float(
                DISPLAY_RADIUS_FLOOR
                + (1.0 - DISPLAY_RADIUS_FLOOR) * normalized_metrics[metric_key]
            )
            for metric_key, _ in METRICS
        }
        display_ready[dataset_name] = {
            **normalized[dataset_name],
            "display_radius_metrics": display_radius_metrics,
        }

    return display_ready


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def determine_highlight_dataset(display_ready: dict, angles: np.ndarray) -> str:
    best_dataset = ""
    best_area = -1.0
    best_mean = -1.0

    for dataset_name, _ in DATASET_SPECS:
        values = np.array(
            [
                display_ready[dataset_name]["display_radius_metrics"][metric_key]
                for metric_key, _ in METRICS
            ]
        )
        area = polygon_area(polygon_points(angles, values))
        mean_value = float(np.mean(values))
        if area > best_area or (np.isclose(area, best_area) and mean_value > best_mean):
            best_dataset = dataset_name
            best_area = area
            best_mean = mean_value

    return best_dataset


def determine_innermost_dataset(display_ready: dict, angles: np.ndarray) -> str:
    inner_dataset = ""
    inner_area = float("inf")
    inner_mean = float("inf")

    for dataset_name, _ in DATASET_SPECS:
        values = np.array(
            [
                display_ready[dataset_name]["display_radius_metrics"][metric_key]
                for metric_key, _ in METRICS
            ]
        )
        area = polygon_area(polygon_points(angles, values))
        mean_value = float(np.mean(values))
        if area < inner_area or (np.isclose(area, inner_area) and mean_value < inner_mean):
            inner_dataset = dataset_name
            inner_area = area
            inner_mean = mean_value

    return inner_dataset


def build_style_map(highlight_dataset: str, inner_red_dataset: str) -> dict[str, dict[str, float | str]]:
    style_map: dict[str, dict[str, float | str]] = {}
    non_highlight_color_iter = iter(REFERENCE_NON_HIGHLIGHT_COLORS)

    for dataset_name, _ in DATASET_SPECS:
        if dataset_name == highlight_dataset:
            style_map[dataset_name] = {
                "color": HIGHLIGHT_COLOR,
                "linewidth": HIGHLIGHT_LINEWIDTH,
                "fill_alpha": HIGHLIGHT_FILL_ALPHA,
            }
        elif dataset_name == inner_red_dataset:
            style_map[dataset_name] = {
                "color": INNER_RED_COLOR,
                "linewidth": DEFAULT_LINEWIDTH,
                "fill_alpha": DEFAULT_FILL_ALPHA,
            }
        else:
            style_map[dataset_name] = {
                "color": next(non_highlight_color_iter),
                "linewidth": DEFAULT_LINEWIDTH,
                "fill_alpha": DEFAULT_FILL_ALPHA,
            }

    return style_map


def axis_alignment(x: float, y: float) -> tuple[str, str]:
    if x > 0.15:
        ha = "left"
    elif x < -0.15:
        ha = "right"
    else:
        ha = "center"

    if y > 0.15:
        va = "bottom"
    elif y < -0.15:
        va = "top"
    else:
        va = "center"

    return ha, va


def polygon_points(angles: np.ndarray, values: np.ndarray) -> np.ndarray:
    x = values * np.cos(angles)
    y = values * np.sin(angles)
    return np.column_stack([x, y])


def closed_polygon(points: np.ndarray) -> np.ndarray:
    return np.vstack([points, points[0]])


def vertex_label_position(vertex: np.ndarray, index: int) -> np.ndarray:
    position = OUTER_RING_LABEL_RADIUS * vertex
    if index == 0:
        position = position + np.array([0.0, -0.005])
    elif index == 3:
        position = position + np.array([0.0, 0.005])
    return position


def format_raw_value(value: float) -> str:
    return f"{value:.2f}"


def value_label_position(point: np.ndarray, angle: float, dataset_index: int, total_datasets: int) -> np.ndarray:
    radial = np.array([np.cos(angle), np.sin(angle)])
    tangential = np.array([-np.sin(angle), np.cos(angle)])
    centered_index = dataset_index - (total_datasets - 1) / 2.0
    radial_offset = (
        VALUE_LABEL_OUTER_RADIAL_OFFSET
        if np.linalg.norm(point) >= VALUE_LABEL_OUTER_THRESHOLD
        else VALUE_LABEL_RADIAL_OFFSET
    )
    return (
        point
        + radial_offset * radial
        + centered_index * VALUE_LABEL_TANGENTIAL_STEP * tangential
    )


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
        }
    )


def draw_radar(
    ax: plt.Axes,
    display_ready: dict,
    highlight_dataset: str,
    *,
    legend_anchor: tuple[float, float] = (0.5, -0.02),
    legend_font_size: float = LEGEND_FONT_SIZE,
    legend_box: bool = False,
) -> None:
    num_axes = len(METRICS)
    angles = np.pi / 2 - np.arange(num_axes) * 2 * np.pi / num_axes
    outer_vertices = polygon_points(angles, np.ones(num_axes))
    inner_red_dataset = determine_innermost_dataset(display_ready, angles)
    style_map = build_style_map(highlight_dataset, inner_red_dataset)

    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    ax.plot(
        OUTER_RING_RADIUS * np.cos(theta),
        OUTER_RING_RADIUS * np.sin(theta),
        color=OUTER_RING_EDGE_COLOR,
        linewidth=OUTER_RING_LINEWIDTH,
        linestyle=OUTER_RING_LINESTYLE,
        zorder=0,
    )

    for level in GRID_LEVELS:
        ring = closed_polygon(polygon_points(angles, np.full(num_axes, level)))
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color=GRID_COLOR if level < 1.0 else OUTER_GRID_COLOR,
            linewidth=0.8 if level < 1.0 else 1.0,
            linestyle=GRID_LINESTYLE if level < 1.0 else "-",
            zorder=1,
        )

    ring_vertices = polygon_points(angles, np.full(num_axes, OUTER_RING_RADIUS))
    for x, y in ring_vertices:
        ax.plot(
            [0.0, x],
            [0.0, y],
            color=RADIAL_AXIS_COLOR,
            linewidth=0.75,
            linestyle=RADIAL_LINESTYLE,
            zorder=1,
        )

    for idx, (_, label) in enumerate(METRICS):
        label_pos = vertex_label_position(outer_vertices[idx], idx)
        ha, va = axis_alignment(label_pos[0], label_pos[1])
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            fontsize=14,
            fontweight="bold",
            color="black",
            ha=ha,
            va=va,
        )

    legend_handles: list[Line2D] = []
    total_datasets = len(DATASET_SPECS)
    for dataset_index, (dataset_name, _) in enumerate(DATASET_SPECS):
        style = style_map[dataset_name]
        color = str(style["color"])
        linewidth = float(style["linewidth"])
        fill_alpha = float(style["fill_alpha"])
        marker = DATASET_MARKERS[dataset_name]
        values = np.array(
            [
                display_ready[dataset_name]["display_radius_metrics"][metric_key]
                for metric_key, _ in METRICS
            ]
        )
        raw_values = [
            display_ready[dataset_name]["raw_mean_metrics"][metric_key]
            for metric_key, _ in METRICS
        ]
        point_coords = polygon_points(angles, values)
        polygon = closed_polygon(point_coords)
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=color,
            linewidth=linewidth,
            solid_joinstyle="round",
            marker=marker,
            markersize=MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=4 if dataset_name == highlight_dataset else 3,
        )
        if fill_alpha > 0.0:
            ax.fill(
                polygon[:, 0],
                polygon[:, 1],
                color=color,
                alpha=fill_alpha,
                zorder=2,
            )

        for metric_index, point in enumerate(point_coords):
            label_pos = value_label_position(
                point,
                angles[metric_index],
                dataset_index,
                total_datasets,
            )
            ax.text(
                label_pos[0],
                label_pos[1],
                format_raw_value(raw_values[metric_index]),
                fontsize=VALUE_LABEL_FONT_SIZE,
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                zorder=5,
                path_effects=[
                    pe.withStroke(linewidth=2.2, foreground="white", alpha=0.95)
                ],
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=linewidth,
                marker=marker,
                markersize=LEGEND_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=dataset_name,
            )
        )

    ax.set_xlim(-1.36, 1.36)
    ax.set_ylim(-1.34, 1.36)

    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=legend_anchor,
        frameon=legend_box,
        facecolor="white",
        fancybox=False,
        framealpha=1.0,
        fontsize=legend_font_size,
        labelspacing=0.8,
        borderpad=0.8,
        handlelength=LEGEND_HANDLE_LENGTH,
        ncol=3,
        columnspacing=1.6,
    )
    if legend_box and legend.get_frame() is not None:
        legend.get_frame().set_edgecolor(RADAR_LEGEND_EDGE_COLOR)
        legend.get_frame().set_linewidth(RADAR_LEGEND_LINEWIDTH)
    for text in legend.get_texts():
        text.set_fontfamily("serif")


def plot_radar(display_ready: dict, highlight_dataset: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    fig, ax = plt.subplots(figsize=(9.6, 8.8), facecolor="white")
    draw_radar(ax, display_ready, highlight_dataset)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.125)
    fig.savefig(OUTPUT_PNG, dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    positive = np.flatnonzero(labels > 0)
    if positive.size == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = int(positive[0])
    prev = int(positive[0])
    for idx in positive[1:]:
        current = int(idx)
        if current == prev + 1:
            prev = current
            continue
        segments.append((start, prev))
        start = current
        prev = current
    segments.append((start, prev))
    return segments


def merge_nearby_segments(
    segments: list[tuple[int, int]], max_gap: int
) -> list[dict[str, object]]:
    if not segments:
        return []

    merged: list[dict[str, object]] = []
    current_start, current_end = segments[0]
    members = [segments[0]]

    for start, end in segments[1:]:
        if start - current_end - 1 <= int(max_gap):
            current_end = end
            members.append((start, end))
            continue
        merged.append(
            {
                "start": int(current_start),
                "end": int(current_end),
                "members": list(members),
            }
        )
        current_start, current_end = start, end
        members = [(start, end)]

    merged.append(
        {
            "start": int(current_start),
            "end": int(current_end),
            "members": list(members),
        }
    )
    return merged


def compute_signal_prominence(test: np.ndarray, start: int, end: int) -> float:
    segment = np.asarray(test[start : end + 1], dtype=np.float64)
    local_left = max(0, start - EXAMPLE_CONTEXT_MARGIN)
    local_right = min(len(test), end + EXAMPLE_CONTEXT_MARGIN + 1)
    local = np.asarray(test[local_left:local_right], dtype=np.float64)
    baseline = np.median(local, axis=0)
    return float(np.max(np.abs(segment - baseline)))


def choose_example_event(
    test: np.ndarray,
    labels: np.ndarray,
    score: np.ndarray,
) -> dict[str, object]:
    segments = anomaly_segments(labels)
    if not segments:
        center = len(test) // 2
        return {
            "start": int(center),
            "end": int(center),
            "members": [(int(center), int(center))],
            "selection_score": 0.0,
            "score_peak": 0.0,
            "score_contrast": 0.0,
            "event_count": 0,
            "edge_completeness": 0.0,
        }

    best_event = {
        "start": int(segments[0][0]),
        "end": int(segments[0][1]),
        "members": [segments[0]],
        "selection_score": 0.0,
        "peak_index": int(segments[0][0]),
        "peak_alignment": 0.0,
        "score_peak": 0.0,
        "score_contrast": 0.0,
        "event_count": 1,
        "edge_completeness": 0.0,
    }
    best_selection_score = -np.inf
    n_steps = len(labels)

    for start, end in segments:
        left, right = choose_example_window(n_steps, int(start), int(end))
        window_score = np.asarray(score[left:right], dtype=np.float64)
        window_peak_offset = int(np.argmax(window_score))
        window_peak_index = left + window_peak_offset
        anchor_peak_index = int(np.argmax(score[int(start) : int(end) + 1])) + int(start)
        score_peak = float(score[anchor_peak_index])
        local_mask = np.asarray(labels[left:right], dtype=np.int32) > 0
        background = window_score[~local_mask]
        background_mean = float(np.mean(background)) if background.size > 0 else 0.0
        score_contrast = score_peak - background_mean
        score_mass = float(np.sum(np.clip(window_score - background_mean, 0.0, None)))
        event_count = sum(1 for seg_start, seg_end in segments if seg_end >= left and seg_start < right)
        event_span = end - start + 1
        center = 0.5 * (left + right - 1)
        edge_distance = min(center, (n_steps - 1) - center)
        edge_completeness = float(
            np.clip(edge_distance / max(1.0, EXAMPLE_WINDOW_MIN / 2.0), 0.0, 1.0)
        )
        signal_prominence = compute_signal_prominence(test, int(start), int(end))
        anchor_members = [(int(seg_start), int(seg_end)) for seg_start, seg_end in segments if seg_end >= left and seg_start < right]
        if int(start) <= window_peak_index <= int(end):
            peak_alignment = 1.0
        else:
            peak_distance = min(
                abs(window_peak_index - int(start)),
                abs(window_peak_index - int(end)),
            )
            peak_alignment = float(np.exp(-peak_distance / 18.0))
        clutter_penalty = 0.12 * max(0, event_count - 2)

        selection_score = (
            8.0 * peak_alignment
            + 3.0 * score_peak
            + 1.4 * score_contrast
            + 0.10 * score_mass
            + 0.20 * min(event_span, 40)
            + 0.18 * min(event_count, 6)
            + 0.9 * edge_completeness
            + 0.40 * signal_prominence
            - clutter_penalty
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_event = {
                "start": int(start),
                "end": int(end),
                "members": anchor_members,
                "selection_score": float(selection_score),
                "peak_index": int(anchor_peak_index),
                "peak_alignment": float(peak_alignment),
                "score_peak": float(score_peak),
                "score_contrast": float(score_contrast),
                "event_count": int(event_count),
                "edge_completeness": float(edge_completeness),
            }

    return best_event


def choose_example_channel(test: np.ndarray, start: int, end: int) -> int:
    local_left = max(0, start - EXAMPLE_CONTEXT_MARGIN)
    local_right = min(len(test), end + EXAMPLE_CONTEXT_MARGIN + 1)
    local = np.asarray(test[local_left:local_right], dtype=np.float64)
    anom_slice = slice(start - local_left, end - local_left + 1)
    local_mask = np.zeros(len(local), dtype=bool)
    local_mask[anom_slice] = True

    best_channel = 0
    best_score = -np.inf
    for channel in range(local.shape[1]):
        signal = local[:, channel]
        target = signal[anom_slice]
        reference = signal[~local_mask] if np.any(~local_mask) else signal
        baseline = float(np.median(reference))
        scale = float(np.median(np.abs(reference - baseline))) + 1e-6
        peak = float(np.max(np.abs(target - baseline))) / scale
        mean_shift = float(np.mean(np.abs(target - baseline))) / scale
        score = peak + 0.35 * mean_shift
        if score > best_score:
            best_score = score
            best_channel = channel
    return best_channel


def choose_example_window(
    total_steps: int,
    start: int,
    end: int,
    center_index: int | None = None,
) -> tuple[int, int]:
    segment_len = end - start + 1
    window_len = int(np.clip(max(EXAMPLE_WINDOW_MIN, segment_len * 5), EXAMPLE_WINDOW_MIN, EXAMPLE_WINDOW_MAX))
    center = (start + end) // 2 if center_index is None else int(center_index)
    left = max(0, center - window_len // 2)
    right = min(total_steps, left + window_len)
    left = max(0, right - window_len)
    return left, right


def resolve_example_experiment_dir(dataset_key: str) -> Path:
    experiment_base = f"{dataset_key}_baseline"
    candidates = sorted((ROOT / "main-result" / "synthetic").glob(f"{experiment_base}_*"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find synthetic experiment directory for {dataset_key}")
    return candidates[0]


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or window <= 1:
        return values.copy()
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    pad = int(window) // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.size]


def normalize_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    low = float(np.percentile(values, 2.0))
    high = float(np.percentile(values, 98.0))
    if np.isclose(high, low):
        high = low + 1e-6
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def load_example_score(dataset_key: str) -> np.ndarray:
    experiment_dir = resolve_example_experiment_dir(dataset_key)
    score_path = experiment_dir / "test_scores_final_cdf_max.npy"
    if not score_path.exists():
        score_path = experiment_dir / "test_scores_cdf_max.npy"
    if not score_path.exists():
        raise FileNotFoundError(f"Cannot find anomaly score file under {experiment_dir}")
    raw_score = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
    return normalize_score(smooth_series(raw_score, EXAMPLE_SCORE_SMOOTH_WINDOW))


def choose_panel_variant(event: dict[str, object]) -> tuple[float, ...]:
    alignment = float(event.get("peak_alignment", 0.0))
    score_peak = float(event.get("score_peak", 0.0))
    return (
        1.0 if alignment >= EXAMPLE_MIN_ALIGNMENT and score_peak >= EXAMPLE_MIN_SCORE_PEAK else 0.0,
        1.0 if alignment >= EXAMPLE_MIN_ALIGNMENT else 0.0,
        1.0 if score_peak >= EXAMPLE_MIN_SCORE_PEAK else 0.0,
        alignment,
        score_peak,
        float(event.get("selection_score", 0.0)),
    )


def load_example_panels() -> list[dict[str, object]]:
    panels: list[dict[str, object]] = []
    for display_name, dataset_keys in EXAMPLE_PANEL_SPECS:
        best_panel: dict[str, object] | None = None
        best_key: tuple[float, ...] | None = None

        for dataset_key in dataset_keys:
            raw_bundle = load_raw_dataset_bundle(dataset_key, ROOT / "dataset" / "anomaly_detect")
            test = np.asarray(raw_bundle.test, dtype=np.float32)
            labels = np.asarray(raw_bundle.test_labels, dtype=np.int32).reshape(-1)
            score = load_example_score(dataset_key)
            event = choose_example_event(test, labels, score)
            start = int(event["start"])
            end = int(event["end"])
            channel = choose_example_channel(test, start, end)
            peak_index = int(event.get("peak_index", (start + end) // 2))
            left, right = choose_example_window(len(test), start, end)
            candidate_panel = {
                "title": display_name,
                "dataset_key": dataset_key,
                "channel": channel,
                "x": np.arange(left, right, dtype=np.int64),
                "signal": np.asarray(test[left:right, channel], dtype=np.float64),
                "score": np.asarray(score[left:right], dtype=np.float64),
                "labels": np.asarray(labels[left:right], dtype=np.int32),
                "event_start": start,
                "event_end": end,
                "window_start": int(left),
                "window_end": int(right - 1),
                "event_members": [(int(s), int(e)) for s, e in event["members"]],
                "selection_score": float(event["selection_score"]),
                "peak_index": peak_index,
                "peak_alignment": float(event.get("peak_alignment", 0.0)),
                "score_peak": float(event["score_peak"]),
                "score_contrast": float(event["score_contrast"]),
                "event_count": int(event["event_count"]),
                "edge_completeness": float(event["edge_completeness"]),
            }
            candidate_key = choose_panel_variant(event)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_panel = candidate_panel

        if best_panel is None:
            raise RuntimeError(f"Failed to build example panel for {display_name}")
        panels.append(best_panel)
    return panels


def write_example_windows(panels: list[dict[str, object]]) -> None:
    payload = []
    for panel in panels:
        payload.append(
            {
                "title": panel["title"],
                "dataset_key": panel["dataset_key"],
                "channel": int(panel["channel"]),
                "window": {
                    "start": int(panel["window_start"]),
                    "end": int(panel["window_end"]),
                },
                "selected_event": {
                    "start": int(panel["event_start"]),
                    "end": int(panel["event_end"]),
                    "members": [
                        {"start": int(start), "end": int(end)}
                        for start, end in panel["event_members"]
                    ],
                },
                "selection_score": float(panel["selection_score"]),
                "peak_index": int(panel["peak_index"]),
                "peak_alignment": float(panel["peak_alignment"]),
                "score_peak": float(panel["score_peak"]),
                "score_contrast": float(panel["score_contrast"]),
                "event_count": int(panel["event_count"]),
                "edge_completeness": float(panel["edge_completeness"]),
            }
        )
    with OUTPUT_EXAMPLE_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def style_example_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(EXAMPLE_PANEL_FACE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(EXAMPLE_PANEL_EDGE)


def draw_example_panel(ax_signal: plt.Axes, ax_score: plt.Axes, panel: dict[str, object], show_y_labels: bool) -> None:
    x = np.asarray(panel["x"], dtype=np.int64)
    signal = np.asarray(panel["signal"], dtype=np.float64)
    score = np.asarray(panel["score"], dtype=np.float64)
    labels = np.asarray(panel["labels"], dtype=np.int32)
    mask = labels > 0

    style_example_axis(ax_signal)
    style_example_axis(ax_score)

    if np.any(mask):
        padded = np.concatenate([[False], mask, [False]])
        changes = np.diff(padded.astype(np.int32))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        filled_score = np.full_like(score, np.nan, dtype=np.float64)
        for start_idx, end_idx in zip(starts, ends):
            x0 = float(x[start_idx])
            x1 = float(x[end_idx])
            ax_signal.axvspan(x0, x1, color=EXAMPLE_ANOMALY_COLOR, alpha=0.55, zorder=0)
            ax_score.axvspan(x0, x1, color=EXAMPLE_ANOMALY_COLOR, alpha=0.40, zorder=0)
            local_peak = int(np.argmax(score[start_idx : end_idx + 1])) + start_idx
            fill_start = max(start_idx, local_peak - EXAMPLE_PEAK_FILL_HALF_WIDTH)
            fill_end = min(end_idx, local_peak + EXAMPLE_PEAK_FILL_HALF_WIDTH)
            filled_score[fill_start : fill_end + 1] = score[fill_start : fill_end + 1]
    else:
        filled_score = np.full_like(score, np.nan, dtype=np.float64)

    ax_signal.plot(x, signal, color=EXAMPLE_SIGNAL_COLOR, linewidth=EXAMPLE_SIGNAL_LINEWIDTH, zorder=2)
    ax_signal.set_title(str(panel["title"]), fontsize=12, fontweight="bold", pad=4)
    ax_signal.tick_params(axis="x", labelbottom=False, length=0)
    ax_signal.tick_params(axis="y", labelsize=8, length=2.5, colors="#444444")
    if not show_y_labels:
        ax_signal.set_yticklabels([])
    else:
        ax_signal.set_ylabel(
            "Time-Series Value",
            fontsize=EXAMPLE_YLABEL_FONT_SIZE,
            fontweight="bold",
            rotation=90,
            labelpad=EXAMPLE_YLABEL_PAD,
        )

    ax_score.fill_between(x, 0.0, filled_score, color=EXAMPLE_SCORE_FILL, alpha=0.92, linewidth=0.0, zorder=1)
    ax_score.plot(x, score, color=EXAMPLE_SCORE_LINE, linewidth=1.1, zorder=2)
    ax_score.set_ylim(0.0, 1.05)
    ax_score.set_yticks([0.0, 0.5, 1.0] if show_y_labels else [])
    ax_score.tick_params(axis="y", labelsize=7, length=2.0, colors="#555555")
    ax_score.tick_params(axis="x", labelsize=7, length=2.0, colors="#555555")
    if show_y_labels:
        ax_score.set_ylabel(
            "Anomaly\nScore",
            fontsize=EXAMPLE_YLABEL_FONT_SIZE - 1,
            fontweight="bold",
            rotation=90,
            labelpad=EXAMPLE_SCORE_YLABEL_PAD,
        )
    ax_score.set_xlabel(f"{panel['dataset_key']} · ch{int(panel['channel']) + 1}", fontsize=7, labelpad=2)


def plot_examples_with_radar(display_ready: dict, highlight_dataset: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    example_panels = load_example_panels()
    write_example_windows(example_panels)

    fig = plt.figure(figsize=(18.0, 8.6), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.45, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.13,
        hspace=0.08,
    )
    left_grid = grid[:, 0].subgridspec(2, 3, wspace=0.14, hspace=0.22)
    radar_ax = fig.add_subplot(grid[:, 1])

    for idx, panel in enumerate(example_panels):
        row = idx // 3
        col = idx % 3
        cell = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=left_grid[row, col],
            height_ratios=[4.3, 1.05],
            hspace=0.03,
        )
        ax_signal = fig.add_subplot(cell[0, 0])
        ax_label = fig.add_subplot(cell[1, 0], sharex=ax_signal)
        draw_example_panel(ax_signal, ax_label, panel, show_y_labels=(col == 0))

    anomaly_legend = fig.legend(
        handles=[
            Patch(
                facecolor=EXAMPLE_ANOMALY_COLOR,
                edgecolor="none",
                alpha=0.55,
                label="Ground Truth Anomaly",
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.285, 0.055),
        frameon=False,
        ncol=1,
        fontsize=10,
        handlelength=2.2,
        handletextpad=0.5,
    )
    for text in anomaly_legend.get_texts():
        text.set_fontfamily("serif")

    draw_radar(
        radar_ax,
        display_ready,
        highlight_dataset,
        legend_anchor=(0.5, -0.005),
        legend_font_size=12,
        legend_box=True,
    )
    radar_ax.set_xlim(-1.34, 1.34)
    radar_ax.set_ylim(-1.30, 1.34)

    fig.subplots_adjust(left=0.03, right=0.985, top=0.965, bottom=0.11)
    fig.savefig(OUTPUT_COMPOSITE_PNG, dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT_COMPOSITE_PDF, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def load_mechanism_curves(dataset_key: str) -> dict[str, np.ndarray]:
    experiment_dir = resolve_example_experiment_dir(dataset_key)
    diagnostic_path = experiment_dir / "test_diagnostic_scores.npz"
    if not diagnostic_path.exists():
        raise FileNotFoundError(f"Cannot find diagnostic score file under {experiment_dir}")

    curve_map: dict[str, np.ndarray] = {}
    with np.load(diagnostic_path) as data:
        for key, _, _ in MECHANISM_COLUMN_SPECS:
            if key == "signal":
                continue
            source_key = key
            if source_key not in data.files and key == "cdf_max_score":
                source_key = "final_fused_score"
            if source_key not in data.files:
                raise KeyError(f"Missing '{key}' in {diagnostic_path}")
            curve_map[key] = normalize_score(
                smooth_series(
                    np.asarray(data[source_key], dtype=np.float64).reshape(-1),
                    MECHANISM_CURVE_SMOOTH_WINDOW,
                )
            )
    return curve_map


def choose_mechanism_event(
    test: np.ndarray,
    labels: np.ndarray,
    curves: dict[str, np.ndarray],
) -> dict[str, object]:
    base_segments = anomaly_segments(labels)
    if not base_segments:
        center = len(test) // 2
        return {
            "start": int(center),
            "end": int(center),
            "members": [(int(center), int(center))],
            "selection_score": 0.0,
            "peak_index": int(center),
            "curve_peaks": {key: 0.0 for key in curves},
            "curve_contrasts": {key: 0.0 for key in curves},
            "edge_completeness": 0.0,
        }
    merged_events = [
        {
            "start": int(start),
            "end": int(end),
            "members": [(int(start), int(end))],
        }
        for start, end in base_segments
    ]

    best_event: dict[str, object] | None = None
    best_selection_score = -np.inf
    total_steps = len(labels)

    for event in merged_events:
        start = int(event["start"])
        end = int(event["end"])
        members = [(int(seg_start), int(seg_end)) for seg_start, seg_end in event["members"]]
        anchor_curve = curves["cdf_max_score"]
        peak_index = int(np.argmax(anchor_curve[start : end + 1])) + start
        left, right = choose_example_window(total_steps, start, end, center_index=peak_index)
        window_labels = np.asarray(labels[left:right], dtype=np.int32) > 0

        curve_peaks: dict[str, float] = {}
        curve_contrasts: dict[str, float] = {}
        response_strength = 0.0
        for key, weight in (
            ("state_novelty", 1.05),
            ("knn_distance", 1.10),
            ("cdf_max_score", 1.25),
        ):
            curve = curves[key]
            inside = np.asarray(curve[start : end + 1], dtype=np.float64)
            window = np.asarray(curve[left:right], dtype=np.float64)
            background = window[~window_labels]
            peak = float(np.max(inside)) if inside.size > 0 else 0.0
            inside_mean = float(np.mean(inside)) if inside.size > 0 else 0.0
            background_mean = float(np.mean(background)) if background.size > 0 else 0.0
            contrast = max(0.0, peak - background_mean)
            curve_peaks[key] = peak
            curve_contrasts[key] = contrast
            response_strength += weight * (1.55 * peak + 0.75 * inside_mean + 0.90 * contrast)

        event_span = end - start + 1
        signal_prominence = compute_signal_prominence(test, start, end)
        center = 0.5 * (left + right - 1)
        edge_distance = min(center, (total_steps - 1) - center)
        edge_completeness = float(
            np.clip(edge_distance / max(1.0, EXAMPLE_WINDOW_MIN / 2.0), 0.0, 1.0)
        )
        selection_score = (
            response_strength
            + 0.30 * min(event_span, 40)
            + 0.32 * signal_prominence
            + 0.85 * edge_completeness
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_event = {
                "start": start,
                "end": end,
                "members": members,
                "selection_score": float(selection_score),
                "peak_index": int(peak_index),
                "curve_peaks": curve_peaks,
                "curve_contrasts": curve_contrasts,
                "edge_completeness": float(edge_completeness),
            }

    if best_event is None:
        raise RuntimeError("Failed to select a mechanism event window.")
    return best_event


def load_example_panels() -> list[dict[str, object]]:
    panels: list[dict[str, object]] = []
    for display_name, dataset_keys in MECHANISM_PANEL_SPECS:
        best_panel: dict[str, object] | None = None
        best_key: tuple[float, ...] | None = None

        for dataset_key in dataset_keys:
            raw_bundle = load_raw_dataset_bundle(dataset_key, ROOT / "dataset" / "anomaly_detect")
            test = np.asarray(raw_bundle.test, dtype=np.float32)
            labels = np.asarray(raw_bundle.test_labels, dtype=np.int32).reshape(-1)
            curves = load_mechanism_curves(dataset_key)
            event = choose_mechanism_event(test, labels, curves)
            start = int(event["start"])
            end = int(event["end"])
            channel = choose_example_channel(test, start, end)
            peak_index = int(event.get("peak_index", (start + end) // 2))
            left, right = choose_example_window(len(test), start, end, center_index=peak_index)
            candidate_panel = {
                "title": display_name,
                "dataset_key": dataset_key,
                "channel": channel,
                "x": np.arange(left, right, dtype=np.int64),
                "signal": np.asarray(test[left:right, channel], dtype=np.float64),
                "labels": np.asarray(labels[left:right], dtype=np.int32),
                "curves": {
                    key: np.asarray(curves[key][left:right], dtype=np.float64)
                    for key in curves
                },
                "event_start": start,
                "event_end": end,
                "window_start": int(left),
                "window_end": int(right - 1),
                "event_members": [(int(s), int(e)) for s, e in event["members"]],
                "selection_score": float(event["selection_score"]),
                "peak_index": peak_index,
                "curve_peaks": {
                    key: float(value) for key, value in dict(event["curve_peaks"]).items()
                },
                "curve_contrasts": {
                    key: float(value) for key, value in dict(event["curve_contrasts"]).items()
                },
                "edge_completeness": float(event["edge_completeness"]),
            }
            candidate_key = (
                float(event["selection_score"]),
                float(event["curve_peaks"]["cdf_max_score"]),
                float(event["curve_peaks"]["knn_distance"]),
                float(event["curve_peaks"]["state_novelty"]),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_panel = candidate_panel

        if best_panel is None:
            raise RuntimeError(f"Failed to build example panel for {display_name}")
        panels.append(best_panel)
    return panels


def write_example_windows(panels: list[dict[str, object]]) -> None:
    payload = []
    for panel in panels:
        payload.append(
            {
                "title": panel["title"],
                "dataset_key": panel["dataset_key"],
                "channel": int(panel["channel"]),
                "window": {
                    "start": int(panel["window_start"]),
                    "end": int(panel["window_end"]),
                },
                "selected_event": {
                    "start": int(panel["event_start"]),
                    "end": int(panel["event_end"]),
                    "members": [
                        {"start": int(start), "end": int(end)}
                        for start, end in panel["event_members"]
                    ],
                },
                "selection_score": float(panel["selection_score"]),
                "peak_index": int(panel["peak_index"]),
                "curve_peaks": {
                    key: float(value)
                    for key, value in dict(panel["curve_peaks"]).items()
                },
                "curve_contrasts": {
                    key: float(value)
                    for key, value in dict(panel["curve_contrasts"]).items()
                },
                "edge_completeness": float(panel["edge_completeness"]),
            }
        )
    with OUTPUT_EXAMPLE_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def add_example_anomaly_spans(ax: plt.Axes, x: np.ndarray, labels: np.ndarray, *, alpha: float) -> None:
    mask = np.asarray(labels, dtype=np.int32) > 0
    if not np.any(mask):
        return

    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int32))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    for start_idx, end_idx in zip(starts, ends):
        x0 = float(x[start_idx]) - 0.5
        x1 = float(x[end_idx]) + 0.5
        ax.axvspan(x0, x1, color=EXAMPLE_ANOMALY_COLOR, alpha=alpha, zorder=0)


def add_target_event_spans(
    ax: plt.Axes,
    x: np.ndarray,
    event_members: list[tuple[int, int]],
    *,
    alpha: float,
) -> None:
    if len(x) == 0:
        return
    x_min = int(x[0])
    x_max = int(x[-1])
    for start, end in event_members:
        span_start = max(int(start), x_min)
        span_end = min(int(end), x_max)
        if span_start > span_end:
            continue
        ax.axvspan(
            float(span_start) - 0.5,
            float(span_end) + 0.5,
            color=EXAMPLE_ANOMALY_COLOR,
            alpha=alpha,
            zorder=0,
        )


def draw_mechanism_panel(
    ax: plt.Axes,
    panel: dict[str, object],
    column_key: str,
    column_title: str,
    color: str,
    *,
    show_column_title: bool,
    show_row_label: bool,
    show_x_label: bool,
    show_y_ticks: bool,
) -> None:
    x = np.asarray(panel["x"], dtype=np.int64)
    labels = np.asarray(panel["labels"], dtype=np.int32)

    style_example_axis(ax)
    add_example_anomaly_spans(ax, x, labels, alpha=0.42 if column_key == "signal" else 0.35)

    if column_key == "signal":
        y = np.asarray(panel["signal"], dtype=np.float64)
        ax.plot(
            x,
            y,
            color=EXAMPLE_SIGNAL_COLOR,
            linewidth=EXAMPLE_SIGNAL_LINEWIDTH,
            zorder=2,
        )
        ax.tick_params(axis="y", labelsize=8, length=2.5, colors="#444444")
    else:
        y = np.asarray(panel["curves"][column_key], dtype=np.float64)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=MECHANISM_LINEWIDTH,
            marker="o",
            markersize=MECHANISM_MARKER_SIZE,
            markevery=max(1, len(x) // 9),
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=2,
        )
        event_mask = (x >= int(panel["event_start"])) & (x <= int(panel["event_end"]))
        if np.any(event_mask):
            local_idx = np.flatnonzero(event_mask)
            peak_idx = int(np.argmax(y[event_mask])) + int(local_idx[0])
            ax.scatter(
                [x[peak_idx]],
                [y[peak_idx]],
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        ax.set_ylim(-0.02, 1.03)
        ax.set_yticks([0.0, 0.5, 1.0] if show_y_ticks else [])
        ax.tick_params(axis="y", labelsize=8, length=2.0, colors="#555555")

    if show_column_title:
        ax.set_title(column_title, fontsize=12.5, fontweight="bold", pad=6)

    if show_row_label:
        ax.text(
            MECHANISM_ROW_LABEL_X,
            0.5,
            str(panel["title"]),
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#2F2F2F",
        )
        ax.text(
            0.03,
            0.06,
            f"{panel['dataset_key']} / ch{int(panel['channel']) + 1}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#5A5A5A",
        )

    if not show_y_ticks:
        ax.set_yticklabels([])

    ax.tick_params(axis="x", labelsize=8, length=2.0, colors="#555555")
    if show_x_label:
        ax.set_xlabel("Time Index", fontsize=9, labelpad=3)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def plot_examples_with_radar(display_ready: dict, highlight_dataset: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    example_panels = load_example_panels()
    write_example_windows(example_panels)

    fig = plt.figure(figsize=(20.0, 9.0), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.78, 1.02],
        wspace=0.10,
    )
    left_grid = grid[0, 0].subgridspec(3, 4, wspace=0.16, hspace=0.24)
    radar_ax = fig.add_subplot(grid[0, 1])

    for row, panel in enumerate(example_panels):
        for col, (column_key, column_title, column_color) in enumerate(MECHANISM_COLUMN_SPECS):
            ax = fig.add_subplot(left_grid[row, col])
            draw_mechanism_panel(
                ax,
                panel,
                column_key,
                column_title,
                column_color,
                show_column_title=(row == 0),
                show_row_label=(col == 0),
                show_x_label=(row == len(example_panels) - 1),
                show_y_ticks=(col == 0),
            )

    anomaly_legend = fig.legend(
        handles=[
            Patch(
                facecolor=EXAMPLE_ANOMALY_COLOR,
                edgecolor="none",
                alpha=0.55,
                label="Ground Truth Anomaly",
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.31, 0.045),
        frameon=False,
        ncol=1,
        fontsize=10,
        handlelength=2.2,
        handletextpad=0.5,
    )
    for text in anomaly_legend.get_texts():
        text.set_fontfamily("serif")

    draw_radar(
        radar_ax,
        display_ready,
        highlight_dataset,
        legend_anchor=(0.5, -0.01),
        legend_font_size=12,
        legend_box=True,
    )
    radar_ax.set_xlim(-1.34, 1.34)
    radar_ax.set_ylim(-1.30, 1.34)

    fig.subplots_adjust(left=0.04, right=0.985, top=0.965, bottom=0.10)
    fig.savefig(OUTPUT_COMPOSITE_PNG, dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT_COMPOSITE_PDF, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int32))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def fit_pca_basis(sample_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_x = np.asarray(sample_x, dtype=np.float64)
    mean_vec = sample_x.mean(axis=0, keepdims=True)
    centered = sample_x - mean_vec
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    explained = (s[:2] ** 2) / max(np.sum(s ** 2), 1e-12)
    return mean_vec, components, explained


def project_pca(values: np.ndarray, mean_vec: np.ndarray, components: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - mean_vec
    return centered @ components.T


def choose_retrieval_query_bounds(
    panel: dict[str, object],
    seq_len: int,
    total_steps: int,
) -> tuple[int, int]:
    members = [
        (int(start), int(end))
        for start, end in panel.get(
            "event_members",
            [(int(panel["event_start"]), int(panel["event_end"]))],
        )
    ]
    event_start = min(start for start, _ in members)
    event_end = max(end for _, end in members)

    x = np.asarray(panel.get("x", []), dtype=np.int64)
    labels = np.asarray(panel.get("labels", []), dtype=np.int32)
    local_segments: list[tuple[int, int]] = []
    if x.size == labels.size and x.size > 0:
        mask = labels > 0
        if np.any(mask):
            padded = np.concatenate([[False], mask, [False]])
            changes = np.diff(padded.astype(np.int32))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1) - 1
            local_segments = [(int(x[s]), int(x[e])) for s, e in zip(starts, ends)]

    if local_segments:
        overlap_ids = [
            idx
            for idx, (seg_start, seg_end) in enumerate(local_segments)
            if not (seg_end < event_start or seg_start > event_end)
        ]
        if overlap_ids:
            left_idx = min(overlap_ids)
            right_idx = max(overlap_ids)
            cover_start = local_segments[left_idx][0]
            cover_end = local_segments[right_idx][1]
            preferred_margin = max(8, int(round(seq_len * 0.16)))

            while left_idx > 0:
                prev_start, prev_end = local_segments[left_idx - 1]
                candidate_span = cover_end - prev_start + 1
                if cover_start - prev_end - 1 > RETRIEVAL_COVER_GAP:
                    break
                if candidate_span + 2 * preferred_margin > seq_len:
                    break
                cover_start = prev_start
                left_idx -= 1

            while right_idx < len(local_segments) - 1:
                next_start, next_end = local_segments[right_idx + 1]
                candidate_span = next_end - cover_start + 1
                if next_start - cover_end - 1 > RETRIEVAL_COVER_GAP:
                    break
                if candidate_span + 2 * preferred_margin > seq_len:
                    break
                cover_end = next_end
                right_idx += 1

            event_start = int(cover_start)
            event_end = int(cover_end)

    center = int(panel.get("peak_index", (event_start + event_end) // 2))
    event_len = max(1, event_end - event_start + 1)
    available_slack = max(0, seq_len - event_len)
    preferred_margin = max(8, int(round(seq_len * 0.16)))
    margin = min(preferred_margin, available_slack // 2)

    min_start = max(0, event_end + margin - seq_len + 1)
    max_start = min(max(0, total_steps - seq_len), event_start - margin)

    centered_start = center - seq_len // 2
    if min_start <= max_start:
        start = int(np.clip(centered_start, min_start, max_start))
    else:
        start = max(0, min(total_steps - seq_len, centered_start))

    end = start + seq_len
    return int(start), int(end)


def to_numpy_array(value, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)


def load_experiment_runtime(dataset_key: str) -> tuple[CoReMADConfig, CoReMADTrainer, object, object]:
    experiment_dir = resolve_example_experiment_dir(dataset_key)
    config = CoReMADConfig.load(experiment_dir / "config.json")
    config.artifact_root = str(experiment_dir.parent)
    config.experiment_name = experiment_dir.name
    config.data_root = str(ROOT / "dataset" / "anomaly_detect")
    config.device = "cpu"
    config.num_workers = 0
    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    memory = trainer.load_memory_bank()
    return config, trainer, model, normalizer, memory


def build_retrieval_row_payload(panel: dict[str, object]) -> dict[str, object]:
    dataset_key = str(panel["dataset_key"])
    config, trainer, model, normalizer, memory = load_experiment_runtime(dataset_key)
    raw_bundle = load_raw_dataset_bundle(dataset_key, ROOT / "dataset" / "anomaly_detect")
    raw_test = np.asarray(raw_bundle.test, dtype=np.float32)
    raw_labels = np.asarray(raw_bundle.test_labels, dtype=np.int32).reshape(-1)
    test_norm = normalizer.transform(raw_test)

    query_start, query_end = choose_retrieval_query_bounds(panel, int(config.seq_len), len(raw_test))
    x_query = torch.from_numpy(test_norm[query_start:query_end]).unsqueeze(0).to(trainer.device)
    with torch.no_grad():
        encoded, _ = model.deterministic_completion_scores(x_query)
        memory_out = memory.query_preencoded(
            encoded["state_vec"],
            encoded["z"],
            encoded["c"],
            return_details=True,
        )

    state_bank = memory.state_bank.cpu().numpy().astype(np.float64, copy=False)
    rng = np.random.RandomState(42)
    sample_size = min(SCATTER_PCA_SAMPLE, len(state_bank))
    if len(state_bank) > sample_size:
        sample_ids = np.sort(rng.choice(np.arange(len(state_bank)), size=sample_size, replace=False))
    else:
        sample_ids = np.arange(len(state_bank), dtype=np.int64)
    mean_vec, components, explained = fit_pca_basis(state_bank[sample_ids])

    background_size = min(SCATTER_MAX_POINTS, len(state_bank))
    if len(state_bank) > background_size:
        background_ids = np.sort(rng.choice(np.arange(len(state_bank)), size=background_size, replace=False))
    else:
        background_ids = np.arange(len(state_bank), dtype=np.int64)
    background_proj = project_pca(state_bank[background_ids], mean_vec, components)

    coarse_windows = to_numpy_array(memory_out["coarse_windows"][0], dtype=np.int64)
    coarse_windows = np.unique(coarse_windows[coarse_windows >= 0])

    query_state = encoded["state_vec"][0].detach().cpu().numpy().astype(np.float64, copy=False)

    fine_scale_idx = 0
    patch_size = int(config.patch_sizes[fine_scale_idx])
    heat_dist = to_numpy_array(memory_out["neighbor_context_distances"][fine_scale_idx][0], dtype=np.float64)
    heat_valid = to_numpy_array(memory_out["neighbor_valid_masks"][fine_scale_idx][0], dtype=bool)
    heat_sim = np.zeros_like(heat_dist, dtype=np.float64)
    valid_values = heat_dist[heat_valid]
    if valid_values.size > 0:
        sigma = max(float(np.median(valid_values)), 1e-6)
        heat_sim[heat_valid] = np.exp(-heat_dist[heat_valid] / sigma)
    heatmap = heat_sim.T

    query_labels = raw_labels[query_start:query_end]
    n_patches = heat_dist.shape[0]
    patch_label_len = n_patches * patch_size
    patch_labels = query_labels[:patch_label_len].reshape(n_patches, patch_size).max(axis=1) > 0

    support_ids: set[int] = set()
    for scale_ids, scale_valid in zip(
        memory_out["neighbor_window_ids"],
        memory_out["neighbor_valid_masks"],
    ):
        ids = to_numpy_array(scale_ids[0], dtype=np.int64)
        valid = to_numpy_array(scale_valid[0], dtype=bool)
        for value in ids[valid].tolist():
            if int(value) >= 0:
                support_ids.add(int(value))
    support_windows = np.array(sorted(support_ids), dtype=np.int64)

    focus_windows = np.unique(
        np.concatenate(
            [
                coarse_windows.astype(np.int64, copy=False),
                support_windows.astype(np.int64, copy=False),
            ]
        )
        if coarse_windows.size > 0 or support_windows.size > 0
        else sample_ids.astype(np.int64, copy=False)
    )
    if focus_windows.size >= 2:
        fit_states = np.vstack([state_bank[focus_windows], query_state[None, :]])
        mean_vec, components, explained = fit_pca_basis(fit_states)
        background_proj = project_pca(state_bank[background_ids], mean_vec, components)

    coarse_proj = (
        project_pca(state_bank[coarse_windows], mean_vec, components)
        if coarse_windows.size > 0
        else np.zeros((0, 2), dtype=np.float64)
    )
    support_proj = (
        project_pca(state_bank[support_windows], mean_vec, components)
        if support_windows.size > 0
        else np.zeros((0, 2), dtype=np.float64)
    )
    query_proj = project_pca(query_state[None, :], mean_vec, components)[0]

    final_score_context = np.asarray(panel["curves"]["cdf_max_score"], dtype=np.float64)

    payload = {
        "panel": panel,
        "query_start": int(query_start),
        "query_end": int(query_end),
        "query_patch_size": patch_size,
        "query_patch_mask": patch_labels.astype(bool),
        "background_proj": background_proj,
        "coarse_proj": coarse_proj,
        "support_proj": support_proj,
        "query_proj": query_proj,
        "explained": explained,
        "heatmap": heatmap,
        "counts": {
            "Memory Windows": int(len(state_bank)),
            "State Top-M": int(len(coarse_windows)),
            "Patch Top-K": int(config.top_K),
        },
        "final_score_context": final_score_context,
    }

    del x_query, encoded, memory_out, model, memory, trainer
    gc.collect()
    return payload


def draw_query_signal_axis(
    ax: plt.Axes,
    payload: dict[str, object],
    *,
    show_title: bool,
    show_row_label: bool,
    show_xlabel: bool,
) -> None:
    panel = dict(payload["panel"])
    x = np.asarray(panel["x"], dtype=np.int64)
    signal = np.asarray(panel["signal"], dtype=np.float64)

    style_example_axis(ax)
    add_target_event_spans(ax, x, list(panel["event_members"]), alpha=0.40)
    ax.plot(x, signal, color=EXAMPLE_SIGNAL_COLOR, linewidth=EXAMPLE_SIGNAL_LINEWIDTH, zorder=2)
    query_patch = ax.axvspan(
        float(payload["query_start"]),
        float(int(payload["query_end"]) - 1),
        facecolor=RETRIEVAL_QUERY_FILL,
        edgecolor=RETRIEVAL_QUERY_EDGE,
        linewidth=1.2,
        alpha=0.26,
        zorder=1,
    )
    query_patch.set_linestyle((0, (4, 2)))
    ax.tick_params(axis="y", labelsize=8, length=2.5, colors="#444444")
    ax.tick_params(axis="x", labelsize=8, length=2.0, colors="#555555")
    if show_title:
        ax.set_title("Query Signal", fontsize=12.5, fontweight="bold", pad=6)
    if show_row_label:
        ax.text(
            MECHANISM_ROW_LABEL_X,
            0.5,
            str(panel["title"]),
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#2F2F2F",
        )
        ax.text(
            0.03,
            0.06,
            f"{panel['dataset_key']} / ch{int(panel['channel']) + 1}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#5A5A5A",
        )
    if show_xlabel:
        ax.set_xlabel("Time Index", fontsize=9, labelpad=3)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def draw_state_scatter_axis(
    ax: plt.Axes,
    payload: dict[str, object],
    *,
    show_title: bool,
    show_xlabel: bool,
    show_legend: bool,
) -> None:
    style_example_axis(ax)
    bg = np.asarray(payload["background_proj"], dtype=np.float64)
    coarse = np.asarray(payload["coarse_proj"], dtype=np.float64)
    support = np.asarray(payload["support_proj"], dtype=np.float64)
    query = np.asarray(payload["query_proj"], dtype=np.float64)
    explained = np.asarray(payload["explained"], dtype=np.float64)

    if bg.size > 0:
        ax.scatter(bg[:, 0], bg[:, 1], s=10, alpha=0.22, color=SCATTER_BACKGROUND, edgecolors="none")
    if coarse.size > 0:
        ax.scatter(
            coarse[:, 0],
            coarse[:, 1],
            s=94,
            alpha=0.42,
            facecolors=SCATTER_COARSE,
            color=SCATTER_COARSE,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
    if support.size > 0:
        ax.scatter(
            support[:, 0],
            support[:, 1],
            s=9,
            alpha=0.90,
            color=SCATTER_SUPPORT,
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )
    ax.scatter(
        [query[0]],
        [query[1]],
        s=140,
        marker="*",
        color=SCATTER_QUERY,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    ax.tick_params(axis="both", labelsize=7, length=2.0, colors="#555555")
    if show_title:
        ax.set_title("Two-Level Retrieval Map", fontsize=12.5, fontweight="bold", pad=6)
    ax.text(
        0.03,
        0.97,
        f"local PCA {100 * float(explained.sum()):.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#5F5F5F",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
    )
    if show_xlabel:
        ax.set_xlabel("PC1", fontsize=9, labelpad=3)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=SCATTER_COARSE, markeredgecolor="white", markeredgewidth=0.7, alpha=0.42, markersize=9.2, label="State Top-M"),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=SCATTER_SUPPORT, markeredgecolor="white", markeredgewidth=0.35, markersize=3.9, label="Patch Supports"),
            Line2D([0], [0], marker="*", linestyle="None", markerfacecolor=SCATTER_QUERY, markeredgecolor="white", markersize=11, label="Query State"),
        ]
        legend = ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.90,
            fontsize=8,
            borderpad=0.35,
            handletextpad=0.4,
        )
        legend.get_frame().set_edgecolor("#D8D8D8")
        legend.get_frame().set_linewidth(0.8)


def draw_patch_heatmap_axis(
    ax: plt.Axes,
    payload: dict[str, object],
    *,
    show_title: bool,
    show_xlabel: bool,
) -> None:
    style_example_axis(ax)
    heatmap = np.asarray(payload["heatmap"], dtype=np.float64)
    patch_mask = np.asarray(payload["query_patch_mask"], dtype=bool)
    ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=HEATMAP_CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    for start, end in contiguous_regions(patch_mask):
        ax.axvspan(start - 0.5, end + 0.5, color=EXAMPLE_ANOMALY_COLOR, alpha=0.22, zorder=2)
    ax.set_yticks([0, max(0, heatmap.shape[0] // 2), max(0, heatmap.shape[0] - 1)])
    ax.set_yticklabels(
        [1, max(1, heatmap.shape[0] // 2 + 1), heatmap.shape[0]],
        fontsize=7,
    )
    n_patches = heatmap.shape[1]
    xticks = np.linspace(0, max(0, n_patches - 1), num=min(5, n_patches), dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(tick) + 1) for tick in xticks], fontsize=7)
    ax.tick_params(axis="both", length=2.0, colors="#555555")
    if show_title:
        ax.set_title("Level-2 Patch Support", fontsize=12.5, fontweight="bold", pad=6)
    ax.text(
        0.03,
        0.97,
        f"patch={int(payload['query_patch_size'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#5F5F5F",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
    )
    if show_xlabel:
        ax.set_xlabel("Query Patch Index", fontsize=9, labelpad=3)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Neighbor Rank", fontsize=9, labelpad=4)


def draw_candidate_funnel_axis(
    ax: plt.Axes,
    payload: dict[str, object],
    *,
    show_title: bool,
) -> None:
    style_example_axis(ax)
    counts_dict = dict(payload["counts"])
    labels = list(counts_dict.keys())
    values = np.array([max(1, int(counts_dict[label])) for label in labels], dtype=np.int64)
    colors = ["#AFAFAF", SCATTER_COARSE, "#D89A67"]
    positions = np.arange(len(labels), dtype=np.float64)

    ax.barh(positions, values, color=colors[: len(labels)], alpha=0.90, height=0.58)
    ax.set_xscale("log")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=7, length=2.0, colors="#555555")
    if show_title:
        ax.set_title("Two-Stage Funnel", fontsize=12.5, fontweight="bold", pad=6)
    for pos, raw_value in zip(positions, [int(counts_dict[label]) for label in labels]):
        ax.text(
            max(raw_value, 1) * 1.15,
            pos,
            f"{raw_value}",
            va="center",
            ha="left",
            fontsize=8,
            color="#3F3F3F",
            fontweight="bold",
        )
    ax.grid(True, axis="x", linestyle=(0, (2, 2)), linewidth=0.7, color="#DEDEDE")


def draw_local_score_axis(
    ax: plt.Axes,
    payload: dict[str, object],
    *,
    show_xlabel: bool,
) -> None:
    panel = dict(payload["panel"])
    x = np.asarray(panel["x"], dtype=np.int64)
    score = np.asarray(payload["final_score_context"], dtype=np.float64)

    style_example_axis(ax)
    add_target_event_spans(ax, x, list(panel["event_members"]), alpha=0.35)
    query_patch = ax.axvspan(
        float(payload["query_start"]),
        float(int(payload["query_end"]) - 1),
        facecolor=RETRIEVAL_QUERY_FILL,
        edgecolor=RETRIEVAL_QUERY_EDGE,
        linewidth=1.0,
        alpha=0.20,
        zorder=1,
    )
    query_patch.set_linestyle((0, (4, 2)))
    ax.plot(
        x,
        score,
        color=INNER_RED_COLOR,
        linewidth=2.2,
        marker="o",
        markersize=4.0,
        markevery=max(1, len(x) // 9),
        markerfacecolor=INNER_RED_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=2,
    )
    ax.set_ylim(-0.02, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", labelsize=7, length=2.0, colors="#555555")
    ax.set_ylabel("Score", fontsize=9, labelpad=4)
    if show_xlabel:
        ax.set_xlabel("Time Index", fontsize=9, labelpad=3)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def plot_examples_with_radar(display_ready: dict, highlight_dataset: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    example_panels = load_example_panels()
    write_example_windows(example_panels)

    fig = plt.figure(figsize=(20.6, 9.4), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.95, 1.02],
        wspace=0.08,
    )
    left_grid = grid[0, 0].subgridspec(3, 4, wspace=0.18, hspace=0.28)
    radar_ax = fig.add_subplot(grid[0, 1])

    for row, panel in enumerate(example_panels):
        payload = build_retrieval_row_payload(panel)

        ax_query = fig.add_subplot(left_grid[row, 0])
        draw_query_signal_axis(
            ax_query,
            payload,
            show_title=(row == 0),
            show_row_label=True,
            show_xlabel=(row == len(example_panels) - 1),
        )

        ax_scatter = fig.add_subplot(left_grid[row, 1])
        draw_state_scatter_axis(
            ax_scatter,
            payload,
            show_title=(row == 0),
            show_xlabel=(row == len(example_panels) - 1),
            show_legend=(row == 0),
        )

        ax_heatmap = fig.add_subplot(left_grid[row, 2])
        draw_patch_heatmap_axis(
            ax_heatmap,
            payload,
            show_title=(row == 0),
            show_xlabel=(row == len(example_panels) - 1),
        )

        col4 = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=left_grid[row, 3],
            height_ratios=[1.0, 1.18],
            hspace=0.16,
        )
        ax_funnel = fig.add_subplot(col4[0, 0])
        draw_candidate_funnel_axis(
            ax_funnel,
            payload,
            show_title=(row == 0),
        )

        ax_score = fig.add_subplot(col4[1, 0], sharex=ax_query)
        draw_local_score_axis(
            ax_score,
            payload,
            show_xlabel=(row == len(example_panels) - 1),
        )

    left_legend = fig.legend(
        handles=[
            Patch(facecolor=EXAMPLE_ANOMALY_COLOR, edgecolor="none", alpha=0.55, label="Ground Truth Anomaly"),
            Patch(facecolor=RETRIEVAL_QUERY_FILL, edgecolor=RETRIEVAL_QUERY_EDGE, linewidth=1.0, alpha=0.45, label="Retrieval Query Window"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.315, 0.045),
        frameon=False,
        ncol=2,
        fontsize=10,
        handlelength=2.4,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    for text in left_legend.get_texts():
        text.set_fontfamily("serif")

    draw_radar(
        radar_ax,
        display_ready,
        highlight_dataset,
        legend_anchor=(0.5, -0.01),
        legend_font_size=12,
        legend_box=True,
    )
    radar_ax.set_xlim(-1.34, 1.34)
    radar_ax.set_ylim(-1.30, 1.34)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.965, bottom=0.10)
    fig.savefig(OUTPUT_COMPOSITE_PNG, dpi=500, facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT_COMPOSITE_PDF, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_aggregated_json(
    display_ready: dict, metric_ranges: dict, highlight_dataset: str
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_order": [{key: label} for key, label in METRICS],
        "aggregation": "mean_of_two_runs_per_synthetic_type",
        "score_source": "cdf_max",
        "normalization": {
            "method": "min_max_per_metric_across_six_synthetic_types",
            "constant_metric_fallback": 0.5,
            "metric_ranges": metric_ranges,
        },
        "display_mapping": {
            "method": "radius_floor_after_min_max_normalization",
            "display_radius_floor": DISPLAY_RADIUS_FLOOR,
            "formula": "display_radius = floor + (1 - floor) * normalized_value",
        },
        "highlight_rule": {
            "method": "largest_polygon_area_on_display_radar",
            "highlighted_dataset": highlight_dataset,
            "highlight_color": HIGHLIGHT_COLOR,
        },
        "datasets": display_ready,
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    summary = load_summary(SUMMARY_PATH)
    aggregated = aggregate_metrics(summary)
    normalized, metric_ranges = normalize_metrics(aggregated)
    display_ready = apply_display_radius_floor(normalized)
    angles = np.pi / 2 - np.arange(len(METRICS)) * 2 * np.pi / len(METRICS)
    highlight_dataset = determine_highlight_dataset(display_ready, angles)
    write_aggregated_json(display_ready, metric_ranges, highlight_dataset)
    plot_radar(display_ready, highlight_dataset)
    plot_examples_with_radar(display_ready, highlight_dataset)
    print(f"Saved averaged metrics to: {OUTPUT_JSON}")
    print(f"Saved example window metadata to: {OUTPUT_EXAMPLE_JSON}")
    print(f"Saved radar chart PNG to: {OUTPUT_PNG}")
    print(f"Saved radar chart PDF to: {OUTPUT_PDF}")
    print(f"Saved composite radar+examples PNG to: {OUTPUT_COMPOSITE_PNG}")
    print(f"Saved composite radar+examples PDF to: {OUTPUT_COMPOSITE_PDF}")


if __name__ == "__main__":
    main()
