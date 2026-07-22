from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.io import loadmat

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_paper_mechanism_figures import (  # noqa: E402
    EVIDENCE_KEYS,
    EVIDENCE_STYLE,
    MODE_ORDER,
    MODE_STYLE,
    SUMMARY_METRICS,
    _add_panel_label,
    _build_fault_evidence_summary,
    _covariance_ellipse,
    _compute_retrieval_heat,
    _finite,
    _finite_positive,
    _format_float,
    _load_metrics,
    _reduce_state_embedding,
    _save_figure,
    _style_axis,
    configure_paper_style,
    copy_metadata_files,
    plot_evidence_summary_axes,
    plot_modewise_exceedance_axis,
    plot_normal_violin_axis,
)
from plot_mechanism_figures import (  # noqa: E402
    EVIDENCE_LABELS,
    _calibrate_fault_sequence_rows,
    _load_experiment_payload,
)
from tep_common import empirical_percentile, load_train_state_meta  # noqa: E402


RETRIEVAL_VARIANTS = {
    "Global retrieval": "tep_ablation_global_retrieval_20260419_123135",
    "State-only retrieval": "tep_ablation_state_only_retrieval_20260419_123135",
    "Context-only retrieval": "tep_ablation_context_only_retrieval_20260419_123135",
    "Full two-stage retrieval": "tep_ablation_full_20260418_193550",
}

SCALE_VARIANTS = {
    "Single short scale": "tep_ablation_single_scale_short_20260418_195059",
    "Single long scale": "tep_ablation_single_scale_long_20260418_200322",
    "Multi-scale": "tep_ablation_multi_scale_20260418_201442",
}

RETRIEVAL_METRICS = [
    ("SMR@K", "higher", "Same-mode retrieval ratio"),
    ("delta_mem_mode", "higher", "Fault-normal gap"),
    ("Mode-FPR-Std", "lower", "Mode-wise FPR std"),
    ("Evidence-Dom-Consistency", "higher", "Evidence consistency"),
]

SCALE_METRICS = [
    ("SMR@K", "higher", "Same-mode retrieval ratio"),
    ("delta_mem_mode", "higher", "Fault-normal gap"),
    ("Mode-FPR-Std", "lower", "Mode-wise FPR std"),
    ("SFR", "higher", "State separation ratio"),
]

CASE_COLORS = {
    "query": "#EE7875",
    "ref1": "#7EA8F8",
    "ref2": "#F4B36A",
    "ref3": "#A8D39B",
}

SOFT_BLUE = "#D8E5FF"
SOFT_RED = "#EE7875"
SOFT_ORANGE = "#F4B36A"
SOFT_GREEN = "#A8D39B"
MUTED_BLUE = "#5E7FC4"
MUTED_RED = "#D9666C"
MUTED_ORANGE = "#E8A65F"
MUTED_GREEN = "#87BC7F"
PALE_TEXT = "#6A6A6A"
PALE_EDGE = "#D8DDE7"
NOTE_BLACK = "#111111"

METADATA_FILES = [
    "mechanism_metrics.json",
    "mechanism_metrics.csv",
    "export_meta.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a complete TEP mechanism figure suite for the paper and appendix.",
    )
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_subdir", type=str, default="tep_mechanism")
    parser.add_argument("--embedding_method", type=str, default="tsne", choices=["pca", "tsne"])
    parser.add_argument("--raw_data_dir", type=Path, default=REPO_ROOT / "TEP-DATA" / "TEP_Selected_Data")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--case_top_k", type=int, default=3)
    parser.add_argument("--num_case_studies", type=int, default=4)
    return parser.parse_args()


def _variant_dir_map(experiment_dir: Path, mapping: dict[str, str]) -> dict[str, Path]:
    base_dir = experiment_dir.parent
    resolved: dict[str, Path] = {}
    for label, dirname in mapping.items():
        path = base_dir / dirname
        if path.exists():
            resolved[label] = path
    return resolved


def _friendly_metric_name(metric: str) -> str:
    for key, _, label in RETRIEVAL_METRICS + SCALE_METRICS:
        if key == metric:
            return label
    return metric


def _metric_value_to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    lines = [",".join(header)]
    for row in rows:
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.8f}")
            else:
                vals.append(str(value))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_panel_footer(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.5,
        -0.24,
        f"{label}. {title}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=NOTE_BLACK,
    )


def _add_metric_box(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        color=NOTE_BLACK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "boxstyle": "round,pad=0.18"},
    )


def _enlarge_main_axis_text(ax: plt.Axes) -> None:
    ax.xaxis.label.set_size(13.2)
    ax.yaxis.label.set_size(13.2)
    ax.tick_params(axis="both", labelsize=11.2)


def _stacked_retrieval_matrix(heat: np.ndarray) -> np.ndarray:
    row_sum = np.sum(heat, axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return heat / row_sum


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(_finite(values))
    if arr.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    y = np.arange(1, arr.size + 1, dtype=np.float64) / float(arr.size)
    return arr, y


def _load_raw_sequence(raw_data_dir: Path, file_id: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if file_id not in cache:
        path = raw_data_dir / f"{file_id}.mat"
        payload = loadmat(path)
        cache[file_id] = np.asarray(payload[file_id], dtype=np.float64)
    return cache[file_id]


def _dominant_mode(row_modes: np.ndarray) -> int:
    valid = [int(v) for v in np.asarray(row_modes, dtype=np.int32).reshape(-1).tolist() if int(v) >= 0]
    if not valid:
        return -1
    counts = Counter(valid)
    return min(counts.keys(), key=lambda key: (-counts[key], key))


def _case_selection_score(row: dict[str, Any]) -> float:
    values = sorted(
        (_metric_value_to_float(row.get(key, float("nan"))) for key in EVIDENCE_KEYS),
        reverse=True,
    )
    if not values:
        return float("-inf")
    if len(values) == 1:
        return values[0]
    if values[0] >= 0.999 and values[1] >= 0.999:
        return -0.1
    return values[0] - values[1]


def _select_case_sequences(summary_rows: list[dict[str, Any]], max_cases: int) -> list[int]:
    rows_by_dominant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_dominant[str(row["dominant_evidence"])].append(row)

    selected: list[int] = []
    preferred = ["state_novelty", "completion_scale32", "completion_scale8", "memory_distance"]
    for dominant in preferred:
        candidates = rows_by_dominant.get(dominant, [])
        if not candidates:
            continue
        best = max(candidates, key=_case_selection_score)
        fault_id = int(best["fault_id"])
        if fault_id not in selected:
            selected.append(fault_id)

    if len(selected) < max_cases:
        remaining = sorted(summary_rows, key=_case_selection_score, reverse=True)
        for row in remaining:
            fault_id = int(row["fault_id"])
            if fault_id not in selected:
                selected.append(fault_id)
            if len(selected) >= max_cases:
                break
    return selected[:max_cases]


def _channel_rankings(query: np.ndarray, refs: list[np.ndarray], num_channels: int = 2) -> list[int]:
    if not refs:
        return list(range(min(num_channels, query.shape[1])))
    ref_mean = np.mean(np.stack(refs, axis=0), axis=0)
    score = np.mean(np.abs(query - ref_mean), axis=0)
    order = np.argsort(score)[::-1]
    return order[:num_channels].tolist()


def _standardize_traces(traces: list[np.ndarray]) -> list[np.ndarray]:
    stacked = np.concatenate([trace.reshape(-1, 1) for trace in traces], axis=1)
    mean = np.mean(stacked)
    std = np.std(stacked)
    if not np.isfinite(std) or std <= 1e-8:
        std = 1.0
    return [((trace - mean) / std).astype(np.float64) for trace in traces]


def _choose_sequence_for_fault(payload: dict[str, Any], fault_id: int) -> tuple[str, int]:
    logs = payload["fault_logs"]
    file_ids = np.asarray(logs["file_id"], dtype=object)
    fault_ids = np.asarray(logs["fault_id"], dtype=np.int32)
    final_scores = np.asarray(logs["final"], dtype=np.float64)
    mask = fault_ids == int(fault_id)
    if not np.any(mask):
        raise KeyError(f"Could not find windows for fault_id={fault_id}")
    local_idx = np.where(mask)[0]
    best_idx = int(local_idx[np.nanargmax(final_scores[mask])])
    return str(file_ids[best_idx]), best_idx


def _build_case_panels(
    payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    raw_data_dir: Path,
    train_state_meta: dict[str, np.ndarray],
    case_top_k: int,
    num_cases: int,
) -> list[dict[str, Any]]:
    selected_faults = _select_case_sequences(summary_rows, max_cases=num_cases)
    logs = payload["fault_logs"]
    starts = np.asarray(logs["start"], dtype=np.int64)
    neighbor_ids = np.asarray(logs["topk_neighbor_window_ids"], dtype=np.int64)
    neighbor_dists = np.asarray(logs["topk_neighbor_distances"], dtype=np.float64)
    mode_ids = np.asarray(logs["mode_id"], dtype=np.int32)
    final_scores = np.asarray(logs["final"], dtype=np.float64)

    train_lookup: dict[int, dict[str, Any]] = {}
    train_window_ids = np.asarray(train_state_meta["window_id"], dtype=np.int64)
    train_starts = np.asarray(train_state_meta["start"], dtype=np.int64)
    train_modes = np.asarray(train_state_meta["mode_id"], dtype=np.int32)
    train_files = np.asarray(train_state_meta["file_id"], dtype=object)
    file_offsets: dict[str, int] = {}
    for file_id in sorted(set(str(file_name) for file_name in train_files.tolist())):
        local_mask = train_files == file_id
        file_offsets[file_id] = int(np.min(train_starts[local_mask])) if np.any(local_mask) else 0
    for window_id, start, mode_id, file_id in zip(
        train_window_ids.tolist(),
        train_starts.tolist(),
        train_modes.tolist(),
        train_files.tolist(),
    ):
        file_id_str = str(file_id)
        local_start = int(start) - int(file_offsets.get(file_id_str, 0))
        train_lookup[int(window_id)] = {
            "start": local_start,
            "mode_id": int(mode_id),
            "file_id": file_id_str,
        }

    raw_cache: dict[str, np.ndarray] = {}
    case_panels: list[dict[str, Any]] = []
    summary_map = {int(row["fault_id"]): row for row in summary_rows}

    for fault_id in selected_faults:
        file_id, best_idx = _choose_sequence_for_fault(payload, fault_id)
        query_start = int(starts[best_idx])
        query_mode = int(mode_ids[best_idx])
        query_score = float(final_scores[best_idx])
        query_data = _load_raw_sequence(raw_data_dir, file_id, raw_cache)
        query_window = query_data[query_start : query_start + 128]

        refs: list[np.ndarray] = []
        ref_meta: list[dict[str, Any]] = []
        for neighbor_id, distance in zip(neighbor_ids[best_idx].tolist(), neighbor_dists[best_idx].tolist()):
            if int(neighbor_id) < 0 or not np.isfinite(float(distance)):
                continue
            meta = train_lookup.get(int(neighbor_id))
            if meta is None:
                continue
            ref_data = _load_raw_sequence(raw_data_dir, meta["file_id"], raw_cache)
            ref_window = ref_data[meta["start"] : meta["start"] + 128]
            if ref_window.shape[0] != 128:
                continue
            refs.append(ref_window)
            ref_meta.append(
                {
                    "window_id": int(neighbor_id),
                    "distance": float(distance),
                    "mode_id": int(meta["mode_id"]),
                    "file_id": str(meta["file_id"]),
                    "start": int(meta["start"]),
                }
            )
            if len(refs) >= case_top_k:
                break
        if query_window.shape[0] != 128 or not refs:
            continue

        channel_ids = _channel_rankings(query_window, refs, num_channels=2)
        dominant = str(summary_map.get(int(fault_id), {}).get("dominant_evidence", "unknown"))
        case_panels.append(
            {
                "fault_id": int(fault_id),
                "file_id": file_id,
                "query_mode": query_mode,
                "query_start": query_start,
                "query_score": query_score,
                "query_window": query_window,
                "refs": refs,
                "ref_meta": ref_meta,
                "channel_ids": channel_ids,
                "dominant_evidence": dominant,
            }
        )
    return case_panels


def _representative_sequence_id(summary_rows: list[dict[str, Any]]) -> int:
    preferred_fault_order = [10, 4, 27, 14, 7, 1]
    available = {int(row["fault_id"]) for row in summary_rows}
    for fault_id in preferred_fault_order:
        if fault_id in available:
            return fault_id
    return int(summary_rows[0]["fault_id"])


def _plot_retrieval_mode_consistency(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    top_k: int,
) -> list[Path]:
    heat = _compute_retrieval_heat(payload, top_k=top_k)
    fig = plt.figure(figsize=(10.5, 4.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.15], wspace=0.28)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_stack = fig.add_subplot(gs[0, 1])

    im = ax_heat.imshow(heat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="equal")
    ax_heat.set_xticks(np.arange(len(MODE_ORDER)), [f"Mode {mode}" for mode in MODE_ORDER])
    ax_heat.set_yticks(np.arange(len(MODE_ORDER)), [f"Mode {mode}" for mode in MODE_ORDER])
    ax_heat.set_xlabel("Retrieved reference mode")
    ax_heat.set_ylabel("Query mode")
    ax_heat.set_title("Mode-consistent retrieval heatmap", loc="left", fontweight="bold")
    ax_heat.set_xticks(np.arange(-0.5, len(MODE_ORDER), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(MODE_ORDER), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.1)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = float(heat[row_idx, col_idx])
            ax_heat.text(
                col_idx,
                row_idx,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=9.3,
                fontweight="bold",
                color="white" if value >= 0.56 else "#1f1f1f",
            )
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    _add_panel_label(ax_heat, "A")

    _style_axis(ax_stack, grid_axis="y")
    proportions = _stacked_retrieval_matrix(heat)
    x = np.arange(len(MODE_ORDER), dtype=np.float64)
    bottom = np.zeros(len(MODE_ORDER), dtype=np.float64)
    for col_idx, retrieved_mode in enumerate(MODE_ORDER):
        vals = proportions[:, col_idx]
        ax_stack.bar(
            x,
            vals,
            bottom=bottom,
            width=0.62,
            color=MODE_STYLE[retrieved_mode]["color"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label=f"Retrieved Mode {retrieved_mode}",
        )
        bottom += vals
    ax_stack.set_xticks(x, [f"Query Mode {mode}" for mode in MODE_ORDER])
    ax_stack.set_ylim(0.0, 1.03)
    ax_stack.set_ylabel("Share within top-K references")
    ax_stack.set_title("Top-K retrieved mode composition", loc="left", fontweight="bold")
    ax_stack.legend(loc="upper center", bbox_to_anchor=(0.52, 1.11), ncol=3, frameon=False)
    ax_stack.text(
        0.02,
        0.98,
        f"SMR@K = {_format_float(metrics.get('SMR@K'), 4)}",
        transform=ax_stack.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.94, "boxstyle": "round,pad=0.26"},
    )
    for idx, value in enumerate(np.diag(proportions)):
        ax_stack.text(idx, min(0.985, value + 0.02), f"{value:.0%}", ha="center", va="bottom", fontsize=8.6)
    _add_panel_label(ax_stack, "B")

    fig.suptitle("Retrieved local references align with the true operating mode", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    return _save_figure(fig, output_dir / "02_retrieval_mode_consistency.png")


def _plot_gap_core_axis(
    ax: plt.Axes,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    panel_label: str | None = None,
) -> None:
    audit_logs = payload["audit_logs"]
    fault_logs = payload["fault_logs"]
    audit_memory = _finite_positive(np.asarray(audit_logs["memory_distance"], dtype=np.float64))
    fault_memory = _finite_positive(np.asarray(fault_logs["memory_distance"], dtype=np.float64))

    _style_axis(ax, grid_axis="y")
    parts = ax.violinplot([audit_memory, fault_memory], positions=[1, 2], showmedians=True, widths=0.78)
    for body, color in zip(parts["bodies"], [SOFT_BLUE, SOFT_RED]):
        body.set_facecolor(color)
        body.set_edgecolor("#4d4d4d")
        body.set_alpha(0.82)
        body.set_linewidth(0.8)
    parts["cmedians"].set_color("#202020")
    parts["cmedians"].set_linewidth(1.3)
    for key in ("cbars", "cmaxes", "cmins"):
        parts[key].set_color("#4d4d4d")
        parts[key].set_linewidth(0.8)
    ax.set_yscale("log")
    ax.set_xticks([1, 2], ["Audit\nnormal", "Fault"])
    ax.set_ylabel("Memory distance (log scale)")
    ax.set_title("Fault queries stay farther from compatible normal references", loc="left", fontweight="bold", pad=8)
    ax.text(
        0.02,
        0.98,
        f"Fault-normal gap = {_format_float(metrics.get('delta_mem_mode'), 2)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.94, "boxstyle": "round,pad=0.24"},
    )
    ax.text(
        0.98,
        0.08,
        "Distances are computed against retrieved\nmode-compatible normal references.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        color="#555555",
    )
    if panel_label:
        _add_panel_label(ax, panel_label)


def _plot_gap_ecdf_main_axis(
    ax: plt.Axes,
    payload: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    audit_logs = payload["audit_logs"]
    fault_logs = payload["fault_logs"]
    audit_memory = _finite_positive(np.asarray(audit_logs["memory_distance"], dtype=np.float64))
    fault_memory = _finite_positive(np.asarray(fault_logs["memory_distance"], dtype=np.float64))
    x_normal, y_normal = _ecdf(audit_memory)
    x_fault, y_fault = _ecdf(fault_memory)

    _style_axis(ax, grid_axis="both")
    ax.plot(x_normal, y_normal, color=MUTED_BLUE, linewidth=2.3, label="Audit normal")
    ax.plot(x_fault, y_fault, color=SOFT_RED, linewidth=2.3, label="Fault")
    if x_normal.size > 0:
        ax.axvline(float(np.median(x_normal)), color=MUTED_BLUE, linestyle=":", linewidth=1.0, alpha=0.9)
    if x_fault.size > 0:
        ax.axvline(float(np.median(x_fault)), color=SOFT_RED, linestyle=":", linewidth=1.0, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Memory distance (log scale)")
    ax.set_ylabel("ECDF")
    _enlarge_main_axis_text(ax)
    gap_handles = [
        Line2D([0], [0], color=MUTED_BLUE, linewidth=2.3, label="Audit normal"),
        Line2D([0], [0], color=SOFT_RED, linewidth=2.3, label="Fault"),
        Line2D([0], [0], color="#666666", linewidth=1.0, linestyle=":", label="Median distance"),
    ]
    ax.legend(
        handles=gap_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=10.2,
        handlelength=2.2,
        borderaxespad=0.0,
    )


def _plot_calibration_chain_axis(
    ax: plt.Axes,
    metrics: dict[str, Any],
) -> None:
    curve_payload = metrics.get("exceedance_curve", {})
    q_values = sorted(float(key) for key in curve_payload.keys())
    q_array = np.asarray(q_values, dtype=np.float64)
    theory = 1.0 - q_array

    _style_axis(ax, grid_axis="both")
    ax.plot(q_array, theory, color="#333333", linestyle="--", linewidth=1.8, label="Theory (1-q)")

    for mode_id in MODE_ORDER:
        y = np.asarray(
            [float(curve_payload.get(f"{q:.2f}", {}).get(str(mode_id), np.nan)) for q in q_values],
            dtype=np.float64,
        )
        ax.plot(
            q_array,
            y,
            color=MODE_STYLE[mode_id]["color"],
            linewidth=2.2,
            marker="o",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=f"Mode {mode_id}",
        )
    for mark_q in (0.95, 0.99):
        ax.axvline(mark_q, color="#C9CED8", linewidth=0.9, linestyle=":")
    ax.set_xlim(0.898, 0.9945)
    ax.set_xlabel("Global threshold quantile q")
    ax.set_ylabel("Pr(score > q-threshold | audit normal)")
    _enlarge_main_axis_text(ax)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        fontsize=9.3,
        title="Operating modes",
        title_fontsize=9.8,
        ncol=1,
        handlelength=2.0,
        columnspacing=1.0,
        borderaxespad=0.0,
    )


def _plot_main_overview_figure(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    embedding_method: str,
    top_k: int,
) -> list[Path]:
    audit_logs = payload["audit_logs"]
    audit_state = np.asarray(audit_logs["state_vec"], dtype=np.float64)
    audit_mode = np.asarray(audit_logs["mode_id"], dtype=np.int32)

    rng = np.random.default_rng(42)
    selected_indices = []
    for mode_id in MODE_ORDER:
        idx = np.where(audit_mode == mode_id)[0]
        if len(idx) > 820:
            idx = rng.choice(idx, size=820, replace=False)
        selected_indices.append(np.sort(idx))
    selected = np.concatenate(selected_indices, axis=0)
    coords = _reduce_state_embedding(audit_state[selected], embedding_method)
    heat = _compute_retrieval_heat(payload, top_k=top_k)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.3, 7.9),
        gridspec_kw={"width_ratios": [1.16, 1.0], "height_ratios": [1.0, 1.0], "wspace": 0.17, "hspace": 0.40},
    )
    ax_state, ax_retrieval, ax_gap, ax_calibration = axes.reshape(-1)

    _style_axis(ax_state, grid_axis="both")
    for mode_id in MODE_ORDER:
        mask = audit_mode[selected] == mode_id
        points = coords[mask]
        centroid = np.mean(points, axis=0)
        ax_state.scatter(
            points[:, 0],
            points[:, 1],
            s=9,
            color=MODE_STYLE[mode_id]["color"],
            alpha=0.22,
            linewidths=0.0,
            rasterized=True,
        )
        if len(points) >= 3:
            width, height, angle = _covariance_ellipse(points, n_std=2.25)
            ellipse = Ellipse(
                xy=centroid,
                width=width,
                height=height,
                angle=angle,
                facecolor=MODE_STYLE[mode_id]["color"],
                edgecolor=MODE_STYLE[mode_id]["color"],
                alpha=0.12,
                linewidth=1.15,
            )
            ax_state.add_patch(ellipse)
        ax_state.scatter(
            centroid[0],
            centroid[1],
            s=28,
            color=MODE_STYLE[mode_id]["color"],
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
    x_low, x_high = np.percentile(coords[:, 0], [0.5, 99.5])
    y_low, y_high = np.percentile(coords[:, 1], [0.5, 99.5])
    x_pad = max(0.08, float((x_high - x_low) * 0.08))
    y_pad = max(0.08, float((y_high - y_low) * 0.10))
    ax_state.set_xlim(float(x_low - x_pad), float(x_high + x_pad))
    ax_state.set_ylim(float(y_low - y_pad), float(y_high + y_pad))
    ax_state.set_xlabel("Embedding dim 1")
    ax_state.set_ylabel("Embedding dim 2")
    _enlarge_main_axis_text(ax_state)
    ax_state.xaxis.labelpad = 2
    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6.2,
            markerfacecolor=MODE_STYLE[mode_id]["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"Mode {mode_id}",
        )
        for mode_id in MODE_ORDER
    ]
    ax_state.legend(
        handles=state_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        fontsize=9.8,
        title="State modes",
        title_fontsize=10.0,
        handlelength=1.0,
        borderaxespad=0.0,
    )
    _add_panel_footer(
        ax_state,
        "A",
        f"Recover operating states (SMC@K={_format_float(metrics.get('SMC@K'), 4)}, SFR={_format_float(metrics.get('SFR'), 2)})",
    )

    heat_cmap = LinearSegmentedColormap.from_list(
        "main_retrieval_heat",
        ["#FAF7F0", "#F2F5DF", "#D9ED9B", "#7EA8F8", "#3750A2"],
    )
    ax_retrieval.imshow(heat, cmap=heat_cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax_retrieval.set_xticks(np.arange(len(MODE_ORDER)), [f"Mode {mode}" for mode in MODE_ORDER])
    ax_retrieval.set_yticks(np.arange(len(MODE_ORDER)), [f"Mode {mode}" for mode in MODE_ORDER])
    ax_retrieval.set_xlabel("Retrieved reference mode")
    ax_retrieval.set_ylabel("Query mode")
    _enlarge_main_axis_text(ax_retrieval)
    ax_retrieval.xaxis.labelpad = 2
    ax_retrieval.set_xticks(np.arange(-0.5, len(MODE_ORDER), 1), minor=True)
    ax_retrieval.set_yticks(np.arange(-0.5, len(MODE_ORDER), 1), minor=True)
    ax_retrieval.grid(which="minor", color="white", linewidth=1.35)
    ax_retrieval.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = float(heat[row_idx, col_idx])
            ax_retrieval.text(
                col_idx,
                row_idx,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=10.4,
                fontweight="bold" if row_idx == col_idx else "semibold",
                color="white" if value >= 0.52 else "#2F3640",
            )
    _add_panel_footer(
        ax_retrieval,
        "B",
        f"Retrieve mode-compatible references (SMR@K={_format_float(metrics.get('SMR@K'), 4)})",
    )

    ax_gap.xaxis.labelpad = 2
    _plot_gap_ecdf_main_axis(ax_gap, payload, metrics)
    _add_panel_footer(
        ax_gap,
        "C",
        f"Separate faults from compatible normals (Gap={_format_float(metrics.get('delta_mem_mode'), 2)})",
    )

    ax_calibration.xaxis.labelpad = 2
    _plot_calibration_chain_axis(ax_calibration, metrics)
    _add_panel_footer(
        ax_calibration,
        "D",
        (
            "Calibrate scores across modes "
            f"(FPR std={_format_float(metrics.get('Mode-FPR-Std'), 4)}, "
            f"q95={_format_float(metrics.get('EE95'), 4)}, "
            f"q99={_format_float(metrics.get('Tail@0.99_error'), 4)})"
        ),
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.17, wspace=0.16, hspace=0.38)
    saved = _save_figure(fig, output_dir / "00_main_mechanism_overview.png")
    alias_paths: list[Path] = []
    for path in saved:
        alias = path.with_name(path.name.replace("00_main_mechanism_overview", "00_main_mechanism_overview_publication"))
        alias.write_bytes(path.read_bytes())
        alias_paths.append(alias)
    return saved + alias_paths


def _plot_fault_normal_gap(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    audit_logs = payload["audit_logs"]
    fault_logs = payload["fault_logs"]
    audit_mode = np.asarray(audit_logs["mode_id"], dtype=np.int32)
    fault_mode = np.asarray(fault_logs["mode_id"], dtype=np.int32)
    audit_memory = np.asarray(audit_logs["memory_distance"], dtype=np.float64)
    fault_memory = np.asarray(fault_logs["memory_distance"], dtype=np.float64)
    fault_ids = np.asarray(fault_logs["fault_id"], dtype=np.int32)

    fig = plt.figure(figsize=(13.2, 4.7))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1.0, 1.05], wspace=0.28)
    ax_violin = fig.add_subplot(gs[0, 0])
    ax_ecdf = fig.add_subplot(gs[0, 1])
    ax_fault = fig.add_subplot(gs[0, 2])

    _style_axis(ax_violin, grid_axis="y")
    positions = []
    data = []
    colors = []
    labels = []
    for mode_idx, mode_id in enumerate(MODE_ORDER):
        normal = _finite_positive(audit_memory[audit_mode == mode_id])
        fault = _finite_positive(fault_memory[fault_mode == mode_id])
        base = mode_idx * 3
        positions.extend([base + 1, base + 2])
        data.extend([normal, fault])
        colors.extend([SOFT_BLUE, MODE_STYLE[mode_id]["color"]])
        labels.extend([f"M{mode_id}\nNormal", f"M{mode_id}\nFault"])
    parts = ax_violin.violinplot(data, positions=positions, showmedians=True, widths=0.82)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#4d4d4d")
        body.set_alpha(0.8)
        body.set_linewidth(0.8)
    parts["cmedians"].set_color("#202020")
    parts["cmedians"].set_linewidth(1.2)
    for key in ("cbars", "cmaxes", "cmins"):
        parts[key].set_color("#4d4d4d")
        parts[key].set_linewidth(0.8)
    ax_violin.set_yscale("log")
    ax_violin.set_xticks(positions, labels)
    ax_violin.set_ylabel("Memory distance (log scale)")
    ax_violin.set_title("Gap by operating mode", loc="left", fontweight="bold")
    ax_violin.text(
        0.02,
        0.98,
        f"Fault-normal gap = {_format_float(metrics.get('delta_mem_mode'), 2)}",
        transform=ax_violin.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.94, "boxstyle": "round,pad=0.26"},
    )
    _add_panel_label(ax_violin, "A")

    _style_axis(ax_ecdf, grid_axis="both")
    x_normal, y_normal = _ecdf(_finite_positive(audit_memory))
    x_fault, y_fault = _ecdf(_finite_positive(fault_memory))
    ax_ecdf.plot(x_normal, y_normal, color=MUTED_BLUE, linewidth=2.0, label="Audit normal")
    ax_ecdf.plot(x_fault, y_fault, color=MUTED_RED, linewidth=2.0, label="Fault")
    if x_normal.size > 0 and x_fault.size > 0:
        ax_ecdf.axvline(float(np.median(x_normal)), color=MUTED_BLUE, linestyle=":", linewidth=0.9)
        ax_ecdf.axvline(float(np.median(x_fault)), color=MUTED_RED, linestyle=":", linewidth=0.9)
    ax_ecdf.set_xscale("log")
    ax_ecdf.set_xlabel("Memory distance")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.set_title("Global separation distribution", loc="left", fontweight="bold")
    ax_ecdf.legend(frameon=False, loc="lower right")
    _add_panel_label(ax_ecdf, "B")

    _style_axis(ax_fault, grid_axis="x")
    fault_rows = []
    for fault_id in sorted(np.unique(fault_ids).tolist()):
        values = _finite_positive(fault_memory[fault_ids == fault_id])
        if values.size == 0:
            continue
        fault_rows.append((int(fault_id), float(np.median(values))))
    fault_rows.sort(key=lambda item: item[1], reverse=True)
    y_pos = np.arange(len(fault_rows), dtype=np.float64)
    ax_fault.barh(
        y_pos,
        [row[1] for row in fault_rows],
        color="#A5B6D8",
        edgecolor="#6E86B3",
        height=0.72,
    )
    ax_fault.axvline(float(np.median(_finite_positive(audit_memory))), color=SOFT_RED, linestyle="--", linewidth=1.0)
    ax_fault.set_yticks(y_pos, [f"d{row[0]:02d}" for row in fault_rows])
    ax_fault.invert_yaxis()
    ax_fault.set_xlabel("Median distance")
    ax_fault.set_title("Per-fault median distance", loc="left", fontweight="bold")
    _add_panel_label(ax_fault, "C")

    fig.suptitle("Fault queries stay farther from compatible normal references", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.94])
    return _save_figure(fig, output_dir / "03_fault_normal_gap_distributions.png")


def _modewise_fpr_from_logs(final_scores: np.ndarray, mode_ids: np.ndarray, quantiles: list[float]) -> dict[float, dict[int, float]]:
    out: dict[float, dict[int, float]] = {}
    for q in quantiles:
        threshold = float(np.quantile(final_scores, q))
        out[q] = {}
        for mode_id in MODE_ORDER:
            mask = mode_ids == mode_id
            out[q][mode_id] = float(np.mean(final_scores[mask] > threshold)) if np.any(mask) else float("nan")
    return out


def _plot_cross_mode_calibration(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    audit_logs = payload["audit_logs"]
    mode_ids = np.asarray(audit_logs["mode_id"], dtype=np.int32)
    final_scores = np.asarray(audit_logs["final"], dtype=np.float64)
    fpr_rows = _modewise_fpr_from_logs(final_scores, mode_ids, quantiles=[0.95, 0.99])

    fig = plt.figure(figsize=(13.0, 4.7))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.95, 0.95, 1.35], wspace=0.28)
    ax_violin = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[0, 2])

    plot_normal_violin_axis(ax_violin, payload, metrics)
    _add_panel_label(ax_violin, "A")

    _style_axis(ax_bar, grid_axis="y")
    x = np.arange(len(MODE_ORDER), dtype=np.float64)
    width = 0.32
    colors = {0.95: MUTED_BLUE, 0.99: SOFT_RED}
    for offset_idx, q in enumerate([0.95, 0.99]):
        vals = np.asarray([fpr_rows[q][mode_id] for mode_id in MODE_ORDER], dtype=np.float64)
        ax_bar.bar(
            x + (offset_idx - 0.5) * width,
            vals,
            width=width,
            color=colors[q],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.8,
            label=f"Global q={q:.2f}",
        )
        for idx, value in enumerate(vals):
            ax_bar.text(
                x[idx] + (offset_idx - 0.5) * width,
                value + 0.002,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )
    ax_bar.set_xticks(x, [f"Mode {mode_id}" for mode_id in MODE_ORDER])
    ax_bar.set_ylabel("False positive rate")
    ax_bar.set_title("FPR under q=0.95 and q=0.99", loc="left", fontweight="bold", pad=10)
    ax_bar.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=1, borderaxespad=0.0)
    ax_bar.text(
        0.02,
        0.98,
        f"Mode-FPR std = {_format_float(metrics.get('Mode-FPR-Std'), 4)}",
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.94, "boxstyle": "round,pad=0.26"},
    )
    _add_panel_label(ax_bar, "B")

    plot_modewise_exceedance_axis(ax_curve, metrics)
    _add_panel_label(ax_curve, "C")

    fig.suptitle("Final anomaly scores remain calibrated across operating regimes", fontsize=12.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    return _save_figure(fig, output_dir / "04_cross_mode_calibration.png")


def _plot_fault_evidence_channels(
    summary_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(10.2, 5.3))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4.8, 1.85], wspace=0.26)
    sub_gs = gs[0, 0].subgridspec(1, 2, width_ratios=[5.0, 1.45], wspace=0.08)
    ax_heat = fig.add_subplot(sub_gs[0, 0])
    ax_std = fig.add_subplot(sub_gs[0, 1], sharey=ax_heat)
    ax_dom = fig.add_subplot(gs[0, 1])

    plot_evidence_summary_axes(ax_heat, ax_std, summary_rows, metrics, show_label_note=True)
    _add_panel_label(ax_heat, "A")

    counts = Counter(str(row["dominant_evidence"]) for row in summary_rows)
    order = [key for key in EVIDENCE_KEYS if counts.get(key, 0) > 0]
    _style_axis(ax_dom, grid_axis="x")
    y_pos = np.arange(len(order), dtype=np.float64)
    ax_dom.barh(
        y_pos,
        [counts[key] for key in order],
        color=[EVIDENCE_STYLE[key]["color"] for key in order],
        edgecolor="#47515e",
        height=0.68,
    )
    ax_dom.set_yticks(y_pos, [EVIDENCE_LABELS[key] for key in order])
    ax_dom.invert_yaxis()
    ax_dom.set_xlabel("Fault count")
    ax_dom.set_title("Dominant channel count", loc="left", fontweight="bold")
    for idx, key in enumerate(order):
        ax_dom.text(counts[key] + 0.08, idx, str(counts[key]), va="center", ha="left", fontsize=8.5)
    ax_dom.text(
        0.02,
        0.98,
        f"Consistency = {_format_float(metrics.get('Evidence-Dom-Consistency'), 4)}",
        transform=ax_dom.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.94, "boxstyle": "round,pad=0.26"},
    )
    _add_panel_label(ax_dom, "B")

    fig.suptitle("Different fault types activate different evidence channels", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.95])
    return _save_figure(fig, output_dir / "05_fault_specific_evidence_channels.png")


def _plot_case_studies(
    case_panels: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    n_cases = len(case_panels)
    if n_cases == 0:
        return []

    fig = plt.figure(figsize=(12.4, max(6.4, 2.7 * n_cases)))
    gs = GridSpec(n_cases, 2, figure=fig, wspace=0.20, hspace=0.44)

    for row_idx, panel in enumerate(case_panels):
        query_window = np.asarray(panel["query_window"], dtype=np.float64)
        refs = [np.asarray(ref, dtype=np.float64) for ref in panel["refs"]]
        channel_ids = panel["channel_ids"]
        ref_meta = panel["ref_meta"]
        dominant = str(panel["dominant_evidence"])

        for col_idx, channel_id in enumerate(channel_ids):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            _style_axis(ax, grid_axis="y")
            traces = [query_window[:, channel_id]] + [ref[:, channel_id] for ref in refs]
            z_traces = _standardize_traces(traces)
            x = np.arange(len(z_traces[0]), dtype=np.float64)
            ax.plot(x, z_traces[0], color=CASE_COLORS["query"], linewidth=2.2, label="Fault query")
            for ref_idx, trace in enumerate(z_traces[1:4], start=1):
                meta = ref_meta[ref_idx - 1]
                label = f"Ref {ref_idx}: {meta['file_id']} ({meta['distance']:.2f})"
                ax.plot(
                    x,
                    trace,
                    color=CASE_COLORS[f"ref{ref_idx}"],
                    linewidth=1.6,
                    alpha=0.92,
                    linestyle="--",
                    label=label,
                )
            ax.set_xlim(0, len(x) - 1)
            ax.set_xlabel("Time index within window")
            ax.set_ylabel("Local z-score")
            ax.set_title(f"Fault d{panel['fault_id']:02d} / Channel {channel_id}", loc="left", fontweight="bold")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", frameon=False, fontsize=7.8)
                _add_panel_label(ax, "A")
            elif row_idx == 0 and col_idx == 1:
                _add_panel_label(ax, "B")
            text = (
                f"Query: {panel['file_id']} | Mode {panel['query_mode']} | start {panel['query_start']}\n"
                f"Dominant evidence: {EVIDENCE_LABELS.get(dominant, dominant)} | score {panel['query_score']:.3f}"
            )
            ax.text(
                0.01,
                0.98,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.7,
                bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.94, "boxstyle": "round,pad=0.24"},
            )

    fig.suptitle("Qualitative retrieval cases: fault queries versus retrieved normal windows", fontsize=12.6, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Each row uses the two most discriminative channels for one fault sequence; retrieved references come from the training normal memory bank.",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4a4a4a",
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    return _save_figure(fig, output_dir / "06_qualitative_retrieval_cases.png")


def _plot_variant_comparison(
    variant_metrics: dict[str, dict[str, Any]],
    metric_specs: list[tuple[str, str, str]],
    title: str,
    output_path: Path,
) -> list[Path]:
    labels = list(variant_metrics.keys())
    n_metrics = len(metric_specs)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    axes_list = axes.reshape(-1)

    colors = []
    for label in labels:
        lower = label.lower()
        if "full" in lower or "multi" in lower:
            colors.append(MUTED_GREEN)
        elif "global" in lower:
            colors.append(MUTED_RED)
        elif "state-only" in lower:
            colors.append(MUTED_BLUE)
        elif "context-only" in lower:
            colors.append(SOFT_ORANGE)
        elif "long" in lower:
            colors.append(MUTED_BLUE)
        elif "short" in lower:
            colors.append(SOFT_RED)
        else:
            colors.append(MUTED_GREEN)

    for ax, (metric, direction, label) in zip(axes_list, metric_specs):
        _style_axis(ax, grid_axis="y")
        values = np.asarray([_metric_value_to_float(variant_metrics[name].get(metric)) for name in labels], dtype=np.float64)
        x = np.arange(len(labels), dtype=np.float64)
        ax.bar(x, values, color=colors, alpha=0.88, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title(f"{label} ({'up' if direction == 'higher' else 'down'})", loc="left", fontweight="bold")
        for idx, value in enumerate(values):
            ax.text(idx, value, _format_float(value, 3), ha="center", va="bottom", fontsize=8.0)
        if metric == "Mode-FPR-Std":
            ax.set_ylim(0.0, max(0.07, float(np.nanmax(values) * 1.20)))
        else:
            ax.set_ylim(0.0, float(np.nanmax(values) * 1.22))

    for ax in axes_list[n_metrics:]:
        ax.axis("off")

    axes_list[0].text(
        0.02,
        0.98,
        "All values come from each variant's own exported TEP mechanism metrics.",
        transform=axes_list[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.92, "boxstyle": "round,pad=0.24"},
    )
    _add_panel_label(axes_list[0], "A")
    _add_panel_label(axes_list[1], "B")
    _add_panel_label(axes_list[2], "C")
    _add_panel_label(axes_list[3], "D")

    fig.suptitle(title, fontsize=12.6, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return _save_figure(fig, output_path)


def _plot_state_evolution(
    payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    fault_id = _representative_sequence_id(summary_rows)
    logs = payload["fault_logs"]
    audit_logs = payload["audit_logs"]
    file_ids = np.asarray(logs["file_id"], dtype=object)
    fault_ids = np.asarray(logs["fault_id"], dtype=np.int32)
    starts = np.asarray(logs["start"], dtype=np.int64)
    state_vec = np.asarray(logs["state_vec"], dtype=np.float64)
    final_scores = np.asarray(logs["final"], dtype=np.float64)
    memory_distance = np.asarray(logs["memory_distance"], dtype=np.float64)
    state_novelty = np.asarray(logs["state_novelty"], dtype=np.float64)
    query_modes = np.asarray(logs["mode_id"], dtype=np.int32)
    neighbor_modes = np.asarray(logs["topk_neighbor_mode_ids"], dtype=np.int32)[:, :10]

    mask = fault_ids == fault_id
    if not np.any(mask):
        return []
    sequence_names = file_ids[mask]
    sequence_name = str(sequence_names[0])
    seq_mask = np.logical_and(mask, file_ids == sequence_name)
    x = starts[seq_mask].astype(np.float64)
    seq_state = state_vec[seq_mask]
    seq_final = final_scores[seq_mask]
    seq_memory = memory_distance[seq_mask]
    seq_novelty = state_novelty[seq_mask]
    seq_modes = query_modes[seq_mask]
    seq_retrieved = neighbor_modes[seq_mask]

    audit_state = np.asarray(audit_logs["state_vec"], dtype=np.float64)
    audit_modes = np.asarray(audit_logs["mode_id"], dtype=np.int32)
    centroids = {
        mode_id: np.mean(audit_state[audit_modes == mode_id], axis=0)
        for mode_id in MODE_ORDER
        if np.any(audit_modes == mode_id)
    }
    centroid_dist = {
        mode_id: np.linalg.norm(seq_state - centroid, axis=1)
        for mode_id, centroid in centroids.items()
    }

    final_pct = empirical_percentile(np.asarray(audit_logs["final"], dtype=np.float64), seq_final)
    memory_pct = empirical_percentile(np.asarray(audit_logs["memory_distance"], dtype=np.float64), seq_memory)
    novelty_pct = empirical_percentile(np.asarray(audit_logs["state_novelty"], dtype=np.float64), seq_novelty)
    dominant_retrieved = np.asarray([_dominant_mode(row) for row in seq_retrieved], dtype=np.int32)

    fig = plt.figure(figsize=(11.8, 6.6))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.1], hspace=0.30)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)

    _style_axis(ax_top, grid_axis="both")
    ax_top.plot(x, final_pct, color=SOFT_RED, linewidth=2.2, label="Final score percentile")
    ax_top.plot(x, memory_pct, color=MUTED_BLUE, linewidth=1.7, linestyle="--", label="Memory percentile")
    ax_top.plot(x, novelty_pct, color=SOFT_ORANGE, linewidth=1.7, linestyle="--", label="Novelty percentile")
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.set_ylabel("Audit percentile")
    ax_top.set_title(f"State evolution on {sequence_name} (fault d{fault_id:02d})", loc="left", fontweight="bold")
    ax_top.legend(loc="upper left", frameon=False, ncol=3)
    mode_strip = np.asarray([MODE_ORDER.index(mode) if mode in MODE_ORDER else np.nan for mode in dominant_retrieved], dtype=np.float64)
    ax_top.scatter(
        x,
        np.full_like(x, -0.01, dtype=np.float64),
        c=mode_strip,
        cmap=plt.get_cmap("viridis", len(MODE_ORDER)),
        s=9,
        alpha=0.9,
        clip_on=False,
    )
    ax_top.text(
        0.01,
        0.98,
        f"True mode = {int(seq_modes[0])}; retrieved mode strip uses the dominant mode within top-K neighbors.",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.94, "boxstyle": "round,pad=0.24"},
    )
    _add_panel_label(ax_top, "A")

    _style_axis(ax_bottom, grid_axis="both")
    for mode_id in MODE_ORDER:
        if mode_id not in centroid_dist:
            continue
        ax_bottom.plot(
            x,
            centroid_dist[mode_id],
            color=MODE_STYLE[mode_id]["color"],
            linewidth=2.0,
            label=f"Distance to mode {mode_id} centroid",
        )
    ax_bottom.set_xlabel("Window start index")
    ax_bottom.set_ylabel("State-centroid distance")
    ax_bottom.set_title("Distance to mode centroids across time", loc="left", fontweight="bold")
    ax_bottom.legend(loc="upper right", frameon=False)
    _add_panel_label(ax_bottom, "B")

    fig.suptitle("Within-sequence state dynamics stay interpretable under fault progression", fontsize=12.6, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    return _save_figure(fig, output_dir / "09_state_representation_evolution.png")


def _sensitivity_status(repo_root: Path) -> dict[str, Any]:
    manifest_path = repo_root / "main-result" / "sensitivity_summary_main5_target_closest_default_stable_retrieval_fixed_v3_manifest.csv"
    if not manifest_path.exists():
        return {
            "status": "not_available",
            "reason": "sensitivity_manifest_missing",
            "source": str(manifest_path),
        }
    text = manifest_path.read_text(encoding="utf-8", errors="ignore")
    if "TEP" not in text and "tep" not in text:
        return {
            "status": "not_available",
            "reason": "no_tep_rows_found_in_sensitivity_manifest",
            "source": str(manifest_path),
        }
    return {
        "status": "available",
        "reason": "tep_rows_found",
        "source": str(manifest_path),
    }


def _write_suite_manifest(
    output_dir: Path,
    experiment_dir: Path,
    metrics: dict[str, Any],
    retrieval_variant_dirs: dict[str, Path],
    scale_variant_dirs: dict[str, Path],
    exported_paths: list[Path],
    sensitivity_status: dict[str, Any],
) -> None:
    figure_notes = [
        ("00_main_mechanism_overview.png", "Recommended main-text four-panel mechanism figure for the TEP subsection."),
        ("00_main_mechanism_overview_publication.png", "Alias of the polished main-text figure, useful when you want a clearly named publication-ready version."),
        ("01_state_space_mode_recovery.png", "Expanded state-space visualization; better used as appendix support."),
        ("02_retrieval_mode_consistency.png", "Strongest standalone retrieval-mechanism figure; suitable for main text if used separately."),
        ("03_fault_normal_gap_distributions.png", "Expanded gap figure; main text should emphasize the global/mode-wise core rather than the per-fault tail."),
        ("04_cross_mode_calibration.png", "Calibration figure with improved panel-B layout; suitable for main text or appendix."),
        ("05_fault_specific_evidence_channels.png", "Evidence-channel diversity analysis; recommended for appendix."),
        ("06_qualitative_retrieval_cases.png", "Qualitative retrieval case studies for appendix."),
        ("07_retrieval_strategy_mechanism_comparison.png", "Strategy comparison with caution: emphasizes trade-offs rather than uniform superiority."),
        ("08_multiscale_mechanism_comparison.png", "Scale comparison with caution: emphasizes balanced calibration rather than uniform superiority."),
        ("09_state_representation_evolution.png", "Additional within-sequence state plot; not recommended by default."),
    ]
    lines = [
        "# TEP Mechanism Figure Suite",
        "",
        f"- experiment: `{experiment_dir}`",
        f"- exported_dir: `{output_dir}`",
        f"- summary metrics: `{', '.join(f'{key}={_format_float(metrics.get(key), 4)}' for key in SUMMARY_METRICS)}`",
        "",
        "## Figures",
    ]
    for filename, note in figure_notes:
        lines.append(f"- `{filename}`: {note}")
    lines.extend(
        [
            "",
            "## Recommended Paper Use",
            "- Main text: `Table 3` plus `00_main_mechanism_overview.png` (or the identical alias `00_main_mechanism_overview_publication.png`).",
            "- If you prefer separate main-text figures instead of the four-panel overview: `02_retrieval_mode_consistency.png`, `03_fault_normal_gap_distributions.png` (mainly panel A/B), and `04_cross_mode_calibration.png`.",
            "- Appendix recommended: `01_state_space_mode_recovery.png`, `05_fault_specific_evidence_channels.png`, `06_qualitative_retrieval_cases.png`.",
            "- Appendix with cautious interpretation: `07_retrieval_strategy_mechanism_comparison.png`, `08_multiscale_mechanism_comparison.png`.",
            "- Not recommended by default: `09_state_representation_evolution.png`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Variant inputs",
            f"- retrieval variants: `{', '.join(str(path.name) for path in retrieval_variant_dirs.values())}`",
            f"- scale variants: `{', '.join(str(path.name) for path in scale_variant_dirs.values())}`",
            "",
            "## Sensitivity availability",
            f"- status: `{sensitivity_status['status']}`",
            f"- reason: `{sensitivity_status['reason']}`",
            f"- source: `{sensitivity_status['source']}`",
            "",
            "## Exported files",
        ]
    )
    for path in sorted(exported_paths, key=lambda item: item.name):
        lines.append(f"- `{path.name}`")
    (output_dir / "suite_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "summary_metrics": {key: metrics.get(key) for key in SUMMARY_METRICS},
        "retrieval_variants": {label: str(path) for label, path in retrieval_variant_dirs.items()},
        "scale_variants": {label: str(path) for label, path in scale_variant_dirs.items()},
        "sensitivity_status": sensitivity_status,
        "exported_files": [str(path) for path in sorted(exported_paths, key=lambda item: item.name)],
    }
    (output_dir / "suite_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_state_space_mode_recovery(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    embedding_method: str,
) -> list[Path]:
    audit_logs = payload["audit_logs"]
    fault_logs = payload["fault_logs"]
    audit_state = np.asarray(audit_logs["state_vec"], dtype=np.float64)
    audit_mode = np.asarray(audit_logs["mode_id"], dtype=np.int32)
    fault_state = np.asarray(fault_logs["state_vec"], dtype=np.float64)
    fault_mode = np.asarray(fault_logs["mode_id"], dtype=np.int32)

    rng = np.random.default_rng(42)
    audit_idx = []
    fault_idx = []
    for mode_id in MODE_ORDER:
        a = np.where(audit_mode == mode_id)[0]
        f = np.where(fault_mode == mode_id)[0]
        if len(a) > 800:
            a = rng.choice(a, size=800, replace=False)
        if len(f) > 520:
            f = rng.choice(f, size=520, replace=False)
        audit_idx.append(np.sort(a))
        fault_idx.append(np.sort(f))
    audit_idx_arr = np.concatenate(audit_idx, axis=0)
    fault_idx_arr = np.concatenate(fault_idx, axis=0)

    joint = np.concatenate([audit_state[audit_idx_arr], fault_state[fault_idx_arr]], axis=0)
    coords = _reduce_state_embedding(joint, embedding_method)
    split = len(audit_idx_arr)
    audit_coords = coords[:split]
    fault_coords = coords[split:]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    ax_left, ax_right = axes

    _style_axis(ax_left, grid_axis="both")
    for mode_id in MODE_ORDER:
        mask = audit_mode[audit_idx_arr] == mode_id
        points = audit_coords[mask]
        ax_left.scatter(
            points[:, 0],
            points[:, 1],
            s=10,
            color=MODE_STYLE[mode_id]["color"],
            alpha=0.32,
            linewidths=0.0,
            label=f"Mode {mode_id}",
            rasterized=True,
        )
    ax_left.set_xlabel("Embedding dim 1")
    ax_left.set_ylabel("Embedding dim 2")
    ax_left.set_title("Audit-normal states by true mode", loc="left", fontweight="bold")
    ax_left.legend(loc="upper right", frameon=False)
    ax_left.text(
        0.02,
        0.98,
        f"SMC@K = {_format_float(metrics.get('SMC@K'), 4)}\nSFR = {_format_float(metrics.get('SFR'), 2)}",
        transform=ax_left.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.94, "boxstyle": "round,pad=0.24"},
    )
    _add_panel_label(ax_left, "A")

    _style_axis(ax_right, grid_axis="both")
    for mode_id in MODE_ORDER:
        mask_a = audit_mode[audit_idx_arr] == mode_id
        mask_f = fault_mode[fault_idx_arr] == mode_id
        ax_right.scatter(
            audit_coords[mask_a, 0],
            audit_coords[mask_a, 1],
            s=9,
            color=MODE_STYLE[mode_id]["color"],
            alpha=0.18,
            linewidths=0.0,
            rasterized=True,
        )
        ax_right.scatter(
            fault_coords[mask_f, 0],
            fault_coords[mask_f, 1],
            s=15,
            facecolors="none",
            edgecolors=MODE_STYLE[mode_id]["color"],
            linewidths=0.75,
            alpha=0.76,
            label=f"Fault in Mode {mode_id}",
            rasterized=True,
        )
    ax_right.set_xlabel("Embedding dim 1")
    ax_right.set_ylabel("Embedding dim 2")
    ax_right.set_title("Fault windows remain mode-aware in state space", loc="left", fontweight="bold")
    ax_right.legend(loc="upper right", frameon=False)
    x_low, x_high = np.percentile(fault_coords[:, 0], [1.0, 99.0])
    y_low, y_high = np.percentile(fault_coords[:, 1], [1.0, 99.0])
    x_pad = max(0.4, float((x_high - x_low) * 0.08))
    y_pad = max(0.3, float((y_high - y_low) * 0.08))
    ax_right.set_xlim(float(x_low - x_pad), float(x_high + x_pad))
    ax_right.set_ylim(float(y_low - y_pad), float(y_high + y_pad))
    _add_panel_label(ax_right, "B")

    fig.suptitle("State representation recovers the underlying operating-mode structure", fontsize=12.6, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    return _save_figure(fig, output_dir / "01_state_space_mode_recovery.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_paper_style()

    stale_files = [
        args.output_dir / "00_mechanism_validation_overview.png",
        args.output_dir / "00_mechanism_validation_overview.pdf",
    ]
    for stale in stale_files:
        if stale.exists():
            stale.unlink()

    payload = _load_experiment_payload(args.experiment_dir, args.log_subdir)
    metrics = _load_metrics(args.experiment_dir, args.log_subdir)
    summary_rows = _build_fault_evidence_summary(payload)
    train_state_meta = load_train_state_meta(args.experiment_dir / args.log_subdir)

    retrieval_variant_dirs = _variant_dir_map(args.experiment_dir, RETRIEVAL_VARIANTS)
    scale_variant_dirs = _variant_dir_map(args.experiment_dir, SCALE_VARIANTS)

    retrieval_metrics: dict[str, dict[str, Any]] = {}
    for label, exp_dir in retrieval_variant_dirs.items():
        retrieval_metrics[label] = _load_metrics(exp_dir, args.log_subdir)

    scale_metrics: dict[str, dict[str, Any]] = {}
    for label, exp_dir in scale_variant_dirs.items():
        scale_metrics[label] = _load_metrics(exp_dir, args.log_subdir)

    exported_paths: list[Path] = []
    exported_paths.extend(copy_metadata_files(args.experiment_dir / args.log_subdir, args.output_dir))
    exported_paths.extend(_plot_main_overview_figure(payload, metrics, args.output_dir, args.embedding_method, args.top_k))
    exported_paths.extend(_plot_state_space_mode_recovery(payload, metrics, args.output_dir, args.embedding_method))
    exported_paths.extend(_plot_retrieval_mode_consistency(payload, metrics, args.output_dir, args.top_k))
    exported_paths.extend(_plot_fault_normal_gap(payload, metrics, args.output_dir))
    exported_paths.extend(_plot_cross_mode_calibration(payload, metrics, args.output_dir))
    exported_paths.extend(_plot_fault_evidence_channels(summary_rows, metrics, args.output_dir))

    case_panels = _build_case_panels(
        payload=payload,
        summary_rows=summary_rows,
        raw_data_dir=args.raw_data_dir,
        train_state_meta=train_state_meta,
        case_top_k=args.case_top_k,
        num_cases=args.num_case_studies,
    )
    exported_paths.extend(_plot_case_studies(case_panels, args.output_dir))
    exported_paths.extend(
        _plot_variant_comparison(
            variant_metrics=retrieval_metrics,
            metric_specs=RETRIEVAL_METRICS,
            title="Mechanism comparison across retrieval strategies",
            output_path=args.output_dir / "07_retrieval_strategy_mechanism_comparison.png",
        )
    )
    exported_paths.extend(
        _plot_variant_comparison(
            variant_metrics=scale_metrics,
            metric_specs=SCALE_METRICS,
            title="Mechanism comparison across short, long, and multi-scale variants",
            output_path=args.output_dir / "08_multiscale_mechanism_comparison.png",
        )
    )
    exported_paths.extend(_plot_state_evolution(payload, summary_rows, args.output_dir))

    retrieval_rows = [
        [label] + [_metric_value_to_float(metrics_row.get(metric)) for metric, _, _ in RETRIEVAL_METRICS]
        for label, metrics_row in retrieval_metrics.items()
    ]
    _write_csv(
        args.output_dir / "07_retrieval_strategy_metrics.csv",
        ["variant"] + [metric for metric, _, _ in RETRIEVAL_METRICS],
        retrieval_rows,
    )
    exported_paths.append(args.output_dir / "07_retrieval_strategy_metrics.csv")

    scale_rows = [
        [label] + [_metric_value_to_float(metrics_row.get(metric)) for metric, _, _ in SCALE_METRICS]
        for label, metrics_row in scale_metrics.items()
    ]
    _write_csv(
        args.output_dir / "08_multiscale_metrics.csv",
        ["variant"] + [metric for metric, _, _ in SCALE_METRICS],
        scale_rows,
    )
    exported_paths.append(args.output_dir / "08_multiscale_metrics.csv")

    evidence_rows = [
        [
            int(row["fault_id"]),
            row["dominant_evidence"],
            _metric_value_to_float(row["memory_distance"]),
            _metric_value_to_float(row["state_novelty"]),
            _metric_value_to_float(row["completion_scale8"]),
            _metric_value_to_float(row["completion_scale32"]),
            _metric_value_to_float(row["selected_calibrated_std"]),
        ]
        for row in summary_rows
    ]
    _write_csv(
        args.output_dir / "05_fault_evidence_summary.csv",
        [
            "fault_id",
            "dominant_evidence",
            "memory_distance",
            "state_novelty",
            "completion_scale8",
            "completion_scale32",
            "selected_calibrated_std",
        ],
        evidence_rows,
    )
    exported_paths.append(args.output_dir / "05_fault_evidence_summary.csv")

    sensitivity_status = _sensitivity_status(REPO_ROOT)
    (args.output_dir / "10_sensitivity_status.json").write_text(
        json.dumps(sensitivity_status, indent=2),
        encoding="utf-8",
    )
    exported_paths.append(args.output_dir / "10_sensitivity_status.json")

    _write_suite_manifest(
        output_dir=args.output_dir,
        experiment_dir=args.experiment_dir,
        metrics=metrics,
        retrieval_variant_dirs=retrieval_variant_dirs,
        scale_variant_dirs=scale_variant_dirs,
        exported_paths=[Path(path) for path in exported_paths],
        sensitivity_status=sensitivity_status,
    )
    exported_paths.extend(
        [
            args.output_dir / "suite_manifest.md",
            args.output_dir / "suite_manifest.json",
        ]
    )
    print(f"[TEP Suite] wrote {len(exported_paths)} files into {args.output_dir}")


if __name__ == "__main__":
    main()
