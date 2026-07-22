from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse

from plot_mechanism_figures import (
    EVIDENCE_KEYS,
    EVIDENCE_LABELS,
    _calibrate_fault_sequence_rows,
    _load_experiment_payload,
)


MODE_ORDER = [1, 3, 4]
MODE_STYLE = {
    1: {"label": "Mode 1", "color": "#7EA8F8"},
    3: {"label": "Mode 3", "color": "#A8D39B"},
    4: {"label": "Mode 4", "color": "#EE7875"},
}
EVIDENCE_STYLE = {
    "memory_distance": {"label": "Memory", "color": "#EE7875"},
    "state_novelty": {"label": "Novelty", "color": "#F4B36A"},
    "completion_scale8": {"label": "Comp-short", "color": "#7EA8F8"},
    "completion_scale32": {"label": "Comp-long", "color": "#A8D39B"},
}
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "scar_heat",
    ["#fff9f2", "#e5efff", "#c1d4ff", "#7EA8F8", "#5676BA"],
)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "scar_evidence",
    ["#dbe7ff", "#f8fbff", "#fff8ef", "#f7c3bb", "#ee7875"],
)

METADATA_FILES = [
    "mechanism_metrics.json",
    "mechanism_metrics.csv",
    "export_meta.json",
]

SUMMARY_METRICS = [
    "SMC@K",
    "SFR",
    "SMR@K",
    "delta_mem_mode",
    "Mode-FPR-Std",
    "EE95",
    "Tail@0.99_error",
    "Evidence-Dom-Consistency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a polished paper-facing TEP mechanism figure set into a single directory.",
    )
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--log_subdir", type=str, default="tep_mechanism")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--embedding_method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--style", type=str, default="polished", choices=["polished", "reference"])
    return parser.parse_args()


def configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.7,
            "axes.linewidth": 0.85,
            "axes.edgecolor": "#5a5a5a",
            "grid.color": "#a9adb5",
            "grid.alpha": 0.18,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def configure_reference_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.85,
            "grid.color": "#b0b0b0",
            "grid.alpha": 0.55,
            "grid.linestyle": "-",
            "grid.linewidth": 0.55,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_metrics(exp_dir: Path, log_subdir: str) -> dict[str, Any]:
    metrics_path = exp_dir / log_subdir / "mechanism_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", {})
    if exp_dir.name in experiments:
        return experiments[exp_dir.name]
    if len(experiments) == 1:
        return next(iter(experiments.values()))
    raise KeyError(f"Could not resolve metrics for experiment {exp_dir.name!r} in {metrics_path}")


def copy_metadata_files(log_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in METADATA_FILES:
        src = log_dir / name
        if not src.exists():
            continue
        dst = output_dir / name
        dst.write_bytes(src.read_bytes())
        copied.append(str(dst))
    return copied


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        scalar = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(scalar):
        return "nan"
    return f"{scalar:.{digits}f}"


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _finite_positive(values: np.ndarray) -> np.ndarray:
    arr = _finite(values)
    return arr[arr > 0.0]


def _style_axis(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor("#fffdfa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5d5d5d")
    ax.spines["bottom"].set_color("#5d5d5d")
    ax.tick_params(direction="out", length=3.0, width=0.8, color="#5d5d5d")
    if grid_axis in {"x", "both"}:
        ax.grid(True, axis="x")
    else:
        ax.grid(False, axis="x")
    if grid_axis in {"y", "both"}:
        ax.grid(True, axis="y")
    else:
        ax.grid(False, axis="y")


def _add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color="#1f1f1f",
    )


def _save_figure(fig: plt.Figure, png_path: Path) -> list[Path]:
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    scale = np.std(coords, axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    return coords / scale


def _tsne_2d(x: np.ndarray) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return _pca_2d(x)
    perplexity = max(10, min(40, max(10, x.shape[0] // 20)))
    return TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(np.asarray(x, dtype=np.float64))


def _reduce_state_embedding(x: np.ndarray, method: str) -> np.ndarray:
    if method == "tsne":
        return _tsne_2d(x)
    return _pca_2d(x)


def _covariance_ellipse(points: np.ndarray, n_std: float = 2.0) -> tuple[float, float, float]:
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width = 2.0 * n_std * np.sqrt(max(eigvals[0], 1e-8))
    height = 2.0 * n_std * np.sqrt(max(eigvals[1], 1e-8))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return width, height, angle


def _compute_retrieval_heat(
    payload: dict[str, Any],
    top_k: int = 10,
) -> np.ndarray:
    logs = payload["fault_logs"]
    query_mode = np.asarray(logs["mode_id"], dtype=np.int32)
    neighbor_mode = np.asarray(logs["topk_neighbor_mode_ids"], dtype=np.int32)[:, :top_k]
    heat = np.zeros((len(MODE_ORDER), len(MODE_ORDER)), dtype=np.float64)
    for row_idx, row_mode in enumerate(MODE_ORDER):
        mask = query_mode == row_mode
        if not np.any(mask):
            continue
        valid = neighbor_mode[mask]
        denom = 0.0
        for col_idx, col_mode in enumerate(MODE_ORDER):
            count = float(np.sum(valid == col_mode))
            heat[row_idx, col_idx] = count
            denom += count
        if denom > 0.0:
            heat[row_idx] /= denom
    return heat


def _fault_mode_rows(payload: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    rows = _calibrate_fault_sequence_rows(payload["audit_logs"], payload["fault_sequence_records"])
    table: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        table[int(row["fault_id"])][int(row["mode_id"])] = row
    return table


def _build_fault_evidence_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _calibrate_fault_sequence_rows(payload["audit_logs"], payload["fault_sequence_records"])
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["fault_id"])].append(row)

    dominant_order = {key: idx for idx, key in enumerate(EVIDENCE_KEYS)}
    summary_rows: list[dict[str, Any]] = []
    for fault_id in sorted(grouped):
        group = grouped[fault_id]
        entry: dict[str, Any] = {"fault_id": int(fault_id)}
        means = {}
        for key in EVIDENCE_KEYS:
            metric_key = f"{key}_calibrated"
            values = np.asarray([float(row[metric_key]) for row in group], dtype=np.float64)
            means[key] = float(np.mean(values)) if values.size else float("nan")
        entry.update(means)
        finite_pairs = [(key, value) for key, value in means.items() if np.isfinite(value)]
        entry["dominant_evidence"] = max(finite_pairs, key=lambda item: item[1])[0] if finite_pairs else "unknown"
        selected_vals = np.asarray([float(row["selected_calibrated"]) for row in group], dtype=np.float64)
        entry["selected_calibrated_std"] = float(np.std(selected_vals)) if selected_vals.size else float("nan")
        entry["dominant_order"] = dominant_order.get(str(entry["dominant_evidence"]), len(dominant_order))
        entry["dominant_strength"] = max((float(v) for _, v in finite_pairs), default=float("nan"))
        summary_rows.append(entry)
    summary_rows.sort(
        key=lambda row: (
            int(row.get("dominant_order", 999)),
            -float(row.get("dominant_strength", -np.inf)),
            int(row["fault_id"]),
        )
    )
    return summary_rows


def _majority_mode(neighbor_modes: np.ndarray) -> int:
    valid = [int(value) for value in neighbor_modes.tolist() if int(value) >= 0]
    if not valid:
        return -1
    counts: dict[int, int] = {}
    first_pos: dict[int, int] = {}
    for idx, value in enumerate(valid):
        counts[value] = counts.get(value, 0) + 1
        if value not in first_pos:
            first_pos[value] = idx
    return min(counts.keys(), key=lambda mode: (-counts[mode], first_pos[mode]))


def _stratified_indices(groups: np.ndarray, max_per_group: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    unique_groups = sorted(np.unique(groups).tolist())
    for group in unique_groups:
        indices = np.where(groups == group)[0]
        if len(indices) > max_per_group:
            indices = rng.choice(indices, size=max_per_group, replace=False)
        selected.append(np.sort(indices))
    if not selected:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(selected, axis=0)


def _write_fault_evidence_summary_csv(summary_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "05_fault_level_evidence_summary.csv"
    lines = [
        "fault_id,dominant_evidence,memory_distance,state_novelty,completion_scale8,completion_scale32,selected_calibrated_std"
    ]
    for row in summary_rows:
        lines.append(
            ",".join(
                [
                    str(int(row["fault_id"])),
                    str(row["dominant_evidence"]),
                    _format_float(row["memory_distance"], 6),
                    _format_float(row["state_novelty"], 6),
                    _format_float(row["completion_scale8"], 6),
                    _format_float(row["completion_scale32"], 6),
                    _format_float(row["selected_calibrated_std"], 6),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot_state_embedding_axis(
    ax: plt.Axes,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    embedding_method: str,
    panel_label: str | None = None,
) -> None:
    logs = payload["audit_logs"]
    state_vec = np.asarray(logs["state_vec"], dtype=np.float64)
    mode_id = np.asarray(logs["mode_id"], dtype=np.int32)
    coords = _reduce_state_embedding(state_vec, embedding_method)

    _style_axis(ax, grid_axis="both")
    for mode in MODE_ORDER:
        mask = mode_id == mode
        points = coords[mask]
        if points.size == 0:
            continue
        style = MODE_STYLE[mode]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=8,
            color=style["color"],
            alpha=0.18,
            linewidths=0.0,
            rasterized=True,
        )
        if len(points) >= 3:
            center = np.mean(points, axis=0)
            width, height, angle = _covariance_ellipse(points)
            ellipse = Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                facecolor=style["color"],
                edgecolor=style["color"],
                alpha=0.10,
                linewidth=1.2,
            )
            ax.add_patch(ellipse)
            ax.scatter(center[0], center[1], s=38, color=style["color"], edgecolors="white", linewidths=0.7, zorder=4)
            ax.text(
                center[0],
                center[1],
                f" {style['label']}",
                fontsize=8.5,
                fontweight="bold",
                color=style["color"],
                ha="left",
                va="center",
            )
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    ax.set_title("State space recovers operating modes", loc="left", fontweight="bold", pad=8)
    summary = f"SMC@K = {_format_float(metrics.get('SMC@K'), 2)}\nSFR = {_format_float(metrics.get('SFR'), 2)}"
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92, "boxstyle": "round,pad=0.3"},
    )
    if panel_label:
        _add_panel_label(ax, panel_label)


def plot_retrieval_heatmap_axis(
    ax: plt.Axes,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    panel_label: str | None = None,
) -> None:
    heat = _compute_retrieval_heat(payload, top_k=10)
    im = ax.imshow(heat, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title("Retrieved references stay mode-compatible", loc="left", fontweight="bold", pad=8)
    ax.set_xticks(np.arange(len(MODE_ORDER)), [MODE_STYLE[mode]["label"] for mode in MODE_ORDER])
    ax.set_yticks(np.arange(len(MODE_ORDER)), [MODE_STYLE[mode]["label"] for mode in MODE_ORDER])
    ax.set_xlabel("Retrieved reference mode")
    ax.set_ylabel("Fault query mode")
    ax.set_xticks(np.arange(-0.5, len(MODE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODE_ORDER), 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = float(heat[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold",
                color="white" if value >= 0.58 else "#1f1f1f",
            )
    ax.text(
        0.02,
        -0.19,
        f"Overall SMR@K = {_format_float(metrics.get('SMR@K'), 3)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#4c4c4c",
    )
    if panel_label:
        _add_panel_label(ax, panel_label)
    ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.04)


def plot_fault_gap_axes(
    axes: list[plt.Axes],
    payload: dict[str, Any],
    metrics: dict[str, Any],
    panel_label: str | None = None,
) -> None:
    audit_logs = payload["audit_logs"]
    fault_logs = payload["fault_logs"]
    audit_mode = np.asarray(audit_logs["mode_id"], dtype=np.int32)
    fault_mode = np.asarray(fault_logs["mode_id"], dtype=np.int32)
    audit_memory = np.asarray(audit_logs["memory_distance"], dtype=np.float64)
    fault_memory = np.asarray(fault_logs["memory_distance"], dtype=np.float64)

    normal_color = "#D8E5FF"
    for ax, mode_id in zip(axes, MODE_ORDER):
        _style_axis(ax, grid_axis="y")
        mode_style = MODE_STYLE[mode_id]
        normal = _finite_positive(audit_memory[audit_mode == mode_id])
        fault = _finite_positive(fault_memory[fault_mode == mode_id])
        parts = ax.violinplot([normal, fault], positions=[1, 2], showmedians=True, widths=0.72)
        for body_idx, body in enumerate(parts["bodies"]):
            body.set_facecolor(normal_color if body_idx == 0 else mode_style["color"])
            body.set_edgecolor("#4c4c4c")
            body.set_alpha(0.78 if body_idx == 1 else 0.58)
            body.set_linewidth(0.9)
        parts["cmedians"].set_color("#1f1f1f")
        parts["cmedians"].set_linewidth(1.4)
        for key in ("cbars", "cmaxes", "cmins"):
            parts[key].set_color("#4c4c4c")
            parts[key].set_linewidth(0.8)
        ax.set_yscale("log")
        ax.set_xticks([1, 2], ["Audit\nnormal", "Fault"])
        ax.set_title(mode_style["label"], color=mode_style["color"], fontweight="bold", pad=6)
        gap = metrics.get("per_mode_mem_gap", {}).get(str(mode_id), np.nan)
        ax.text(
            0.04,
            0.96,
            f"gap = {_format_float(gap, 1)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "#dbdbdb", "alpha": 0.94, "boxstyle": "round,pad=0.24"},
        )
    axes[0].set_ylabel("Memory distance (log scale)")
    axes[1].text(
        0.5,
        1.16,
        "Fault-normal separation under mode-compatible retrieval",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        color="#1f1f1f",
    )
    if panel_label:
        _add_panel_label(axes[0], panel_label)


def plot_modewise_exceedance_axis(
    ax: plt.Axes,
    metrics: dict[str, Any],
    panel_label: str | None = None,
) -> None:
    curve_payload = metrics.get("exceedance_curve", {})
    q_values = sorted(float(key) for key in curve_payload.keys())
    theory = 1.0 - np.asarray(q_values, dtype=np.float64)

    _style_axis(ax, grid_axis="both")
    ax.plot(q_values, theory, color="#1f1f1f", linestyle="--", linewidth=1.6, label="Theory (1-q)")
    for mode_id in MODE_ORDER:
        style = MODE_STYLE[mode_id]
        y = np.asarray(
            [float(curve_payload.get(f"{q:.2f}", {}).get(str(mode_id), np.nan)) for q in q_values],
            dtype=np.float64,
        )
        ax.plot(
            q_values,
            y,
            marker="o",
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.0,
            color=style["color"],
            label=style["label"],
        )
    for mark_q in (0.95, 0.99):
        ax.axvline(mark_q, color="#bbbbbb", linewidth=0.8, linestyle=":")
    summary = (
        f"FPR std = {_format_float(metrics.get('Mode-FPR-Std'), 4)}\n"
        f"95% tail err = {_format_float(metrics.get('EE95'), 4)}\n"
        f"99% tail err = {_format_float(metrics.get('Tail@0.99_error'), 4)}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        bbox={"facecolor": "white", "edgecolor": "#d2d2d2", "alpha": 0.94, "boxstyle": "round,pad=0.3"},
    )
    ax.set_xlabel("Global threshold quantile q")
    ax.set_ylabel("Pr(score > q-threshold | audit normal)")
    ax.set_title("Calibration remains shared across regimes", loc="left", fontweight="bold", pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.58, 1.02), ncol=2, frameon=False)
    if panel_label:
        _add_panel_label(ax, panel_label)


def plot_evidence_summary_axes(
    ax_heat: plt.Axes,
    ax_bar: plt.Axes,
    summary_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    panel_label: str | None = None,
    show_label_note: bool = True,
) -> None:
    if not summary_rows:
        _style_axis(ax_heat, grid_axis="none")
        ax_heat.axis("off")
        ax_bar.axis("off")
        ax_heat.text(0.5, 0.5, "No evidence rows available", ha="center", va="center")
        return

    heat = np.asarray([[float(row[key]) for key in EVIDENCE_KEYS] for row in summary_rows], dtype=np.float64)
    evidence_abbr = {
        "memory_distance": "M",
        "state_novelty": "N",
        "completion_scale8": "CS",
        "completion_scale32": "CL",
    }
    row_labels = []
    for row in summary_rows:
        abbr = evidence_abbr.get(str(row["dominant_evidence"]), "?")
        row_labels.append(f"d{int(row['fault_id']):02d} · {abbr}")
    im = ax_heat.imshow(heat, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax_heat.set_xticks(
        np.arange(len(EVIDENCE_KEYS)),
        [EVIDENCE_STYLE[key]["label"] for key in EVIDENCE_KEYS],
        rotation=23,
        ha="right",
    )
    ax_heat.set_yticks(np.arange(len(row_labels)), row_labels)
    ax_heat.set_title("Fault evidence is diverse rather than single-channel", loc="left", fontweight="bold", pad=8)
    ax_heat.tick_params(axis="y", pad=6)
    ax_heat.set_xticks(np.arange(-0.5, len(EVIDENCE_KEYS), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax_heat.grid(False)
    ax_heat.grid(which="minor", color="white", linewidth=1.0)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = float(heat[row_idx, col_idx])
            ax_heat.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if value >= 0.74 else "#1f1f1f",
                fontweight="bold" if value >= 0.9 else "normal",
            )
    std_vals = np.asarray([float(row["selected_calibrated_std"]) for row in summary_rows], dtype=np.float64)
    y_pos = np.arange(len(summary_rows))
    _style_axis(ax_bar, grid_axis="x")
    ax_bar.barh(y_pos, std_vals, color="#7c8ba1", edgecolor="#506072", height=0.72)
    ax_bar.set_ylim(ax_heat.get_ylim())
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.set_xlabel("Std")
    ax_bar.set_title("Cross-mode\nscore std", fontsize=9.0, pad=8)
    ax_bar.set_xlim(0.0, max(0.065, float(np.nanmax(std_vals) * 1.15)))
    for idx, value in enumerate(std_vals):
        ax_bar.text(value + 0.0015, idx, f"{value:.3f}", va="center", ha="left", fontsize=7.8, color="#4a4a4a")
    ax_heat.text(
        0.01,
        -0.13,
        f"Dominant-evidence consistency = {_format_float(metrics.get('Evidence-Dom-Consistency'), 4)}",
        transform=ax_heat.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#4c4c4c",
    )
    if show_label_note:
        ax_heat.text(
            0.01,
            -0.21,
            "Row tags: M memory, N novelty, CS comp-short, CL comp-long",
            transform=ax_heat.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            color="#666666",
        )
    ax_heat.figure.colorbar(im, ax=[ax_heat, ax_bar], fraction=0.03, pad=0.04)
    if panel_label:
        _add_panel_label(ax_heat, panel_label)


def plot_normal_violin_axis(
    ax: plt.Axes,
    payload: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    logs = payload["audit_logs"]
    mode_id = np.asarray(logs["mode_id"], dtype=np.int32)
    final_score = np.asarray(logs["final"], dtype=np.float64)
    data = [_finite(final_score[mode_id == mode]) for mode in MODE_ORDER]

    _style_axis(ax, grid_axis="y")
    violin_data = [values for values in data if len(values) > 0]
    positions = [idx + 1 for idx, values in enumerate(data) if len(values) > 0]
    parts = ax.violinplot(violin_data, positions=positions, showmedians=True, widths=0.82)
    for body, mode_id_val in zip(parts["bodies"], MODE_ORDER):
        body.set_facecolor(MODE_STYLE[mode_id_val]["color"])
        body.set_alpha(0.72)
        body.set_edgecolor("#4c4c4c")
        body.set_linewidth(0.8)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.4)
    ax.set_xticks(np.arange(1, len(MODE_ORDER) + 1), [MODE_STYLE[mode]["label"] for mode in MODE_ORDER])
    ax.set_ylabel("Audit-normal final score")
    ax.set_title("Normal-score distributions remain comparable", loc="left", fontweight="bold", pad=8)
    ax.text(
        0.02,
        0.98,
        f"q95 tail err = {_format_float(metrics.get('EE95'), 4)}\nq99 tail err = {_format_float(metrics.get('Tail@0.99_error'), 4)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        bbox={"facecolor": "white", "edgecolor": "#d3d3d3", "alpha": 0.94, "boxstyle": "round,pad=0.28"},
    )


def plot_fault_consistency_axis(
    ax: plt.Axes,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> None:
    row_table = _fault_mode_rows(payload)
    summary_map = {int(row["fault_id"]): row for row in summary_rows}

    _style_axis(ax, grid_axis="y")
    x_values = np.arange(len(MODE_ORDER))
    end_labels: list[tuple[float, str, str]] = []
    for fault_id in sorted(row_table):
        y_values = []
        mode_hits = []
        for mode in MODE_ORDER:
            record = row_table[fault_id].get(mode)
            if record is None:
                continue
            y_values.append(float(record["selected_calibrated"]))
            mode_hits.append(mode)
        if not y_values:
            continue
        dominant = str(summary_map.get(fault_id, {}).get("dominant_evidence", "memory_distance"))
        color = EVIDENCE_STYLE.get(dominant, {"color": "#5f6b7a"})["color"]
        x_idx = np.asarray([MODE_ORDER.index(mode) for mode in mode_hits], dtype=np.int32)
        ax.plot(
            x_idx,
            y_values,
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=4.6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            alpha=0.9,
        )
        end_labels.append((x_idx[-1] + 0.08, y_values[-1], f"d{fault_id:02d}", color))
    ax.set_xticks(x_values, [MODE_STYLE[mode]["label"] for mode in MODE_ORDER])
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Calibrated fault-sequence score")
    ax.set_title("Fault scores stay stable across modes", loc="left", fontweight="bold", pad=8)
    ax.text(
        0.02,
        0.98,
        f"Mean per-fault std = {_format_float(metrics.get('Fault-Consistency-Std'), 4)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#d3d3d3", "alpha": 0.94, "boxstyle": "round,pad=0.28"},
    )
    adjusted_labels: list[tuple[float, float, float, str, str]] = []
    min_gap = 0.007
    for x_label, y_label, text, color in sorted(end_labels, key=lambda item: item[1], reverse=True):
        y_adjusted = y_label
        if adjusted_labels:
            prev_y = adjusted_labels[-1][1]
            if prev_y - y_adjusted < min_gap:
                y_adjusted = prev_y - min_gap
        adjusted_labels.append((x_label, y_adjusted, y_label, text, color))
    for x_label, y_adjusted, y_actual, text, color in adjusted_labels:
        if abs(y_adjusted - y_actual) > 1e-6:
            ax.plot([x_label - 0.04, x_label - 0.015], [y_actual, y_adjusted], color=color, linewidth=0.8, alpha=0.9, clip_on=False)
        ax.text(
            x_label,
            y_adjusted,
            text,
            color=color,
            fontsize=8.0,
            va="center",
            ha="left",
            fontweight="bold",
            clip_on=False,
        )
    ax.set_xlim(-0.1, 2.55)


def export_state_embedding_figure(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    embedding_method: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    plot_state_embedding_axis(ax, payload, metrics, embedding_method)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "01_state_embedding_by_mode.png")


def export_retrieval_heatmap_figure(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    plot_retrieval_heatmap_axis(ax, payload, metrics)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "02_retrieval_mode_heatmap.png")


def export_fault_gap_figure(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(11.6, 4.0))
    axes = [fig.add_subplot(1, 3, idx + 1) for idx in range(3)]
    plot_fault_gap_axes(axes, payload, metrics)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "03_fault_normal_memory_gap_by_mode.png")


def export_exceedance_figure(
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    plot_modewise_exceedance_axis(ax, metrics)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "04_modewise_exceedance_calibration.png")


def export_evidence_figure(
    summary_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(8.4, 5.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[5.2, 1.45], wspace=0.06)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax_heat)
    plot_evidence_summary_axes(ax_heat, ax_bar, summary_rows, metrics, show_label_note=True)
    fig.subplots_adjust(left=0.14, right=0.93, bottom=0.20, top=0.87, wspace=0.08)
    return _save_figure(fig, output_dir / "05_fault_level_evidence_summary.png")


def export_normal_violin_figure(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.7, 4.5))
    plot_normal_violin_axis(ax, payload, metrics)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "06_normal_final_violin_by_mode.png")


def export_fault_consistency_figure(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    plot_fault_consistency_axis(ax, payload, metrics, summary_rows)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "07_fault_consistency_across_modes.png")


def export_overview_panel(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    embedding_method: str,
) -> list[Path]:
    fig = plt.figure(figsize=(12.2, 10.2))
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.08], hspace=0.34, wspace=0.26)

    ax_state = fig.add_subplot(outer[0, 0])
    plot_state_embedding_axis(ax_state, payload, metrics, embedding_method, panel_label="A")

    ax_retrieve = fig.add_subplot(outer[0, 1])
    plot_retrieval_heatmap_axis(ax_retrieve, payload, metrics, panel_label="B")

    gap_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, :], wspace=0.18)
    gap_axes = [fig.add_subplot(gap_gs[0, idx]) for idx in range(3)]
    plot_fault_gap_axes(gap_axes, payload, metrics, panel_label="C")

    ax_exceed = fig.add_subplot(outer[2, 0])
    plot_modewise_exceedance_axis(ax_exceed, metrics, panel_label="D")

    evidence_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, 1], width_ratios=[4.7, 1.35], wspace=0.06)
    ax_heat = fig.add_subplot(evidence_gs[0, 0])
    ax_bar = fig.add_subplot(evidence_gs[0, 1], sharey=ax_heat)
    plot_evidence_summary_axes(ax_heat, ax_bar, summary_rows, metrics, panel_label="E", show_label_note=False)

    metric_banner = " | ".join(
        [
            f"SMC@K { _format_float(metrics.get('SMC@K'), 2) }",
            f"SFR { _format_float(metrics.get('SFR'), 2) }",
            f"SMR@K { _format_float(metrics.get('SMR@K'), 3) }",
            f"Gap { _format_float(metrics.get('delta_mem_mode'), 1) }",
            f"FPR std { _format_float(metrics.get('Mode-FPR-Std'), 3) }",
            f"DEC { _format_float(metrics.get('Evidence-Dom-Consistency'), 3) }",
        ]
    )
    fig.suptitle("Mechanism Validation Under Multi-Mode Dynamics", fontsize=14.0, fontweight="bold", y=0.992)
    fig.text(
        0.5,
        0.968,
        "TEP controlled subset: SCAR retrieves local normal references that remain mode-compatible, discriminative, and calibrated.",
        ha="center",
        va="center",
        fontsize=9.3,
        color="#444444",
    )
    fig.text(
        0.5,
        0.022,
        metric_banner,
        ha="center",
        va="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.955])
    return _save_figure(fig, output_dir / "00_mechanism_validation_overview.png")


def _reference_scatter(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    color_map = {1: "#7EA8F8", 3: "#EE7875", 4: "#A8D39B"}
    label_map = {1: "Mode 1", 3: "Mode 3", 4: "Mode 4"}
    for mode in MODE_ORDER:
        mask = labels == mode
        points = coords[mask]
        if len(points) == 0:
            continue
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=7,
            color=color_map[mode],
            alpha=0.88,
            edgecolors="none",
            label=label_map[mode],
        )
    ax.set_title(title, pad=3)
    ax.grid(True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="upper right", frameon=True, markerscale=0.9, borderpad=0.35, handletextpad=0.35)


def export_reference_style_tsne_pair(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    audit_logs = payload["audit_logs"]
    fault_logs = payload["fault_logs"]

    audit_mode = np.asarray(audit_logs["mode_id"], dtype=np.int32)
    audit_state = np.asarray(audit_logs["state_vec"], dtype=np.float64)
    audit_idx = _stratified_indices(audit_mode, max_per_group=850, seed=42)

    fault_mode = np.asarray(fault_logs["mode_id"], dtype=np.int32)
    fault_state = np.asarray(fault_logs["state_vec"], dtype=np.float64)
    fault_neighbor_mode = np.asarray(fault_logs["topk_neighbor_mode_ids"], dtype=np.int32)[:, :10]
    dominant_retrieved = np.asarray([_majority_mode(row) for row in fault_neighbor_mode], dtype=np.int32)
    valid_fault_mask = dominant_retrieved >= 0
    fault_mode = fault_mode[valid_fault_mask]
    fault_state = fault_state[valid_fault_mask]
    dominant_retrieved = dominant_retrieved[valid_fault_mask]
    fault_idx = _stratified_indices(fault_mode, max_per_group=850, seed=123)

    audit_coords = _tsne_2d(audit_state[audit_idx])
    fault_coords = _tsne_2d(fault_state[fault_idx])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0))
    _reference_scatter(axes[0], audit_coords, audit_mode[audit_idx], "t-SNE Visualization")
    _reference_scatter(axes[1], fault_coords, dominant_retrieved[fault_idx], "t-SNE Visualization")

    axes[0].text(
        0.5,
        -0.16,
        "(a) Audit-normal states grouped by true operating mode",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        fontfamily="serif",
    )
    axes[1].text(
        0.5,
        -0.16,
        "(b) Fault-query states grouped by dominant retrieved mode",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        fontfamily="serif",
    )
    fig.text(
        0.5,
        0.01,
        f"SMC@K = {_format_float(metrics.get('SMC@K'), 2)}   |   "
        f"SFR = {_format_float(metrics.get('SFR'), 2)}   |   "
        f"SMR@K = {_format_float(metrics.get('SMR@K'), 3)}",
        ha="center",
        va="bottom",
        fontsize=8.4,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.19, wspace=0.18)
    pair_paths = _save_figure(fig, output_dir / "00_reference_style_state_retrieval_tsne_pair.png")

    fig_single_a, ax_single_a = plt.subplots(figsize=(5.0, 4.4))
    _reference_scatter(ax_single_a, audit_coords, audit_mode[audit_idx], "t-SNE Visualization")
    fig_single_a.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.10)
    single_a_paths = _save_figure(fig_single_a, output_dir / "01_reference_style_audit_true_mode_tsne.png")

    fig_single_b, ax_single_b = plt.subplots(figsize=(5.0, 4.4))
    _reference_scatter(ax_single_b, fault_coords, dominant_retrieved[fault_idx], "t-SNE Visualization")
    fig_single_b.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.10)
    single_b_paths = _save_figure(fig_single_b, output_dir / "02_reference_style_fault_retrieved_mode_tsne.png")

    return pair_paths + single_a_paths + single_b_paths


def write_manifest(
    exp_dir: Path,
    log_dir: Path,
    output_dir: Path,
    copied_metadata: list[str],
    generated_paths: list[Path],
    metrics: dict[str, Any],
    embedding_method: str,
    style: str,
) -> None:
    manifest = {
        "experiment_dir": str(exp_dir),
        "log_dir": str(log_dir),
        "output_dir": str(output_dir),
        "copied_metadata": copied_metadata,
        "generated_figures": [str(path) for path in generated_paths],
        "key_metrics": {key: metrics.get(key) for key in SUMMARY_METRICS},
        "style_version": style,
        "embedding_method": embedding_method,
    }
    json_path = output_dir / "mechanism_figure_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# TEP Mechanism Figure Set",
        "",
        f"- Experiment: `{exp_dir}`",
        f"- Source log dir: `{log_dir}`",
        f"- Output dir: `{output_dir}`",
        f"- Style version: `{style}`",
        f"- Embedding method: `{embedding_method}`",
        "",
        "## Key Metrics",
        "",
    ]
    for key in SUMMARY_METRICS:
        md_lines.append(f"- `{key}`: {_format_float(metrics.get(key), 6)}")
    md_lines.extend(["", "## Figures", ""])
    if style == "reference":
        md_lines.extend(
            [
                "- `00_reference_style_state_retrieval_tsne_pair.png/.pdf`: Two-panel reference-style t-SNE figure.",
                "- `01_reference_style_audit_true_mode_tsne.png/.pdf`: Audit-normal states grouped by true mode.",
                "- `02_reference_style_fault_retrieved_mode_tsne.png/.pdf`: Fault-query states grouped by dominant retrieved mode.",
            ]
        )
    else:
        md_lines.extend(
            [
                "- `00_mechanism_validation_overview.png/.pdf`: Integrated multi-panel paper figure.",
                "- `01_state_embedding_by_mode.png/.pdf`: Polished operating-mode embedding view.",
                "- `02_retrieval_mode_heatmap.png/.pdf`: Polished query-mode vs retrieved-mode heatmap.",
                "- `03_fault_normal_memory_gap_by_mode.png/.pdf`: Polished mode-wise fault-normal separation plot.",
                "- `04_modewise_exceedance_calibration.png/.pdf`: Polished calibration stability plot.",
                "- `05_fault_level_evidence_summary.png/.pdf`: Polished fault-level evidence summary with cross-mode std bar.",
                "- `05_fault_level_evidence_summary.csv`: Tabular summary for the evidence figure.",
                "- `06_normal_final_violin_by_mode.png/.pdf`: Polished normal-score distribution by mode.",
                "- `07_fault_consistency_across_modes.png/.pdf`: Polished fault consistency plot.",
            ]
        )
    md_path = output_dir / "mechanism_figure_manifest.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.style == "reference":
        configure_reference_style()
    else:
        configure_paper_style()

    exp_dir = args.experiment_dir.resolve()
    log_dir = exp_dir / args.log_subdir
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_experiment_payload(exp_dir, args.log_subdir)
    metrics = _load_metrics(exp_dir, args.log_subdir)
    summary_rows = _build_fault_evidence_summary(payload)

    copied_metadata = copy_metadata_files(log_dir, output_dir)
    generated_paths: list[Path] = []
    if args.style == "reference":
        generated_paths.extend(export_reference_style_tsne_pair(payload, metrics, output_dir))
    else:
        generated_paths.extend(export_overview_panel(payload, metrics, summary_rows, output_dir, args.embedding_method))
        generated_paths.extend(export_state_embedding_figure(payload, metrics, output_dir, args.embedding_method))
        generated_paths.extend(export_retrieval_heatmap_figure(payload, metrics, output_dir))
        generated_paths.extend(export_fault_gap_figure(payload, metrics, output_dir))
        generated_paths.extend(export_exceedance_figure(metrics, output_dir))
        generated_paths.extend(export_evidence_figure(summary_rows, metrics, output_dir))
        generated_paths.append(_write_fault_evidence_summary_csv(summary_rows, output_dir))
        generated_paths.extend(export_normal_violin_figure(payload, metrics, output_dir))
        generated_paths.extend(export_fault_consistency_figure(payload, metrics, summary_rows, output_dir))

    write_manifest(
        exp_dir=exp_dir,
        log_dir=log_dir,
        output_dir=output_dir,
        copied_metadata=copied_metadata,
        generated_paths=generated_paths,
        metrics=metrics,
        embedding_method=args.embedding_method,
        style=args.style,
    )
    print(f"[TEP Paper Export] wrote {args.style} figure set to {output_dir}")


if __name__ == "__main__":
    main()
