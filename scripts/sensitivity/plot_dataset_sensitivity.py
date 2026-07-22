from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


PAPER_COLORS = {
    "blue": "#1f4e79",
    "red": "#b23a48",
    "green": "#3a7d44",
    "gold": "#d9a441",
    "gray": "#6c757d",
    "black": "#1f1f1f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot single-dataset sensitivity-analysis results.")
    parser.add_argument("--summary-json", type=str, required=True, help="Path to *_summary.json from run_dataset_sensitivity.py")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional figure output directory. Defaults to <summary-dir>/figures.",
    )
    return parser.parse_args()


def configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid summary payload: {path}")
    return payload


def rows_for_param(payload: dict[str, Any], param: str) -> list[dict[str, Any]]:
    rows = [row for row in payload.get("rows", []) if row.get("param") == param]
    specs = payload.get("specs", {})
    spec = specs.get(param, {})
    value_order = spec.get("values", [])
    order_map = {str(value): idx for idx, value in enumerate(value_order)}
    return sorted(rows, key=lambda row: order_map.get(str(row.get("value")), 10**9))


def roc_values(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row.get("roc_auc", float("nan"))))
        except Exception:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def finite_concat(series_list: list[np.ndarray]) -> np.ndarray:
    finite_parts = [series[np.isfinite(series)] for series in series_list if np.isfinite(series).any()]
    return np.concatenate(finite_parts, axis=0) if finite_parts else np.asarray([], dtype=np.float64)


def maybe_set_ylim(ax, series_list: list[np.ndarray]) -> None:
    finite = finite_concat(series_list)
    if finite.size == 0:
        ax.set_ylim(0.0, 1.0)
        return
    lo = float(finite.min())
    hi = float(finite.max())
    pad = max(0.003, (hi - lo) * 0.18)
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))


def style_axis(ax) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)


def add_panel_label(ax, label: str) -> None:
    ax.text(0.0, 1.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")


def mark_best_point(ax, x: np.ndarray, y: np.ndarray) -> None:
    finite_mask = np.isfinite(y)
    if not finite_mask.any():
        return
    finite_indices = np.where(finite_mask)[0]
    best_local_idx = int(np.nanargmax(y[finite_mask]))
    best_idx = int(finite_indices[best_local_idx])
    ax.scatter(
        [x[best_idx]],
        [y[best_idx]],
        marker="*",
        s=95,
        facecolor=PAPER_COLORS["gold"],
        edgecolor=PAPER_COLORS["black"],
        linewidth=0.7,
        zorder=5,
    )


def mark_default_point(ax, x: np.ndarray, y: np.ndarray, default_idx: int) -> None:
    if default_idx < 0 or default_idx >= len(x) or not np.isfinite(y[default_idx]):
        return
    ax.scatter(
        [x[default_idx]],
        [y[default_idx]],
        marker="D",
        s=42,
        facecolor="white",
        edgecolor=PAPER_COLORS["black"],
        linewidth=0.9,
        zorder=5,
    )


def save_figure(fig, base_path: Path) -> list[Path]:
    saved_paths: list[Path] = []
    for suffix in (".png", ".pdf"):
        save_path = base_path.with_suffix(suffix)
        fig.savefig(save_path, dpi=250 if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(save_path)
    plt.close(fig)
    return saved_paths


def format_param_axis_label(param: str) -> str:
    mapping = {
        "top_M": r"Coarse Retrieval ($M$)",
        "top_K": r"Re-ranking Candidates ($K$)",
        "knn_k": r"kNN Neighbors ($k$)",
        "d_z": r"Representation Dim. ($d_z$)",
        "mask_ratio": "Mask Ratio",
        "seq_len": "Window Length",
        "coreset_max_patches_per_scale": "Memory Budget",
    }
    return mapping.get(param, param)


def plot_single_numeric_sweep(ax, rows: list[dict[str, Any]], default_value: float, color: str, xlabel: str) -> None:
    x = np.asarray([float(row["value"]) for row in rows], dtype=np.float64)
    y = roc_values(rows)
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.0,
        marker="o",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.axvline(default_value, color=PAPER_COLORS["gray"], linestyle="--", linewidth=1.0, alpha=0.7)
    default_candidates = np.where(np.isclose(x, default_value))[0]
    if default_candidates.size > 0:
        mark_default_point(ax, x, y, int(default_candidates[0]))
    mark_best_point(ax, x, y)
    style_axis(ax)
    maybe_set_ylim(ax, [y])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROC-AUC")


def plot_main_figure(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 5, figsize=(23.0, 4.4))
    study_name = str(payload.get("study_name", "dataset_sensitivity"))
    dataset_display = str(payload.get("dataset_display", payload.get("dataset", "Dataset")))
    specs = payload.get("specs", {})
    panel_specs = [
        ("(a)", "top_M", "Coarse Retrieval", PAPER_COLORS["blue"]),
        ("(b)", "top_K", "Re-ranking Candidates", PAPER_COLORS["red"]),
        ("(c)", "knn_k", "kNN Neighbors", PAPER_COLORS["green"]),
        ("(d)", "d_z", "Representation Dimension", PAPER_COLORS["blue"]),
        ("(e)", "mask_ratio", "Masking Ratio", PAPER_COLORS["green"]),
    ]
    for ax, (panel_label, param, title, color) in zip(axes, panel_specs):
        add_panel_label(ax, panel_label)
        rows = rows_for_param(payload, param)
        if not rows:
            ax.set_visible(False)
            continue
        plot_single_numeric_sweep(ax, rows, float(specs[param]["default"]), color, format_param_axis_label(param))
        ax.set_title(title, pad=8)
        if param == "mask_ratio":
            ax.set_xticks([float(row["value"]) for row in rows])
            ax.set_xticklabels([f"{float(row['value']):.2f}" for row in rows])

    fig.suptitle(f"Sensitivity Analysis on {dataset_display}", fontsize=12.5, y=1.04)
    fig.text(0.995, -0.02, "Diamond: default setting. Star: best result.", ha="right", va="top", fontsize=8.5, color=PAPER_COLORS["gray"])
    fig.tight_layout()
    return save_figure(fig, output_dir / f"{study_name}_main")


def plot_appendix_figure(payload: dict[str, Any], output_dir: Path) -> list[Path] | None:
    appendix_params = [param for param in ["seq_len", "coreset_max_patches_per_scale"] if param in payload.get("specs", {})]
    if not appendix_params:
        return None

    fig, axes = plt.subplots(1, len(appendix_params), figsize=(6.4 * len(appendix_params), 4.2))
    if len(appendix_params) == 1:
        axes = [axes]
    study_name = str(payload.get("study_name", "dataset_sensitivity"))
    dataset_display = str(payload.get("dataset_display", payload.get("dataset", "Dataset")))
    specs = payload.get("specs", {})

    for ax, param, panel_label, color in zip(
        axes,
        appendix_params,
        ["(a)", "(b)"],
        [PAPER_COLORS["green"], PAPER_COLORS["red"]],
    ):
        add_panel_label(ax, panel_label)
        rows = rows_for_param(payload, param)
        if not rows:
            ax.set_visible(False)
            continue

        if param == "coreset_max_patches_per_scale":
            x = np.arange(1, len(rows) + 1, dtype=np.int32)
            y = roc_values(rows)
            x_labels = [str(row.get("value_label", row.get("value"))) for row in rows]
            ax.plot(
                x,
                y,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5.8,
                markerfacecolor="white",
                markeredgewidth=1.0,
            )
            default_value = str(specs[param]["default"])
            default_idx = next((idx for idx, row in enumerate(rows) if str(row.get("value")) == default_value), -1)
            mark_default_point(ax, x, y, default_idx)
            mark_best_point(ax, x, y)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel(format_param_axis_label(param))
            ax.set_title("Memory Budget", pad=8)
            maybe_set_ylim(ax, [y])
            style_axis(ax)
            ax.set_ylabel("ROC-AUC")
        else:
            plot_single_numeric_sweep(ax, rows, float(specs[param]["default"]), color, format_param_axis_label(param))
            ax.set_title("Window Length", pad=8)

    fig.suptitle(f"Appendix Sensitivity Curves on {dataset_display}", fontsize=12.5, y=1.03)
    fig.text(0.995, -0.02, "Diamond: default setting. Star: best result.", ha="right", va="top", fontsize=8.5, color=PAPER_COLORS["gray"])
    fig.tight_layout()
    return save_figure(fig, output_dir / f"{study_name}_appendix")


def main() -> None:
    if not HAS_MATPLOTLIB or plt is None:
        raise RuntimeError("matplotlib is required to plot sensitivity figures.")

    configure_paper_style()
    args = parse_args()
    summary_path = Path(args.summary_json)
    payload = load_payload(summary_path)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_paths = plot_main_figure(payload, output_dir)
    appendix_paths = plot_appendix_figure(payload, output_dir)

    for path in main_paths:
        print(f"[SensitivityPlot] main_figure={path}")
    if appendix_paths is not None:
        for path in appendix_paths:
            print(f"[SensitivityPlot] appendix_figure={path}")


if __name__ == "__main__":
    main()
