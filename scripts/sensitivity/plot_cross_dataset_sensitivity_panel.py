from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sensitivity.plot_cross_dataset_sensitivity import (  # noqa: E402
    PAPER_COLORS,
    PARAM_DISPLAY,
    build_series,
    collect_param_values,
    configure_style,
    filter_rows,
    format_value_label,
    load_rows,
    make_waterfall_polygons,
    style_3d_axis,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required to build the combined sensitivity figure.") from exc


DEFAULT_DATASETS = ["MSL", "PSM", "SMAP", "SMD", "SWAT"]
DEFAULT_PARAMS = ["top_M", "top_K", "knn_k", "d_z", "mask_ratio"]
PANEL_TITLES = {
    "top_M": "(a) Top-M",
    "top_K": "(b) Top-K",
    "knn_k": "(c) kNN-k",
    "d_z": "(d) $d_z$",
    "mask_ratio": "(e) Mask Ratio",
}
DEFAULT_POINT_NOTE = "Defaults: top_M=50, top_K=20, knn_k=5, d_z=128, mask_ratio=0.25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a polished multi-panel sensitivity figure.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="main-result/sensitivity_summary_main5_target_closest_default_stable_scores.csv",
        help="Input sensitivity CSV.",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="target_closest_default_stable",
        help="Score key to filter from the CSV.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "pr_auc", "point_best_f1", "pa_best_f1"],
        help="Metric to visualize.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset order to display.",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=DEFAULT_PARAMS,
        help="Parameter order to display.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="main-result/figures/cross_dataset_sensitivity_main5_target_closest_default_stable_roc_auc_raw_scale/combined_sensitivity_panel.png",
        help="Output path for the combined figure. A PDF is also saved alongside it.",
    )
    return parser.parse_args()


def compute_global_z_limits(series_map: dict[str, list[np.ndarray]]) -> tuple[float, float]:
    values: list[float] = []
    for series_list in series_map.values():
        for arr in series_list:
            finite = arr[np.isfinite(arr)]
            values.extend(float(v) for v in finite)
    if not values:
        return 0.0, 1.0
    z_min = min(values)
    z_max = max(values)
    lower = max(0.0, z_min - 0.035)
    upper = min(1.0, z_max + 0.03)
    return lower, upper


def draw_waterfall_panel(
    ax,
    param: str,
    values: list[str],
    dataset_labels: list[str],
    dataset_keys: list[str],
    series: list[np.ndarray],
    colors: list[str],
    metric: str,
    zlim: tuple[float, float],
) -> None:
    x = np.arange(len(values), dtype=np.float64)
    y_positions = np.arange(len(dataset_labels), dtype=np.float64)

    poly = PolyCollection(
        make_waterfall_polygons(x, series),
        facecolors=colors,
        edgecolors=colors,
        linewidths=0.9,
        alpha=0.16,
    )
    ax.add_collection3d(poly, zs=y_positions, zdir="y")

    for y_pos, z, color in zip(y_positions, series, colors):
        z_safe = np.where(np.isfinite(z), z, np.nan)
        finite_mask = np.isfinite(z_safe)
        ax.plot(
            x[finite_mask],
            np.full(int(finite_mask.sum()), y_pos, dtype=np.float64),
            z_safe[finite_mask],
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=3.8,
        )

    _, _, x_label = PARAM_DISPLAY.get(param, (param, param, param))
    ax.set_title(PANEL_TITLES.get(param, param), loc="left", pad=10, fontweight="bold")
    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel("Dataset", labelpad=10)
    ax.set_zlabel(metric.upper().replace("_", "-"), labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([format_value_label(v) for v in values], rotation=0, ha="center")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(dataset_labels)
    ax.set_xlim(float(x.min()), float(x.max()) if len(x) > 1 else float(x.min()) + 1.0)
    ax.set_ylim(-0.35, float(len(dataset_labels) - 1) + 0.35)
    ax.set_zlim(*zlim)
    ax.set_zticks(np.linspace(zlim[0], zlim[1], 4))
    ax.view_init(elev=27, azim=-58)
    style_3d_axis(ax)


def add_info_panel(ax, dataset_keys: list[str], colors: list[str], source_map: dict[str, str]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.0, 0.96, "Panel Notes", fontsize=16, fontweight="bold", ha="left", va="top")
    ax.text(0.0, 0.88, "Metric: ROC-AUC", fontsize=11, ha="left", va="top")
    ax.text(0.0, 0.82, "Scale: raw (0-1)", fontsize=11, ha="left", va="top")
    ax.text(0.0, 0.76, DEFAULT_POINT_NOTE, fontsize=10.5, ha="left", va="top")

    ax.text(0.0, 0.66, "Datasets", fontsize=12, fontweight="bold", ha="left", va="top")
    y = 0.60
    for dataset, color in zip(dataset_keys, colors):
        ax.add_line(Line2D([0.02, 0.12], [y, y], color=color, linewidth=3.2))
        ax.plot([0.07], [y], marker="o", color=color, markersize=5)
        ax.text(0.16, y, dataset, fontsize=11, va="center", ha="left")
        y -= 0.07

    ax.text(0.50, 0.66, "Selected Source", fontsize=12, fontweight="bold", ha="left", va="top")
    y = 0.60
    for dataset in dataset_keys:
        source_label = source_map.get(dataset, "NA")
        ax.text(0.50, y, f"{dataset}: {source_label}", fontsize=10.5, va="center", ha="left")
        y -= 0.07

    note_y = 0.16
    note_text = (
        "Layout rationale:\n"
        "Top row groups retrieval controls.\n"
        "Bottom row shows representation controls\n"
        "plus a shared note/legend panel."
    )
    ax.text(0.0, note_y, note_text, fontsize=10.5, ha="left", va="top", linespacing=1.4)

    ax.add_patch(
        plt.Rectangle((0.0, 0.02), 0.98, 0.94, fill=False, linewidth=1.0, edgecolor="#c7c7c7")
    )


def main() -> None:
    args = parse_args()
    configure_style()

    rows = load_rows(Path(args.input_csv))
    filtered = filter_rows(rows, score_key=args.score_key, metric=args.metric)
    if not filtered:
        raise RuntimeError(f"No rows found for score_key={args.score_key} in {args.input_csv}")

    datasets = args.datasets
    params = args.params
    colors = [PAPER_COLORS[idx % len(PAPER_COLORS)] for idx in range(len(datasets))]

    series_map: dict[str, list[np.ndarray]] = {}
    values_map: dict[str, list[str]] = {}
    labels_map: dict[str, list[str]] = {}
    for param in params:
        values = collect_param_values(filtered, param, datasets)
        labels, series = build_series(filtered, param, datasets, args.metric, values, scale=1.0)
        values_map[param] = values
        labels_map[param] = labels
        series_map[param] = series

    source_map: dict[str, str] = {}
    for row in filtered:
        source = row.get("original_score_key", "")
        dataset = row.get("dataset", "")
        if dataset and source:
            source_map[dataset] = source

    zlim = compute_global_z_limits(series_map)

    fig = plt.figure(figsize=(18.5, 11.2))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.04], height_ratios=[1.0, 1.0])

    axes = {
        "top_M": fig.add_subplot(grid[0, 0], projection="3d"),
        "top_K": fig.add_subplot(grid[0, 1], projection="3d"),
        "knn_k": fig.add_subplot(grid[0, 2], projection="3d"),
        "d_z": fig.add_subplot(grid[1, 0], projection="3d"),
        "mask_ratio": fig.add_subplot(grid[1, 1], projection="3d"),
    }
    info_ax = fig.add_subplot(grid[1, 2])

    for param, ax in axes.items():
        draw_waterfall_panel(
            ax=ax,
            param=param,
            values=values_map[param],
            dataset_labels=labels_map[param],
            dataset_keys=datasets,
            series=series_map[param],
            colors=colors,
            metric=args.metric,
            zlim=zlim,
        )

    add_info_panel(info_ax, datasets, colors, source_map)
    fig.suptitle("Sensitivity Analysis Across Five Main Datasets", fontsize=18, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.04, top=0.93, wspace=0.05, hspace=0.14)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PanelPlot] wrote: {output_path}")
    print(f"[PanelPlot] wrote: {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
