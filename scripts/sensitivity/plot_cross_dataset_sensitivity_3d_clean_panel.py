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
    raise RuntimeError("matplotlib is required to build the 3D sensitivity panel.") from exc


DEFAULT_DATASETS = ["MSL", "PSM", "SMAP", "SMD", "SWAT"]
DEFAULT_PARAMS = ["top_M", "top_K", "knn_k", "d_z", "mask_ratio"]
PANEL_TITLES = {
    "top_M": "(a) Top-M",
    "top_K": "(b) Top-K",
    "knn_k": "(c) kNN-k",
    "d_z": "(d) $d_z$",
    "mask_ratio": "(e) Mask Ratio",
}
X_LABELS = {
    "top_M": "Coarse Retrieval ($M$)",
    "top_K": "Fine Re-ranking ($K$)",
    "knn_k": "kNN Neighbors ($k$)",
    "d_z": "Latent Dimension ($d_z$)",
    "mask_ratio": "Mask Ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaner 3D multi-panel sensitivity figure.")
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
        default="main-result/figures/cross_dataset_sensitivity_main5_target_closest_default_stable_roc_auc_raw_scale/combined_sensitivity_3d_clean_panel.png",
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
    lower = max(0.0, min(values) - 0.03)
    upper = min(1.0, max(values) + 0.02)
    return lower, upper


def draw_panel(
    ax,
    param: str,
    values: list[str],
    dataset_labels: list[str],
    series: list[np.ndarray],
    colors: list[str],
    metric: str,
    zlim: tuple[float, float],
    show_zlabel: bool,
) -> None:
    x = np.arange(len(values), dtype=np.float64)
    y_positions = np.arange(len(dataset_labels), dtype=np.float64)

    poly = PolyCollection(
        make_waterfall_polygons(x, series),
        facecolors=colors,
        edgecolors=colors,
        linewidths=0.8,
        alpha=0.14,
    )
    ax.add_collection3d(poly, zs=y_positions, zdir="y")

    for y_pos, z, color in zip(y_positions, series, colors):
        finite_mask = np.isfinite(z)
        ax.plot(
            x[finite_mask],
            np.full(int(finite_mask.sum()), y_pos, dtype=np.float64),
            z[finite_mask],
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=3.6,
        )

    ax.set_title(PANEL_TITLES[param], loc="left", pad=8, fontweight="bold", fontsize=13)
    ax.set_xlabel(X_LABELS[param], labelpad=8)
    ax.set_ylabel("")
    ax.set_zlabel(metric.upper().replace("_", "-") if show_zlabel else "", labelpad=7)
    ax.set_xticks(x)
    ax.set_xticklabels([format_value_label(v) for v in values], rotation=0, ha="center")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(dataset_labels)
    ax.set_xlim(float(x.min()), float(x.max()) if len(x) > 1 else float(x.min()) + 1.0)
    ax.set_ylim(-0.3, float(len(dataset_labels) - 1) + 0.3)
    ax.set_zlim(*zlim)
    ax.set_zticks(np.linspace(zlim[0], zlim[1], 4))
    ax.view_init(elev=26, azim=-56)
    style_3d_axis(ax)


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

    zlim = compute_global_z_limits(series_map)

    fig = plt.figure(figsize=(18.2, 9.8))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.98])

    axes = {
        "top_M": fig.add_subplot(grid[0, 0:2], projection="3d"),
        "top_K": fig.add_subplot(grid[0, 2:4], projection="3d"),
        "knn_k": fig.add_subplot(grid[0, 4:6], projection="3d"),
        "d_z": fig.add_subplot(grid[1, 1:3], projection="3d"),
        "mask_ratio": fig.add_subplot(grid[1, 3:5], projection="3d"),
    }

    for idx, param in enumerate(params):
        draw_panel(
            ax=axes[param],
            param=param,
            values=values_map[param],
            dataset_labels=labels_map[param],
            series=series_map[param],
            colors=colors,
            metric=args.metric,
            zlim=zlim,
            show_zlabel=idx in {0, 3},
        )

    legend_handles = [
        Line2D([0], [0], color=color, lw=2.0, marker="o", markersize=5, label=dataset)
        for dataset, color in zip(datasets, colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(datasets),
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.8,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.92, bottom=0.06, wspace=0.0, hspace=0.10)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[3DCleanPanel] wrote: {output_path}")
    print(f"[3DCleanPanel] wrote: {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
