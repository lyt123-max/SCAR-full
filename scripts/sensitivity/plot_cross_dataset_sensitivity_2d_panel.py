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
    to_float,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required to build the 2D sensitivity panel.") from exc


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
DEFAULT_POINT_NOTE = "Default points: top_M=50, top_K=20, knn_k=5, d_z=128, mask_ratio=0.25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 2D multi-panel cross-dataset sensitivity figure.")
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
        default="main-result/figures/cross_dataset_sensitivity_main5_target_closest_default_stable_roc_auc_raw_scale/combined_sensitivity_2d_panel.png",
        help="Output path for the combined figure. A PDF is also saved alongside it.",
    )
    return parser.parse_args()


def compute_global_limits(series_map: dict[str, list[np.ndarray]]) -> tuple[float, float]:
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


def infer_default_value(param: str) -> str:
    defaults = {
        "top_M": "50",
        "top_K": "20",
        "knn_k": "5",
        "d_z": "128",
        "mask_ratio": "0.25",
    }
    return defaults[param]


def add_info_panel(ax, datasets: list[str], colors: list[str], source_map: dict[str, str], metric: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.0, 0.96, "Legend & Notes", fontsize=14, fontweight="bold", ha="left", va="top")
    ax.text(0.0, 0.88, f"Metric: {metric.upper().replace('_', '-')}", fontsize=10.5, ha="left", va="top")
    ax.text(0.0, 0.82, "Scale: raw (0-1)", fontsize=10.5, ha="left", va="top")
    ax.text(0.0, 0.76, DEFAULT_POINT_NOTE, fontsize=9.8, ha="left", va="top")

    y = 0.64
    for dataset, color in zip(datasets, colors):
        ax.add_line(Line2D([0.02, 0.12], [y, y], color=color, linewidth=2.6))
        ax.plot([0.07], [y], marker="o", color=color, markersize=5)
        ax.text(0.16, y, dataset, fontsize=10.5, va="center", ha="left")
        ax.text(0.48, y, source_map.get(dataset, "NA"), fontsize=10.0, va="center", ha="left", color="#444444")
        y -= 0.09

    ax.text(
        0.0,
        0.18,
        "Dashed vertical line marks the default setting\nfor each parameter sweep.",
        fontsize=9.8,
        ha="left",
        va="top",
        linespacing=1.45,
        color="#444444",
    )
    ax.add_patch(plt.Rectangle((0.0, 0.02), 0.98, 0.94, fill=False, linewidth=0.9, edgecolor="#c7c7c7"))


def main() -> None:
    args = parse_args()
    configure_style()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#6f6f6f",
            "axes.linewidth": 0.8,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
        }
    )

    rows = load_rows(Path(args.input_csv))
    filtered = filter_rows(rows, score_key=args.score_key, metric=args.metric)
    if not filtered:
        raise RuntimeError(f"No rows found for score_key={args.score_key} in {args.input_csv}")

    datasets = args.datasets
    params = args.params
    colors = [PAPER_COLORS[idx % len(PAPER_COLORS)] for idx in range(len(datasets))]

    series_map: dict[str, list[np.ndarray]] = {}
    values_map: dict[str, list[str]] = {}
    for param in params:
        values = collect_param_values(filtered, param, datasets)
        _, series = build_series(filtered, param, datasets, args.metric, values, scale=1.0)
        values_map[param] = values
        series_map[param] = series

    source_map: dict[str, str] = {}
    for row in filtered:
        dataset = row.get("dataset", "")
        source = row.get("original_score_key", "")
        if dataset and source:
            source_map[dataset] = source

    y_min, y_max = compute_global_limits(series_map)

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.8), constrained_layout=False)
    axes = np.asarray(axes, dtype=object)
    layout = {
        "top_M": axes[0, 0],
        "top_K": axes[0, 1],
        "knn_k": axes[0, 2],
        "d_z": axes[1, 0],
        "mask_ratio": axes[1, 1],
    }
    info_ax = axes[1, 2]

    for param, ax in layout.items():
        values = values_map[param]
        x = np.arange(len(values), dtype=np.float64)
        default_value = infer_default_value(param)
        default_index = values.index(default_value) if default_value in values else None

        for dataset, color, series in zip(datasets, colors, series_map[param]):
            finite_mask = np.isfinite(series)
            ax.plot(
                x[finite_mask],
                series[finite_mask],
                color=color,
                linewidth=2.1,
                marker="o",
                markersize=4.8,
                label=dataset,
            )

        if default_index is not None:
            ax.axvline(default_index, color="#8a8a8a", linestyle="--", linewidth=1.0, alpha=0.9)

        ax.set_title(PANEL_TITLES[param], loc="left", fontsize=14, fontweight="bold")
        ax.set_xlabel(X_LABELS[param])
        ax.set_ylabel(args.metric.upper().replace("_", "-"))
        ax.set_xticks(x)
        ax.set_xticklabels([format_value_label(v) for v in values])
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", alpha=0.9)
        ax.grid(True, axis="x", alpha=0.45)
        ax.tick_params(axis="both", labelsize=9.5)

    add_info_panel(info_ax, datasets, colors, source_map, args.metric)

    fig.suptitle("Sensitivity Analysis Across Five Main Datasets", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.08, wspace=0.18, hspace=0.28)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[2DPanelPlot] wrote: {output_path}")
    print(f"[2DPanelPlot] wrote: {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
