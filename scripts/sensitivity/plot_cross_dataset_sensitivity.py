from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    PolyCollection = None
    Line2D = None
    HAS_MATPLOTLIB = False


DEFAULT_PARAMS = ["d_z", "knn_k", "mask_ratio", "top_K", "top_M"]
DEFAULT_DATASET_ORDER = ["GENESIS", "MSL", "PSM", "SMAP", "SMD", "GECCO", "SWAT"]

PAPER_COLORS = [
    "#1f4e79",
    "#b23a48",
    "#3a7d44",
    "#d9a441",
    "#6a4c93",
    "#ff7f50",
    "#264653",
]

PARAM_DISPLAY = {
    "top_M": ("Top-M", "Coarse Retrieval Windows", "Coarse Retrieval ($M$)"),
    "top_K": ("Top-K", "Fine Re-ranking Candidates", "Fine Re-ranking ($K$)"),
    "knn_k": ("kNN-k", "kNN Neighbors", "kNN Neighbors ($k$)"),
    "d_z": ("$d_z$", "Latent Dimension", "Latent Dimension ($d_z$)"),
    "mask_ratio": ("Mask Ratio", "Mask Ratio", "Mask Ratio"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cross-dataset sensitivity-analysis figures from the aggregated summary CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="main-result/sensitivity_summary_all.csv",
        help="Path to the aggregated sensitivity CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="main-result/figures/cross_dataset_sensitivity",
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=DEFAULT_PARAMS,
        help="Parameter groups to plot.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional explicit dataset list. If omitted, uses datasets shared by all requested params.",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="cdf_mean",
        help="Score key to filter by, e.g. cdf_mean.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "pr_auc", "point_best_f1", "pa_best_f1"],
        help="Metric column to visualize.",
    )
    parser.add_argument(
        "--raw-scale",
        action="store_true",
        help="Plot metric values on their original scale instead of multiplying by 100.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Global font scaling factor for titles, axes, ticks, and legend.",
    )
    parser.add_argument(
        "--annotation-fontsize",
        type=float,
        default=8.5,
        help="Font size for per-point numeric annotations.",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=27.0,
        help="3D view elevation angle in degrees.",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=-58.0,
        help="3D view azimuth angle in degrees.",
    )
    parser.add_argument(
        "--title-style",
        type=str,
        default="full",
        choices=["full", "short", "none"],
        help="Whether to use full titles, short titles, or no titles.",
    )
    parser.add_argument(
        "--y-spacing",
        type=float,
        default=1.0,
        help="Spacing multiplier between dataset layers on the y axis.",
    )
    parser.add_argument(
        "--extra-gap-after-dataset",
        type=str,
        default="",
        help="Insert an additional y-axis gap after this dataset key, e.g. PSM.",
    )
    parser.add_argument(
        "--extra-gap-size",
        type=float,
        default=0.0,
        help="Additional y-axis gap size inserted after --extra-gap-after-dataset.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def configure_style(font_scale: float = 1.0) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 10 * font_scale,
            "axes.titlesize": 24 * font_scale,
            "axes.labelsize": 13 * font_scale,
            "xtick.labelsize": 11 * font_scale,
            "ytick.labelsize": 11.5 * font_scale,
            "legend.fontsize": 11 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def value_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def format_value_label(value: str) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.2f}"


def filter_rows(rows: list[dict[str, str]], score_key: str, metric: str | None = None) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("score_key") != score_key:
            continue
        if metric is not None:
            metric_value = to_float(row.get(metric))
            if not np.isfinite(metric_value):
                continue
        filtered.append(row)
    return filtered


def available_datasets_by_param(rows: list[dict[str, str]], params: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        param = row.get("param", "")
        dataset = row.get("dataset", "")
        if param in params and dataset:
            grouped[param].add(dataset)
    return {param: sorted(grouped.get(param, set())) for param in params}


def infer_dataset_order(shared_datasets: set[str]) -> list[str]:
    ordered = [dataset for dataset in DEFAULT_DATASET_ORDER if dataset in shared_datasets]
    remaining = sorted(shared_datasets.difference(ordered))
    return ordered + remaining


def choose_datasets(rows: list[dict[str, str]], params: list[str], requested: list[str] | None) -> list[str]:
    by_param = available_datasets_by_param(rows, params)
    if requested:
        return requested
    shared: set[str] | None = None
    for param in params:
        current = set(by_param.get(param, []))
        shared = current if shared is None else shared.intersection(current)
    return infer_dataset_order(shared or set())


def collect_param_values(rows: list[dict[str, str]], param: str, datasets: list[str]) -> list[str]:
    values = {
        str(row["value"])
        for row in rows
        if row.get("param") == param and row.get("dataset") in set(datasets)
    }
    return sorted(values, key=value_sort_key)


def build_series(
    rows: list[dict[str, str]],
    param: str,
    datasets: list[str],
    metric: str,
    values: list[str],
    scale: float,
) -> tuple[list[str], list[np.ndarray]]:
    row_map: dict[tuple[str, str], float] = {}
    dataset_display: dict[str, str] = {}
    for row in rows:
        if row.get("param") != param:
            continue
        dataset = row.get("dataset", "")
        value = str(row.get("value"))
        if dataset not in datasets or value not in values:
            continue
        row_map[(dataset, value)] = to_float(row.get(metric))
        dataset_display[dataset] = row.get("dataset_display", dataset) or dataset

    labels: list[str] = []
    series: list[np.ndarray] = []
    for dataset in datasets:
        labels.append(dataset_display.get(dataset, dataset))
        y = np.asarray([row_map.get((dataset, value), float("nan")) * scale for value in values], dtype=np.float64)
        series.append(y)
    return labels, series


def make_waterfall_polygons(x: np.ndarray, series: list[np.ndarray]) -> list[list[tuple[float, float]]]:
    verts: list[list[tuple[float, float]]] = []
    for y in series:
        points = [(float(xi), float(zi if np.isfinite(zi) else 0.0)) for xi, zi in zip(x, y)]
        verts.append([(float(x[0]), 0.0), *points, (float(x[-1]), 0.0)])
    return verts


def style_3d_axis(ax) -> None:
    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.04)
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.6
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.6
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.6
    ax.xaxis._axinfo["grid"]["alpha"] = 0.25
    ax.yaxis._axinfo["grid"]["alpha"] = 0.20
    ax.zaxis._axinfo["grid"]["alpha"] = 0.30


def line_aligned_label_offset(
    x: np.ndarray,
    z: np.ndarray,
    index: int,
    raw_scale: bool,
    tangential_step: float,
    vertical_lift: float,
) -> tuple[float, float]:
    if len(x) <= 1:
        return (-tangential_step, vertical_lift)

    if index == 0:
        dx = float(x[1] - x[0])
        dz = float(z[1] - z[0])
    elif index == len(x) - 1:
        dx = float(x[-1] - x[-2])
        dz = float(z[-1] - z[-2])
    else:
        dx = float(x[index + 1] - x[index - 1])
        dz = float(z[index + 1] - z[index - 1])

    z_scale = 24.0 if raw_scale else 1.0
    vx = dx
    vz = dz * z_scale
    norm = max((vx * vx + vz * vz) ** 0.5, 1e-9)
    ux = vx / norm
    uz = vz / norm

    # Move against the local line direction so the label sits slightly up-left of the point.
    x_offset = -ux * tangential_step
    z_offset = -uz * (tangential_step / z_scale) + vertical_lift
    return x_offset, z_offset


def annotate_series(
    ax,
    x: np.ndarray,
    y_pos: float,
    z: np.ndarray,
    color: str,
    raw_scale: bool,
    annotation_fontsize: float,
    dataset_key: str,
    dataset_index: int,
    dataset_count: int,
    y_spacing: float,
) -> None:
    centered = dataset_index - (dataset_count - 1) / 2.0
    x_offset = centered * 0.075
    y_offset = centered * 0.12 * y_spacing
    base_z_offset = 0.012 if raw_scale else 0.7
    z_offset = base_z_offset + (abs(centered) * (0.008 if raw_scale else 0.35))
    ha = "center"
    if centered < -0.25:
        ha = "right"
    elif centered > 0.25:
        ha = "left"
    if dataset_key == "SMAP":
        # Keep SMAP nudged away from PSM, but with a smaller depth shift to avoid looking "pushed back".
        x_offset += 0.016
        y_offset += 0.10 * y_spacing
        z_offset += 0.002 if raw_scale else 0.10
        ha = "left"
    if dataset_key == "SWAT":
        # Pull SWAT labels a bit closer to the line.
        x_offset *= 0.68
        y_offset *= 0.40
        z_offset -= 0.006 if raw_scale else 0.24
        ha = "left"
    for index, (xi, zi) in enumerate(zip(x, z)):
        if not np.isfinite(zi):
            continue
        point_x_offset = x_offset
        point_y_offset = y_offset
        point_z_offset = z_offset
        point_ha = ha
        if dataset_key == "MSL":
            point_x_offset, point_z_offset = line_aligned_label_offset(
                x,
                z,
                index,
                raw_scale,
                tangential_step=0.18,
                vertical_lift=base_z_offset + (0.008 if raw_scale else 0.24),
            )
            point_y_offset = 0.0
            point_ha = "right"
        elif dataset_key == "PSM":
            point_x_offset, point_z_offset = line_aligned_label_offset(
                x,
                z,
                index,
                raw_scale,
                tangential_step=0.165,
                vertical_lift=base_z_offset + (0.007 if raw_scale else 0.22),
            )
            point_y_offset = 0.0
            point_ha = "right"
        ax.scatter([xi], [y_pos], [zi], color=color, s=18, depthshade=False)
        label = f"{zi:.3f}" if raw_scale else f"{zi:.1f}"
        ax.text(
            xi + point_x_offset,
            y_pos + point_y_offset,
            zi + point_z_offset,
            label,
            fontsize=annotation_fontsize,
            ha=point_ha,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.15},
        )


def plot_param_figure(
    param: str,
    values: list[str],
    dataset_labels: list[str],
    dataset_keys: list[str],
    series: list[np.ndarray],
    output_dir: Path,
    metric: str,
    raw_scale: bool,
    annotation_fontsize: float,
    view_elev: float,
    view_azim: float,
    title_style: str,
    y_spacing: float,
    extra_gap_after_dataset: str,
    extra_gap_size: float,
) -> list[Path]:
    fig = plt.figure(figsize=(11.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(len(values), dtype=np.float64)
    y_positions = np.arange(len(dataset_labels), dtype=np.float64) * y_spacing
    if extra_gap_after_dataset:
        try:
            gap_index = dataset_keys.index(extra_gap_after_dataset)
        except ValueError:
            gap_index = -1
        if gap_index >= 0 and extra_gap_size > 0:
            y_positions[gap_index + 1 :] += extra_gap_size
    colors = [PAPER_COLORS[idx % len(PAPER_COLORS)] for idx in range(len(dataset_labels))]

    poly = PolyCollection(
        make_waterfall_polygons(x, series),
        facecolors=colors,
        edgecolors=colors,
        linewidths=1.0,
        alpha=0.18,
    )
    ax.add_collection3d(poly, zs=y_positions, zdir="y")

    legend_handles: list[Line2D] = []
    z_max = 0.0
    for dataset_index, (dataset_key, y_pos, label, z, color) in enumerate(
        zip(dataset_keys, y_positions, dataset_labels, series, colors)
    ):
        z_safe = np.where(np.isfinite(z), z, np.nan)
        finite_mask = np.isfinite(z_safe)
        if not finite_mask.any():
            print(f"[Plot] warning param={param}, dataset={dataset_key}: all {metric} values are missing")
        if finite_mask.any():
            z_max = max(z_max, float(np.nanmax(z_safe)))
        ax.plot(
            x[finite_mask],
            np.full(int(finite_mask.sum()), y_pos, dtype=np.float64),
            z_safe[finite_mask],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
        )
        annotate_series(
            ax,
            x[finite_mask],
            y_pos,
            z_safe[finite_mask],
            color,
            raw_scale,
            annotation_fontsize,
            dataset_key,
            dataset_index,
            len(dataset_labels),
            y_spacing,
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=2.0, marker="o", markersize=5, label=label))

    short_title, title_text, x_label = PARAM_DISPLAY.get(param, (param, param, param))
    if title_style == "short":
        ax.set_title(short_title, pad=6)
    elif title_style == "full":
        ax.set_title(f"{title_text} Sensitivity Across Datasets", pad=10)
    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel("Datasets", labelpad=14)
    z_label_suffix = "" if raw_scale else " (%)"
    ax.set_zlabel(metric.upper().replace("_", "-") + z_label_suffix, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([format_value_label(value) for value in values], rotation=0, ha="center")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(dataset_labels)
    if len(x) > 1:
        ax.set_xlim(float(x.min()) - 0.55, float(x.max()) + 0.15)
    else:
        ax.set_xlim(float(x.min()) - 0.55, float(x.min()) + 1.0)
    ax.set_ylim(-0.4 * y_spacing, float(y_positions[-1]) + 0.4 * y_spacing)
    if raw_scale:
        ax.set_zlim(0.0, min(1.0, max(0.05, z_max + 0.08)))
    else:
        ax.set_zlim(0.0, min(100.0, max(5.0, z_max + 8.0)))
    ax.view_init(elev=view_elev, azim=view_azim)
    style_3d_axis(ax)
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.92)

    base = output_dir / f"{param}_{metric}_waterfall"
    saved_paths: list[Path] = []
    for suffix in (".png", ".pdf"):
        save_path = base.with_suffix(suffix)
        fig.savefig(save_path, dpi=250 if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(save_path)
    plt.close(fig)
    return saved_paths


def main() -> None:
    args = parse_args()
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting but is not available.")

    configure_style(args.font_scale)
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    score_filtered = filter_rows(rows, score_key=args.score_key)
    filtered = filter_rows(rows, score_key=args.score_key, metric=args.metric)
    params = [param for param in args.params if param in PARAM_DISPLAY]
    datasets = choose_datasets(score_filtered, params, args.datasets)
    if not datasets:
        raise RuntimeError("No datasets available for the requested params after filtering.")

    print(f"[Plot] input_csv={input_csv}")
    print(f"[Plot] score_key={args.score_key}, metric={args.metric}")
    print(f"[Plot] raw_scale={args.raw_scale}")
    print(f"[Plot] font_scale={args.font_scale}, annotation_fontsize={args.annotation_fontsize}")
    print(f"[Plot] view_elev={args.view_elev}, view_azim={args.view_azim}")
    print(f"[Plot] title_style={args.title_style}")
    print(f"[Plot] y_spacing={args.y_spacing}")
    print(
        f"[Plot] extra_gap_after_dataset={args.extra_gap_after_dataset or 'None'}, "
        f"extra_gap_size={args.extra_gap_size}"
    )
    print(f"[Plot] datasets={datasets}")

    for param in params:
        values = collect_param_values(score_filtered, param, datasets)
        if not values:
            print(f"[Plot] skip param={param}: no rows found")
            continue
        scale = 1.0 if args.raw_scale else 100.0
        dataset_labels, series = build_series(score_filtered, param, datasets, args.metric, values, scale)
        saved_paths = plot_param_figure(
            param=param,
            values=values,
            dataset_labels=dataset_labels,
            dataset_keys=datasets,
            series=series,
            output_dir=output_dir,
            metric=args.metric,
            raw_scale=args.raw_scale,
            annotation_fontsize=args.annotation_fontsize,
            view_elev=args.view_elev,
            view_azim=args.view_azim,
            title_style=args.title_style,
            y_spacing=args.y_spacing,
            extra_gap_after_dataset=args.extra_gap_after_dataset,
            extra_gap_size=args.extra_gap_size,
        )
        print(
            f"[Plot] saved param={param}, values={len(values)}, datasets={len(dataset_labels)} -> "
            + ", ".join(str(path) for path in saved_paths)
        )


if __name__ == "__main__":
    main()
