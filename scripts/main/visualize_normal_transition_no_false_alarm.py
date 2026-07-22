from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coremad import CoReMADConfig
from coremad.data import load_raw_dataset_bundle


MAIN5_EXPERIMENTS = {
    "MSL": "msl_baseline_20260412_123051",
    "PSM": "psm_baseline_20260409_134926",
    "SMAP": "smap_baseline_20260412_153337",
    "SMD": "smd_baseline_20260412_120852",
    "SWAT": "swat_baseline_20260412_140435",
}
DEFAULT_SELECTED_SCORE_KEYS = {
    "MSL": "zscore_mean",
    "PSM": "knn_distance",
    "SMAP": "zscore_mean",
    "SMD": "cdf_mean",
    "SWAT": "completion_scale32",
}
SCORE_NPZ_KEYS = {
    "zscore_mean": "zscore_mean_score",
    "cdf_mean": "cdf_mean_score",
    "cdf_max": "cdf_max_score",
    "cdf_softmax": "cdf_softmax_score",
    "cdf_mean_soft_support": "cdf_mean_soft_support_score",
    "raw_max": "raw_max_score",
    "knn_distance": "knn_distance",
    "completion_scale8": "completion_scale8",
    "completion_scale32": "completion_scale32",
    "state_novelty": "state_novelty",
    "final_fused": "final_fused_score",
}
DATASET_COLORS = {
    "MSL": "#355070",
    "PSM": "#6D597A",
    "SMAP": "#B56576",
    "SMD": "#E56B6F",
    "SWAT": "#EAAC8B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find and visualize normal high-change candidate segments where the selected "
            "CoReM-AD score stays below the normal alarm threshold."
        )
    )
    parser.add_argument("--result-root", type=Path, default=ROOT / "main-result")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "main-result" / "sensitivity_summary_main5_mixed_selected_scores_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "main-result" / "figures" / "normal_transition_no_false_alarm",
    )
    parser.add_argument("--datasets", nargs="+", default=list(MAIN5_EXPERIMENTS.keys()))
    parser.add_argument("--left-right-window", type=int, default=96)
    parser.add_argument("--score-radius", type=int, default=16)
    parser.add_argument("--normal-margin", type=int, default=128)
    parser.add_argument("--candidate-step", type=int, default=8)
    parser.add_argument("--min-distance", type=int, default=384)
    parser.add_argument("--top-cases", type=int, default=2)
    parser.add_argument("--plot-pad", type=int, default=320)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--low-score-quantile", type=float, default=0.80)
    parser.add_argument("--smooth-score-window", type=int, default=7)
    parser.add_argument("--max-channels", type=int, default=5)
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": ":",
            "legend.frameon": False,
        }
    )


def read_selected_score_manifest(path: Path) -> dict[str, str]:
    selected = dict(DEFAULT_SELECTED_SCORE_KEYS)
    if not path.exists():
        return selected
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = str(row.get("dataset", "")).strip().upper()
            score = str(row.get("selected_source_score_key", "")).strip()
            if dataset and score:
                selected[dataset] = score
    return selected


def resolve_exp_dir(result_root: Path, dataset: str) -> Path:
    expected = result_root / MAIN5_EXPERIMENTS[dataset]
    if expected.exists():
        return expected
    prefix = dataset.lower()
    candidates = sorted(result_root.glob(f"{prefix}_baseline_*"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find baseline result directory for {dataset} under {result_root}")
    return candidates[-1]


def load_dataset_payload(exp_dir: Path, score_key: str) -> dict[str, Any]:
    config = CoReMADConfig.load(exp_dir / "config.json")
    if not Path(config.data_root).exists():
        local_root = ROOT / "dataset" / "anomaly_detect"
        if local_root.exists():
            config.data_root = str(local_root)
    raw = load_raw_dataset_bundle(config.dataset, config.data_root, config=config)
    diag_path = exp_dir / "test_diagnostic_scores.npz"
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing diagnostic scores: {diag_path}")
    diag = np.load(diag_path, allow_pickle=True)
    npz_key = SCORE_NPZ_KEYS.get(score_key, score_key)
    if npz_key not in diag.files:
        raise KeyError(f"{exp_dir}: selected score {score_key!r} maps to missing key {npz_key!r}")
    labels = np.asarray(diag["labels"], dtype=np.int32)
    test = np.asarray(raw.test, dtype=np.float32)
    score = np.asarray(diag[npz_key], dtype=np.float64)
    if len(test) != len(labels) or len(score) != len(labels):
        raise RuntimeError(
            f"{exp_dir}: length mismatch test={len(test)} labels={len(labels)} score={len(score)}"
        )
    return {
        "dataset": str(config.dataset).upper(),
        "exp_dir": exp_dir,
        "score_key": score_key,
        "npz_key": npz_key,
        "test": test,
        "labels": labels,
        "score": score,
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(arr, kernel, mode="same")


def robust_zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    median = np.nanmedian(arr, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(arr, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = (arr - median[None, :]) / scale[None, :]
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def window_means(z: np.ndarray, centers: np.ndarray, half_window: int) -> tuple[np.ndarray, np.ndarray]:
    n = z.shape[0]
    prefix = np.vstack([np.zeros((1, z.shape[1]), dtype=np.float64), np.cumsum(z, axis=0, dtype=np.float64)])
    left_start = np.clip(centers - half_window, 0, n)
    left_end = np.clip(centers, 0, n)
    right_start = np.clip(centers, 0, n)
    right_end = np.clip(centers + half_window, 0, n)
    left_len = np.maximum(left_end - left_start, 1)[:, None]
    right_len = np.maximum(right_end - right_start, 1)[:, None]
    left = (prefix[left_end] - prefix[left_start]) / left_len
    right = (prefix[right_end] - prefix[right_start]) / right_len
    return left, right


def labels_are_normal(labels: np.ndarray, centers: np.ndarray, margin: int) -> np.ndarray:
    y = (np.asarray(labels, dtype=np.int32) > 0).astype(np.int32)
    prefix = np.r_[0, np.cumsum(y)]
    n = len(labels)
    lo = np.clip(centers - margin, 0, n)
    hi = np.clip(centers + margin, 0, n)
    return (prefix[hi] - prefix[lo]) == 0


def score_max_around(score: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
    score_arr = np.asarray(score, dtype=np.float64)
    out = np.full(centers.shape, np.nan, dtype=np.float64)
    n = len(score_arr)
    for idx, center in enumerate(centers.tolist()):
        lo = max(0, int(center) - int(radius))
        hi = min(n, int(center) + int(radius) + 1)
        out[idx] = float(np.nanmax(score_arr[lo:hi])) if hi > lo else float("nan")
    return out


def select_candidates(payload: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    test = np.asarray(payload["test"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int32)
    score = np.asarray(payload["score"], dtype=np.float64)
    normal_score = score[labels == 0]
    alarm_threshold = float(np.quantile(normal_score, args.threshold_quantile))
    quiet_threshold = float(np.quantile(normal_score, args.low_score_quantile))

    z = robust_zscore(test)
    n = len(z)
    start = max(args.left_right_window, args.normal_margin, args.score_radius)
    end = min(n - max(args.left_right_window, args.normal_margin, args.score_radius), n)
    if end <= start:
        return []
    centers = np.arange(start, end, max(1, int(args.candidate_step)), dtype=np.int64)
    left, right = window_means(z, centers, int(args.left_right_window))
    delta = right - left
    transition_strength = np.sqrt(np.mean(delta * delta, axis=1))
    top_channel_delta = np.max(np.abs(delta), axis=1)
    normal_mask = labels_are_normal(labels, centers, int(args.normal_margin))
    score_peak = score_max_around(score, centers, int(args.score_radius))
    no_alarm = score_peak < alarm_threshold
    quiet = score_peak < quiet_threshold
    eligible = normal_mask & no_alarm & np.isfinite(transition_strength)
    eligible_centers = centers[eligible]
    if eligible_centers.size == 0:
        return []
    eligible_rows = []
    eligible_indices = np.where(eligible)[0]
    for idx in eligible_indices.tolist():
        eligible_rows.append(
            {
                "center": int(centers[idx]),
                "transition_strength": float(transition_strength[idx]),
                "top_channel_delta": float(top_channel_delta[idx]),
                "score_peak": float(score_peak[idx]),
                "score_peak_percentile": float(np.mean(normal_score <= score_peak[idx])),
                "quiet_below_q": bool(quiet[idx]),
                "alarm_threshold": alarm_threshold,
                "quiet_threshold": quiet_threshold,
                "delta_vector": delta[idx],
            }
        )
    eligible_rows.sort(key=lambda row: (row["transition_strength"], row["top_channel_delta"]), reverse=True)
    selected: list[dict[str, Any]] = []
    for row in eligible_rows:
        if all(abs(int(row["center"]) - int(prev["center"])) >= int(args.min_distance) for prev in selected):
            selected.append(row)
        if len(selected) >= int(args.top_cases):
            break
    return selected


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    vals = np.asarray(values, dtype=np.float64)
    if ref.size == 0:
        return np.full(vals.shape, np.nan, dtype=np.float64)
    return np.searchsorted(ref, vals, side="right").astype(np.float64) / float(ref.size)


def choose_channels(delta_vector: np.ndarray, max_channels: int) -> list[int]:
    order = np.argsort(np.abs(np.asarray(delta_vector, dtype=np.float64)))[::-1]
    return [int(idx) for idx in order[: max(1, int(max_channels))].tolist()]


def plot_case(payload: dict[str, Any], case: dict[str, Any], case_idx: int, args: argparse.Namespace) -> Path:
    dataset = payload["dataset"]
    test = np.asarray(payload["test"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=np.int32)
    score = np.asarray(payload["score"], dtype=np.float64)
    normal_score = score[labels == 0]
    center = int(case["center"])
    lo = max(0, center - int(args.plot_pad))
    hi = min(len(score), center + int(args.plot_pad))
    x = np.arange(lo, hi)
    channels = choose_channels(np.asarray(case["delta_vector"]), args.max_channels)
    fig = plt.figure(
        figsize=(11.5, max(7.4, 2.2 * len(channels) + 3.2)),
        constrained_layout=True,
    )
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[max(1.6, 0.78 * len(channels)), 1.0, 0.75],
        hspace=0.18,
    )
    color = DATASET_COLORS.get(dataset, "#355070")

    top_grid = outer[0].subgridspec(len(channels), 1, hspace=0.06)
    top_axes: list[plt.Axes] = []
    for idx, channel in enumerate(channels):
        ax = fig.add_subplot(top_grid[idx, 0], sharex=top_axes[0] if top_axes else None)
        vals = test[lo:hi, channel]
        local_color = plt.cm.tab10(idx % 10)
        ax.plot(x, vals, color=local_color, linewidth=1.15)
        ax.axvline(center, color="#F26419", linewidth=1.3)
        ax.axvspan(center - args.left_right_window, center, color="#4D908E", alpha=0.10, linewidth=0)
        ax.axvspan(center, center + args.left_right_window, color="#F9C74F", alpha=0.13, linewidth=0)
        ax.set_ylabel(f"ch{channel}", rotation=0, labelpad=22, va="center")
        if idx == 0:
            ax.set_title(f"{dataset}: raw time-series channels around a normal high-change case")
            ax.text(
                0.995,
                0.96,
                "candidate center",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#F26419",
            )
        if idx != len(channels) - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        top_axes.append(ax)

    ax = fig.add_subplot(outer[1], sharex=top_axes[0])
    pct = empirical_percentile(normal_score, score[lo:hi])
    ax.plot(x, pct, color=color, linewidth=1.8, label=f"{payload['score_key']} percentile")
    ax.plot(x, rolling_mean(pct, args.smooth_score_window), color="#1A1A1A", linewidth=1.1, alpha=0.75, label="smoothed")
    ax.axhline(args.threshold_quantile, color="#B23A48", linestyle="--", linewidth=1.2, label=f"normal q{int(args.threshold_quantile * 100)} alarm line")
    ax.axhline(args.low_score_quantile, color="#758E4F", linestyle=":", linewidth=1.1, label=f"low-score q{int(args.low_score_quantile * 100)}")
    ax.axvline(center, color="#F26419", linewidth=1.2)
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("normal percentile")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    ax = fig.add_subplot(outer[2], sharex=top_axes[0])
    local_labels = labels[lo:hi] > 0
    ax.fill_between(x, 0, local_labels.astype(float), step="mid", color="#B23A48", alpha=0.35)
    ax.axvline(center, color="#F26419", linewidth=1.2)
    ax.set_yticks([0, 1], ["normal", "anomaly"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("test time index")
    ax.set_title(
        "candidate score peak percentile="
        f"{case['score_peak_percentile']:.1%}, transition strength={case['transition_strength']:.3f}"
    )

    out_path = args.output_dir / f"{dataset.lower()}_normal_transition_case_{case_idx}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[NormalTransition] wrote {out_path}")
    return out_path


def plot_overview(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not rows:
        return
    datasets = [row["dataset"] for row in rows]
    strengths = np.asarray([float(row["transition_strength"]) for row in rows], dtype=np.float64)
    score_pct = np.asarray([float(row["score_peak_percentile"]) for row in rows], dtype=np.float64)
    colors = [DATASET_COLORS.get(ds, "#888888") for ds in datasets]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(rows), dtype=np.float64)
    ax.barh(y, strengths, color=colors, alpha=0.86)
    for yi, row, pct in zip(y, rows, score_pct):
        ax.text(
            float(row["transition_strength"]) + max(strengths) * 0.015,
            yi,
            f"score peak {pct:.0%}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y, [f"{row['dataset']} #{row['case_rank']}" for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("raw-signal left/right shift strength")
    ax.set_title("Normal high-change candidates retained below the alarm line")
    fig.tight_layout()
    out_path = args.output_dir / "normal_transition_no_false_alarm_overview.png"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[NormalTransition] wrote {out_path}")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[NormalTransition] wrote {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    selected_scores = read_selected_score_manifest(args.manifest)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "definition": (
            "Candidates are normal-only test regions with large raw-signal left/right mean shift "
            "and selected score peak below the normal alarm quantile. They are heuristic "
            "operating-condition-change candidates, not ground-truth regime labels."
        ),
        "parameters": {
            "left_right_window": args.left_right_window,
            "score_radius": args.score_radius,
            "normal_margin": args.normal_margin,
            "threshold_quantile": args.threshold_quantile,
            "low_score_quantile": args.low_score_quantile,
        },
        "datasets": {},
    }

    for dataset in [str(ds).strip().upper() for ds in args.datasets]:
        if dataset not in MAIN5_EXPERIMENTS:
            print(f"[NormalTransition] skip unknown dataset: {dataset}")
            continue
        exp_dir = resolve_exp_dir(args.result_root, dataset)
        score_key = selected_scores.get(dataset, DEFAULT_SELECTED_SCORE_KEYS[dataset])
        payload = load_dataset_payload(exp_dir, score_key)
        candidates = select_candidates(payload, args)
        summary["datasets"][dataset] = {
            "experiment_dir": str(exp_dir),
            "selected_score_key": score_key,
            "n_candidates": len(candidates),
        }
        for rank, case in enumerate(candidates, start=1):
            figure = plot_case(payload, case, rank, args)
            row = {
                "dataset": dataset,
                "case_rank": rank,
                "experiment_dir": str(exp_dir),
                "selected_score_key": score_key,
                "center": int(case["center"]),
                "transition_strength": float(case["transition_strength"]),
                "top_channel_delta": float(case["top_channel_delta"]),
                "score_peak": float(case["score_peak"]),
                "score_peak_percentile": float(case["score_peak_percentile"]),
                "alarm_threshold": float(case["alarm_threshold"]),
                "quiet_threshold": float(case["quiet_threshold"]),
                "quiet_below_low_score_quantile": bool(case["quiet_below_q"]),
                "figure": str(figure),
            }
            all_rows.append(row)

    write_rows(args.output_dir / "normal_transition_no_false_alarm_cases.csv", all_rows)
    (args.output_dir / "normal_transition_no_false_alarm_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[NormalTransition] wrote {args.output_dir / 'normal_transition_no_false_alarm_summary.json'}")
    plot_overview(all_rows, args)


if __name__ == "__main__":
    main()
