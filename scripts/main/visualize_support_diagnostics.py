from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    HAS_SKLEARN = True
except Exception:
    average_precision_score = None
    roc_auc_score = None
    HAS_SKLEARN = False

from coremad.config import CoReMADConfig
from coremad.data import load_raw_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize soft-support diagnostics for a single experiment or a tau sweep."
    )
    parser.add_argument("--npz", type=str, help="Path to test_diagnostic_scores.npz for single-run visualization.")
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        nargs="*",
        default=None,
        help="Experiment directories for tau sweep plotting.",
    )
    parser.add_argument(
        "--experiment-glob",
        type=str,
        default=None,
        help="Glob pattern for experiment directories, e.g. './artifacts/gecco_tau_test_*'.",
    )
    parser.add_argument("--labels-key", type=str, default="labels")
    parser.add_argument("--score-key", type=str, default="soft_support_score")
    parser.add_argument("--completion-keys", type=str, nargs=2, default=["completion_scale8", "completion_scale32"])
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures. Defaults to the experiment directory / visualization_support.",
    )
    parser.add_argument(
        "--saturation-tol",
        type=float,
        default=1e-3,
        help="Treat scores within this tolerance of -log(eps) as saturated.",
    )
    parser.add_argument(
        "--scatter-max-points",
        type=int,
        default=6000,
        help="Maximum points to draw in completion-vs-support scatter.",
    )
    return parser.parse_args()


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.zeros_like(scores, dtype=np.float64)
    fill_max = float(finite.max())
    fill_min = float(finite.min())
    return np.nan_to_num(scores, nan=0.0, posinf=fill_max, neginf=fill_min)


def resolve_labels(npz_path: Path, data: np.lib.npyio.NpzFile, labels_key: str) -> np.ndarray:
    if labels_key in data.files:
        return np.asarray(data[labels_key], dtype=np.int32)

    config_path = npz_path.parent / "config.json"
    if not config_path.exists():
        raise KeyError(
            f"Missing labels key '{labels_key}' in {npz_path} and cannot find config.json for dataset fallback."
        )
    config = CoReMADConfig.load(config_path)
    raw_bundle = load_raw_dataset_bundle(config.dataset, config.data_root, config=config)
    print(f"[VizSupport] labels missing in npz, loaded from dataset={config.dataset} data_root={config.data_root}")
    return np.asarray(raw_bundle.test_labels, dtype=np.int32)


def load_config_for_npz(npz_path: Path) -> CoReMADConfig | None:
    config_path = npz_path.parent / "config.json"
    if not config_path.exists():
        return None
    return CoReMADConfig.load(config_path)


def ensure_output_dir(base: Path | None, fallback_parent: Path, suffix: str = "visualization_support") -> Path:
    out_dir = base if base is not None else (fallback_parent / suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def compute_saturation_value(config: CoReMADConfig | None) -> float:
    eps = 1e-8 if config is None else float(config.support_score_eps)
    return float(-math.log(max(eps, 1e-300)))


def recover_total_support(scores: np.ndarray, config: CoReMADConfig | None) -> np.ndarray:
    eps = 1e-8 if config is None else float(config.support_score_eps)
    recovered = np.exp(-sanitize_scores(scores)) - eps
    return np.clip(recovered, 0.0, None)


def summarize_by_label(name: str, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    summary = {
        "normal_mean": float(normal.mean()),
        "normal_median": float(np.median(normal)),
        "normal_p95": float(np.percentile(normal, 95)),
        "anomaly_mean": float(anomaly.mean()),
        "anomaly_median": float(np.median(anomaly)),
        "anomaly_p05": float(np.percentile(anomaly, 5)),
    }
    print(
        f"[VizSupport] {name}: "
        f"normal_mean={summary['normal_mean']:.6f}, anomaly_mean={summary['anomaly_mean']:.6f}, "
        f"normal_median={summary['normal_median']:.6f}, anomaly_median={summary['anomaly_median']:.6f}"
    )
    return summary


def plot_histogram(scores: np.ndarray, labels: np.ndarray, title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    bins = 80
    plt.hist(scores[labels == 0], bins=bins, alpha=0.6, density=True, label="Normal", color="#2a9d8f")
    plt.hist(scores[labels == 1], bins=bins, alpha=0.6, density=True, label="Anomaly", color="#e76f51")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_boxplot(scores: np.ndarray, labels: np.ndarray, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    values = [scores[labels == 0], scores[labels == 1]]
    plt.boxplot(values, labels=["Normal", "Anomaly"], showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_support_distribution(
    support: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    log_support = np.log10(np.clip(support, 1e-20, None))
    plt.figure(figsize=(8, 5))
    plt.hist(log_support[labels == 0], bins=80, alpha=0.6, density=True, label="Normal", color="#264653")
    plt.hist(log_support[labels == 1], bins=80, alpha=0.6, density=True, label="Anomaly", color="#f4a261")
    plt.title(title)
    plt.xlabel("log10(total_support)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_completion_scatter(
    completion_avg: np.ndarray,
    support_score: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    max_points: int,
) -> None:
    rng = np.random.RandomState(42)
    idx = np.arange(len(labels))
    if len(idx) > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    plt.figure(figsize=(7, 6))
    sampled_labels = labels[idx]
    plt.scatter(
        completion_avg[idx][sampled_labels == 0],
        support_score[idx][sampled_labels == 0],
        s=8,
        alpha=0.25,
        color="#2a9d8f",
        label="Normal",
    )
    plt.scatter(
        completion_avg[idx][sampled_labels == 1],
        support_score[idx][sampled_labels == 1],
        s=10,
        alpha=0.45,
        color="#e76f51",
        label="Anomaly",
    )
    plt.xlabel("completion_avg")
    plt.ylabel("soft_support_score")
    plt.title("Completion vs Soft Support")
    plt.legend(markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if not HAS_SKLEARN:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def visualize_single_run(args: argparse.Namespace) -> None:
    npz_path = Path(args.npz).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    labels = resolve_labels(npz_path, data, args.labels_key)
    config = load_config_for_npz(npz_path)
    score = sanitize_scores(data[args.score_key])
    saturation_value = compute_saturation_value(config)
    saturation_ratio = float(np.mean(score >= saturation_value - args.saturation_tol))
    total_support = recover_total_support(score, config)
    comp_avg = 0.5 * (
        sanitize_scores(data[args.completion_keys[0]]) + sanitize_scores(data[args.completion_keys[1]])
    )

    output_dir = ensure_output_dir(
        Path(args.output_dir).resolve() if args.output_dir else None,
        npz_path.parent,
    )
    score_summary = summarize_by_label(args.score_key, score, labels)
    support_summary = summarize_by_label("recovered_total_support", total_support, labels)
    auc, pr_auc = compute_metrics(labels, score)

    print(
        f"[VizSupport] score={args.score_key} auc={auc:.6f} pr_auc={pr_auc:.6f} "
        f"saturation_value={saturation_value:.6f} saturation_ratio={saturation_ratio:.4%}"
    )

    plot_histogram(
        score,
        labels,
        title=f"{args.score_key} Distribution",
        xlabel=args.score_key,
        output_path=output_dir / f"{args.score_key}_hist.png",
    )
    plot_boxplot(
        score,
        labels,
        title=f"{args.score_key} Boxplot",
        ylabel=args.score_key,
        output_path=output_dir / f"{args.score_key}_boxplot.png",
    )
    plot_support_distribution(
        total_support,
        labels,
        title="Recovered Total Support Distribution",
        output_path=output_dir / "recovered_total_support_hist.png",
    )
    plot_completion_scatter(
        comp_avg,
        score,
        labels,
        output_path=output_dir / "completion_vs_soft_support.png",
        max_points=args.scatter_max_points,
    )

    summary = {
        "npz_path": str(npz_path),
        "score_key": args.score_key,
        "auc": auc,
        "pr_auc": pr_auc,
        "saturation_value": saturation_value,
        "saturation_ratio": saturation_ratio,
        "score_summary": score_summary,
        "support_summary": support_summary,
    }
    (output_dir / "support_visualization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[VizSupport] saved figures to {output_dir}")


def collect_experiment_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    if args.experiment_dirs:
        dirs.extend(Path(item).resolve() for item in args.experiment_dirs)
    if args.experiment_glob:
        dirs.extend(Path().glob(args.experiment_glob))
        dirs = [path.resolve() for path in dirs]
    unique_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        if path in seen:
            continue
        if not path.exists():
            print(f"[VizSupport] skip missing experiment dir: {path}")
            continue
        seen.add(path)
        unique_dirs.append(path)
    return unique_dirs


def visualize_tau_sweep(args: argparse.Namespace) -> None:
    experiment_dirs = collect_experiment_dirs(args)
    if not experiment_dirs:
        raise ValueError("No valid experiment directories were provided for tau sweep visualization.")

    rows: list[dict[str, float | str]] = []
    for exp_dir in experiment_dirs:
        npz_path = exp_dir / "test_diagnostic_scores.npz"
        config_path = exp_dir / "config.json"
        if not npz_path.exists() or not config_path.exists():
            print(f"[VizSupport] skip {exp_dir}: missing test_diagnostic_scores.npz or config.json")
            continue
        config = CoReMADConfig.load(config_path)
        data = np.load(npz_path)
        labels = resolve_labels(npz_path, data, args.labels_key)
        score = sanitize_scores(data[args.score_key])
        auc, pr_auc = compute_metrics(labels, score)
        sat_value = compute_saturation_value(config)
        sat_ratio = float(np.mean(score >= sat_value - args.saturation_tol))
        total_support = recover_total_support(score, config)
        rows.append(
            {
                "experiment_dir": str(exp_dir),
                "tau": float(config.support_score_tau),
                "auc": auc,
                "pr_auc": pr_auc,
                "saturation_ratio": sat_ratio,
                "normal_median": float(np.median(score[labels == 0])),
                "anomaly_median": float(np.median(score[labels == 1])),
                "normal_support_median": float(np.median(total_support[labels == 0])),
                "anomaly_support_median": float(np.median(total_support[labels == 1])),
            }
        )

    if not rows:
        raise RuntimeError("No valid tau sweep experiments were found.")

    rows = sorted(rows, key=lambda item: float(item["tau"]))
    output_dir = ensure_output_dir(
        Path(args.output_dir).resolve() if args.output_dir else None,
        Path(rows[0]["experiment_dir"]).parent,
        suffix="visualization_support_tau",
    )

    tau = np.asarray([float(row["tau"]) for row in rows], dtype=np.float64)
    auc = np.asarray([float(row["auc"]) for row in rows], dtype=np.float64)
    pr_auc = np.asarray([float(row["pr_auc"]) for row in rows], dtype=np.float64)
    sat_ratio = np.asarray([float(row["saturation_ratio"]) for row in rows], dtype=np.float64)
    normal_median = np.asarray([float(row["normal_median"]) for row in rows], dtype=np.float64)
    anomaly_median = np.asarray([float(row["anomaly_median"]) for row in rows], dtype=np.float64)

    plt.figure(figsize=(8, 5))
    plt.semilogx(tau, auc, marker="o", label="AUROC", color="#264653")
    plt.semilogx(tau, pr_auc, marker="s", label="PR-AUC", color="#e76f51")
    plt.xlabel("support_score_tau")
    plt.ylabel("Metric")
    plt.title("Tau Sweep Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tau_metrics.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogx(tau, sat_ratio, marker="o", color="#f4a261")
    plt.xlabel("support_score_tau")
    plt.ylabel("Saturation Ratio")
    plt.title("Tau Sweep Saturation Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "tau_saturation_ratio.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogx(tau, normal_median, marker="o", label="Normal Median", color="#2a9d8f")
    plt.semilogx(tau, anomaly_median, marker="s", label="Anomaly Median", color="#e76f51")
    plt.xlabel("support_score_tau")
    plt.ylabel(args.score_key)
    plt.title("Tau Sweep Score Medians")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tau_score_medians.png", dpi=160)
    plt.close()

    (output_dir / "tau_sweep_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[VizSupport] saved tau sweep figures to {output_dir}")
    for row in rows:
        print(
            f"[VizSupport] tau={float(row['tau']):.6g} auc={float(row['auc']):.6f} "
            f"pr_auc={float(row['pr_auc']):.6f} sat_ratio={float(row['saturation_ratio']):.4%}"
        )


def main() -> None:
    args = parse_args()
    if not args.npz and not args.experiment_dirs and not args.experiment_glob:
        raise ValueError("Provide either --npz for a single run or --experiment-dirs/--experiment-glob for tau sweep.")

    if args.npz:
        visualize_single_run(args)
    if args.experiment_dirs or args.experiment_glob:
        visualize_tau_sweep(args)


if __name__ == "__main__":
    main()
