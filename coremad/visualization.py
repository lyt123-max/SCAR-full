from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


def _prepare_path(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_or_skip(fig, save_path: str | Path, name: str) -> bool:
    if not HAS_MATPLOTLIB or plt is None:
        print(f"[Viz] skip {name}: matplotlib is not available.")
        return False
    path = _prepare_path(save_path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] saved {name}: {path}")
    return True


def plot_training_curves(history: list[dict], save_path: str | Path) -> bool:
    if not history:
        print("[Viz] skip stage_a_loss_curve: empty training history.")
        return False
    if not HAS_MATPLOTLIB or plt is None:
        print("[Viz] skip stage_a_loss_curve: matplotlib is not available.")
        return False

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item.get("train", {}).get("loss", np.nan)) for item in history]
    val_loss = [float(item.get("val", {}).get("loss", np.nan)) for item in history]
    val_mask_loss = [float(item.get("val", {}).get("mask_loss", np.nan)) for item in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=1.8, alpha=0.9)
    if np.isfinite(val_loss).any():
        ax.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=1.8, alpha=0.9)

    finite_mask_losses = np.asarray(val_mask_loss, dtype=np.float64)
    if np.isfinite(finite_mask_losses).any():
        best_idx = int(np.nanargmin(finite_mask_losses))
        best_epoch = epochs[best_idx]
        ax.axvline(
            x=best_epoch,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"Best Val Mask Loss (epoch {best_epoch})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage A Training Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_or_skip(fig, save_path, "stage_a_loss_curve")


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    title: str = "Score Distribution: Normal vs Anomaly",
    xlabel: str = "Anomaly Score",
    plot_name: str = "score_distribution",
) -> bool:
    if scores.size == 0 or labels.size == 0:
        print(f"[Viz] skip {plot_name}: empty scores or labels.")
        return False
    if not HAS_MATPLOTLIB or plt is None:
        print(f"[Viz] skip {plot_name}: matplotlib is not available.")
        return False

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    if normal_scores.size > 0:
        ax.hist(
            normal_scores,
            bins=100,
            alpha=0.55,
            density=True,
            color="#1f77b4",
            label=f"Normal (n={normal_scores.size})",
        )
    if anomaly_scores.size > 0:
        ax.hist(
            anomaly_scores,
            bins=100,
            alpha=0.55,
            density=True,
            color="#d62728",
            label=f"Anomaly (n={anomaly_scores.size})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_or_skip(fig, save_path, plot_name)


def plot_score_timeline(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    time_range: Optional[tuple[int, int]] = None,
    title: str = "Detection Result",
) -> bool:
    if scores.size == 0 or labels.size == 0:
        print("[Viz] skip score_timeline: empty scores or labels.")
        return False
    if not HAS_MATPLOTLIB or plt is None:
        print("[Viz] skip score_timeline: matplotlib is not available.")
        return False

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if time_range is not None:
        start, end = time_range
        start = max(0, int(start))
        end = min(int(end), len(scores))
        if end <= start:
            print("[Viz] skip score_timeline: invalid time range.")
            return False
        scores = scores[start:end]
        labels = labels[start:end]

    t = np.arange(len(scores))
    ymax = float(np.percentile(scores, 99.5)) if scores.size > 0 else 1.0
    ymax = max(ymax, float(scores.max()) if scores.size > 0 else 1.0, 1e-6)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    axes[0].plot(t, scores, color="#6a3d9a", linewidth=0.7, alpha=0.85)
    axes[0].fill_between(t, 0.0, scores, alpha=0.18, color="#6a3d9a")
    anomaly_mask = labels.astype(bool)
    if anomaly_mask.any():
        axes[0].fill_between(t, 0.0, ymax, where=anomaly_mask, alpha=0.12, color="#d62728")
    axes[0].set_ylabel("Anomaly Score")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.2)

    axes[1].imshow(
        labels.reshape(1, -1),
        aspect="auto",
        cmap="Reds",
        interpolation="nearest",
        extent=[0, len(scores), 0, 1],
    )
    axes[1].set_ylabel("GT")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time Step")

    plt.tight_layout()
    return _save_or_skip(fig, save_path, Path(save_path).name)
