from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad.config import CoReMADConfig
from coremad.memory import MemoryBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize prototype structure in the state_bank with a PCA projection."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment directory containing config.json and memory.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures. Defaults to <experiment-dir>/prototype_visualization",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help="Maximum number of state vectors to draw in scatter plots.",
    )
    parser.add_argument(
        "--top-prototypes",
        type=int,
        default=12,
        help="How many largest prototypes to highlight individually.",
    )
    return parser.parse_args()


def pca_project(train_x: np.ndarray, query_x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    train_x = np.asarray(train_x, dtype=np.float64)
    mean = train_x.mean(axis=0, keepdims=True)
    centered = train_x - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    explained = (s[:2] ** 2) / max(np.sum(s ** 2), 1e-12)
    train_proj = centered @ components.T
    query_proj = None
    if query_x is not None:
        query_proj = (np.asarray(query_x, dtype=np.float64) - mean) @ components.T
    return train_proj, query_proj, explained


def ensure_output_dir(args: argparse.Namespace, experiment_dir: Path) -> Path:
    out_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "prototype_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def sample_indices(n: int, limit: int, seed: int = 42) -> np.ndarray:
    if n <= limit or limit <= 0:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def plot_overview_scatter(
    state_proj: np.ndarray,
    centers_proj: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    max_points: int,
) -> None:
    idx = sample_indices(len(state_proj), max_points)
    labels_sample = labels[idx]
    unique_labels = np.unique(labels_sample)
    cmap = plt.cm.get_cmap("tab20", max(20, int(labels.max()) + 1))

    plt.figure(figsize=(9, 7))
    for proto_id in unique_labels:
        mask = labels_sample == proto_id
        plt.scatter(
            state_proj[idx][mask, 0],
            state_proj[idx][mask, 1],
            s=7,
            alpha=0.28,
            color=cmap(int(proto_id) % cmap.N),
            label=f"P{int(proto_id)}" if len(unique_labels) <= 20 else None,
        )
    plt.scatter(
        centers_proj[:, 0],
        centers_proj[:, 1],
        s=140,
        marker="X",
        color="black",
        edgecolor="white",
        linewidth=0.8,
        label="Prototype Centers",
    )
    if len(unique_labels) <= 20:
        plt.legend(loc="best", fontsize=8, markerscale=1.4)
    plt.title("State Bank PCA with Prototype Centers")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_size_bar(proto_sizes: np.ndarray, output_path: Path) -> None:
    order = np.argsort(proto_sizes)[::-1]
    plt.figure(figsize=(10, 4.8))
    plt.bar(np.arange(len(proto_sizes)), proto_sizes[order], color="#457b9d")
    plt.title("Prototype Member Counts")
    plt.xlabel("Prototype Rank (sorted by size)")
    plt.ylabel("Number of Windows")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_top_prototypes(
    state_proj: np.ndarray,
    centers_proj: np.ndarray,
    labels: np.ndarray,
    proto_sizes: np.ndarray,
    output_path: Path,
    top_k: int,
    max_points: int,
) -> None:
    order = np.argsort(proto_sizes)[::-1][:top_k]
    idx = sample_indices(len(state_proj), max_points)
    background = state_proj[idx]

    n_cols = 3
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.0 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for panel_idx, proto_id in enumerate(order):
        ax = axes.ravel()[panel_idx]
        ax.axis("on")
        ax.scatter(background[:, 0], background[:, 1], s=6, alpha=0.08, color="#b0b0b0")
        member_mask = labels[idx] == proto_id
        ax.scatter(
            state_proj[idx][member_mask, 0],
            state_proj[idx][member_mask, 1],
            s=8,
            alpha=0.45,
            color="#2a9d8f",
        )
        ax.scatter(
            centers_proj[proto_id, 0],
            centers_proj[proto_id, 1],
            s=120,
            marker="X",
            color="#e63946",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(f"Prototype {int(proto_id)}  size={int(proto_sizes[proto_id])}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle("Largest Prototypes in State Space", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = CoReMADConfig.load(config_path)
    if not config.memory_path.exists():
        raise FileNotFoundError(f"Memory artifact not found: {config.memory_path}")

    memory = MemoryBank.load(config.memory_path, config)
    output_dir = ensure_output_dir(args, experiment_dir)

    state_bank = memory.state_bank.cpu().numpy().astype(np.float64, copy=False)
    proto_centers = memory.prototype_centers.cpu().numpy().astype(np.float64, copy=False)
    proto_labels = memory.prototype_labels.cpu().numpy().astype(np.int64, copy=False)
    proto_sizes = np.asarray([int(member.numel()) for member in memory.prototype_members], dtype=np.int64)

    state_proj, centers_proj, explained = pca_project(state_bank, proto_centers)
    explained_msg = f"PCA explained variance ratio: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}, sum={explained.sum():.4f}"
    print(f"[ProtoViz] experiment_dir={experiment_dir}")
    print(f"[ProtoViz] num_windows={len(state_bank)} num_prototypes={len(proto_centers)}")
    print(f"[ProtoViz] prototype_size_min={int(proto_sizes.min())} median={float(np.median(proto_sizes)):.1f} max={int(proto_sizes.max())}")
    print(f"[ProtoViz] {explained_msg}")

    plot_overview_scatter(
        state_proj=state_proj,
        centers_proj=centers_proj,
        labels=proto_labels,
        output_path=output_dir / "prototype_overview_pca.png",
        max_points=args.max_points,
    )
    plot_size_bar(
        proto_sizes=proto_sizes,
        output_path=output_dir / "prototype_size_bar.png",
    )
    plot_top_prototypes(
        state_proj=state_proj,
        centers_proj=centers_proj,
        labels=proto_labels,
        proto_sizes=proto_sizes,
        output_path=output_dir / "prototype_top_panels_pca.png",
        top_k=max(1, args.top_prototypes),
        max_points=args.max_points,
    )

    summary = {
        "experiment_dir": str(experiment_dir),
        "num_windows": int(len(state_bank)),
        "num_prototypes": int(len(proto_centers)),
        "prototype_sizes": proto_sizes.tolist(),
        "pca_explained_variance_ratio": explained.tolist(),
        "max_points": int(args.max_points),
        "top_prototypes": int(args.top_prototypes),
    }
    (output_dir / "prototype_visualization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ProtoViz] saved figures to {output_dir}")


if __name__ == "__main__":
    main()
