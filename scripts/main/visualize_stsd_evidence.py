from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from sklearn.metrics import roc_auc_score

    HAS_SKLEARN = True
except Exception:
    roc_auc_score = None
    HAS_SKLEARN = False

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad.config import CoReMADConfig
from coremad.data import build_loader, load_raw_dataset_bundle
from coremad.memory import MemoryBank
from coremad.model import CoReMADModel
from coremad.trainer import CoReMADTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-style evidence figures for STSD."
    )
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--low-freq-cutoff", type=float, default=None)
    parser.add_argument("--embedding-max-points", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=int, nargs="*", default=None)
    parser.add_argument("--max-channels", type=int, default=6)
    parser.add_argument(
        "--spectrum-demean",
        action="store_true",
        default=True,
        help="Remove the per-channel temporal mean before computing spectra.",
    )
    parser.add_argument(
        "--no-spectrum-demean",
        dest="spectrum_demean",
        action="store_false",
        help="Disable mean removal before the spectrum computation.",
    )
    return parser.parse_args()


def ensure_output_dir(args: argparse.Namespace, experiment_dir: Path) -> Path:
    out_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "stsd_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def infer_device(config: CoReMADConfig, override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if str(config.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_split(
    config: CoReMADConfig,
    raw_bundle,
    split: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    if split == "train":
        return raw_bundle.train, None, raw_bundle.train_segment_ranges, int(config.train_stride)
    return raw_bundle.test, np.asarray(raw_bundle.test_labels, dtype=np.int32), raw_bundle.test_segment_ranges, int(config.test_stride)


def build_fallback_start_indices(total_steps: int, seq_len: int, stride: int) -> np.ndarray:
    last_start = int(total_steps) - int(seq_len)
    if last_start < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, last_start + 1, int(stride), dtype=np.int64)


def compute_window_labels(start_indices: np.ndarray, labels: np.ndarray | None, seq_len: int) -> np.ndarray:
    if labels is None:
        return np.zeros(len(start_indices), dtype=np.int32)
    return np.asarray(
        [int(np.asarray(labels[start : start + seq_len], dtype=np.int32).max()) for start in start_indices],
        dtype=np.int32,
    )


def compute_window_anomaly_counts(start_indices: np.ndarray, labels: np.ndarray | None, seq_len: int) -> np.ndarray:
    if labels is None:
        return np.zeros(len(start_indices), dtype=np.int32)
    return np.asarray(
        [int(np.asarray(labels[start : start + seq_len], dtype=np.int32).sum()) for start in start_indices],
        dtype=np.int32,
    )


def stratified_sample_indices(window_labels: np.ndarray, max_windows: int, seed: int) -> np.ndarray:
    total = len(window_labels)
    if max_windows <= 0 or total <= max_windows:
        return np.arange(total, dtype=np.int64)
    rng = np.random.RandomState(seed)
    unique = np.unique(window_labels)
    per_class = max(1, max_windows // max(1, len(unique)))
    sampled: list[np.ndarray] = []
    for label in unique.tolist():
        ids = np.where(window_labels == int(label))[0]
        if ids.size == 0:
            continue
        if ids.size > per_class:
            ids = np.sort(rng.choice(ids, size=per_class, replace=False))
        sampled.append(ids.astype(np.int64))
    merged = np.concatenate(sampled, axis=0) if sampled else np.arange(total, dtype=np.int64)
    if merged.size < max_windows:
        remaining = np.setdiff1d(np.arange(total, dtype=np.int64), merged, assume_unique=False)
        if remaining.size > 0:
            take = min(max_windows - merged.size, remaining.size)
            extra = np.sort(rng.choice(remaining, size=take, replace=False))
            merged = np.concatenate([merged, extra], axis=0)
    return np.sort(merged[:max_windows]).astype(np.int64)


def l2_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    num = float(np.linalg.norm(np.asarray(numerator, dtype=np.float64).reshape(-1), ord=2))
    den = float(np.linalg.norm(np.asarray(denominator, dtype=np.float64).reshape(-1), ord=2))
    return num / max(den, 1e-12)


def preprocess_for_spectrum(arr: np.ndarray, demean: bool) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if demean:
        values = values - values.mean(axis=1, keepdims=True)
    return values


def compute_power_spectrum(arr: np.ndarray, demean: bool = False) -> np.ndarray:
    freq = np.fft.rfft(preprocess_for_spectrum(arr, demean=demean), axis=1)
    return np.mean(np.abs(freq) ** 2, axis=2)


def low_frequency_ratio(arr: np.ndarray, cutoff: float, demean: bool = False) -> np.ndarray:
    power = compute_power_spectrum(arr, demean=demean)
    n_freq = power.shape[1]
    cutoff_idx = int(round(float(cutoff) * max(1, n_freq - 1)))
    cutoff_idx = max(0, min(cutoff_idx, n_freq - 1))
    num = np.sum(power[:, : cutoff_idx + 1], axis=1)
    den = np.sum(power, axis=1)
    return num / np.maximum(den, 1e-12)


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.22)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
    )


def draw_spectrum_axis(
    ax,
    freq_grid: np.ndarray,
    payload: dict[str, np.ndarray],
    cutoff: float,
) -> None:
    color_map = {"X": "#334155", "S": "#2563eb", "R": "#dc2626"}
    for key in ("X", "S", "R"):
        if key not in payload:
            continue
        spectrum = np.maximum(np.asarray(payload[key], dtype=np.float64), 1e-12)
        ax.plot(freq_grid, spectrum, linewidth=1.6, label=key, color=color_map[key])
    ax.axvline(float(cutoff), color="#7c3aed", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Mean Power (log scale)")
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax, grid_axis="both")


def draw_lowfreq_box_axis(
    ax,
    values: list[np.ndarray],
    labels: list[str],
) -> None:
    colors = ["#334155", "#2563eb", "#dc2626"]
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors[: len(values)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_ylabel("Low-Frequency Energy Ratio")
    style_axis(ax, grid_axis="y")


def draw_lowfreq_grouped_axis(ax, ratio_payload: dict[str, dict[str, np.ndarray]]) -> None:
    groups = [group for group in ("all", "normal", "anomaly") if group in ratio_payload]
    colors = {"X": "#334155", "S": "#2563eb", "R": "#dc2626"}
    group_centers: list[float] = []
    positions: list[float] = []
    values: list[np.ndarray] = []
    box_colors: list[str] = []
    current = 1.0
    for group in groups:
        payload = ratio_payload[group]
        local_positions = [current, current + 0.8, current + 1.6]
        for pos, key in zip(local_positions, ("X", "S", "R")):
            if key not in payload:
                continue
            positions.append(pos)
            values.append(np.asarray(payload[key], dtype=np.float64))
            box_colors.append(colors[key])
        group_centers.append(np.mean(local_positions))
        current += 3.2

    if not values:
        return

    box = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([group.capitalize() for group in groups])
    ax.set_ylabel("Low-Frequency Energy Ratio")
    style_axis(ax, grid_axis="y")
    legend_handles = [
        plt.Line2D([0], [0], color=colors[key], linewidth=8, alpha=0.45, label=key)
        for key in ("X", "S", "R")
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)


def draw_anti_collapse_hist_axis(ax, s_ratio: np.ndarray, r_ratio: np.ndarray) -> None:
    ax.hist(s_ratio, bins=40, alpha=0.55, density=True, color="#2563eb", label="||S|| / ||X||")
    ax.hist(r_ratio, bins=40, alpha=0.55, density=True, color="#dc2626", label="||R|| / ||X||")
    ax.axvline(float(np.median(s_ratio)), color="#1d4ed8", linestyle="--", linewidth=1.0)
    ax.axvline(float(np.median(r_ratio)), color="#b91c1c", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax, grid_axis="y")


def draw_anti_collapse_scatter_axis(ax, s_ratio: np.ndarray, r_ratio: np.ndarray, labels: np.ndarray) -> None:
    mask_normal = np.asarray(labels, dtype=np.int32) == 0
    mask_anomaly = np.asarray(labels, dtype=np.int32) == 1
    ax.scatter(s_ratio[mask_normal], r_ratio[mask_normal], s=16, alpha=0.35, color="#2a9d8f", label="Normal")
    if np.any(mask_anomaly):
        ax.scatter(s_ratio[mask_anomaly], r_ratio[mask_anomaly], s=18, alpha=0.45, color="#e76f51", label="Anomaly")
    ax.set_xlabel("||S|| / ||X||")
    ax.set_ylabel("||R|| / ||X||")
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax, grid_axis="both")


def draw_state_projection_axis(ax, proj: np.ndarray, labels: np.ndarray) -> None:
    mask_normal = labels == 0
    mask_anomaly = labels == 1
    ax.scatter(proj[mask_normal, 0], proj[mask_normal, 1], s=12, alpha=0.35, color="#2a9d8f", label="Normal")
    if np.any(mask_anomaly):
        ax.scatter(proj[mask_anomaly, 0], proj[mask_anomaly, 1], s=14, alpha=0.55, color="#e76f51", label="Anomaly")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    style_axis(ax, grid_axis="both")


def draw_state_novelty_compare_axis(
    ax,
    novelty_raw: np.ndarray,
    novelty_stsd: np.ndarray,
    labels: np.ndarray,
) -> None:
    grouped = [
        np.asarray(novelty_raw[labels == 0], dtype=np.float64),
        np.asarray(novelty_raw[labels == 1], dtype=np.float64) if np.any(labels == 1) else np.zeros(1, dtype=np.float64),
        np.asarray(novelty_stsd[labels == 0], dtype=np.float64),
        np.asarray(novelty_stsd[labels == 1], dtype=np.float64) if np.any(labels == 1) else np.zeros(1, dtype=np.float64),
    ]
    positions = [1.0, 2.0, 4.0, 5.0]
    box = ax.boxplot(grouped, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    palette = ["#94a3b8", "#cbd5e1", "#60a5fa", "#fca5a5"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(["Raw X", "S"])
    ax.set_ylabel("state_novelty")
    style_axis(ax, grid_axis="y")
    legend_handles = [
        plt.Line2D([0], [0], color="#94a3b8", linewidth=8, alpha=0.65, label="Normal"),
        plt.Line2D([0], [0], color="#fca5a5", linewidth=8, alpha=0.65, label="Anomaly"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)


def shade_anomaly_region(ax, t: np.ndarray, label_mask: np.ndarray | None, anomaly_onset: int | None) -> None:
    if label_mask is not None and np.any(label_mask):
        ax.fill_between(t, 0.0, 1.05, where=label_mask, transform=ax.get_xaxis_transform(), color="#fee2e2", alpha=0.40)
    if anomaly_onset is not None:
        ax.axvline(anomaly_onset, color="#b91c1c", linestyle="--", linewidth=1.1, alpha=0.9)


def choose_channels(residual_norm: np.ndarray, requested_channels: list[int] | None, max_channels: int) -> list[int]:
    n_channels = residual_norm.shape[1]
    if requested_channels:
        channels = [int(ch) for ch in requested_channels]
        for channel in channels:
            if channel < 0 or channel >= n_channels:
                raise IndexError(f"Channel {channel} is out of range [0, {n_channels - 1}].")
        return channels
    residual_energy = np.mean(np.abs(residual_norm), axis=0)
    order = np.argsort(residual_energy)[::-1]
    top_k = max(1, min(int(max_channels), n_channels))
    return [int(idx) for idx in order[:top_k]]


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def pca_project(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.asarray(train_x, dtype=np.float64)
    mean = train_x.mean(axis=0, keepdims=True)
    centered = train_x - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    explained = (s[:2] ** 2) / max(np.sum(s ** 2), 1e-12)
    train_proj = centered @ components.T
    return train_proj, explained


def encode_without_stsd(model: CoReMADModel, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    state_vec = model.state_encoder(x)
    patch_outputs = model.patch_encoder(x, x, state_vec)
    patch_outputs["slow"] = x
    patch_outputs["residual"] = x
    patch_outputs["state_vec"] = state_vec
    return patch_outputs


def deterministic_completion_scores_from_encoded(
    model: CoReMADModel,
    encoded: dict[str, torch.Tensor | list[torch.Tensor]],
    x: torch.Tensor,
) -> list[torch.Tensor]:
    if not model.config.use_completion_head:
        return []
    scores: list[torch.Tensor] = []
    for scale_idx, patch_size in enumerate(model.config.patch_sizes):
        u = encoded["u"][scale_idx]
        target = encoded["x_patches"][scale_idx]
        accum = x.new_zeros(u.size(0), u.size(1))
        counts = x.new_zeros(u.size(0), u.size(1))
        positions = torch.arange(u.size(1), device=x.device)
        for group in range(model.config.n_mask_groups):
            mask = (positions % model.config.n_mask_groups == group).unsqueeze(0).expand(u.size(0), -1)
            if not mask.any():
                continue
            pred = model.completion_heads[scale_idx](u, mask).reshape(
                u.size(0), u.size(1), patch_size, model.config.n_channels
            )
            err = model._completion_error(pred, target)
            accum = accum + err * mask.float()
            counts = counts + mask.float()
        scores.append(accum / counts.clamp(min=1.0))
    return scores


def expand_patch_score(score: torch.Tensor, patch_size: int, seq_len: int) -> torch.Tensor:
    expanded = score.repeat_interleave(patch_size, dim=1)
    if expanded.size(1) < seq_len:
        pad = expanded[:, -1:].expand(-1, seq_len - expanded.size(1))
        expanded = torch.cat([expanded, pad], dim=1)
    return expanded[:, :seq_len]


def scale_weights(config: CoReMADConfig) -> list[float]:
    weights = [1.0 / float(size) for size in config.patch_sizes]
    total = sum(weights)
    return [weight / total for weight in weights]


def fuse_patch_scores(patch_scores: list[torch.Tensor], config: CoReMADConfig) -> torch.Tensor:
    point_scores = None
    weights = scale_weights(config)
    for score, patch_size, weight in zip(patch_scores, config.patch_sizes, weights):
        expanded = expand_patch_score(score, patch_size, config.seq_len)
        point_scores = expanded * weight if point_scores is None else point_scores + expanded * weight
    if point_scores is None:
        raise ValueError("patch_scores must not be empty.")
    return point_scores


def build_feature_dict(
    config: CoReMADConfig,
    encoded: dict[str, torch.Tensor | list[torch.Tensor]],
    comp_scores: list[torch.Tensor],
    memory: MemoryBank,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    memory_out = memory.query_preencoded(encoded["state_vec"], encoded["z"], encoded["c"])
    novelty = memory_out["state_novelty"].to(encoded["state_vec"].device).unsqueeze(-1).expand(-1, config.seq_len)
    features = {
        "knn_distance": fuse_patch_scores(memory_out["mem_scores"], config),
        "soft_support_score": fuse_patch_scores(memory_out["soft_mem_scores"], config),
        "state_novelty": novelty,
    }
    for score_name, patch_size, comp_score in zip(config.completion_score_names(), config.patch_sizes, comp_scores):
        features[score_name] = expand_patch_score(comp_score, patch_size, config.seq_len)
    return features, memory_out


def choose_local_window_id(start_indices: np.ndarray, labels: np.ndarray | None, seq_len: int) -> int:
    if labels is None:
        return 0
    anomaly_counts = compute_window_anomaly_counts(start_indices, labels, seq_len)
    onset_ids = np.where((anomaly_counts > 0) & (anomaly_counts < seq_len))[0]
    if onset_ids.size > 0:
        target = seq_len // 3
        return int(onset_ids[np.argmin(np.abs(anomaly_counts[onset_ids] - target))])
    anomaly_ids = np.where(anomaly_counts > 0)[0]
    if anomaly_ids.size > 0:
        return int(anomaly_ids[np.argmax(anomaly_counts[anomaly_ids])])
    return 0


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return None
    if HAS_SKLEARN:
        return float(roc_auc_score(y, score_arr))
    pos = score_arr[y == 1]
    neg = score_arr[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    combined = np.concatenate([pos, neg], axis=0)
    ranks = np.argsort(np.argsort(combined, kind="mergesort"), kind="mergesort") + 1
    pos_ranks = ranks[: pos.size]
    auc = (pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0) / max(pos.size * neg.size, 1.0)
    return float(auc)


def plot_spectrum_comparison(
    dataset_name: str,
    freq_grid: np.ndarray,
    grouped_specs: dict[str, dict[str, np.ndarray]],
    cutoff: float,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDEvidence] skip spectrum figure: matplotlib is not available.")
        return
    groups = [group for group, payload in grouped_specs.items() if payload]
    n_cols = max(1, len(groups))
    fig, axes = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.4), squeeze=False)
    for ax, group in zip(axes[0], groups):
        draw_spectrum_axis(ax, freq_grid=freq_grid, payload=grouped_specs[group], cutoff=cutoff)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lowfreq_ratio_boxplot(dataset_name: str, ratio_payload: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDEvidence] skip low-frequency ratio figure: matplotlib is not available.")
        return
    groups = [group for group in ("all", "normal", "anomaly") if group in ratio_payload]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 4.5), squeeze=False)
    colors = ["#334155", "#2563eb", "#dc2626"]
    for ax, group in zip(axes[0], groups):
        payload = ratio_payload[group]
        values = [np.asarray(payload[key], dtype=np.float64) for key in ("X", "S", "R") if key in payload]
        labels = [key for key in ("X", "S", "R") if key in payload]
        if not values:
            continue
        draw_lowfreq_box_axis(ax, values=values, labels=labels)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_anti_collapse_stats(
    dataset_name: str,
    s_ratio: np.ndarray,
    r_ratio: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDEvidence] skip anti-collapse figure: matplotlib is not available.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), squeeze=False)
    ax_hist = axes[0, 0]
    ax_scatter = axes[0, 1]

    draw_anti_collapse_hist_axis(ax_hist, s_ratio=s_ratio, r_ratio=r_ratio)
    draw_anti_collapse_scatter_axis(ax_scatter, s_ratio=s_ratio, r_ratio=r_ratio, labels=labels)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_state_retrieval_figure(
    dataset_name: str,
    state_stsd: np.ndarray,
    state_raw: np.ndarray,
    novelty_stsd: np.ndarray,
    novelty_raw: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    max_points: int,
) -> dict[str, float | None]:
    if not HAS_MATPLOTLIB:
        print("[STSDEvidence] skip state retrieval figure: matplotlib is not available.")
        return {}
    rng = np.random.RandomState(42)
    n_points = len(labels)
    if n_points > max_points:
        idx = np.sort(rng.choice(np.arange(n_points), size=max_points, replace=False))
    else:
        idx = np.arange(n_points, dtype=np.int64)

    proj_stsd, explained_stsd = pca_project(state_stsd[idx])
    proj_raw, explained_raw = pca_project(state_raw[idx])

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8), squeeze=False)
    for ax, proj in (
        (axes[0, 0], proj_raw),
        (axes[0, 1], proj_stsd),
    ):
        draw_state_projection_axis(ax, proj=proj, labels=labels[idx])

    auc_raw = compute_auc(labels, novelty_raw)
    auc_stsd = compute_auc(labels, novelty_stsd)
    for ax, values, colors in (
        (axes[0, 2], novelty_raw, ["#94a3b8", "#cbd5e1"]),
        (axes[0, 3], novelty_stsd, ["#60a5fa", "#fca5a5"]),
    ):
        per_group = [
            values[labels == 0],
            values[labels == 1] if np.any(labels == 1) else np.zeros(1, dtype=np.float64),
        ]
        box = ax.boxplot(
            per_group,
            tick_labels=["Normal", "Anomaly"],
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.60)
        ax.set_ylabel("state_novelty")
        style_axis(ax, grid_axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"auc_raw_state_novelty": auc_raw, "auc_slow_state_novelty": auc_stsd}


def plot_local_downstream_figure(
    dataset_name: str,
    raw_signal: np.ndarray,
    labels: np.ndarray | None,
    score_raw: np.ndarray,
    score_stsd: np.ndarray,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDEvidence] skip local downstream figure: matplotlib is not available.")
        return
    t = np.arange(raw_signal.shape[0], dtype=np.int64)
    signal_mean_abs = np.mean(np.abs(raw_signal), axis=1)
    score_raw_norm = minmax_normalize(score_raw)
    score_stsd_norm = minmax_normalize(score_stsd)

    fig, axes = plt.subplots(3, 1, figsize=(12.6, 7.8), sharex=True, squeeze=False)
    label_mask = None if labels is None else (np.asarray(labels, dtype=np.int32).reshape(-1) > 0)
    anomaly_onset = None
    if label_mask is not None and np.any(label_mask):
        anomaly_onset = int(np.where(label_mask)[0][0])
    for ax in axes[:, 0]:
        shade_anomaly_region(ax, t=t, label_mask=label_mask, anomaly_onset=anomaly_onset)

    axes[0, 0].plot(t, minmax_normalize(signal_mean_abs), color="#334155", linewidth=1.4)
    axes[0, 0].set_ylabel("Norm. |X|")
    style_axis(axes[0, 0], grid_axis="both")

    axes[1, 0].plot(t, score_raw_norm, color="#64748b", linewidth=1.5)
    axes[1, 0].set_ylabel("Raw-X score")
    style_axis(axes[1, 0], grid_axis="both")

    axes[2, 0].plot(t, score_stsd_norm, color="#dc2626", linewidth=1.6)
    axes[2, 0].set_ylabel("R-based score")
    axes[2, 0].set_xlabel("Timestep in Window")
    style_axis(axes[2, 0], grid_axis="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_validity_paper_panel(
    dataset_name: str,
    freq_grid: np.ndarray,
    grouped_specs: dict[str, dict[str, np.ndarray]],
    ratio_payload: dict[str, dict[str, np.ndarray]],
    cutoff: float,
    s_ratio: np.ndarray,
    r_ratio: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[STSDEvidence] skip validity paper panel: matplotlib is not available.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), squeeze=False)

    spectrum_payload = grouped_specs.get("all")
    if spectrum_payload is None and grouped_specs:
        spectrum_payload = next(iter(grouped_specs.values()))
    if spectrum_payload is not None:
        draw_spectrum_axis(axes[0, 0], freq_grid=freq_grid, payload=spectrum_payload, cutoff=cutoff)
    add_panel_label(axes[0, 0], "(a)")

    draw_lowfreq_grouped_axis(axes[0, 1], ratio_payload=ratio_payload)
    add_panel_label(axes[0, 1], "(b)")

    draw_anti_collapse_hist_axis(axes[1, 0], s_ratio=s_ratio, r_ratio=r_ratio)
    add_panel_label(axes[1, 0], "(c)")

    draw_anti_collapse_scatter_axis(axes[1, 1], s_ratio=s_ratio, r_ratio=r_ratio, labels=labels)
    add_panel_label(axes[1, 1], "(d)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_downstream_paper_panel(
    dataset_name: str,
    state_stsd: np.ndarray,
    state_raw: np.ndarray,
    novelty_stsd: np.ndarray,
    novelty_raw: np.ndarray,
    window_labels: np.ndarray,
    raw_signal: np.ndarray,
    local_labels: np.ndarray | None,
    score_raw: np.ndarray,
    score_stsd: np.ndarray,
    output_path: Path,
    max_points: int,
) -> dict[str, float | None]:
    if not HAS_MATPLOTLIB:
        print("[STSDEvidence] skip downstream paper panel: matplotlib is not available.")
        return {}

    rng = np.random.RandomState(42)
    n_points = len(window_labels)
    if n_points > max_points:
        idx = np.sort(rng.choice(np.arange(n_points), size=max_points, replace=False))
    else:
        idx = np.arange(n_points, dtype=np.int64)

    proj_stsd, _ = pca_project(state_stsd[idx])
    proj_raw, _ = pca_project(state_raw[idx])
    auc_raw = compute_auc(window_labels, novelty_raw)
    auc_stsd = compute_auc(window_labels, novelty_stsd)

    t = np.arange(raw_signal.shape[0], dtype=np.int64)
    signal_mean_abs = minmax_normalize(np.mean(np.abs(raw_signal), axis=1))
    label_mask = None if local_labels is None else (np.asarray(local_labels, dtype=np.int32).reshape(-1) > 0)
    anomaly_onset = None
    if label_mask is not None and np.any(label_mask):
        anomaly_onset = int(np.where(label_mask)[0][0])

    # Do not share x-axes across rows here: the top row mixes scatter/box plots
    # while the bottom row uses temporal axes, and shared x-limits can corrupt
    # the time-series panels for datasets with large PCA outliers.
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8), squeeze=False)

    draw_state_projection_axis(axes[0, 0], proj=proj_raw, labels=window_labels[idx])
    add_panel_label(axes[0, 0], "(a)")

    draw_state_projection_axis(axes[0, 1], proj=proj_stsd, labels=window_labels[idx])
    add_panel_label(axes[0, 1], "(b)")

    draw_state_novelty_compare_axis(axes[0, 2], novelty_raw=novelty_raw, novelty_stsd=novelty_stsd, labels=window_labels)
    add_panel_label(axes[0, 2], "(c)")

    shade_anomaly_region(axes[1, 0], t=t, label_mask=label_mask, anomaly_onset=anomaly_onset)
    axes[1, 0].plot(t, signal_mean_abs, color="#334155", linewidth=1.4)
    axes[1, 0].set_ylabel("Norm. |X|")
    axes[1, 0].set_xlabel("Timestep in Window")
    axes[1, 0].set_xlim(float(t[0]), float(t[-1]) if t.size > 0 else 1.0)
    style_axis(axes[1, 0], grid_axis="both")
    add_panel_label(axes[1, 0], "(d)")

    shade_anomaly_region(axes[1, 1], t=t, label_mask=label_mask, anomaly_onset=anomaly_onset)
    axes[1, 1].plot(t, minmax_normalize(score_raw), color="#64748b", linewidth=1.5)
    axes[1, 1].set_ylabel("Raw-X score")
    axes[1, 1].set_xlabel("Timestep in Window")
    axes[1, 1].set_xlim(float(t[0]), float(t[-1]) if t.size > 0 else 1.0)
    style_axis(axes[1, 1], grid_axis="both")
    add_panel_label(axes[1, 1], "(e)")

    shade_anomaly_region(axes[1, 2], t=t, label_mask=label_mask, anomaly_onset=anomaly_onset)
    axes[1, 2].plot(t, minmax_normalize(score_stsd), color="#dc2626", linewidth=1.6)
    axes[1, 2].set_ylabel("R-based score")
    axes[1, 2].set_xlabel("Timestep in Window")
    axes[1, 2].set_xlim(float(t[0]), float(t[-1]) if t.size > 0 else 1.0)
    style_axis(axes[1, 2], grid_axis="both")
    add_panel_label(axes[1, 2], "(f)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"auc_raw_state_novelty": auc_raw, "auc_slow_state_novelty": auc_stsd}


def build_caption_text(summary: dict[str, object], dataset_name: str, cutoff: float, spectrum_demean: bool) -> str:
    spectrum_note = "after per-window mean removal" if spectrum_demean else "without additional mean removal"
    state_metrics = summary.get("downstream_state", {})
    auc_raw = state_metrics.get("auc_raw_state_novelty")
    auc_slow = state_metrics.get("auc_slow_state_novelty")
    local_meta = summary.get("downstream_local", {})
    lines = [
        f"# STSD Figure Notes ({dataset_name})",
        "",
        "## Frequency-Domain Evidence",
        (
            f"Figure ({dataset_name}): The mean power spectra of X, S, and R are shown for all, normal, and anomalous windows. "
            f"Spectra are computed {spectrum_note}, and the y-axis is log-scaled. "
            f"The dashed vertical line marks the low-frequency cutoff used for the energy-ratio statistic (cutoff={cutoff:.3f})."
        ),
        "",
        "## Low-Frequency Energy Concentration",
        (
            f"Figure ({dataset_name}): Boxplots summarize the fraction of spectral energy below the cutoff for X, S, and R. "
            "A consistently higher ratio for S indicates stronger low-frequency dominance, while a lower ratio for R "
            "shows that the residual preserves more non-low-frequency deviation cues."
        ),
        "",
        "## Non-Collapse Evidence",
        (
            f"Figure ({dataset_name}): The left panel reports the distributions of ||S||/||X|| and ||R||/||X|| over windows, while the right panel "
            "shows their joint scatter for normal and anomalous samples. A broad, non-degenerate spread away from the trivial corner "
            "supports that the decomposition does not collapse to S~=X and R~=0."
        ),
        "",
        "## Downstream Benefit I",
        (
            f"Figure ({dataset_name}): PCA projections compare window-level state embeddings derived from raw X and from the slow component S. "
            "Separate novelty boxplots use independent y-scales for raw-X and S-based retrieval. "
            f"The corresponding AUC values are raw={auc_raw:.3f} and slow={auc_slow:.3f}."
            if auc_raw is not None and auc_slow is not None
            else f"Figure ({dataset_name}): PCA projections compare window-level state embeddings derived from raw X and from the slow component S. "
                 "Separate novelty boxplots use independent y-scales for raw-X and S-based retrieval."
        ),
        "",
        "## Downstream Benefit II",
        (
            f"Figure ({dataset_name}): A representative anomalous window is shown together with patch-level retrieval scores without STSD and with the residual R. "
            "The shaded region marks anomalous points, and the dashed vertical line marks the anomaly onset. "
            f"The illustrated window starts at t={int(local_meta.get('start', 0))} and contains {int(local_meta.get('anomaly_points', 0))} anomalous points."
        ),
        "",
        "## Suggested Main-Text Naming",
        "- Fig. A: Frequency-Domain Evidence of STSD",
        "- Fig. B: Non-Collapse Evidence for the STSD Decomposition",
        "- Fig. C: STSD Improves Window-Level State Retrieval and Local Anomaly Localization",
        "",
        "## Paper Panel Recommendation",
        f"- {summary.get('split', 'test')}_paper_panel_validity.png: combine frequency separation, low-frequency concentration, and anti-collapse evidence into one validity-oriented panel.",
        f"- {summary.get('split', 'test')}_paper_panel_downstream.png: combine state retrieval and local anomaly localization into one downstream-utility panel.",
    ]
    return "\n".join(lines)


def summarize_component(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def rebase_config_to_experiment_dir(config: CoReMADConfig, experiment_dir: Path) -> CoReMADConfig:
    config.artifact_root = str(experiment_dir.parent)
    config.experiment_name = experiment_dir.name
    return config


def encode_with_completion(
    model: CoReMADModel,
    x: torch.Tensor,
) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], list[torch.Tensor]]:
    if model.config.use_completion_head:
        return model.deterministic_completion_scores(x)
    return model.encode(x), []


def group_mask_payload(values: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    payload = {"all": np.asarray(values)}
    normal_mask = labels == 0
    anomaly_mask = labels > 0
    if np.any(normal_mask):
        payload["normal"] = np.asarray(values)[normal_mask]
    if np.any(anomaly_mask):
        payload["anomaly"] = np.asarray(values)[anomaly_mask]
    return payload


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = rebase_config_to_experiment_dir(CoReMADConfig.load(config_path), experiment_dir)
    dataset_name = str(config.dataset)
    if args.device is not None:
        config.device = str(args.device)
    config.num_workers = 0
    config.seed = int(args.seed)

    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    memory = trainer.load_memory_bank()
    device = trainer.device

    raw_bundle = load_raw_dataset_bundle(config.dataset, config.data_root, config=config)
    raw_data, point_labels, segment_ranges, stride = select_split(config, raw_bundle, args.split)
    normalized_data = normalizer.transform(raw_data)

    full_loader = build_loader(
        data=normalized_data,
        labels=point_labels,
        seq_len=config.seq_len,
        stride=stride,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        max_windows=0,
        drop_last=False,
        segment_ranges=segment_ranges,
    )
    full_dataset = full_loader.dataset
    if hasattr(full_dataset, "start_indices") and full_dataset.start_indices is not None:
        start_indices = np.asarray(full_dataset.start_indices, dtype=np.int64)
    else:
        start_indices = build_fallback_start_indices(
            total_steps=len(normalized_data),
            seq_len=int(config.seq_len),
            stride=int(stride),
        )
    if start_indices.size == 0:
        raise RuntimeError("No valid windows available for the selected split.")

    window_labels = compute_window_labels(start_indices, point_labels, int(config.seq_len))
    sampled_ids = stratified_sample_indices(window_labels, int(args.max_windows), int(args.seed))
    sampled_starts = start_indices[sampled_ids]
    sampled_window_labels = window_labels[sampled_ids]

    sample_loader = build_loader(
        data=normalized_data,
        labels=point_labels,
        seq_len=config.seq_len,
        stride=stride,
        batch_size=max(1, int(args.batch_size)),
        num_workers=0,
        shuffle=False,
        max_windows=0,
        drop_last=False,
        start_indices=sampled_starts,
    )

    cutoff = float(config.stsd_lowpass_center if args.low_freq_cutoff is None else args.low_freq_cutoff)
    cutoff = max(0.0, min(cutoff, 1.0))
    output_dir = ensure_output_dir(args, experiment_dir)

    x_spectra_list: list[np.ndarray] = []
    s_spectra_list: list[np.ndarray] = []
    r_spectra_list: list[np.ndarray] = []
    x_lowfreq_list: list[np.ndarray] = []
    s_lowfreq_list: list[np.ndarray] = []
    r_lowfreq_list: list[np.ndarray] = []
    s_ratio_list: list[np.ndarray] = []
    r_ratio_list: list[np.ndarray] = []
    state_stsd_list: list[np.ndarray] = []
    state_raw_list: list[np.ndarray] = []
    novelty_stsd_list: list[np.ndarray] = []
    novelty_raw_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in sample_loader:
            x = batch["x"].to(device, non_blocking=True)
            encoded_stsd, comp_scores_stsd = encode_with_completion(model, x)
            encoded_raw = encode_without_stsd(model, x)
            comp_scores_raw = deterministic_completion_scores_from_encoded(model, encoded_raw, x)
            _, memory_out_stsd = build_feature_dict(config, encoded_stsd, comp_scores_stsd, memory)
            _, memory_out_raw = build_feature_dict(config, encoded_raw, comp_scores_raw, memory)

            x_np = x.detach().cpu().numpy().astype(np.float64)
            slow_np = encoded_stsd["slow"].detach().cpu().numpy().astype(np.float64)
            resid_np = encoded_stsd["residual"].detach().cpu().numpy().astype(np.float64)

            x_spectra_list.append(compute_power_spectrum(x_np, demean=bool(args.spectrum_demean)))
            s_spectra_list.append(compute_power_spectrum(slow_np, demean=bool(args.spectrum_demean)))
            r_spectra_list.append(compute_power_spectrum(resid_np, demean=bool(args.spectrum_demean)))
            x_lowfreq_list.append(low_frequency_ratio(x_np, cutoff, demean=bool(args.spectrum_demean)))
            s_lowfreq_list.append(low_frequency_ratio(slow_np, cutoff, demean=bool(args.spectrum_demean)))
            r_lowfreq_list.append(low_frequency_ratio(resid_np, cutoff, demean=bool(args.spectrum_demean)))

            x_norm = np.linalg.norm(x_np.reshape(x_np.shape[0], -1), axis=1)
            s_norm = np.linalg.norm(slow_np.reshape(slow_np.shape[0], -1), axis=1)
            r_norm = np.linalg.norm(resid_np.reshape(resid_np.shape[0], -1), axis=1)
            s_ratio_list.append(s_norm / np.maximum(x_norm, 1e-12))
            r_ratio_list.append(r_norm / np.maximum(x_norm, 1e-12))

            state_stsd_list.append(encoded_stsd["state_vec"].detach().cpu().numpy().astype(np.float64))
            state_raw_list.append(encoded_raw["state_vec"].detach().cpu().numpy().astype(np.float64))
            novelty_stsd_list.append(memory_out_stsd["state_novelty"].detach().cpu().numpy().astype(np.float64))
            novelty_raw_list.append(memory_out_raw["state_novelty"].detach().cpu().numpy().astype(np.float64))

    x_spectra = np.concatenate(x_spectra_list, axis=0)
    s_spectra = np.concatenate(s_spectra_list, axis=0)
    r_spectra = np.concatenate(r_spectra_list, axis=0)
    x_lowfreq = np.concatenate(x_lowfreq_list, axis=0)
    s_lowfreq = np.concatenate(s_lowfreq_list, axis=0)
    r_lowfreq = np.concatenate(r_lowfreq_list, axis=0)
    s_ratio = np.concatenate(s_ratio_list, axis=0)
    r_ratio = np.concatenate(r_ratio_list, axis=0)
    state_stsd = np.concatenate(state_stsd_list, axis=0)
    state_raw = np.concatenate(state_raw_list, axis=0)
    novelty_stsd = np.concatenate(novelty_stsd_list, axis=0)
    novelty_raw = np.concatenate(novelty_raw_list, axis=0)

    freq_grid = np.linspace(0.0, 1.0, x_spectra.shape[1], dtype=np.float64)
    grouped_specs: dict[str, dict[str, np.ndarray]] = {}
    ratio_payload: dict[str, dict[str, np.ndarray]] = {}
    for group_name, mask in (
        ("all", np.ones(len(sampled_window_labels), dtype=bool)),
        ("normal", sampled_window_labels == 0),
        ("anomaly", sampled_window_labels > 0),
    ):
        if not np.any(mask):
            continue
        grouped_specs[group_name] = {
            "X": x_spectra[mask].mean(axis=0),
            "S": s_spectra[mask].mean(axis=0),
            "R": r_spectra[mask].mean(axis=0),
        }
        ratio_payload[group_name] = {
            "X": x_lowfreq[mask],
            "S": s_lowfreq[mask],
            "R": r_lowfreq[mask],
        }

    plot_spectrum_comparison(
        dataset_name=dataset_name,
        freq_grid=freq_grid,
        grouped_specs=grouped_specs,
        cutoff=cutoff,
        output_path=output_dir / f"{args.split}_spectrum_comparison.png",
    )
    plot_lowfreq_ratio_boxplot(
        dataset_name=dataset_name,
        ratio_payload=ratio_payload,
        output_path=output_dir / f"{args.split}_lowfreq_ratio_boxplot.png",
    )
    plot_anti_collapse_stats(
        dataset_name=dataset_name,
        s_ratio=s_ratio,
        r_ratio=r_ratio,
        labels=sampled_window_labels,
        output_path=output_dir / f"{args.split}_anti_collapse_stats.png",
    )
    state_metrics = plot_state_retrieval_figure(
        dataset_name=dataset_name,
        state_stsd=state_stsd,
        state_raw=state_raw,
        novelty_stsd=novelty_stsd,
        novelty_raw=novelty_raw,
        labels=sampled_window_labels,
        output_path=output_dir / f"{args.split}_downstream_state_retrieval.png",
        max_points=max(10, int(args.embedding_max_points)),
    )

    local_window_id = choose_local_window_id(start_indices, point_labels, int(config.seq_len))
    local_item = full_dataset[int(local_window_id)]
    local_start = int(local_item["start"])
    local_end = local_start + int(config.seq_len)
    local_x = local_item["x"].unsqueeze(0).to(device)
    local_labels = None if point_labels is None else np.asarray(point_labels[local_start:local_end], dtype=np.int32)
    local_raw_signal = np.asarray(raw_data[local_start:local_end], dtype=np.float64)

    with torch.no_grad():
        encoded_stsd_local, comp_scores_stsd_local = encode_with_completion(model, local_x)
        encoded_raw_local = encode_without_stsd(model, local_x)
        comp_scores_raw_local = deterministic_completion_scores_from_encoded(model, encoded_raw_local, local_x)
        features_stsd_local, _ = build_feature_dict(config, encoded_stsd_local, comp_scores_stsd_local, memory)
        features_raw_local, _ = build_feature_dict(config, encoded_raw_local, comp_scores_raw_local, memory)

    plot_local_downstream_figure(
        dataset_name=dataset_name,
        raw_signal=local_raw_signal,
        labels=local_labels,
        score_raw=features_raw_local["knn_distance"].squeeze(0).detach().cpu().numpy().astype(np.float64),
        score_stsd=features_stsd_local["knn_distance"].squeeze(0).detach().cpu().numpy().astype(np.float64),
        output_path=output_dir / f"{args.split}_downstream_localization.png",
    )
    plot_validity_paper_panel(
        dataset_name=dataset_name,
        freq_grid=freq_grid,
        grouped_specs=grouped_specs,
        ratio_payload=ratio_payload,
        cutoff=cutoff,
        s_ratio=s_ratio,
        r_ratio=r_ratio,
        labels=sampled_window_labels,
        output_path=output_dir / f"{args.split}_paper_panel_validity.png",
    )
    plot_downstream_paper_panel(
        dataset_name=dataset_name,
        state_stsd=state_stsd,
        state_raw=state_raw,
        novelty_stsd=novelty_stsd,
        novelty_raw=novelty_raw,
        window_labels=sampled_window_labels,
        raw_signal=local_raw_signal,
        local_labels=local_labels,
        score_raw=features_raw_local["knn_distance"].squeeze(0).detach().cpu().numpy().astype(np.float64),
        score_stsd=features_stsd_local["knn_distance"].squeeze(0).detach().cpu().numpy().astype(np.float64),
        output_path=output_dir / f"{args.split}_paper_panel_downstream.png",
        max_points=max(10, int(args.embedding_max_points)),
    )

    summary = {
        "experiment_dir": str(experiment_dir),
        "dataset": dataset_name,
        "split": str(args.split),
        "sampled_window_count": int(len(sampled_window_labels)),
        "sampled_normal_windows": int(np.sum(sampled_window_labels == 0)),
        "sampled_anomaly_windows": int(np.sum(sampled_window_labels > 0)),
        "low_frequency_cutoff": float(cutoff),
        "spectrum_demean": bool(args.spectrum_demean),
        "frequency_evidence": {
            "lowfreq_ratio": {
                "X": summarize_component(x_lowfreq),
                "S": summarize_component(s_lowfreq),
                "R": summarize_component(r_lowfreq),
            }
        },
        "anti_collapse": {
            "s_over_x": summarize_component(s_ratio),
            "r_over_x": summarize_component(r_ratio),
        },
        "downstream_state": state_metrics,
        "downstream_local": {
            "window_id": int(local_window_id),
            "start": int(local_start),
            "end": int(local_end),
            "anomaly_points": int(local_labels.sum()) if local_labels is not None else 0,
        },
        "notes": {
            "state_baseline": "raw-X proxy with the same trained state/patch encoders and memory bank",
            "local_baseline": "raw-X proxy kNN patch retrieval compared against STSD residual-based retrieval",
            "spectrum_setting": "per-window mean removal before RFFT with log-scaled y-axis" if args.spectrum_demean else "raw-window RFFT with log-scaled y-axis",
            "paper_panels": [
                f"{args.split}_paper_panel_validity.png",
                f"{args.split}_paper_panel_downstream.png",
            ],
        },
    }
    for group_name, payload in ratio_payload.items():
        summary["frequency_evidence"][f"{group_name}_lowfreq_ratio"] = {
            key: summarize_component(value) for key, value in payload.items()
        }

    summary_path = output_dir / f"{args.split}_stsd_evidence_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    caption_text = build_caption_text(
        summary=summary,
        dataset_name=dataset_name,
        cutoff=cutoff,
        spectrum_demean=bool(args.spectrum_demean),
    )
    caption_path = output_dir / f"{args.split}_paper_figure_notes.md"
    caption_path.write_text(caption_text, encoding="utf-8")
    print(f"[STSDEvidence] experiment_dir={experiment_dir}")
    print(f"[STSDEvidence] split={args.split} sampled_windows={len(sampled_window_labels)} cutoff={cutoff:.3f}")
    print(f"[STSDEvidence] saved figures to {output_dir}")
    print(f"[STSDEvidence] summary saved to {summary_path}")
    print(f"[STSDEvidence] figure notes saved to {caption_path}")


if __name__ == "__main__":
    main()
