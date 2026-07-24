from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tep_common import (
    empirical_percentile,
    load_fault_log_fields,
    load_sequence_scores_with_meta,
    load_window_logs,
)

try:
    from sklearn.manifold import TSNE

    HAS_SKLEARN = True
except Exception:
    TSNE = None
    HAS_SKLEARN = False

try:
    import umap

    HAS_UMAP = True
except Exception:
    umap = None
    HAS_UMAP = False


MODE_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#17becf",
}

EVIDENCE_KEYS = ["memory_distance", "state_novelty", "completion_scale8", "completion_scale32"]
EVIDENCE_LABELS = {
    "memory_distance": "Memory",
    "state_novelty": "Novelty",
    "completion_scale8": "Comp-short",
    "completion_scale32": "Comp-long",
}


def _available_modes(
    exp_payloads: list[tuple[str, dict[str, Any]]],
    log_key: str = "audit_logs",
) -> list[int]:
    modes: set[int] = set()
    for _, payload in exp_payloads:
        logs = payload[log_key]
        modes.update(int(value) for value in np.unique(np.asarray(logs["mode_id"])).tolist())
    return sorted(modes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot key offline TEP mechanism figures from exported logs.")
    parser.add_argument("--experiment_dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--short_dir", type=Path, default=None)
    parser.add_argument("--long_dir", type=Path, default=None)
    parser.add_argument("--multi_dir", type=Path, default=None)
    parser.add_argument("--log_subdir", type=str, default="tep_mechanism")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--embedding_method", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--smr_k", type=int, default=10)
    return parser.parse_args()


def _reduce_2d(x: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(x), "UMAP"
    if HAS_SKLEARN:
        perplexity = max(5, min(30, max(6, x.shape[0] // 10)))
        reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
        return reducer.fit_transform(x), "t-SNE"
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, "PCA"


def _load_experiment_payload(exp_dir: Path, log_subdir: str) -> dict[str, Any]:
    log_dir = exp_dir / log_subdir
    return {
        "audit_logs": load_window_logs(log_dir, prefix="audit_normal_window_logs"),
        "fault_logs": load_window_logs(log_dir, prefix="fault_window_logs"),
        "fault_metric_logs": load_fault_log_fields(
            log_dir,
            fields=(
                "mode_id",
                "fault_id",
                "memory_distance",
                "topk_neighbor_mode_ids",
            ),
        ),
        "fault_sequence_records": load_sequence_scores_with_meta(log_dir, filename="fault_sequence_scores_with_meta.json"),
    }


def _calibrate_fault_sequence_rows(
    audit_logs: dict[str, np.ndarray],
    fault_sequence_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audit_reference = {
        "selected": np.asarray(audit_logs["final"], dtype=np.float64),
        "memory_distance": np.asarray(audit_logs["memory_distance"], dtype=np.float64),
        "state_novelty": np.asarray(audit_logs["state_novelty"], dtype=np.float64),
        "completion_scale8": np.asarray(audit_logs["completion_scale8"], dtype=np.float64),
        "completion_scale32": np.asarray(audit_logs["completion_scale32"], dtype=np.float64),
    }
    rows: list[dict[str, Any]] = []
    for record in fault_sequence_records:
        row = dict(record)
        if "memory_distance" not in row and "knn_distance" in row:
            row["memory_distance"] = float(row["knn_distance"])
        for score_key, reference in audit_reference.items():
            value = float(row.get(score_key, float("nan")))
            if score_key == "memory_distance" and not np.isfinite(value):
                value = float(row.get("knn_distance", float("nan")))
            calibrated = float(empirical_percentile(reference, np.asarray([value], dtype=np.float64))[0])
            row[f"{score_key}_calibrated"] = calibrated
        rows.append(row)
    return rows


def plot_state_embedding(exp_payloads: list[tuple[str, dict[str, Any]]], output_dir: Path, method: str) -> None:
    fig, axes = plt.subplots(1, len(exp_payloads), figsize=(6 * len(exp_payloads), 5), squeeze=False)
    for col_idx, (label, payload) in enumerate(exp_payloads):
        ax = axes[0, col_idx]
        logs = payload["audit_logs"]
        x = np.asarray(logs["state_vec"], dtype=np.float64)
        mode_id = np.asarray(logs["mode_id"], dtype=np.int32)
        if len(x) == 0:
            ax.set_title(f"{label}\n(no audit-normal windows)")
            ax.axis("off")
            continue
        coords, used_method = _reduce_2d(x, method)
        for current_mode in sorted(np.unique(mode_id).tolist()):
            group = coords[mode_id == current_mode]
            ax.scatter(group[:, 0], group[:, 1], s=8, alpha=0.65, label=f"mode {current_mode}", color=MODE_COLORS.get(current_mode))
        ax.set_title(f"{label}\n{used_method}")
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "state_embedding.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[TEP Plot] wrote {path}")


def plot_normal_violin(exp_payloads: list[tuple[str, dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(exp_payloads), figsize=(5 * len(exp_payloads), 5), squeeze=False)
    mode_order = _available_modes(exp_payloads, "audit_logs")
    for col_idx, (label, payload) in enumerate(exp_payloads):
        ax = axes[0, col_idx]
        logs = payload["audit_logs"]
        mode_id = np.asarray(logs["mode_id"], dtype=np.int32)
        final_score = np.asarray(logs["final"], dtype=np.float64)
        data = [final_score[mode_id == current_mode] for current_mode in mode_order]
        violin_data = []
        violin_pos = []
        violin_modes = []
        for pos, (current_mode, values) in enumerate(zip(mode_order, data), start=1):
            if len(values) >= 2:
                violin_data.append(values)
                violin_pos.append(pos)
                violin_modes.append(current_mode)
            elif len(values) == 1:
                ax.scatter([pos], values, s=28, color=MODE_COLORS.get(current_mode, "#888888"), zorder=3)
        if violin_data:
            parts = ax.violinplot(violin_data, positions=violin_pos, showmedians=True, widths=0.8)
            for body, current_mode in zip(parts["bodies"], violin_modes):
                body.set_facecolor(MODE_COLORS.get(current_mode, "#888888"))
                body.set_alpha(0.55)
        ax.set_xticks(np.arange(1, len(mode_order) + 1), [f"mode {mode}" for mode in mode_order])
        ax.set_ylabel("audit-normal final score")
        ax.set_title(label)
    fig.tight_layout()
    path = output_dir / "normal_final_violin_by_mode.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[TEP Plot] wrote {path}")


def plot_retrieval_confusion(exp_payloads: list[tuple[str, dict[str, Any]]], output_dir: Path, smr_k: int) -> None:
    mode_order = _available_modes(exp_payloads, "fault_logs")
    fig, axes = plt.subplots(1, len(exp_payloads), figsize=(5 * len(exp_payloads), 4.5), squeeze=False)
    for col_idx, (label, payload) in enumerate(exp_payloads):
        ax = axes[0, col_idx]
        logs = payload.get("fault_metric_logs", payload["fault_logs"])
        query_mode = np.asarray(logs["mode_id"], dtype=np.int32)
        neighbor_mode = np.asarray(logs["topk_neighbor_mode_ids"], dtype=np.int32)[:, :smr_k]
        heat = np.zeros((len(mode_order), len(mode_order)), dtype=np.float64)
        for row_mode_idx, row_mode in enumerate(mode_order):
            mask = query_mode == row_mode
            if not np.any(mask):
                continue
            valid = neighbor_mode[mask]
            denom = 0.0
            for col_mode_idx, col_mode in enumerate(mode_order):
                count = float(np.sum(valid == col_mode))
                heat[row_mode_idx, col_mode_idx] = count
                denom += count
            if denom > 0:
                heat[row_mode_idx] /= denom
        im = ax.imshow(heat, cmap="YlOrRd", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(mode_order)), [f"m{mode}" for mode in mode_order])
        ax.set_yticks(np.arange(len(mode_order)), [f"m{mode}" for mode in mode_order])
        ax.set_xlabel("retrieved mode")
        ax.set_ylabel("fault query mode")
        ax.set_title(label)
        for i in range(len(mode_order)):
            for j in range(len(mode_order)):
                ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    fig.tight_layout()
    path = output_dir / "retrieval_mode_confusion_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[TEP Plot] wrote {path}")


def plot_exceedance(exp_payloads: list[tuple[str, dict[str, Any]]], output_dir: Path) -> None:
    q_grid = np.linspace(0.90, 0.99, 10)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(q_grid, 1.0 - q_grid, color="black", linestyle="--", linewidth=1.5, label="theory (1-q)")
    for label, payload in exp_payloads:
        logs = payload["audit_logs"]
        mode_id = np.asarray(logs["mode_id"], dtype=np.int32)
        final_score = np.asarray(logs["final"], dtype=np.float64)
        curve = []
        for q in q_grid.tolist():
            threshold = float(np.quantile(final_score, q))
            per_mode = []
            for current_mode in sorted(np.unique(mode_id).tolist()):
                group_mask = mode_id == current_mode
                if np.any(group_mask):
                    per_mode.append(float(np.mean(final_score[group_mask] > threshold)))
            curve.append(float(np.mean(per_mode)) if per_mode else np.nan)
        ax.plot(q_grid, curve, marker="o", linewidth=1.6, label=label)
    ax.set_xlabel("q")
    ax.set_ylabel("Pr(score > global q-threshold | audit normal)")
    ax.set_title("Exceedance Plot")
    ax.legend()
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    path = output_dir / "exceedance_plot.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[TEP Plot] wrote {path}")


def plot_mode_fpr_and_fault_gap(
    exp_payloads: list[tuple[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    mode_order = _available_modes(exp_payloads, "audit_logs")
    fig, axes = plt.subplots(
        len(exp_payloads),
        2,
        figsize=(10, 3.8 * len(exp_payloads)),
        squeeze=False,
    )
    for row_idx, (label, payload) in enumerate(exp_payloads):
        audit_logs = payload["audit_logs"]
        fault_logs = payload.get("fault_metric_logs", payload["fault_logs"])
        audit_mode = np.asarray(audit_logs["mode_id"], dtype=np.int32)
        audit_final = np.asarray(audit_logs["final"], dtype=np.float64)
        audit_memory = np.asarray(audit_logs["memory_distance"], dtype=np.float64)
        fault_mode = np.asarray(fault_logs["mode_id"], dtype=np.int32)
        fault_memory = np.asarray(fault_logs["memory_distance"], dtype=np.float64)

        threshold = float(np.quantile(audit_final, 0.95))
        fpr_values = [
            float(np.mean(audit_final[audit_mode == mode] > threshold))
            if np.any(audit_mode == mode)
            else float("nan")
            for mode in mode_order
        ]
        gap_values = [
            (
                float(np.mean(fault_memory[fault_mode == mode]))
                - float(np.mean(audit_memory[audit_mode == mode]))
            )
            if np.any(fault_mode == mode) and np.any(audit_mode == mode)
            else float("nan")
            for mode in mode_order
        ]
        colors = [MODE_COLORS.get(mode, "#888888") for mode in mode_order]
        x = np.arange(len(mode_order))

        ax_fpr, ax_gap = axes[row_idx]
        ax_fpr.bar(x, fpr_values, color=colors, alpha=0.82)
        ax_fpr.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="target 0.05")
        ax_fpr.set_xticks(x, [f"M{mode}" for mode in mode_order])
        ax_fpr.set_ylabel("normal false-positive rate")
        ax_fpr.set_title(f"{label}: per-mode normal FPR")
        ax_fpr.legend()
        ax_fpr.grid(axis="y", alpha=0.25, linestyle=":")

        ax_gap.bar(x, gap_values, color=colors, alpha=0.82)
        ax_gap.axhline(0.0, color="black", linewidth=1.0)
        ax_gap.set_xticks(x, [f"M{mode}" for mode in mode_order])
        ax_gap.set_ylabel("mean fault memory distance - normal")
        ax_gap.set_title(f"{label}: fault-normal gap")
        ax_gap.grid(axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    path = output_dir / "mode_fpr_and_fault_normal_gap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[TEP Plot] wrote {path}")


def plot_fault_evidence_heatmap(exp_payloads: list[tuple[str, dict[str, Any]]], output_dir: Path) -> None:
    mode_order = _available_modes(exp_payloads, "fault_logs")
    fig, axes = plt.subplots(
        len(exp_payloads),
        len(EVIDENCE_KEYS),
        figsize=(3.3 * len(EVIDENCE_KEYS), 0.28 * 28 * len(exp_payloads) + 2.5),
        squeeze=False,
    )
    images = []
    for row_idx, (label, payload) in enumerate(exp_payloads):
        rows = _calibrate_fault_sequence_rows(payload["audit_logs"], payload["fault_sequence_records"])
        if not rows:
            for ax in axes[row_idx]:
                ax.set_title(f"{label}\n(no fault sequences)")
                ax.axis("off")
            continue
        fault_order = sorted({int(row["fault_id"]) for row in rows})
        row_lookup = {
            (int(row["fault_id"]), int(row["mode_id"])): row
            for row in rows
        }
        for col_idx, evidence_key in enumerate(EVIDENCE_KEYS):
            ax = axes[row_idx, col_idx]
            heat = np.full((len(fault_order), len(mode_order)), np.nan, dtype=np.float64)
            for fault_idx, fault_id in enumerate(fault_order):
                for mode_idx, mode_id in enumerate(mode_order):
                    record = row_lookup.get((fault_id, mode_id))
                    if record is not None:
                        heat[fault_idx, mode_idx] = float(
                            record.get(f"{evidence_key}_calibrated", float("nan"))
                        )
            image = ax.imshow(heat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
            images.append(image)
            ax.set_xticks(np.arange(len(mode_order)), [f"M{mode}" for mode in mode_order])
            ax.set_yticks(
                np.arange(len(fault_order)),
                [f"IDV{fault}" for fault in fault_order],
            )
            ax.set_title(f"{label}: {EVIDENCE_LABELS[evidence_key]}")
            if col_idx > 0:
                ax.tick_params(axis="y", labelleft=False)
    if images:
        fig.colorbar(
            images[-1],
            ax=axes.ravel().tolist(),
            shrink=0.85,
            label="calibrated sequence evidence",
        )
    fig.tight_layout()
    path = output_dir / "fault_evidence_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[TEP Plot] wrote {path}")


def plot_fault_triplet_consistency(exp_payloads: list[tuple[str, dict[str, Any]]], output_dir: Path) -> None:
    mode_order = _available_modes(exp_payloads, "fault_logs")
    fig, axes = plt.subplots(1, len(exp_payloads), figsize=(6 * len(exp_payloads), 4.5), squeeze=False)
    max_fault_count = 0
    for col_idx, (label, payload) in enumerate(exp_payloads):
        ax = axes[0, col_idx]
        rows = _calibrate_fault_sequence_rows(payload["audit_logs"], payload["fault_sequence_records"])
        fault_ids = sorted({int(row["fault_id"]) for row in rows})
        max_fault_count = max(max_fault_count, len(fault_ids))
        if not fault_ids:
            ax.set_title(f"{label}\n(no fault sequences)")
            ax.axis("off")
            continue
        for fault_id in fault_ids:
            points = []
            x_values = []
            for mode_id in mode_order:
                matched = [
                    float(row["selected_calibrated"])
                    for row in rows
                    if int(row["fault_id"]) == fault_id and int(row["mode_id"]) == mode_id
                ]
                if matched:
                    x_values.append(mode_id)
                    points.append(float(np.mean(matched)))
            if len(points) >= 2:
                ax.plot(x_values, points, marker="o", linewidth=1.2, alpha=0.75, label=f"IDV{fault_id}")
            elif len(points) == 1:
                ax.scatter(x_values, points, s=30, alpha=0.75)
        ax.set_xticks(mode_order, [f"mode {mode}" for mode in mode_order])
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("calibrated fault sequence score")
        ax.set_title(label)
        ax.grid(alpha=0.25, linestyle=":")
    if max_fault_count <= 12:
        handles, labels = [], []
        for ax in axes.ravel().tolist():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)))
            fig.tight_layout(rect=[0, 0, 1, 0.92])
        else:
            fig.tight_layout()
    else:
        fig.tight_layout()
    path = output_dir / "fault_triplet_consistency.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[TEP Plot] wrote {path}")


def plot_a4_gain_heatmap(
    short_payload: dict[str, Any],
    long_payload: dict[str, Any],
    multi_payload: dict[str, Any],
    output_dir: Path,
) -> None:
    def build_fault_table(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
        rows = _calibrate_fault_sequence_rows(payload["audit_logs"], payload["fault_sequence_records"])
        return {
            str(row["sequence_name"]): {
                "mode_id": int(row["mode_id"]),
                "fault_id": int(row["fault_id"]),
                "selected_calibrated": float(row["selected_calibrated"]),
            }
            for row in rows
        }

    short_table = build_fault_table(short_payload)
    long_table = build_fault_table(long_payload)
    multi_table = build_fault_table(multi_payload)
    common = sorted(set(short_table) & set(long_table) & set(multi_table))
    if not common:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.set_title("Fault-File Gain Heatmap")
        ax.text(0.5, 0.5, "No common fault sequences", ha="center", va="center")
        ax.axis("off")
        path = output_dir / "fault_file_gain_heatmap.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[TEP Plot] wrote {path}")
        return
    modes = sorted({int(multi_table[name]["mode_id"]) for name in common})
    faults = sorted({multi_table[name]["fault_id"] for name in common})
    heat = np.full((len(modes), len(faults)), np.nan, dtype=np.float64)
    for name in common:
        mode_id = multi_table[name]["mode_id"]
        fault_id = multi_table[name]["fault_id"]
        gain = multi_table[name]["selected_calibrated"] - max(
            short_table[name]["selected_calibrated"],
            long_table[name]["selected_calibrated"],
        )
        heat[modes.index(mode_id), faults.index(fault_id)] = float(gain)

    fig, ax = plt.subplots(figsize=(1.2 * len(faults) + 2.5, 4.2))
    im = ax.imshow(heat, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(faults)), [f"IDV{fault}" for fault in faults], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(modes)), [f"mode {mode}" for mode in modes])
    ax.set_title("Fault-File Gain Heatmap")
    for i in range(len(modes)):
        for j in range(len(faults)):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, label="multi - max(short, long)")
    fig.tight_layout()
    path = output_dir / "fault_file_gain_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[TEP Plot] wrote {path}")


def main() -> None:
    args = parse_args()
    exp_dirs = list(args.experiment_dirs)
    trio = [args.short_dir, args.long_dir, args.multi_dir]
    for path in trio:
        if path is not None:
            exp_dirs.append(path)
    exp_dirs = [path for path in exp_dirs if path is not None]
    if not exp_dirs:
        raise ValueError("Please provide --experiment_dirs or the A4 trio.")

    unique_dirs = []
    seen = set()
    for path in exp_dirs:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_dirs.append(path)
    exp_dirs = unique_dirs

    output_dir = args.output_dir or (exp_dirs[0] / args.log_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dirs = list(args.experiment_dirs)
    if not base_dirs and args.multi_dir is not None:
        base_dirs = [args.multi_dir]
    exp_payloads = [(exp_dir.name, _load_experiment_payload(exp_dir, args.log_subdir)) for exp_dir in base_dirs]
    if exp_payloads:
        plot_state_embedding(exp_payloads, output_dir, args.embedding_method)
        plot_normal_violin(exp_payloads, output_dir)
        plot_retrieval_confusion(exp_payloads, output_dir, args.smr_k)
        plot_exceedance(exp_payloads, output_dir)
        plot_mode_fpr_and_fault_gap(exp_payloads, output_dir)
        plot_fault_evidence_heatmap(exp_payloads, output_dir)
        plot_fault_triplet_consistency(exp_payloads, output_dir)

    if args.short_dir and args.long_dir and args.multi_dir:
        plot_a4_gain_heatmap(
            short_payload=_load_experiment_payload(args.short_dir, args.log_subdir),
            long_payload=_load_experiment_payload(args.long_dir, args.log_subdir),
            multi_payload=_load_experiment_payload(args.multi_dir, args.log_subdir),
            output_dir=output_dir,
        )
        summary = {
            "short_dir": str(args.short_dir),
            "long_dir": str(args.long_dir),
            "multi_dir": str(args.multi_dir),
        }
        (output_dir / "a4_plot_inputs.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
