from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    HAS_SKLEARN = True
except Exception:
    average_precision_score = None
    precision_recall_curve = None
    roc_auc_score = None
    HAS_SKLEARN = False

try:
    from vus.metrics import get_metrics as vus_get_metrics

    HAS_VUS = True
except Exception:
    vus_get_metrics = None
    HAS_VUS = False

from coremad import CoReMADConfig
from coremad.data import load_raw_dataset_bundle


DEFAULT_SCORE_KEYS = [
    "completion_scale8",
    "completion_scale32",
    "knn_distance",
    "state_novelty",
    "cdf_max_score",
    "cdf_mean_score",
    "cdf_softmax_score",
    "final_fused_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved diagnostic anomaly scores from test_diagnostic_scores.npz")
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to test_diagnostic_scores.npz",
    )
    parser.add_argument(
        "--labels-key",
        type=str,
        default="labels",
        help="Label key inside the npz file",
    )
    parser.add_argument(
        "--score-keys",
        type=str,
        nargs="*",
        default=DEFAULT_SCORE_KEYS,
        help="Score keys to evaluate",
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


def normalize_scores_01(scores: np.ndarray) -> np.ndarray:
    scores = sanitize_scores(scores)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.zeros_like(scores, dtype=np.float64)
    score_min = float(finite.min())
    score_max = float(finite.max())
    if score_max - score_min < 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return np.clip((scores - score_min) / (score_max - score_min), 0.0, 1.0)


def load_score_array(data: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in data.files:
            return sanitize_scores(data[key])
    return None


def find_anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(np.asarray(labels, dtype=np.int32)):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


def estimate_vus_window(labels: np.ndarray) -> int:
    # Use the median anomaly segment length as a stable dataset-specific sliding window.
    seg_lengths = [end - start for start, end in find_anomaly_segments(labels)]
    if not seg_lengths:
        return 1
    return max(1, int(np.median(np.asarray(seg_lengths, dtype=np.float64))))


def compute_vus_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if not HAS_VUS:
        return {
            "aff_f1": float("nan"),
            "range_f1": float("nan"),
            "r_auc_roc": float("nan"),
            "r_auc_pr": float("nan"),
            "vus_roc": float("nan"),
            "vus_pr": float("nan"),
        }
    vus_window = estimate_vus_window(labels)
    results = vus_get_metrics(
        normalize_scores_01(scores),
        np.asarray(labels, dtype=np.int32),
        metric="all",
        slidingWindow=vus_window,
    )
    aff_precision = float(results.get("Affiliation_Precision", float("nan")))
    aff_recall = float(results.get("Affiliation_Recall", float("nan")))
    range_precision = float(results.get("Range_Precision", results.get("Rprecision", float("nan"))))
    range_recall = float(results.get("Range_Recall", results.get("Rrecall", float("nan"))))
    range_f1 = float(results.get("Range_F", results.get("RF", float("nan"))))
    if not np.isfinite(range_f1):
        range_f1 = 2.0 * range_precision * range_recall / max(range_precision + range_recall, 1e-12)
    return {
        "aff_f1": 2.0 * aff_precision * aff_recall / max(aff_precision + aff_recall, 1e-12),
        "range_f1": float(range_f1),
        "r_auc_roc": float(results.get("R_AUC_ROC", float("nan"))),
        "r_auc_pr": float(results.get("R_AUC_PR", float("nan"))),
        "vus_roc": float(results.get("VUS_ROC", float("nan"))),
        "vus_pr": float(results.get("VUS_PR", float("nan"))),
    }


def best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for best-F1 threshold search.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.clip(precision + recall, 1e-8, None)
    best_idx = int(np.nanargmax(f1))
    if thresholds.size == 0:
        threshold = 0.5
    elif best_idx < thresholds.size:
        threshold = float(thresholds[best_idx])
    else:
        threshold = float(thresholds[-1])
    return float(np.nanmax(f1)), threshold


def evaluate_one(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, dict[str, float]]:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required to compute ROC-AUC / PR-AUC.")
    auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    f1, _ = best_f1(labels, scores)
    vus_metrics = compute_vus_metrics(labels, scores)
    return auc, pr_auc, f1, vus_metrics


def print_metric_row(name: str, labels: np.ndarray, scores: np.ndarray) -> None:
    auc, pr_auc, f1, vus_metrics = evaluate_one(labels, scores)
    print(
        f"{name:30s} AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  best-F1={f1:.4f}  "
        f"Aff-F1={vus_metrics['aff_f1']:.4f}  "
        f"Range-F1={vus_metrics['range_f1']:.4f}  "
        f"R-AUC-ROC={vus_metrics['r_auc_roc']:.4f}  "
        f"R-AUC-PR={vus_metrics['r_auc_pr']:.4f}  "
        f"VUS-ROC={vus_metrics['vus_roc']:.4f}  VUS-PR={vus_metrics['vus_pr']:.4f}"
    )


def resolve_labels(npz_path: Path, data: np.lib.npyio.NpzFile, labels_key: str) -> np.ndarray:
    available_keys = set(data.files)
    if labels_key in available_keys:
        return np.asarray(data[labels_key], dtype=np.int32)

    experiment_dir = npz_path.parent
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise KeyError(
            f"Missing labels key '{labels_key}'. Available keys: {sorted(available_keys)}. "
            f"Also missing config.json for fallback: {config_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = str(config.get("dataset", "MSL"))
    data_root = str(config.get("data_root", "./dataset/anomaly_detect"))
    config_kwargs = {
        "dataset": dataset,
        "data_root": data_root,
        "seq_len": int(config.get("seq_len", 128)),
        "val_ratio": float(config.get("val_ratio", 0.15)),
        "val_gap": bool(config.get("val_gap", True)),
        "val_min_train_windows": int(config.get("val_min_train_windows", 50)),
        "val_split_mode": str(config.get("val_split_mode", "tail")),
        "train_stride": int(config.get("train_stride", 1)),
        "test_stride": int(config.get("test_stride", 1)),
        "memory_build_stride": int(config.get("memory_build_stride", 1)),
    }
    raw_bundle = load_raw_dataset_bundle(dataset, data_root, config=CoReMADConfig(**config_kwargs))
    labels = np.asarray(raw_bundle.test_labels, dtype=np.int32)
    print(f"[DiagEval] labels missing in npz, loaded from dataset={dataset} data_root={data_root}")
    return labels


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is not available in the current environment.")
    if not HAS_VUS:
        print("[DiagEval] warning: `vus` is not installed, Aff-F1 / Range-F1 / R-AUC / VUS metrics will be NaN")

    data = np.load(npz_path)
    labels = resolve_labels(npz_path, data, args.labels_key)
    print(f"[DiagEval] file={npz_path}")
    print(f"[DiagEval] labels_key={args.labels_key} positives={int(labels.sum())} total={len(labels)}")

    for name in args.score_keys:
        if name not in data.files:
            print(f"{name:30s} SKIP missing key")
            continue
        scores = sanitize_scores(data[name])
        print_metric_row(name, labels, scores)

    comp8 = load_score_array(data, "completion_scale8")
    comp32 = load_score_array(data, "completion_scale32")
    if comp8 is not None and comp32 is not None:
        comp_avg = 0.5 * comp8 + 0.5 * comp32
        print_metric_row("comp8+comp32 (simple avg)", labels, comp_avg)

    knn = load_score_array(data, "knn_distance")
    if comp8 is not None and comp32 is not None and knn is not None:
        comp_mem = (comp8 + comp32 + knn) / 3.0
        print_metric_row("comp8+comp32+knn (avg)", labels, comp_mem)


if __name__ == "__main__":
    main()
