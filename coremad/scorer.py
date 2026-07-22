from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class _LinearCDFInterpolator:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
            raise ValueError("CDF interpolator expects non-empty 1D x/y arrays with matching lengths.")
        self.x = x
        self.y = y

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.x.size == 1:
            out = np.full_like(values, self.y[0], dtype=np.float64)
            out = np.where(values < self.x[0], 0.0, out)
            out = np.where(values > self.x[0], 1.0, out)
            return out
        return np.interp(values, self.x, self.y, left=0.0, right=1.0)


class CDFPITFusion:
    """
    Fit empirical CDFs on training diagnostics, calibrate each sub-score to [0, 1],
    then fuse with max, mean, or softmax-weighted averaging at test time.
    """

    DEFAULT_SCORE_NAMES = [
        "knn_distance",
        "state_novelty",
        "completion_scale8",
        "completion_scale32",
    ]

    def __init__(self, score_names: list[str] | None = None) -> None:
        self.score_names = list(score_names or self.DEFAULT_SCORE_NAMES)
        self.cdfs: dict[str, _LinearCDFInterpolator] = {}
        self.stats: dict[str, dict[str, float | int]] = {}
        self._fitted = False

    def fit(self, train_diagnostic_scores: dict[str, np.ndarray]) -> None:
        self.cdfs = {}
        self.stats = {}

        for name in self.score_names:
            if name not in train_diagnostic_scores:
                print(f"[CDFFusion] warning: {name} not found, skipping")
                continue

            scores = np.asarray(train_diagnostic_scores[name], dtype=np.float64).reshape(-1)
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                print(f"[CDFFusion] warning: {name} has no finite values, skipping")
                continue

            sorted_scores = np.sort(scores)
            unique_vals, unique_idx = np.unique(sorted_scores, return_index=True)
            cdf_vals = unique_idx.astype(np.float64) / float(len(sorted_scores))

            self.cdfs[name] = _LinearCDFInterpolator(unique_vals, cdf_vals)
            self.stats[name] = {
                "median": float(np.median(scores)),
                "iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "n_samples": int(len(scores)),
            }

        self._fitted = True
        print(f"[CDFFusion] fitted on {len(self.cdfs)} sub-scores: {list(self.cdfs.keys())}")

    def calibrate(
        self,
        diagnostic_scores: dict[str, np.ndarray],
        score_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Must call fit() or load() before calibrate().")

        target_names = list(score_names or self.score_names)
        calibrated: dict[str, np.ndarray] = {}
        for name in target_names:
            if name not in self.cdfs or name not in diagnostic_scores:
                continue
            raw = np.asarray(diagnostic_scores[name], dtype=np.float64)
            calibrated[name] = np.clip(self.cdfs[name](raw), 0.0, 1.0)
        return calibrated

    def fuse(
        self,
        diagnostic_scores: dict[str, np.ndarray],
        mode: str = "max",
        score_names: list[str] | None = None,
    ) -> np.ndarray:
        calibrated = self.calibrate(diagnostic_scores, score_names=score_names)
        if not calibrated:
            raise ValueError("No valid calibrated scores available for fusion.")

        stacked = np.stack(list(calibrated.values()), axis=0)
        if mode == "max":
            return np.max(stacked, axis=0)
        if mode == "mean":
            return np.mean(stacked, axis=0)
        if mode == "softmax":
            temperature = 0.1
            logits = stacked / temperature
            logits = logits - np.max(logits, axis=0, keepdims=True)
            weights = np.exp(logits)
            weights = weights / np.sum(weights, axis=0, keepdims=True)
            return np.sum(weights * stacked, axis=0)
        raise ValueError(f"Unknown fusion mode: {mode}")

    def save(self, path: str | Path) -> None:
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        cdf_data: dict[str, np.ndarray] = {}
        for name, func in self.cdfs.items():
            cdf_data[f"{name}_x"] = func.x
            cdf_data[f"{name}_y"] = func.y
        np.savez(base_path.with_suffix(".npz"), **cdf_data)

        meta = {"stats": self.stats, "score_names": list(self.cdfs.keys())}
        base_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[CDFFusion] saved to {base_path}")

    def load(self, path: str | Path) -> None:
        base_path = Path(path)
        data = np.load(base_path.with_suffix(".npz"))
        meta = json.loads(base_path.with_suffix(".json").read_text(encoding="utf-8"))

        self.stats = meta["stats"]
        self.score_names = list(meta.get("score_names", self.DEFAULT_SCORE_NAMES))
        self.cdfs = {}
        for name in self.score_names:
            self.cdfs[name] = _LinearCDFInterpolator(
                data[f"{name}_x"],
                data[f"{name}_y"],
            )
        self._fitted = True
        print(f"[CDFFusion] loaded from {base_path}, scores: {list(self.cdfs.keys())}")


class ZScoreMeanFusion:
    """
    Fit per-score mean/std on training diagnostics, standardize each sub-score at
    test time, then average the standardized scores.
    """

    DEFAULT_SCORE_NAMES = [
        "knn_distance",
        "state_novelty",
        "completion_scale8",
        "completion_scale32",
    ]

    def __init__(self, score_names: list[str] | None = None) -> None:
        self.score_names = list(score_names or self.DEFAULT_SCORE_NAMES)
        self.stats: dict[str, dict[str, float | int]] = {}
        self._fitted = False

    def fit(self, train_diagnostic_scores: dict[str, np.ndarray]) -> None:
        self.stats = {}

        for name in self.score_names:
            if name not in train_diagnostic_scores:
                print(f"[ZScoreFusion] warning: {name} not found, skipping")
                continue

            scores = np.asarray(train_diagnostic_scores[name], dtype=np.float64).reshape(-1)
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                print(f"[ZScoreFusion] warning: {name} has no finite values, skipping")
                continue

            mean = float(np.mean(scores))
            std = float(np.std(scores))
            if std < 1e-12:
                std = 1.0

            self.stats[name] = {
                "mean": mean,
                "std": std,
                "min": float(scores.min()),
                "max": float(scores.max()),
                "n_samples": int(scores.size),
            }

        self._fitted = True
        print(f"[ZScoreFusion] fitted on {len(self.stats)} sub-scores: {list(self.stats.keys())}")

    def transform(self, diagnostic_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Must call fit() or load() before transform().")

        transformed: dict[str, np.ndarray] = {}
        for name in self.score_names:
            if name not in self.stats or name not in diagnostic_scores:
                continue
            raw = np.asarray(diagnostic_scores[name], dtype=np.float64)
            mean = float(self.stats[name]["mean"])
            std = float(self.stats[name]["std"])
            transformed[name] = (raw - mean) / max(std, 1e-12)
        return transformed

    def fuse(self, diagnostic_scores: dict[str, np.ndarray]) -> np.ndarray:
        transformed = self.transform(diagnostic_scores)
        if not transformed:
            raise ValueError("No valid standardized scores available for fusion.")
        stacked = np.stack(list(transformed.values()), axis=0)
        return np.mean(stacked, axis=0)

    def save(self, path: str | Path) -> None:
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stats": self.stats, "score_names": list(self.stats.keys())}
        base_path.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[ZScoreFusion] saved to {base_path}")

    def load(self, path: str | Path) -> None:
        base_path = Path(path)
        payload = json.loads(base_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.stats = payload["stats"]
        self.score_names = list(payload.get("score_names", self.DEFAULT_SCORE_NAMES))
        self._fitted = True
        print(f"[ZScoreFusion] loaded from {base_path}, scores: {list(self.stats.keys())}")


def fuse_raw_max(
    diagnostic_scores: dict[str, np.ndarray],
    score_names: list[str],
) -> np.ndarray:
    raw_scores = [
        np.asarray(diagnostic_scores[name], dtype=np.float64)
        for name in score_names
        if name in diagnostic_scores
    ]
    if not raw_scores:
        raise ValueError("No valid raw scores available for max fusion.")
    return np.max(np.stack(raw_scores, axis=0), axis=0)
