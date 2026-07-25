from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rebuttal"
    / "baselines"
    / "run_classical_adapter.py"
)
CLASSICAL = None
SKLEARN_ERROR: Exception | None = None
try:
    from sklearn.preprocessing import StandardScaler

    SPEC = importlib.util.spec_from_file_location("run_classical_adapter", SCRIPT)
    CLASSICAL = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(CLASSICAL)
except (ImportError, ValueError) as error:
    SKLEARN_ERROR = error


@unittest.skipIf(CLASSICAL is None, f"compatible scikit-learn unavailable: {SKLEARN_ERROR}")
class ClassicalBaselineTests(unittest.TestCase):
    def test_deterministic_subsample_preserves_endpoints(self) -> None:
        values = np.arange(20, dtype=np.float32)[:, None]
        first = CLASSICAL.deterministic_subsample(values, 5)
        second = CLASSICAL.deterministic_subsample(values, 5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(float(first[0, 0]), 0.0)
        self.assertEqual(float(first[-1, 0]), 19.0)

    def test_knn_and_lof_emit_finite_nonconstant_scores_and_scalability(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(size=(256, 4)).astype(np.float32)
        query = np.concatenate(
            [
                rng.normal(size=(96, 4)),
                rng.normal(loc=4.0, size=(32, 4)),
            ],
            axis=0,
        ).astype(np.float32)
        scaler = StandardScaler().fit(reference)
        reference_scaled = scaler.transform(reference).astype(np.float32)
        query_scaled = scaler.transform(query).astype(np.float32)
        for method, neighbors in (("KNN", 5), ("LOF", 20)):
            detector = CLASSICAL.fit_detector(
                method,
                reference_scaled,
                neighbors=neighbors,
                n_jobs=1,
            )
            scores = CLASSICAL.score_detector(
                method,
                detector,
                query_scaled,
                chunk_size=32,
            )
            self.assertEqual(scores.shape, (128,))
            self.assertTrue(np.isfinite(scores).all())
            self.assertGreater(float(np.ptp(scores)), 0.0)
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                payload = CLASSICAL.write_scalability(
                    output_dir=output,
                    method=method,
                    scaler=scaler,
                    reference_pool=reference_scaled,
                    test_scaled=query_scaled,
                    neighbors=neighbors,
                    n_jobs=1,
                    chunk_size=32,
                    query_cap=128,
                )
                self.assertEqual(len(payload["rows"]), 7)
                self.assertTrue((output / "scalability.csv").is_file())
                self.assertTrue((output / "scalability.json").is_file())
