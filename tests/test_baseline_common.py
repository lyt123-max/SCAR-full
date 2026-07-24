from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.rebuttal.baselines.common import (
    measure_inference,
    overlap_average,
    save_standard_outputs,
)


class BaselineCommonTests(unittest.TestCase):
    def test_overlap_average_restores_point_length(self) -> None:
        result = overlap_average(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            series_length=4,
        )
        np.testing.assert_allclose(result, [1.0, 3.0, 4.0, 6.0])

    def test_formal_timing_calls_once_plus_three(self) -> None:
        calls = []

        def infer() -> np.ndarray:
            calls.append(1)
            return np.asarray([0.0, 1.0])

        scores, timing = measure_inference(infer)
        self.assertEqual(len(calls), 4)
        self.assertEqual(timing["timed_runs"], 3)
        np.testing.assert_array_equal(scores, [0.0, 1.0])

    def test_standard_outputs_require_exact_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                save_standard_outputs(
                    output_dir=Path(temporary),
                    method="x",
                    dataset="MSL",
                    seed=42,
                    scores=np.zeros(2),
                    labels=np.zeros(3),
                    timing={},
                    implementation={},
                )


if __name__ == "__main__":
    unittest.main()
