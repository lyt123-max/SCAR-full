from __future__ import annotations

import unittest

import numpy as np

from scripts.rebuttal.muqn.reevaluate_zero_contamination import masked_binary_metrics


class ZeroContaminationReuseTests(unittest.TestCase):
    def test_masked_metrics_exclude_injection_fold_points(self) -> None:
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        scores = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7, 0.6, 0.1])
        result, mask = masked_binary_metrics(labels, scores, [(1, 3)])
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        self.assertEqual(result["n_evaluation_points"], 6)

    def test_score_length_must_match_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            masked_binary_metrics(np.zeros(5), np.zeros(4), [])


if __name__ == "__main__":
    unittest.main()
