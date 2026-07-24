from __future__ import annotations

import unittest

import numpy as np

from scripts.rebuttal.muqn.baselines.evaluate_baseline_scores import (
    align_labels_and_scores,
)


class BaselineScoreAlignmentTests(unittest.TestCase):
    def test_test_only_scores_crop_full_labels(self) -> None:
        labels = np.arange(10)
        scores = np.arange(6)
        aligned_labels, aligned_scores = align_labels_and_scores(labels, scores, train_end=4)
        np.testing.assert_array_equal(aligned_labels, labels[4:])
        np.testing.assert_array_equal(aligned_scores, scores)

    def test_full_scores_and_labels_crop_both(self) -> None:
        labels = np.arange(10)
        scores = np.arange(10)
        aligned_labels, aligned_scores = align_labels_and_scores(labels, scores, train_end=4)
        np.testing.assert_array_equal(aligned_labels, labels[4:])
        np.testing.assert_array_equal(aligned_scores, scores[4:])

    def test_irreconcilable_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot align"):
            align_labels_and_scores(np.zeros(10), np.zeros(5), train_end=4)


if __name__ == "__main__":
    unittest.main()
