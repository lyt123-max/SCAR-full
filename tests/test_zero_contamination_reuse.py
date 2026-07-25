from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.experiments.score_outputs import resolve_score_files
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

    def test_materializes_diagnostic_only_subscore(self) -> None:
        with TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            np.save(experiment_dir / "test_scores_selected.npy", np.array([0.1, 0.9]))
            np.savez(
                experiment_dir / "test_diagnostic_scores.npz",
                labels=np.array([0, 1]),
                soft_support_score=np.array([0.2, 0.8]),
            )
            resolved = resolve_score_files(
                experiment_dir,
                {"score_files": {"selected": "test_scores_selected.npy"}},
                ["selected", "soft_support_score"],
            )

            self.assertEqual(
                resolved["soft_support_score"],
                "test_scores_soft_support_score.npy",
            )
            np.testing.assert_allclose(
                np.load(experiment_dir / resolved["soft_support_score"]),
                np.array([0.2, 0.8]),
            )

    def test_missing_diagnostic_subscore_fails_clearly(self) -> None:
        with TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            np.savez(
                experiment_dir / "test_diagnostic_scores.npz",
                labels=np.array([0, 1]),
            )
            with self.assertRaisesRegex(KeyError, "soft_support_score"):
                resolve_score_files(
                    experiment_dir,
                    {"score_files": {"selected": "test_scores_selected.npy"}},
                    ["selected", "soft_support_score"],
                )


if __name__ == "__main__":
    unittest.main()
