from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from coremad import temporal_metrics


class TemporalMetricCompatibilityTests(unittest.TestCase):
    def test_tsb_ad_1p5_key_schema_and_signature(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0], dtype=np.int32)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3], dtype=np.float64)
        predictions = np.asarray([0, 0, 1, 1, 0], dtype=np.int32)

        def current_api(score, label, slidingWindow=100, pred=None, version="opt", thre=250):
            self.assertEqual(slidingWindow, 2)
            np.testing.assert_array_equal(pred, predictions)
            return {
                "Affiliation-F": 0.7,
                "VUS-ROC": 0.8,
                "VUS-PR": 0.6,
            }

        with (
            patch.object(temporal_metrics, "temporal_get_metrics", current_api),
            patch.object(temporal_metrics, "HAS_TEMPORAL_METRICS", True),
            patch.object(
                temporal_metrics,
                "_affiliation_precision_recall",
                return_value=(0.75, 0.5),
            ),
        ):
            result = temporal_metrics.compute_temporal_metrics(
                labels,
                scores,
                sliding_window=2,
                predictions=predictions,
            )
        self.assertAlmostEqual(result["aff_precision"], 0.75)
        self.assertAlmostEqual(result["aff_recall"], 0.5)
        self.assertAlmostEqual(result["aff_f1"], 0.6)
        self.assertAlmostEqual(result["vus_roc"], 0.8)
        self.assertAlmostEqual(result["vus_pr"], 0.6)

    def test_legacy_key_schema_and_signature(self) -> None:
        labels = np.asarray([0, 1, 0], dtype=np.int32)
        scores = np.asarray([0.1, 0.9, 0.2], dtype=np.float64)

        def legacy_api(score, label, metric="all", slidingWindow=100):
            self.assertEqual(metric, "all")
            return {
                "Affiliation_Precision": 0.8,
                "Affiliation_Recall": 0.5,
                "VUS_ROC": 0.9,
                "VUS_PR": 0.7,
                "R_AUC_ROC": 0.85,
                "R_AUC_PR": 0.65,
            }

        with (
            patch.object(temporal_metrics, "temporal_get_metrics", legacy_api),
            patch.object(temporal_metrics, "HAS_TEMPORAL_METRICS", True),
        ):
            result = temporal_metrics.compute_temporal_metrics(
                labels,
                scores,
                sliding_window=3,
            )
        self.assertAlmostEqual(result["aff_f1"], 2 * 0.8 * 0.5 / 1.3)
        self.assertAlmostEqual(result["vus_roc"], 0.9)
        self.assertAlmostEqual(result["r_auc_pr"], 0.65)


if __name__ == "__main__":
    unittest.main()
