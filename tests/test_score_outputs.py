from __future__ import annotations

import unittest

from scripts.experiments.score_outputs import (
    CORE_FUSION_SCORE_KEYS,
    REQUIRED_REPORT_METRIC_KEYS,
    flatten_score_metrics,
    score_metric_groups,
)


def metric(value: float) -> dict[str, float]:
    result = {key: value for key in REQUIRED_REPORT_METRIC_KEYS}
    result["best_f1"] = value
    return result


class ScoreOutputSchemaTests(unittest.TestCase):
    def test_trainer_schema_flattens_core_fusions_and_subscores(self) -> None:
        payload = {
            "selected_score_key": "cdf_mean",
            "selected": metric(0.9),
            **{key: metric(0.5 + index / 10) for index, key in enumerate(CORE_FUSION_SCORE_KEYS)},
            "subscores": {
                "completion_scale8": metric(0.4),
                "knn_distance": metric(0.3),
                "state_novelty": metric(0.2),
            },
        }
        flat = flatten_score_metrics(payload)
        self.assertEqual(flat["selected_score_key"], "cdf_mean")
        self.assertIn("raw_max_roc_auc", flat)
        self.assertIn("cdf_mean_pr_auc", flat)
        self.assertIn("completion_scale8_roc_auc", flat)
        self.assertIn("knn_distance_pr_auc", flat)
        for score_key in (*CORE_FUSION_SCORE_KEYS, "completion_scale8"):
            for metric_key in REQUIRED_REPORT_METRIC_KEYS:
                self.assertIn(f"{score_key}_{metric_key}", flat)

    def test_strategy_schema_detects_peer_subscores(self) -> None:
        scores = {
            "selected": metric(0.9),
            **{key: metric(0.6) for key in CORE_FUSION_SCORE_KEYS},
            "completion_scale32": metric(0.5),
            "state_novelty": metric(0.4),
        }
        groups = score_metric_groups({"selected_score_key": "cdf_mean", "scores": scores})
        self.assertIn("completion_scale32", groups)
        self.assertIn("state_novelty", groups)

    def test_missing_core_fusion_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "cdf_max"):
            score_metric_groups(
                {
                    key: metric(0.5)
                    for key in CORE_FUSION_SCORE_KEYS
                    if key != "cdf_max"
                }
            )

    def test_missing_report_metric_is_rejected(self) -> None:
        incomplete = metric(0.5)
        incomplete.pop("aff_recall")
        with self.assertRaisesRegex(KeyError, "aff_recall"):
            score_metric_groups(
                {
                    **{key: metric(0.5) for key in CORE_FUSION_SCORE_KEYS},
                    "subscores": {"knn_distance": incomplete},
                },
                require_report_metrics=True,
            )


if __name__ == "__main__":
    unittest.main()
