from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.ablation.collect_results import build_records
from scripts.experiments.score_outputs import REQUIRED_REPORT_METRIC_KEYS


class AblationScoreCollectionTests(unittest.TestCase):
    def test_default_collection_discovers_nested_subscores(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            experiment = root / "msl_ablation_full"
            experiment.mkdir()
            metric = {key: 0.8 for key in REQUIRED_REPORT_METRIC_KEYS}
            (experiment / "test_metrics.json").write_text(
                json.dumps(
                    {
                        "scores": {
                            "selected": metric,
                            "raw_max": metric,
                            "zscore_mean": metric,
                            "cdf_mean": metric,
                            "cdf_max": metric,
                            "subscores": {
                                "completion_scale8": metric,
                                "knn_distance": metric,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                artifact_root=root,
                datasets=["MSL"],
                ablations=["full"],
                experiment_name_template="{dataset_lower}_ablation_{ablation}",
                score_keys=None,
                metric_keys=list(REQUIRED_REPORT_METRIC_KEYS),
            )
            records = build_records(args)
            self.assertEqual(
                {record["score_key"] for record in records},
                {
                    "selected",
                    "raw_max",
                    "zscore_mean",
                    "cdf_mean",
                    "cdf_max",
                    "completion_scale8",
                    "knn_distance",
                },
            )
            for record in records:
                for metric_key in REQUIRED_REPORT_METRIC_KEYS:
                    self.assertEqual(record[metric_key], 0.8)


if __name__ == "__main__":
    unittest.main()
