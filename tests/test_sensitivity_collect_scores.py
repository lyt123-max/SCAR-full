from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.sensitivity.collect_results import build_records


class SensitivityScoreCollectionTests(unittest.TestCase):
    def test_default_collection_keeps_all_available_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / "msl_sensitivity" / "summary" / "msl_sensitivity_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "dataset_display": "MSL",
                        "study_name": "msl_sensitivity",
                        "selected_params": ["clean_ratio"],
                        "specs": {"clean_ratio": {}},
                        "rows": [
                            {
                                "param": "clean_ratio",
                                "value": 0.02,
                                "score_metrics": {
                                    "raw_max": {"roc_auc": 0.8},
                                    "cdf_mean": {"roc_auc": 0.9},
                                    "completion_scale8": {"roc_auc": 0.7},
                                    "state_novelty": {"roc_auc": 0.6},
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                artifact_root=root,
                datasets=["MSL"],
                study_name_template="{dataset_lower}_sensitivity",
                score_keys=None,
                metric_keys=["roc_auc"],
            )
            records, missing = build_records(args)
            self.assertFalse(missing)
            self.assertEqual(
                {record["score_key"] for record in records},
                {"raw_max", "cdf_mean", "completion_scale8", "state_novelty"},
            )


if __name__ == "__main__":
    unittest.main()
