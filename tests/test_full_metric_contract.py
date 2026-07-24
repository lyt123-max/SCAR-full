from __future__ import annotations

import importlib.util
import ast
import unittest
from pathlib import Path

import numpy as np

from scripts.ablation.collect_global_auc_table import build_long_rows, build_wide_rows
from scripts.experiments.collect_p1_tables import _is_report_metric_column
from scripts.experiments.score_outputs import REQUIRED_REPORT_METRIC_KEYS


def _load_standalone_evaluation():
    path = Path(__file__).resolve().parents[1] / "coremad" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("standalone_full_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullMetricContractTests(unittest.TestCase):
    def test_standalone_evaluator_returns_all_registered_metrics(self) -> None:
        evaluation = _load_standalone_evaluation()
        metrics = evaluation.binary_point_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.2, 0.8, 0.9]),
        )
        self.assertTrue(set(REQUIRED_REPORT_METRIC_KEYS).issubset(metrics))
        self.assertEqual(metrics["point_best_f1"], 1.0)
        self.assertEqual(metrics["pa_best_f1"], 1.0)

    def test_p1_macro_recognizes_every_registered_metric_column(self) -> None:
        for metric_key in REQUIRED_REPORT_METRIC_KEYS:
            self.assertTrue(_is_report_metric_column(metric_key))
            self.assertTrue(_is_report_metric_column(f"knn_distance_{metric_key}"))
        self.assertFalse(_is_report_metric_column("runtime_seconds"))

    def test_paper_table_defaults_include_all_registered_metrics(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "ablation"
            / "generate_paper_tables.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MAIN_METRIC_KEYS"
                for target in node.targets
            )
        )
        self.assertEqual(tuple(ast.literal_eval(assignment.value)), REQUIRED_REPORT_METRIC_KEYS)

    def test_global_ablation_tables_pair_every_subscore_with_nine_metrics(self) -> None:
        metric = {key: 0.5 for key in REQUIRED_REPORT_METRIC_KEYS}
        records = [
            {
                "dataset": "MSL",
                "ablation_key": "full",
                "ablation_label": "full",
                "experiment_name": "msl_ablation_full",
                "experiment_dir": "/tmp/experiment",
                "artifact_root": "/tmp",
                "evaluation_protocol": "point_level",
                "main_score_key": "cdf_mean",
                "main": metric,
                "subscores": {"knn_distance": metric},
            }
        ]
        wide = build_wide_rows(records, ["knn_distance"])[0]
        long = build_long_rows(records, ["knn_distance"])
        for metric_key in REQUIRED_REPORT_METRIC_KEYS:
            self.assertIn(f"main_{metric_key}", wide)
            self.assertIn(f"knn_distance_{metric_key}", wide)
            self.assertTrue(all(metric_key in row for row in long))


if __name__ == "__main__":
    unittest.main()
