from __future__ import annotations

import csv
import importlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "tsb_ad"
sys.path.insert(0, str(SCRIPT_DIR))

from collect_results import METRIC_KEYS, SCORE_KEYS, collect
from common import MANIFESTS, parse_file_name, read_manifest
from run_benchmark import _is_complete, _required_test_files, _select_shard, validate_protocol


def _import_data_module_without_torch():
    if "coremad.data" in sys.modules:
        return sys.modules["coremad.data"]

    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.from_numpy = lambda array: array
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    utils = types.ModuleType("torch.utils")
    utils_data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    class DataLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    utils_data.Dataset = Dataset
    utils_data.DataLoader = DataLoader
    utils.data = utils_data
    torch.utils = utils
    sys.modules["torch"] = torch
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.data"] = utils_data

    package = types.ModuleType("coremad")
    package.__path__ = [str(PROJECT_ROOT / "coremad")]
    sys.modules["coremad"] = package
    return importlib.import_module("coremad.data")


class TSBADManifestTests(unittest.TestCase):
    def test_official_manifest_counts_and_relationships(self):
        loaded = {(edition, split): read_manifest(edition, split) for edition, split in MANIFESTS}
        for key, (_, expected_count) in MANIFESTS.items():
            self.assertEqual(len(loaded[key]), expected_count)
        self.assertFalse(set(loaded[("M", "tuning")]) & set(loaded[("M", "eval")]))
        self.assertEqual(
            set(loaded[("M", "all")]),
            set(loaded[("M", "tuning")]) | set(loaded[("M", "eval")]),
        )
        self.assertEqual(
            set(loaded[("U", "all")]),
            set(loaded[("U", "tuning")]) | set(loaded[("U", "eval_full")]),
        )
        self.assertLessEqual(set(loaded[("U", "eval")]), set(loaded[("U", "eval_full")]))

    def test_filename_parser(self):
        metadata = parse_file_name("005_MSL_id_4_Sensor_tr_855_1st_2700.csv")
        self.assertEqual(metadata["file_index"], 5)
        self.assertEqual(metadata["source_dataset"], "MSL")
        self.assertEqual(metadata["series_id"], 4)
        self.assertEqual(metadata["train_length"], 855)
        self.assertEqual(metadata["first_anomaly_index"], 2700)


class TSBADLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_module = _import_data_module_without_torch()

    def test_loader_uses_prefix_and_scores_full_series(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file_name = "001_Toy_id_1_Sensor_tr_200_1st_240.csv"
            path = root / file_name
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["A", "B", "Label"])
                for index in range(300):
                    writer.writerow([index / 10.0, index % 7, int(index >= 240)])

            bundle = self.data_module._load_tsb_ad_raw_dataset_bundle(file_name, root)
            self.assertEqual(bundle.train.shape, (200, 2))
            self.assertEqual(bundle.test.shape, (300, 2))
            self.assertEqual(bundle.test_labels.shape, (300,))
            self.assertEqual(int(np.flatnonzero(bundle.test_labels)[0]), 240)
            self.assertEqual(bundle.dataset_metadata["n_channels"], 2)
            self.assertEqual(bundle.dataset_metadata["normalization_scope"], "training_prefix")
            self.assertEqual(bundle.dataset_metadata["training_prefix_anomaly_count"], 0)
            self.assertEqual(bundle.dataset_metadata["training_prefix_anomaly_ratio"], 0.0)
            self.assertFalse(bundle.dataset_metadata["training_prefix_contains_anomalies"])

    def test_loader_retains_contaminated_official_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file_name = "120_Toy_id_5_Environment_tr_200_1st_3.csv"
            path = root / file_name
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["A", "B", "Label"])
                for index in range(300):
                    writer.writerow([index / 10.0, index % 7, int(3 <= index < 13)])

            bundle = self.data_module._load_tsb_ad_raw_dataset_bundle(file_name, root)
            metadata = bundle.dataset_metadata
            self.assertEqual(bundle.train.shape, (200, 2))
            self.assertEqual(metadata["training_prefix_protocol"], "official_tr_N_boundary")
            self.assertEqual(metadata["training_prefix_anomaly_count"], 10)
            self.assertAlmostEqual(metadata["training_prefix_anomaly_ratio"], 0.05)
            self.assertTrue(metadata["training_prefix_contains_anomalies"])

    def test_normalizer_cannot_observe_post_prefix_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file_name = "001_Toy_id_1_Sensor_tr_200_1st_240.csv"
            path = root / file_name
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Data", "Label"])
                for index in range(300):
                    value = float(index) if index < 200 else 1_000_000.0
                    writer.writerow([value, int(index >= 240)])

            config = self.data_module.CoReMADConfig(
                dataset=file_name,
                data_root=str(root),
                data_format="tsb_ad",
                seq_len=32,
                patch_sizes=[8, 32],
                val_min_train_windows=2,
            )
            bundle = self.data_module.build_data_bundle(config)
            self.assertLess(float(bundle.normalizer.mean_[0]), 200.0)
            self.assertGreater(float(bundle.test[-1, 0]), 1000.0)


class TSBADOutputTests(unittest.TestCase):
    def test_manifest_shards_are_disjoint_and_complete(self):
        names = [f"series_{index}" for index in range(23)]
        shards = [_select_shard(names, 4, index) for index in range(4)]
        self.assertEqual(sum(map(len, shards)), len(names))
        self.assertEqual(set().union(*map(set, shards)), set(names))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(set(shards[left]) & set(shards[right]))
        with self.assertRaisesRegex(ValueError, "shard-count"):
            _select_shard(names, 0, 0)
        with self.assertRaisesRegex(ValueError, "shard-index"):
            _select_shard(names, 4, 4)

    def test_formal_runner_rejects_eval_full_and_nonformal_seed(self):
        with self.assertRaisesRegex(ValueError, "eval_full"):
            validate_protocol("formal-rebuttal", "U", "eval_full", [42])
        with self.assertRaisesRegex(ValueError, "seed 42"):
            validate_protocol("formal-rebuttal", "M", "eval", [43])
        validate_protocol("formal-rebuttal", "U", "eval", [42])

    def test_resume_requires_all_score_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            experiment_dir = Path(temp)
            complete_metrics = {
                "roc_auc": 0.5,
                "pr_auc": 0.4,
                "point_best_f1": 0.3,
                "pa_best_f1": 0.3,
                "aff_precision": 0.2,
                "aff_recall": 0.2,
                "aff_f1": 0.2,
                "vus_roc": 0.6,
                "vus_pr": 0.5,
            }
            score_metrics = {key: complete_metrics for key in SCORE_KEYS}
            metrics = {
                "dataset_metadata": {"total_length": 16},
                **score_metrics,
                "subscores": {
                    "completion_scale8": complete_metrics,
                    "completion_scale32": complete_metrics,
                    "knn_distance": complete_metrics,
                    "state_novelty": complete_metrics,
                },
            }
            (experiment_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            for name in _required_test_files([8, 32]):
                if name.endswith(".npy"):
                    np.save(experiment_dir / name, np.arange(16, dtype=np.float32))
            np.savez(experiment_dir / "test_diagnostic_scores.npz", labels=np.zeros(16))
            self.assertTrue(_is_complete(experiment_dir, "full", [8, 32]))
            metrics["subscores"]["knn_distance"] = {
                key: value
                for key, value in complete_metrics.items()
                if key != "aff_recall"
            }
            (experiment_dir / "test_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            self.assertFalse(_is_complete(experiment_dir, "full", [8, 32]))
            metrics["subscores"]["knn_distance"] = complete_metrics
            (experiment_dir / "test_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (experiment_dir / "test_scores_cdf_max.npy").unlink()
            self.assertFalse(_is_complete(experiment_dir, "full", [8, 32]))

    def test_collector_outputs_fusions_and_dynamic_subscores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            experiment_dir = root / "M" / "eval" / "seed_42" / "001_Toy_id_1_Sensor_tr_200_1st_240"
            experiment_dir.mkdir(parents=True)
            score_metrics = {
                key: {
                    "vus_pr": 0.1 + index,
                    "vus_roc": 0.2 + index,
                    "roc_auc": 0.3 + index,
                    "pr_auc": 0.4 + index,
                    "point_best_f1": 0.5 + index,
                    "pa_best_f1": 0.6 + index,
                    "aff_precision": 0.7 + index,
                    "aff_recall": 0.8 + index,
                    "aff_f1": 0.9 + index,
                }
                for index, key in enumerate(SCORE_KEYS)
            }
            payload = {
                "evaluation_protocol": "point_level",
                "point_coverage_ratio": 1.0,
                "dataset_metadata": {
                    "file_name": "001_Toy_id_1_Sensor_tr_200_1st_240.csv",
                    "source_dataset": "Toy",
                },
                **score_metrics,
                "subscores": {
                    "knn_distance": {
                        "vus_pr": 0.5,
                        "vus_roc": 0.6,
                        "roc_auc": 0.7,
                        "pr_auc": 0.8,
                        "point_best_f1": 0.6,
                        "pa_best_f1": 0.5,
                        "aff_precision": 0.4,
                        "aff_recall": 0.3,
                        "aff_f1": 0.35,
                    },
                    "state_novelty": {
                        "vus_pr": 0.4,
                        "vus_roc": 0.5,
                        "roc_auc": 0.6,
                        "pr_auc": 0.7,
                        "point_best_f1": 0.5,
                        "pa_best_f1": 0.4,
                        "aff_precision": 0.3,
                        "aff_recall": 0.2,
                        "aff_f1": 0.24,
                    },
                },
            }
            (experiment_dir / "test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            (experiment_dir / "run_record.json").write_text(
                json.dumps({"runtime_seconds": 12.5}),
                encoding="utf-8",
            )
            historical_dir = (
                root
                / "M"
                / "eval"
                / "seed_43"
                / "001_Toy_id_1_Sensor_tr_200_1st_240"
            )
            shutil.copytree(experiment_dir, historical_dir)

            summary = collect(root, "M", "eval")
            self.assertEqual(summary["long_rows"], 6)
            with (root / "M" / "eval" / "results_long.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            expected_scores = {*SCORE_KEYS, "knn_distance", "state_novelty"}
            self.assertEqual({row["score_key"] for row in rows}, expected_scores)
            primary = [row["score_key"] for row in rows if row["is_primary"] == "1"]
            self.assertEqual(primary, ["cdf_mean"])
            self.assertEqual({row["seed"] for row in rows}, {"42"})
            for file_name in (
                "results_per_series_wide.csv",
                "results_by_dataset_wide.csv",
                "results_official_average_wide.csv",
                "results_dataset_macro_average_wide.csv",
            ):
                path = root / "M" / "eval" / file_name
                self.assertTrue(path.is_file())
                header = path.read_text(encoding="utf-8-sig").splitlines()[0]
                for score_key in expected_scores:
                    for metric_key in METRIC_KEYS:
                        self.assertIn(f"{score_key}_{metric_key}", header)


if __name__ == "__main__":
    unittest.main()
