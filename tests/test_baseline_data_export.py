from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.rebuttal.muqn.baselines.prepare_baseline_data import (
    export_baseline_arrays,
)


class BaselineDataExportTests(unittest.TestCase):
    def test_exports_common_and_memto_npy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = export_baseline_arrays(
                train=np.arange(24, dtype=np.float32).reshape(8, 3),
                test=np.arange(18, dtype=np.float32).reshape(6, 3),
                test_labels=np.asarray([0, 1, 0, 0, 1, 0]),
                dataset="MSL",
                output_dir=root,
            )
            self.assertEqual(manifest["train_end"], 8)
            self.assertEqual(manifest["test_length"], 6)
            self.assertTrue((root / "common" / "labels.npy").is_file())
            self.assertTrue((root / "memto" / "MSL_train.npy").is_file())
            self.assertEqual(len(manifest["array_hashes"]["train"]), 64)

    def test_exports_memto_csv_layout_without_losing_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_baseline_arrays(
                train=np.ones((8, 3), dtype=np.float32),
                test=np.ones((6, 3), dtype=np.float32),
                test_labels=np.asarray([0, 1, 0, 0, 1, 0]),
                dataset="PSM",
                output_dir=root,
            )
            train = np.loadtxt(root / "memto" / "train.csv", delimiter=",", skiprows=1)
            labels = np.loadtxt(
                root / "memto" / "test_label.csv", delimiter=",", skiprows=1
            )
            self.assertEqual(train.shape, (8, 4))
            self.assertEqual(labels.shape, (6, 2))

    def test_rejects_misaligned_test_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "same length"):
                export_baseline_arrays(
                    train=np.ones((8, 2)),
                    test=np.ones((6, 2)),
                    test_labels=np.zeros(5),
                    dataset="SMD",
                    output_dir=Path(temporary),
                )


if __name__ == "__main__":
    unittest.main()
