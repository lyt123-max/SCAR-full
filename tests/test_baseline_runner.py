from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.rebuttal.baselines.common import fixed_seed_inference
from scripts.rebuttal.baselines.run_baseline import FORMAL_METHODS, build_adapter_command
from scripts.rebuttal.baselines.run_memto_adapter import parse_args as parse_memto_args
from scripts.rebuttal.baselines.run_paano_adapter import parse_args as parse_paano_args


class BaselineRunnerTests(unittest.TestCase):
    def test_formal_methods_exclude_reported_catch(self) -> None:
        self.assertEqual(FORMAL_METHODS, ("PaAno", "MEMTO", "PUAD", "PGRF-Net"))

    def test_all_supported_methods_have_project_side_adapters(self) -> None:
        for method in ("PaAno", "MEMTO", "PUAD", "PGRF-Net", "CATCH"):
            command = build_adapter_command(
                method=method,
                baseline_python="/env/python",
                data_dir=Path("/data"),
                output_dir=Path("/out"),
                dataset="MSL",
                seed=42,
                device="cuda:0",
            )
            self.assertEqual(command[0], "/env/python")
            self.assertTrue(command[1].endswith("_adapter.py"))
            self.assertIn("--device", command)

    def test_rejects_nonformal_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed 42"):
            build_adapter_command(
                method="CATCH",
                baseline_python="python",
                data_dir=Path("data"),
                output_dir=Path("out"),
                dataset="MSL",
                seed=7,
                device="cuda:0",
            )

    def test_smoke_commands_reduce_every_training_loop(self) -> None:
        expected = {
            "PaAno": ("--num-iters", "1"),
            "MEMTO": ("--epochs", "1"),
            "PUAD": ("--epochs", "1"),
            "PGRF-Net": ("--epochs-stage1", "1"),
        }
        for method, pair in expected.items():
            command = build_adapter_command(
                method=method,
                baseline_python="/env/python",
                data_dir=Path("/data"),
                output_dir=Path("/output"),
                dataset="MSL",
                seed=42,
                device="cuda:0",
                smoke=True,
            )
            index = command.index(pair[0])
            self.assertEqual(command[index + 1], pair[1])

    def test_paano_adapter_matches_official_multivariate_hyperparameters(self) -> None:
        import sys
        from unittest.mock import patch

        argv = [
            "run_paano_adapter.py",
            "--data-dir",
            "/data",
            "--output-dir",
            "/out",
            "--dataset",
            "MSL",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_paano_args()
        self.assertEqual(args.patch_size, 96)
        self.assertEqual(args.num_iters, 100)
        self.assertEqual(args.batch_size, 512)

    def test_fixed_seed_inference_replays_stochastic_upstream_path(self) -> None:
        infer = fixed_seed_inference(lambda: np.random.random(8), seed=42)
        np.testing.assert_array_equal(infer(), infer())

    def test_memto_adapter_matches_official_memory_and_phase_limits(self) -> None:
        import sys
        from unittest.mock import patch

        argv = [
            "run_memto_adapter.py",
            "--data-dir",
            "/data",
            "--output-dir",
            "/out",
            "--dataset",
            "MSL",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_memto_args()
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.n_memory, 10)
        self.assertEqual(args.batch_size, 64)


if __name__ == "__main__":
    unittest.main()
