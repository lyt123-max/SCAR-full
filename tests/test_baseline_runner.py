from __future__ import annotations

import unittest
from pathlib import Path

from scripts.rebuttal.baselines.run_baseline import FORMAL_METHODS, build_adapter_command
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


if __name__ == "__main__":
    unittest.main()
