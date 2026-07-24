from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSB_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "tsb_ad"
PAANO_RUNNER = (
    PROJECT_ROOT / "scripts" / "rebuttal" / "muqn" / "baselines" / "run_paano.py"
)
CONTAMINATION_RUNNER = (
    PROJECT_ROOT / "scripts" / "rebuttal" / "muqn" / "run_contamination_sweep.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SingleSeedProtocolTests(unittest.TestCase):
    def test_tsb_ad_default_is_single_formal_seed(self) -> None:
        sys.path.insert(0, str(TSB_SCRIPT_DIR))
        try:
            module = _load_module(
                "scar_tsb_single_seed_test",
                TSB_SCRIPT_DIR / "run_benchmark.py",
            )
            with patch.object(
                sys,
                "argv",
                ["run_benchmark.py", "--edition", "M", "--dry-run"],
            ):
                args = module.parse_args()
        finally:
            sys.path.remove(str(TSB_SCRIPT_DIR))

        self.assertEqual(module.FORMAL_SEED, 42)
        self.assertEqual(args.seeds, [42])

    def test_paano_default_is_formal_seed(self) -> None:
        module = _load_module("scar_paano_single_seed_test", PAANO_RUNNER)
        with patch.object(
            sys,
            "argv",
            [
                "run_paano.py",
                "--data_dir",
                "input",
                "--output_dir",
                "output",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(module.FORMAL_SEED, 42)
        self.assertEqual(args.seed, 42)

    def test_contamination_folds_do_not_derive_additional_seeds(self) -> None:
        source = CONTAMINATION_RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("args.seed + fold_index", source)
        self.assertIn("seed=args.seed", source)


if __name__ == "__main__":
    unittest.main()
