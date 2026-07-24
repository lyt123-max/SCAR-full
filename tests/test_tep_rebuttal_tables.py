from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tep" / "generate_rebuttal_tables.py"
SPEC = importlib.util.spec_from_file_location("generate_rebuttal_tables", SCRIPT_PATH)
TABLES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TABLES)


def write_experiment(root: Path, name: str, protocol: str) -> Path:
    exp_dir = root / name
    mechanism_dir = exp_dir / "tep_mechanism"
    mechanism_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(
        json.dumps({"dataset": "TEP", "tep_protocol": protocol}),
        encoding="utf-8",
    )
    metrics = {key: 0.5 for key in TABLES.T9_METRICS}
    (mechanism_dir / "mechanism_metrics.json").write_text(
        json.dumps({"experiments": {name: metrics}}),
        encoding="utf-8",
    )
    return exp_dir


class TEPRebuttalTablesTest(unittest.TestCase):
    def test_selected_baseline_is_discovered_and_t9_has_two_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = write_experiment(
                root / "TEP-abalation",
                "tep_ablation_full_selected",
                "selected",
            )
            full = write_experiment(root / "artifacts", "tep_full", "full")
            previous_root = TABLES.REPO_ROOT
            TABLES.REPO_ROOT = root
            try:
                discovered = TABLES.discover_selected_experiment(full)
                rows = TABLES.build_t9(full, discovered)
            finally:
                TABLES.REPO_ROOT = previous_root
            self.assertEqual(discovered, selected)
            self.assertEqual([row["Protocol"] for row in rows], ["selected", "full"])


if __name__ == "__main__":
    unittest.main()
