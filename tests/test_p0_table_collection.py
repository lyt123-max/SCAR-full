from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.experiments.collect_p0_tables import collect


class P0TableCollectionTests(unittest.TestCase):
    def test_non_strict_collection_reports_missing_inputs_and_writes_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "tables"
            summary = collect(root / "artifacts", output, strict=False)
            self.assertTrue(summary["missing"])
            self.assertEqual(summary["formal_output_mode"], "tables_and_text_only")
            for name in (
                "table_p0_main5.csv",
                "table_p0_baselines.csv",
                "table_p0_efficiency.csv",
                "table_p0_catch.csv",
                "table_p0_e9.csv",
                "table_p0_e10.csv",
                "table_p0_tsb_series.csv",
                "table_p0_tsb_summary.csv",
                "p0_collection_summary.json",
                "p0_collection_summary.txt",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_strict_collection_rejects_incomplete_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                collect(root / "artifacts", root / "tables", strict=True)


if __name__ == "__main__":
    unittest.main()
