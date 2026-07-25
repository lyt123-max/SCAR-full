from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.experiments.reuse_stage_a import copy_stage_a_artifacts, parse_overrides


class ReuseStageATests(unittest.TestCase):
    def test_override_values_use_json_types(self) -> None:
        values = parse_overrides(
            ["clean_ratio=0.005", "top_M=50", "resource_monitor_enabled=true"]
        )
        self.assertEqual(values["clean_ratio"], 0.005)
        self.assertEqual(values["top_M"], 50)
        self.assertIs(values["resource_monitor_enabled"], True)

    def test_copy_refuses_to_replace_different_stage_a(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            (source / "stage_a.pt").write_bytes(b"source")
            (source / "stage_a_last.pt").write_bytes(b"last")
            (target / "stage_a.pt").write_bytes(b"different")
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                copy_stage_a_artifacts(source, target)

    def test_copy_is_idempotent_for_identical_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "stage_a.pt").write_bytes(b"source")
            (source / "stage_a_last.pt").write_bytes(b"last")
            copy_stage_a_artifacts(source, target)
            before = hashlib.sha256((target / "stage_a.pt").read_bytes()).hexdigest()
            copy_stage_a_artifacts(source, target)
            after = hashlib.sha256((target / "stage_a.pt").read_bytes()).hexdigest()
            self.assertEqual(before, after)

    def test_copy_accepts_completed_artifacts_without_last_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "stage_a.pt").write_bytes(b"best")

            copy_stage_a_artifacts(source, target)

            self.assertEqual((target / "stage_a.pt").read_bytes(), b"best")
            self.assertFalse((target / "stage_a_last.pt").exists())

    def test_copy_still_requires_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "stage_a_last.pt").write_bytes(b"last")

            with self.assertRaisesRegex(FileNotFoundError, "stage_a.pt"):
                copy_stage_a_artifacts(source, target)


if __name__ == "__main__":
    unittest.main()
