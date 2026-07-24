from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.experiments.preflight import static_checks


class PreflightTests(unittest.TestCase):
    def test_static_preflight_passes_in_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checks = static_checks(Path(temporary))
        failed = [item for item in checks if not item["ok"]]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
