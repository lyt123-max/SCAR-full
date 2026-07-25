from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.experiments.preflight import static_checks


class PreflightTests(unittest.TestCase):
    def test_static_preflight_passes_in_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checks = static_checks(Path(temporary), require_upstreams=False)
        failed = [item for item in checks if not item["ok"]]
        self.assertEqual(failed, [])

    def test_scar_environment_satisfies_tsb_ad_numpy_constraint(self) -> None:
        environment = (
            Path(__file__).resolve().parents[1]
            / "environments"
            / "scar-paano-cu126.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("numpy=1.26.4", environment)
        self.assertNotIn("numpy=2.3.2", environment)
        self.assertIn("statsmodels=0.14.5", environment)
        self.assertIn("TSB-AD==1.5", environment)


if __name__ == "__main__":
    unittest.main()
