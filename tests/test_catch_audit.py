from __future__ import annotations

import unittest

from scripts.experiments.audit_catch_datasets import ASD, REAL, SYNTHETIC_PREFIXES


class CatchAuditTests(unittest.TestCase):
    def test_protocol_group_counts_are_fixed(self) -> None:
        self.assertEqual(len(ASD), 12)
        self.assertEqual(len(REAL), 6)
        self.assertEqual(len(SYNTHETIC_PREFIXES), 6)


if __name__ == "__main__":
    unittest.main()
