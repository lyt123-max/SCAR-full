from __future__ import annotations

import unittest

import numpy as np

from scripts.rebuttal.muqn.collect_coreset_sweep import candidate_recall


class CoresetCollectorTests(unittest.TestCase):
    def test_candidate_recall_uses_first_valid_oracle_neighbor(self) -> None:
        oracle = np.asarray([[10, 11], [20, 21], [-1, -1]])
        self.assertEqual(candidate_recall(oracle, {10, 21}), 0.5)


if __name__ == "__main__":
    unittest.main()
