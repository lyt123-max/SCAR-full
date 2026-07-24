from __future__ import annotations

import unittest

from scripts.rebuttal.muqn.score_retrieval_strategies import strategy_flags


class RetrievalStrategyScoringTests(unittest.TestCase):
    def test_strategy_flags_cover_all_formal_controls(self) -> None:
        self.assertEqual(strategy_flags("full"), (True, True))
        self.assertEqual(strategy_flags("no_state"), (False, True))
        self.assertEqual(strategy_flags("context_only"), (False, True))
        self.assertEqual(strategy_flags("no_context"), (True, False))
        self.assertEqual(strategy_flags("state_only"), (True, False))
        self.assertEqual(strategy_flags("global"), (False, False))

    def test_unknown_strategy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown retrieval strategy"):
            strategy_flags("mystery")


if __name__ == "__main__":
    unittest.main()
