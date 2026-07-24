from __future__ import annotations

import unittest

from scripts.experiments.match_budget import closest_axis


class BudgetAxisTests(unittest.TestCase):
    def test_axis_selection_uses_relative_error_then_smaller_value(self) -> None:
        selected = closest_axis(
            100.0,
            [
                {"parameters": 90.0, "d_z": 128},
                {"parameters": 110.0, "d_z": 64},
            ],
            "parameters",
            smaller_key="d_z",
        )
        self.assertEqual(selected["d_z"], 64)


if __name__ == "__main__":
    unittest.main()
