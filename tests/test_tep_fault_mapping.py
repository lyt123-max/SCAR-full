from __future__ import annotations

import unittest

from coremad.data import TEP_NUM_DISTURBANCES, tep_file_fault_to_idv, tep_idv_to_file_fault


class TEPFaultMappingTest(unittest.TestCase):
    def test_normal_file_maps_to_normal_identifier(self) -> None:
        self.assertEqual(tep_file_fault_to_idv(0), 0)
        self.assertEqual(tep_idv_to_file_fault(0), 0)

    def test_official_reverse_mapping(self) -> None:
        self.assertEqual(tep_file_fault_to_idv(1), 28)
        self.assertEqual(tep_file_fault_to_idv(28), 1)
        self.assertEqual(tep_idv_to_file_fault(28), 1)
        self.assertEqual(tep_idv_to_file_fault(1), 28)

    def test_all_faults_round_trip(self) -> None:
        for file_fault_id in range(1, TEP_NUM_DISTURBANCES + 1):
            idv = tep_file_fault_to_idv(file_fault_id)
            self.assertEqual(tep_idv_to_file_fault(idv), file_fault_id)

    def test_out_of_range_values_are_rejected(self) -> None:
        for invalid in (-1, TEP_NUM_DISTURBANCES + 1):
            with self.assertRaises(ValueError):
                tep_file_fault_to_idv(invalid)
            with self.assertRaises(ValueError):
                tep_idv_to_file_fault(invalid)


if __name__ == "__main__":
    unittest.main()
