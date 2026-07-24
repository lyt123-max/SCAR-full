from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coremad.data import (
    TEP_FAULT_FILES,
    TEP_NORMAL_FILES,
    _compute_start_indices_from_segment_ranges,
    _validate_tep_fault_length,
    resolve_tep_files,
    tep_protocol_files,
)


class TEPProtocolTest(unittest.TestCase):
    def test_selected_protocol_is_backward_compatible(self) -> None:
        normal_files, fault_files = tep_protocol_files("selected")
        self.assertEqual(normal_files, TEP_NORMAL_FILES)
        self.assertEqual(fault_files, TEP_FAULT_FILES)
        self.assertEqual(len(normal_files), 3)
        self.assertEqual(len(fault_files), 24)

    def test_full_protocol_covers_six_modes_and_all_faults(self) -> None:
        normal_files, fault_files = tep_protocol_files("full")
        self.assertEqual(len(normal_files), 6)
        self.assertEqual(len(fault_files), 168)
        self.assertEqual(normal_files[0], "m1d00.mat")
        self.assertEqual(normal_files[-1], "m6d00.mat")
        self.assertIn("m1d01.mat", fault_files)
        self.assertIn("m6d28.mat", fault_files)

    def test_selected_flat_directory_resolution(self) -> None:
        normal_files, fault_files = tep_protocol_files("selected")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for file_name in (*normal_files, *fault_files):
                (root / file_name).touch()
            paths, actual_normal, actual_fault = resolve_tep_files(root, "selected")
            self.assertEqual(actual_normal, normal_files)
            self.assertEqual(actual_fault, fault_files)
            self.assertEqual(len(paths), 27)
            self.assertEqual(paths["m1d00.mat"], root / "m1d00.mat")

    def test_full_nested_directory_resolution(self) -> None:
        normal_files, fault_files = tep_protocol_files("full")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for file_name in (*normal_files, *fault_files):
                mode = file_name[1 : file_name.index("d")]
                mode_dir = root / f"M{mode}"
                mode_dir.mkdir(exist_ok=True)
                (mode_dir / file_name).touch()
            paths, _, _ = resolve_tep_files(root, "full")
            self.assertEqual(len(paths), 174)
            self.assertEqual(paths["m6d28.mat"], root / "M6" / "m6d28.mat")

    def test_missing_full_file_is_rejected(self) -> None:
        normal_files, fault_files = tep_protocol_files("full")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for file_name in (*normal_files, *fault_files[:-1]):
                mode = file_name[1 : file_name.index("d")]
                mode_dir = root / f"M{mode}"
                mode_dir.mkdir(exist_ok=True)
                (mode_dir / file_name).touch()
            with self.assertRaises(FileNotFoundError):
                resolve_tep_files(root, "full")

    def test_fault_shorter_than_window_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shorter than seq_len=128"):
            _validate_tep_fault_length(127, 128, "m2d03.mat")
        _validate_tep_fault_length(128, 128, "m2d03.mat")

    def test_segment_windows_do_not_cross_mode_boundaries(self) -> None:
        starts = _compute_start_indices_from_segment_ranges(
            seq_len=128,
            stride=1,
            segment_ranges=[(0, 5993), (5993, 11986)],
        )
        self.assertEqual(len(starts), 2 * 5866)
        self.assertFalse(any(5866 <= int(start) < 5993 for start in starts))


if __name__ == "__main__":
    unittest.main()
