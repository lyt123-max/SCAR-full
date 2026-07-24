from __future__ import annotations

import struct
import unittest
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "Multi-mode-Fault-Diagnosis-Datasets-with-TE-process"


def read_mat_v5_shape(path: Path) -> tuple[int, ...]:
    raw_file = path.read_bytes()
    data_type, compressed_size = struct.unpack("<II", raw_file[128:136])
    if data_type != 15:
        raise ValueError(f"{path} does not start with an miCOMPRESSED element.")
    raw = zlib.decompress(raw_file[136 : 136 + compressed_size])
    matrix_type, _ = struct.unpack("<II", raw[:8])
    if matrix_type != 14:
        raise ValueError(f"{path} does not contain an miMATRIX element.")
    position = 8
    _, array_flags_size = struct.unpack("<II", raw[position : position + 8])
    position += 8 + ((array_flags_size + 7) // 8) * 8
    _, dimensions_size = struct.unpack("<II", raw[position : position + 8])
    return struct.unpack(
        "<" + "i" * (dimensions_size // 4),
        raw[position + 8 : position + 8 + dimensions_size],
    )


@unittest.skipUnless(DATA_ROOT.is_dir(), "Local full MMFDD-TEP checkout is unavailable.")
class LocalTEPDatasetTest(unittest.TestCase):
    def test_full_checkout_shape_inventory(self) -> None:
        fault_shapes: list[tuple[int, int]] = []
        for mode_id in range(1, 7):
            mode_dir = DATA_ROOT / f"M{mode_id}"
            files = sorted(mode_dir.glob(f"m{mode_id}d*.mat"))
            self.assertEqual(len(files), 29)
            for path in files:
                shape = read_mat_v5_shape(path)
                self.assertEqual(shape[1], 81)
                if not path.stem.endswith("d00"):
                    fault_shapes.append((int(shape[0]), int(shape[1])))
        self.assertEqual(len(fault_shapes), 168)
        self.assertEqual(sum(rows == 7201 for rows, _ in fault_shapes), 158)
        self.assertEqual(sum(rows < 7201 for rows, _ in fault_shapes), 10)
        self.assertGreaterEqual(min(rows for rows, _ in fault_shapes), 128)

    def test_expected_full_protocol_window_counts(self) -> None:
        seq_len = 128
        val_steps = int(7201 * 0.15)
        gap_steps = seq_len
        train_steps_per_mode = 7201 - val_steps - gap_steps
        self.assertEqual(6 * train_steps_per_mode, 35958)
        self.assertEqual(6 * val_steps, 6480)
        self.assertEqual(6 * (train_steps_per_mode - seq_len + 1), 35196)
        self.assertEqual(6 * (val_steps - seq_len + 1), 5718)

        fault_windows = 0
        for mode_id in range(1, 7):
            for path in (DATA_ROOT / f"M{mode_id}").glob(f"m{mode_id}d*.mat"):
                if path.stem.endswith("d00"):
                    continue
                rows = read_mat_v5_shape(path)[0]
                fault_windows += rows - seq_len + 1
        self.assertEqual(fault_windows, 1_132_009)


if __name__ == "__main__":
    unittest.main()
