from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TEP_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "tep"
if str(TEP_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TEP_SCRIPT_DIR))

from tep_common import iter_fault_log_shards, load_sequence_scores_with_meta, load_window_logs


class TEPLogCompatibilityTest(unittest.TestCase):
    def test_schema_v2_window_log_uses_official_idv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "fault_window_logs.npz",
                start=np.asarray([0, 1], dtype=np.int64),
                mode_id=np.asarray([1, 1], dtype=np.int32),
                fault_id=np.asarray([1, 1], dtype=np.int32),
                file_id=np.asarray(["m1d01", "m1d01"], dtype=object),
                sequence_name=np.asarray(["m1d01", "m1d01"], dtype=object),
                memory_distance=np.asarray([1.0, 2.0]),
            )
            logs = load_window_logs(root, "fault_window_logs")
            self.assertEqual(logs["file_fault_id"].tolist(), [1, 1])
            self.assertEqual(logs["fault_id"].tolist(), [28, 28])

    def test_schema_v2_sequence_json_uses_official_idv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "fault_sequence_scores_with_meta.json"
            path.write_text(
                json.dumps([{"sequence_name": "m4d27", "fault_id": 27}]),
                encoding="utf-8",
            )
            records = load_sequence_scores_with_meta(root, path.name)
            self.assertEqual(records[0]["file_fault_id"], 27)
            self.assertEqual(records[0]["fault_id"], 2)
            self.assertEqual(records[0]["mode_id"], 4)

    def test_schema_v3_metric_shards_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "fault_window_shards"
            shard_dir.mkdir()
            np.savez(
                shard_dir / "m6d28.npz",
                schema_version=np.asarray(3),
                signature=np.asarray("test"),
                sequence_name=np.asarray("m6d28"),
                start=np.asarray([0, 1], dtype=np.int32),
                mode_id=np.asarray([6, 6], dtype=np.int8),
                fault_id=np.asarray([1, 1], dtype=np.int8),
                file_fault_id=np.asarray([28, 28], dtype=np.int8),
                memory_distance=np.asarray([1.0, 2.0], dtype=np.float32),
            )
            (root / "fault_window_shards_manifest.json").write_text(
                json.dumps(
                    {
                        "metric_shard_dir": "fault_window_shards",
                        "sequences": [{"metric_file": "m6d28.npz"}],
                    }
                ),
                encoding="utf-8",
            )
            payloads = list(iter_fault_log_shards(root))
            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0]["fault_id"].tolist(), [1, 1])
            self.assertEqual(payloads[0]["file_fault_id"].tolist(), [28, 28])
            self.assertEqual(payloads[0]["sequence_name"].tolist(), ["m6d28", "m6d28"])


if __name__ == "__main__":
    unittest.main()
