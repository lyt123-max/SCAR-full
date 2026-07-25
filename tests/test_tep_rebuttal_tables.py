from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tep" / "generate_rebuttal_tables.py"
SPEC = importlib.util.spec_from_file_location("generate_rebuttal_tables", SCRIPT_PATH)
TABLES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TABLES)


def write_experiment(root: Path, name: str, protocol: str) -> Path:
    exp_dir = root / name
    mechanism_dir = exp_dir / "tep_mechanism"
    mechanism_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(
        json.dumps({"dataset": "TEP", "tep_protocol": protocol}),
        encoding="utf-8",
    )
    metrics = {key: 0.5 for key in TABLES.T9_METRICS}
    (mechanism_dir / "mechanism_metrics.json").write_text(
        json.dumps({"experiments": {name: metrics}}),
        encoding="utf-8",
    )
    return exp_dir


class TEPRebuttalTablesTest(unittest.TestCase):
    def test_selected_baseline_is_discovered_and_t9_has_two_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = write_experiment(
                root / "TEP-abalation",
                "tep_ablation_full_selected",
                "selected",
            )
            full = write_experiment(root / "artifacts", "tep_full", "full")
            previous_root = TABLES.REPO_ROOT
            TABLES.REPO_ROOT = root
            try:
                discovered = TABLES.discover_selected_experiment(full)
                rows = TABLES.build_t9(full, discovered)
            finally:
                TABLES.REPO_ROOT = previous_root
            self.assertEqual(discovered, selected)
            self.assertEqual([row["Protocol"] for row in rows], ["selected", "full"])

    def test_t10_selects_one_fixed_disturbance_case_per_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "tep_full"
            mechanism = experiment / "tep_mechanism"
            mechanism.mkdir(parents=True)
            np.savez(
                mechanism / "audit_normal_window_logs.npz",
                final=np.linspace(0.0, 1.0, 20),
                memory_distance=np.linspace(0.0, 2.0, 20),
                state_novelty=np.linspace(0.0, 3.0, 20),
                start=np.arange(20),
            )
            neighbor_ids = np.asarray(
                [[mode * 10, mode * 10 + 1] for mode in range(1, 7)],
                dtype=np.int64,
            )
            neighbor_modes = np.asarray(
                [[mode, mode] for mode in range(1, 7)],
                dtype=np.int32,
            )
            np.savez(
                mechanism / "fault_window_logs.npz",
                file_id=np.asarray([f"m{mode}d19" for mode in range(1, 7)]),
                mode_id=np.arange(1, 7, dtype=np.int32),
                fault_id=np.full(6, 10, dtype=np.int32),
                start=np.arange(6, dtype=np.int64) * 100,
                final=np.linspace(1.0, 2.0, 6),
                memory_distance=np.linspace(0.5, 1.0, 6),
                state_novelty=np.linspace(0.25, 0.75, 6),
                topk_neighbor_window_ids=neighbor_ids,
                topk_neighbor_mode_ids=neighbor_modes,
                topk_neighbor_distances=np.tile(
                    np.asarray([[0.1, 0.2]], dtype=np.float64),
                    (6, 1),
                ),
            )
            train_ids = neighbor_ids.reshape(-1)
            train_modes = np.repeat(np.arange(1, 7, dtype=np.int32), 2)
            np.savez(
                mechanism / "train_state_meta.npz",
                window_id=train_ids,
                file_id=np.asarray(
                    [f"m{mode}d0" for mode in range(1, 7) for _ in range(2)]
                ),
                start=np.arange(12, dtype=np.int64) * 128,
                mode_id=train_modes,
            )
            rows = TABLES.build_t10(experiment, fault_id=10, top_k=2)
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                [row["Query mode"] for row in rows],
                [1, 2, 3, 4, 5, 6],
            )
            self.assertTrue(
                all(row["Nearest normal mode"] == row["Query mode"] for row in rows)
            )
            self.assertTrue(all(row["Same-mode ratio@2"] == 1.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
