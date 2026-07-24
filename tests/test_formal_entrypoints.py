from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FormalEntrypointTests(unittest.TestCase):
    def test_main_scar_scripts_enable_audit_and_resource_monitor_explicitly(self) -> None:
        paths = [
            ROOT / "scripts" / "main" / name
            for name in (
                "train_msl.sh",
                "train_psm.sh",
                "train_smap.sh",
                "train_smd.sh",
                "train_swat.sh",
                "_train_detect_dataset.sh",
                "train_tep.sh",
            )
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn('MEMORY_AUDIT_MODE="${MEMORY_AUDIT_MODE:-full}"', text, path.name)
            self.assertIn('RESOURCE_MONITOR="${RESOURCE_MONITOR:-1}"', text, path.name)
            self.assertIn('--memory_audit_mode "${MEMORY_AUDIT_MODE}"', text, path.name)
            self.assertIn('--resource_monitor "${RESOURCE_MONITOR}"', text, path.name)

    def test_formal_tep_entrypoint_does_not_plot_by_default(self) -> None:
        path = ROOT / "scripts" / "main" / "train_tep_full.sh"
        text = path.read_text(encoding="utf-8")
        self.assertIn('TEP_ALLOW_PLOTS="${TEP_ALLOW_PLOTS:-0}"', text)
        self.assertIn('FORMAL_REBUTTAL="${FORMAL_REBUTTAL:-1}"', text)
        self.assertIn("formal rebuttal mode forbids plot generation", text)
        self.assertIn("formal rebuttal mode requires all test sequences", text)
        self.assertIn('if [[ "${TEP_ALLOW_PLOTS}" == "1" ]]', text)

    def test_sensitivity_never_recursively_deletes_experiment_directories(self) -> None:
        path = ROOT / "scripts" / "sensitivity" / "run_dataset_sensitivity.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn("--base-experiment-dir", text)
        self.assertNotIn("shutil.rmtree", text)


if __name__ == "__main__":
    unittest.main()
