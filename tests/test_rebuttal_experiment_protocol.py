from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.experiments.manifest import (
    RunSpec,
    artifact_is_complete,
    artifact_is_reusable,
    stable_run_id,
)
from scripts.experiments.runner import (
    execute_task,
    execute_tasks,
    execution_is_reusable,
    freeze_environment,
    summarize_tasks,
    write_plan,
)
from scripts.experiments.rebuttal import AC_CORE_GROUPS, _apply_data_root_map, select_tasks
from scripts.experiments.budget import select_budget_candidate
from scripts.experiments.protocol import (
    CORESET_KEEP_RATIOS,
    FORMAL_DATASET_PROFILES,
    PURIFICATION_RATIOS,
    build_lite_tasks,
    build_p0_tasks,
    build_p1_tasks,
    validate_formal_request,
)


class FormalProtocolTests(unittest.TestCase):
    def test_main_dataset_profiles_match_frozen_main_runs(self) -> None:
        self.assertEqual(set(FORMAL_DATASET_PROFILES), {"MSL", "PSM", "SMAP", "SMD", "SWAT"})
        self.assertEqual(FORMAL_DATASET_PROFILES["MSL"]["batch_size"], 128)
        self.assertEqual(FORMAL_DATASET_PROFILES["PSM"]["val_split_mode"], "interleaved")
        self.assertEqual(FORMAL_DATASET_PROFILES["PSM"]["lr"], 5e-4)
        self.assertEqual(FORMAL_DATASET_PROFILES["SMD"]["memory_build_stride"], 4)
        self.assertEqual(FORMAL_DATASET_PROFILES["SMD"]["early_stop_patience"], 6)
        self.assertEqual(FORMAL_DATASET_PROFILES["SWAT"]["memory_build_stride"], 2)
        self.assertEqual(FORMAL_DATASET_PROFILES["SWAT"]["completion_dropout"], 0.2)
        for profile in FORMAL_DATASET_PROFILES.values():
            self.assertEqual(profile["seed"], 42)
            self.assertEqual(profile["memory_audit_mode"], "full")
            self.assertEqual(profile["resource_monitor"], 1)
            self.assertEqual(profile["export_visualizations"], 0)

    def test_rebuttal_sweep_grids_are_frozen(self) -> None:
        self.assertEqual(PURIFICATION_RATIOS, (0.0, 0.005, 0.01, 0.02, 0.05, 0.10))
        self.assertEqual(CORESET_KEEP_RATIOS, (1.0, 0.5, 0.25, 0.10, 0.05))

    def test_p0_recomputation_counts_match_registered_protocol(self) -> None:
        tasks = build_p0_tasks(Path("/artifacts"), python_exe="python")
        full = [task for task in tasks if task.compute_kind == "full_model_fit"]
        stage_b = [task for task in tasks if task.compute_kind == "stage_b_test"]
        self.assertEqual(len(full), 659)
        self.assertEqual(len(stage_b), 180)
        self.assertEqual(len({task.run_id for task in tasks}), len(tasks))

    def test_formal_baseline_queue_does_not_rerun_catch(self) -> None:
        tasks = build_p0_tasks(Path("/artifacts"), python_exe="python")
        baselines = [task for task in tasks if task.group == "baselines"]
        self.assertEqual(len(baselines), 25)
        self.assertEqual(
            {task.method for task in baselines},
            {"PaAno", "PUAD", "PGRF-Net", "KNN", "LOF"},
        )
        self.assertFalse(any(task.method == "CATCH" for task in baselines))

    def test_formal_scar_tasks_parallelize_score_metrics(self) -> None:
        tasks = build_p0_tasks(Path("/artifacts"), python_exe="python")
        score_tasks = [
            task
            for task in tasks
            if task.method == "SCAR" and task.stage in {"full", "stage_b_test"}
        ]
        self.assertTrue(score_tasks)
        self.assertTrue(
            all(
                task.metadata.get("environment", {}).get("SCAR_METRIC_WORKERS")
                == "8"
                for task in score_tasks
            )
        )

    def test_e10_uses_one_frozen_protocol_per_dataset(self) -> None:
        tasks = build_p0_tasks(Path("/artifacts"), python_exe="python")
        protocols = [task for task in tasks if task.group == "e10_protocol"]
        e10_runs = [task for task in tasks if task.group == "e10"]
        self.assertEqual(len(protocols), 5)
        self.assertEqual(len(e10_runs), 135)
        protocol_ids = {task.dataset: task.run_id for task in protocols}
        for task in e10_runs:
            self.assertIn("--fold_manifest", task.command)
            self.assertIn(protocol_ids[task.dataset], task.dependencies)

    def test_tep_declares_sequence_level_outputs(self) -> None:
        tasks = build_p0_tasks(Path("/artifacts"), python_exe="python")
        tep = next(task for task in tasks if task.group == "tep")
        self.assertIn("test_sequence_scores_selected.npy", tep.required_artifacts)
        for score_name in (
            "raw_max",
            "zscore_mean",
            "cdf_mean",
            "cdf_max",
            "completion_scale8",
            "completion_scale32",
            "knn_distance",
            "state_novelty",
        ):
            self.assertIn(
                f"test_sequence_scores_{score_name}.npy", tep.required_artifacts
            )
        self.assertNotIn("test_scores_selected.npy", tep.required_artifacts)
        self.assertEqual(tep.metadata["environment"]["FORMAL_REBUTTAL"], "1")
        self.assertEqual(tep.metadata["environment"]["MAX_TEST_SEQUENCES"], "0")
        self.assertEqual(
            Path(tep.metadata["environment"]["ARTIFACT_ROOT"]),
            tep.artifact_dir.parent,
        )

    def test_p0_collector_declares_strategy_and_tep_score_tables(self) -> None:
        tasks = build_p0_tasks(Path("/artifacts"), python_exe="python")
        collector = next(task for task in tasks if task.group == "p0_tables")
        self.assertIn(
            "table_p0_retrieval_strategies.csv", collector.required_artifacts
        )
        self.assertIn("table_p0_tep_scores.csv", collector.required_artifacts)
        self.assertIn(
            "table_p0_classical_scalability.csv", collector.required_artifacts
        )

    def test_formal_protocol_rejects_nonformal_seed_and_u_eval_full(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed 42"):
            validate_formal_request(seed=43)
        with self.assertRaisesRegex(ValueError, "eval_full"):
            validate_formal_request(seed=42, tsb_edition="U", tsb_split="eval_full")
        validate_formal_request(seed=42, tsb_edition="U", tsb_split="eval")

    def test_p1_reuses_existing_default_anchors(self) -> None:
        tasks = build_p1_tasks(Path("/artifacts"), python_exe="python")
        e29_fits = [
            task
            for task in tasks
            if task.group == "e29" and task.compute_kind == "full_model_fit"
        ]
        e30_fits = [
            task
            for task in tasks
            if task.group == "e30" and task.compute_kind == "full_model_fit"
        ]
        self.assertEqual(len(e29_fits), 15)
        self.assertEqual(len(e30_fits), 20)
        self.assertEqual({task.dataset for task in e30_fits}, set(FORMAL_DATASET_PROFILES))
        self.assertNotIn(128, {task.config["seq_len"] for task in e29_fits})
        self.assertTrue(all(len(task.config["patch_sizes"]) == 1 for task in e30_fits))
        for task in e30_fits:
            patch_size = task.config["patch_sizes"][0]
            completion_files = {
                name
                for name in task.required_artifacts
                if name.startswith("test_scores_completion_scale")
            }
            self.assertEqual(
                completion_files,
                {f"test_scores_completion_scale{patch_size}.npy"},
            )
        for task in (task for task in tasks if task.group == "e31" and task.stage == "analysis"):
            for score_name in (
                "raw_max",
                "zscore_mean",
                "cdf_mean",
                "cdf_max",
                "completion_scale8",
                "completion_scale32",
                "knn_distance",
                "state_novelty",
            ):
                self.assertIn(f"scores_{score_name}.npy", task.required_artifacts)

    def test_lite_protocol_is_five_dataset_self_contained_and_nonduplicated(self) -> None:
        tasks = build_lite_tasks(Path("/artifacts"), python_exe="python")
        counts = {
            kind: sum(task.compute_kind == kind for task in tasks)
            for kind in ("full_model_fit", "stage_b_test", "analysis")
        }
        self.assertEqual(len(tasks), 158)
        self.assertEqual(
            counts,
            {"full_model_fit": 30, "stage_b_test": 70, "analysis": 58},
        )
        self.assertEqual(len({task.run_id for task in tasks}), len(tasks))
        self.assertEqual(
            {
                task.dataset
                for task in tasks
                if task.group in {"lite_e29", "lite_e30", "lite_e31"}
            },
            set(FORMAL_DATASET_PROFILES),
        )

    def test_lite_e10_keeps_endpoints_and_three_way_ten_percent_contrast(self) -> None:
        tasks = build_lite_tasks(Path("/artifacts"), python_exe="python")
        e10 = [task for task in tasks if task.group == "lite_e10"]
        zero = [task for task in tasks if task.group == "lite_e10_zero"]
        self.assertEqual(len(e10), 50)
        self.assertEqual(len(zero), 20)
        for dataset in FORMAL_DATASET_PROFILES:
            rows = [task for task in e10 if task.dataset == dataset]
            self.assertEqual(len(rows), 10)
            for fold in (0, 1):
                fold_rows = [task for task in rows if task.config["fold"] == fold]
                observed = {
                    (
                        float(task.config["contamination_ratio"]),
                        float(task.config["clean_ratio"]),
                    )
                    for task in fold_rows
                }
                self.assertEqual(
                    observed,
                    {
                        (0.01, 0.02),
                        (0.05, 0.02),
                        (0.10, 0.0),
                        (0.10, 0.02),
                        (0.10, 0.10),
                    },
                )

    def test_lite_scope_is_selectable_without_p0_or_p1_tasks(self) -> None:
        tasks = select_tasks(
            scope="lite",
            groups=None,
            artifact_root=Path("/artifacts"),
            python_exe="python",
        )
        self.assertEqual(len(tasks), 158)
        self.assertTrue(all(task.group.startswith("lite_") for task in tasks))

    def test_lite_anchor_root_reuses_only_stage_a_and_remaps_dependencies(self) -> None:
        anchor_root = Path("/verified-anchors")
        tasks = select_tasks(
            scope="lite",
            groups=None,
            artifact_root=Path("/artifacts"),
            python_exe="python",
            lite_anchor_root=anchor_root,
        )
        anchors = [task for task in tasks if task.group == "lite_anchors"]
        self.assertEqual(len(anchors), 5)
        self.assertTrue(all(task.stage == "stage_b_test" for task in anchors))
        self.assertTrue(
            all(task.compute_kind == "stage_b_test" for task in anchors)
        )
        self.assertTrue(
            all(
                task.config["base_experiment_dir"]
                == str(
                    anchor_root.resolve()
                    / f"scar_main_{task.dataset.lower()}_seed42"
                )
                for task in anchors
            )
        )
        self.assertTrue(
            all(
                Path(task.command[1]).name == "reuse_stage_a.py"
                for task in anchors
            )
        )
        counts = {
            kind: sum(task.compute_kind == kind for task in tasks)
            for kind in ("full_model_fit", "stage_b_test", "analysis")
        }
        self.assertEqual(
            counts,
            {"full_model_fit": 25, "stage_b_test": 75, "analysis": 58},
        )
        anchor_ids = {task.run_id for task in anchors}
        self.assertTrue(
            all(
                any(dependency in anchor_ids for dependency in task.dependencies)
                for task in tasks
                if task.group == "lite_efficiency_source"
            )
        )


class ManifestContractTests(unittest.TestCase):
    def test_stable_run_id_ignores_dictionary_order(self) -> None:
        first = stable_run_id(
            method="SCAR",
            dataset="MSL",
            seed=42,
            stage="full",
            config={"a": 1, "b": [2, 3]},
        )
        second = stable_run_id(
            method="SCAR",
            dataset="MSL",
            seed=42,
            stage="full",
            config={"b": [2, 3], "a": 1},
        )
        self.assertEqual(first, second)

    def test_complete_artifact_requires_every_declared_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = RunSpec(
                method="SCAR",
                dataset="MSL",
                seed=42,
                stage="analysis",
                group="test",
                compute_kind="analysis",
                config={},
                command=("python", "-V"),
                artifact_dir=root,
                required_artifacts=("scores.npy", "metrics.json"),
            )
            self.assertFalse(artifact_is_complete(task))
            np.save(root / "scores.npy", np.asarray([0.1, 0.2]))
            self.assertFalse(artifact_is_complete(task))
            (root / "metrics.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertTrue(artifact_is_complete(task))
            self.assertFalse(artifact_is_reusable(task))

    def test_corrupt_numpy_and_misaligned_score_artifacts_are_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = RunSpec(
                method="Baseline",
                dataset="MSL",
                seed=42,
                stage="full",
                group="test",
                compute_kind="full_model_fit",
                config={},
                command=("python", "-V"),
                artifact_dir=root,
                required_artifacts=("scores.npy", "labels.npy"),
            )
            (root / "scores.npy").write_bytes(b"not-a-numpy-file")
            np.save(root / "labels.npy", np.asarray([0, 1]))
            self.assertFalse(artifact_is_complete(task))
            np.save(root / "scores.npy", np.asarray([0.1]))
            self.assertFalse(artifact_is_complete(task))
            np.save(root / "scores.npy", np.asarray([0.1, 0.2]))
            self.assertTrue(artifact_is_complete(task))

    def test_scar_and_tep_score_lengths_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scar = RunSpec(
                method="SCAR",
                dataset="MSL",
                seed=42,
                stage="full",
                group="test",
                compute_kind="full_model_fit",
                config={},
                command=("python", "-V"),
                artifact_dir=root,
                required_artifacts=(
                    "test_scores_selected.npy",
                    "test_scores_cdf_mean.npy",
                    "test_diagnostic_scores.npz",
                ),
            )
            np.save(root / "test_scores_selected.npy", np.asarray([0.1, 0.2]))
            np.save(root / "test_scores_cdf_mean.npy", np.asarray([0.1]))
            np.savez(root / "test_diagnostic_scores.npz", labels=np.asarray([0, 1]))
            self.assertFalse(artifact_is_complete(scar))
            np.save(root / "test_scores_cdf_mean.npy", np.asarray([0.1, 0.2]))
            self.assertTrue(artifact_is_complete(scar))

            tep = RunSpec(
                method="SCAR",
                dataset="TEP",
                seed=42,
                stage="full",
                group="test",
                compute_kind="full_model_fit",
                config={},
                command=("python", "-V"),
                artifact_dir=root,
                required_artifacts=(
                    "test_sequence_scores_selected.npy",
                    "test_sequence_scores.csv",
                ),
            )
            np.save(root / "test_sequence_scores_selected.npy", np.asarray([0.1, 0.2]))
            (root / "test_sequence_scores.csv").write_text(
                "name,label,score\none,0,0.1\n",
                encoding="utf-8",
            )
            self.assertFalse(artifact_is_complete(tep))
            (root / "test_sequence_scores.csv").write_text(
                "name,label,score\none,0,0.1\ntwo,1,0.2\n",
                encoding="utf-8",
            )
            self.assertTrue(artifact_is_complete(tep))

    def test_parallel_executor_respects_declared_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_dir = root / "first"
            second_dir = root / "second"
            first = RunSpec(
                method="Toy",
                dataset="Toy",
                seed=42,
                stage="first",
                group="test",
                compute_kind="analysis",
                config={"step": 1},
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path(r'{first_dir / 'ready.json'}').parent.mkdir(parents=True, exist_ok=True); "
                        f"Path(r'{first_dir / 'ready.json'}').write_text('{{}}')"
                    ),
                ),
                artifact_dir=first_dir,
                required_artifacts=("ready.json",),
            )
            second = RunSpec(
                method="Toy",
                dataset="Toy",
                seed=42,
                stage="second",
                group="test",
                compute_kind="analysis",
                config={"step": 2},
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"assert Path(r'{first_dir / 'ready.json'}').is_file(); "
                        f"Path(r'{second_dir / 'done.json'}').parent.mkdir(parents=True, exist_ok=True); "
                        f"Path(r'{second_dir / 'done.json'}').write_text('{{}}')"
                    ),
                ),
                artifact_dir=second_dir,
                required_artifacts=("done.json",),
                dependencies=(first.run_id,),
            )
            results = execute_tasks(
                [second, first], repo_root=root, max_parallel=2
            )
            self.assertEqual([row["status"] for row in results], ["completed", "completed"])

    def test_parallel_executor_assigns_distinct_gpu_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = []
            for index in range(2):
                output = root / f"task_{index}"
                tasks.append(
                    RunSpec(
                        method="Toy",
                        dataset=f"Toy{index}",
                        seed=42,
                        stage="analysis",
                        group="test",
                        compute_kind="analysis",
                        config={"index": index},
                        command=(
                            sys.executable,
                            "-c",
                            (
                                "import json, os; from pathlib import Path; "
                                f"Path(r'{output / 'gpu.json'}').parent.mkdir(parents=True, exist_ok=True); "
                                f"Path(r'{output / 'gpu.json'}').write_text(json.dumps("
                                "{'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}))"
                            ),
                        ),
                        artifact_dir=output,
                        required_artifacts=("gpu.json",),
                    )
                )
            results = execute_tasks(
                tasks,
                repo_root=root,
                max_parallel=2,
                gpu_devices=("2", "3"),
            )
            self.assertEqual({row["assigned_gpu"] for row in results}, {"2", "3"})
            observed = {
                json.loads((task.artifact_dir / "gpu.json").read_text())["gpu"]
                for task in tasks
            }
            self.assertEqual(observed, {"2", "3"})

    def test_gpu_slots_reject_oversubscription(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            execute_tasks(
                [],
                repo_root=Path("."),
                max_parallel=2,
                gpu_devices=("0",),
            )

    def test_plan_summary_separates_recomputation_kinds(self) -> None:
        tasks = build_p0_tasks(Path("/artifacts"), python_exe="python")
        summary = summarize_tasks(tasks)
        self.assertEqual(summary["full_model_fit"], 659)
        self.assertEqual(summary["stage_b_test"], 180)
        self.assertGreater(summary["total"], 0)

    def test_executor_writes_auditable_record_and_validates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = RunSpec(
                method="Toy",
                dataset="Toy",
                seed=42,
                stage="analysis",
                group="test",
                compute_kind="analysis",
                config={"value": 1},
                command=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('result.json').write_text('{\"ok\": true}')",
                ),
                artifact_dir=root,
                required_artifacts=("result.json",),
            )
            result = execute_task(task, repo_root=root)
            self.assertEqual(result["status"], "completed")
            record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
            self.assertEqual(record["run_id"], task.run_id)
            self.assertEqual(record["return_code"], 0)
            self.assertTrue(record["artifacts_complete"])
            self.assertTrue(artifact_is_reusable(task))
            self.assertEqual(len(record["source_hash"]), 64)
            self.assertIn("data_hash", record)
            record["source_hash"] = "stale"
            (root / "run_record.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )
            self.assertFalse(execution_is_reusable(task, root))

    def test_resume_rejects_artifacts_from_a_different_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = RunSpec(
                method="Toy",
                dataset="Toy",
                seed=42,
                stage="analysis",
                group="test",
                compute_kind="analysis",
                config={"value": 2},
                command=(sys.executable, "-c", "raise SystemExit(0)"),
                artifact_dir=root,
                required_artifacts=("result.json",),
            )
            (root / "result.json").write_text("{}", encoding="utf-8")
            (root / "run_record.json").write_text(
                json.dumps(
                    {
                        "run_id": "different",
                        "status": "completed",
                        "artifacts_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(artifact_is_reusable(task))

    def test_resume_propagates_to_model_command_and_shell_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / "run.py"
            result_path = root / "artifacts" / "result.json"
            script.write_text(
                (
                    "import json, os, sys\n"
                    "from pathlib import Path\n"
                    f"path = Path(r'{result_path}')\n"
                    "path.parent.mkdir(parents=True, exist_ok=True)\n"
                    "path.write_text(json.dumps({'argv': sys.argv, "
                    "'resume_env': os.environ.get('RESUME')}))\n"
                ),
                encoding="utf-8",
            )
            task = RunSpec(
                method="SCAR",
                dataset="Toy",
                seed=42,
                stage="full",
                group="test",
                compute_kind="full_model_fit",
                config={},
                command=(sys.executable, str(script)),
                artifact_dir=result_path.parent,
                required_artifacts=("result.json",),
            )
            record = execute_task(task, repo_root=root, resume=True)
            self.assertEqual(record["status"], "completed")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["argv"][-2:], ["--resume", "1"])
            self.assertEqual(payload["resume_env"], "1")

    def test_write_plan_is_deterministic_and_jsonl_addressable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            tasks = build_p0_tasks(Path("/artifacts"), python_exe="python")
            payload = write_plan(tasks, output)
            self.assertEqual(payload["summary"]["full_model_fit"], 659)
            lines = (output / "plan.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(tasks))
            self.assertEqual(json.loads(lines[0])["run_id"], tasks[0].run_id)

    def test_cli_group_selection_does_not_pull_unrelated_work(self) -> None:
        tasks = select_tasks(
            scope="p0",
            groups={"e9"},
            artifact_root=Path("/artifacts"),
            python_exe="python",
        )
        self.assertEqual(len(tasks), 25)
        self.assertEqual({task.group for task in tasks}, {"e9"})

    def test_ac_core_scope_and_shard_filters(self) -> None:
        tasks = select_tasks(
            scope="ac-core",
            groups=None,
            artifact_root=Path("/artifacts"),
            python_exe="python",
        )
        self.assertEqual({task.group for task in tasks}, AC_CORE_GROUPS)
        self.assertEqual(len(tasks), 102)
        self.assertEqual(
            sum(task.compute_kind == "full_model_fit" for task in tasks), 51
        )
        self.assertEqual(
            sum(task.compute_kind == "stage_b_test" for task in tasks), 25
        )
        selected_ids = {task.run_id for task in tasks}
        self.assertTrue(
            all(
                dependency in selected_ids
                for task in tasks
                for dependency in task.dependencies
            )
        )

        shard = select_tasks(
            scope="ac-core",
            groups={"baselines"},
            artifact_root=Path("/artifacts"),
            python_exe="python",
            methods={"PaAno", "PGRF-Net"},
            datasets={"SMD", "SWAT"},
        )
        self.assertEqual(len(shard), 4)
        self.assertEqual({task.method for task in shard}, {"PaAno", "PGRF-Net"})
        self.assertEqual({task.dataset for task in shard}, {"SMD", "SWAT"})

    def test_data_root_map_rewrites_commands_and_dependency_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapping = root / "roots.json"
            mapping.write_text(
                json.dumps({"anomaly_detect": str(root / "detect")}),
                encoding="utf-8",
            )
            tasks = build_p0_tasks(root / "artifacts", python_exe="python")
            anchor = next(
                task
                for task in tasks
                if task.group == "anchors" and task.dataset == "MSL"
            )
            e9 = next(
                task
                for task in tasks
                if task.group == "e9"
                and task.dataset == "MSL"
                and task.config["clean_ratio"] == 0.0
            )
            rewritten = _apply_data_root_map([anchor, e9], mapping)
            new_anchor, new_e9 = rewritten
            self.assertIn(str(root / "detect"), new_anchor.command)
            self.assertEqual(
                new_anchor.config["data_root_override"], str(root / "detect")
            )
            self.assertIn(new_anchor.run_id, new_e9.dependencies)

    def test_environment_freeze_always_writes_auditable_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "freeze"
            freeze_environment(output, repo_root=Path(temp), include_package_commands=False)
            for name in (
                "environment.json",
                "git_diff.patch",
                "untracked_source_files.json",
                "pip_freeze.txt",
                "conda_explicit.txt",
                "nvidia_smi.txt",
            ):
                self.assertTrue((output / name).is_file(), name)


class BudgetMatchingTests(unittest.TestCase):
    def test_budget_selection_uses_log_error_then_declared_tiebreakers(self) -> None:
        target = {"parameters": 100.0, "bank_bytes": 1000.0, "latency_ms": 10.0}
        candidates = [
            {
                "name": "larger",
                "parameters": 100.0,
                "bank_bytes": 1000.0,
                "latency_ms": 11.0,
                "config_size": 2,
            },
            {
                "name": "smaller",
                "parameters": 100.0,
                "bank_bytes": 1000.0,
                "latency_ms": 10.0 / 1.1,
                "config_size": 1,
            },
        ]
        selected = select_budget_candidate(target, candidates, tolerance=0.15)
        self.assertEqual(selected["candidate"]["name"], "smaller")
        self.assertTrue(selected["within_tolerance"])

    def test_budget_selection_falls_back_to_minimum_log_error(self) -> None:
        target = {"parameters": 100.0, "bank_bytes": 1000.0, "latency_ms": 10.0}
        candidates = [
            {
                "name": "far",
                "parameters": 200.0,
                "bank_bytes": 2000.0,
                "latency_ms": 20.0,
                "config_size": 1,
            },
            {
                "name": "near",
                "parameters": 120.0,
                "bank_bytes": 1200.0,
                "latency_ms": 12.0,
                "config_size": 2,
            },
        ]
        selected = select_budget_candidate(target, candidates, tolerance=0.15)
        self.assertEqual(selected["candidate"]["name"], "near")
        self.assertFalse(selected["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
